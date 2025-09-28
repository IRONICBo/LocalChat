from fastapi import FastAPI, HTTPException
import requests
import hashlib
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Union
from uuid import uuid4
import litellm
litellm._turn_on_debug()

from litellm import completion
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import OperatorConfig
import os


# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_proxy.log"),
        logging.StreamHandler()
    ]
)

# Initialize the analyzer and anonymizer engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
deanonymizer = DeanonymizeEngine()
os.environ['GEMINI_API_KEY'] = "AIzaSyD0GpisRje3z9tPHxVjuSQPDnBP0W0pLGA"

# Create a FastAPI application instance
app = FastAPI(title="Enhanced LLM Proxy Server")

# Default model configuration
DEFAULT_MODEL = "gemini/gemini-2.0-flash"

# In-memory cache for sensitive data mapping
# Format: {hash_value: original_value}
sensitive_data_cache = {}

# Track entity counts for numbering
entity_counter = {}

def generate_hash(value: str, entity_type: str) -> str:
    """Generate a unique hash for sensitive data with entity type prefix"""
    entity_counter[entity_type] = entity_counter.get(entity_type, 0) + 1
    return f"{entity_type}_{entity_counter[entity_type]}"

def anonymize_text(text: str) -> Tuple[str, Dict[str, str]]:
    """Anonymize text with entity numbering"""
    # Analyze the text to find PII entities
    results = analyzer.analyze(text=text, language='en')

    # Create a mapping for this operation
    operation_mapping = {}

    # Prepare anonymization operations with numbered entities
    operations = []
    for result in results:
        entity_type = result.entity_type
        original_value = text[result.start:result.end]

        # Generate numbered identifier
        numbered_entity = generate_hash(original_value, entity_type)

        # Store mapping
        operation_mapping[numbered_entity] = original_value

        # Create operator config
        operations.append(
            OperatorConfig(
                "replace",
                {"new_value": f"[{numbered_entity}]"}
            )
        )

    # Anonymize with custom operations
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={result.entity_type: operations[i] for i, result in enumerate(results)}
    )

    # Update global cache
    sensitive_data_cache.update(operation_mapping)

    return anonymized_result.text, operation_mapping

def mask_sensitive_info(content: str) -> str:
    """
    Identify and mask sensitive information in content with numbered entities
    Returns masked content
    """
    masked_text, _ = anonymize_text(content)
    return masked_text

def restore_sensitive_info(content: str) -> str:
    """Restore original sensitive information from numbered placeholders"""
    import re
    # Find all numbered entity patterns in content
    matches = re.findall(r"\[([A-Z_]+_\d+)\]", content)

    for entity_id in matches:
        if entity_id in sensitive_data_cache:
            content = content.replace(f"[{entity_id}]", sensitive_data_cache[entity_id])

    return content

# Define Pydantic models
class Message(BaseModel):
    """Represents a single message in the chat conversation"""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Represents a chat completion request"""
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = False

@app.post("/api/chat", response_model=Dict)
async def proxy_chat(request: ChatRequest):
    """
    Enhanced proxy endpoint with:
    - Sensitive information masking before sending to LLM
    - Numbered entity tracking for repeated values
    - Restoration of sensitive information in responses
    - Logging of original LLM responses
    """
    # Generate a request ID for tracking
    request_id = str(uuid4())[:8]
    logging.info(f"Processing request {request_id}")

    try:
        # Process messages to mask sensitive information
        processed_messages = []

        for msg in request.messages:
            masked_content = mask_sensitive_info(msg.content)
            logging.info(
                f"Request {request_id} - Masked content: {masked_content}"
            )
            processed_messages.append({
                "role": msg.role,
                "content": masked_content
            })

        # Get LLM response
        response = completion(
            model=DEFAULT_MODEL,
            messages=processed_messages,
        )

        # Get and log the original LLM response
        llm_response = response.choices[0]['message']['content']
        logging.info(f"Request {request_id} - Original LLM response: {llm_response}")

        # Restore sensitive information in the response
        restored_response = restore_sensitive_info(llm_response)
        response.choices[0]['message']['content'] = restored_response

        return response.to_dict()

    except requests.exceptions.ConnectionError:
        logging.error(f"Request {request_id} - Unable to connect to LLM API")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to LLM API"
        )
    except Exception as e:
        logging.error(f"Request {request_id} - Proxy server error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Proxy server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
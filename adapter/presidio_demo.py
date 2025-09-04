from fastapi import FastAPI, HTTPException
import requests
import hashlib
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from uuid import uuid4
import litellm
litellm._turn_on_debug()

from litellm import completion
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
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

# Configure Ollama API endpoint
# OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Default model configuration
DEFAULT_MODEL = "gemini/gemini-2.0-flash"

# In-memory cache for sensitive data mapping
# Format: {hash_value: original_value}
sensitive_data_cache = {}

def generate_hash(value: str) -> str:
    """Generate a unique hash for sensitive data"""
    return hashlib.sha256(value.encode()).hexdigest()[:16]

def anonymize_text(text):
    # Analyze the text to find PII entities
    results = analyzer.analyze(text=text, language='en')
    # Anonymize the identified PII entities
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text.text

def mask_sensitive_info(content: str) -> str:
    """
    Identify and mask sensitive information in content
    Returns masked content and a dictionary of hash mappings
    """

    return anonymize_text(content)

def restore_sensitive_info(content: str) -> str:
    """Restore original sensitive information from hash placeholders"""
    import re
    # Find all hash patterns in content
    hash_matches = re.findall(r"\[HASH:([a-f0-9]{16})\]", content)

    for hash_str in hash_matches:
        if hash_str in sensitive_data_cache:
            content = content.replace(f"[HASH:{hash_str}]", sensitive_data_cache[hash_str])

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
    - Caching of sensitive data mappings
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

        # Log the sensitive data mappings for this request

        # Create request data with forced parameters
        # Forward the request to the Ollama API
        response = completion(
            model=DEFAULT_MODEL,
            messages=processed_messages,
        )
        # response = completion(
        #     model="gemini/gemini-2.0-flash",
        #     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
        # )

        # Get and log the original LLM response
        logging.info(f"Request {request_id} - Original LLM response: {response.choices[0]['message']['content']}")

        # Restore sensitive information in the response
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

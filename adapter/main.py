from fastapi import FastAPI, HTTPException
import requests
import hashlib
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from uuid import uuid4

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_proxy.log"),
        logging.StreamHandler()
    ]
)

# Create a FastAPI application instance
app = FastAPI(title="Enhanced LLM Proxy Server")

# Configure Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Default model configuration
DEFAULT_MODEL = "qwen2:0.5b"

# In-memory cache for sensitive data mapping
# Format: {hash_value: original_value}
sensitive_data_cache = {}

# Sensitive patterns to detect (can be expanded as needed)
SENSITIVE_PATTERNS = {
    # Example patterns - adjust according to your needs
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "id_card": r'\b\d{17}[\dXx]\b'  # Chinese ID card example
}

def generate_hash(value: str) -> str:
    """Generate a unique hash for sensitive data"""
    return hashlib.sha256(value.encode()).hexdigest()[:16]

def mask_sensitive_info(content: str) -> Tuple[str, Dict]:
    """
    Identify and mask sensitive information in content
    Returns masked content and a dictionary of hash mappings
    """
    import re
    mappings = {}

    # Check for email patterns
    emails = re.findall(SENSITIVE_PATTERNS["email"], content)
    for email in emails:
        if email not in [v for v in sensitive_data_cache.values()]:
            hash_str = generate_hash(email)
            sensitive_data_cache[hash_str] = email
            mappings[hash_str] = email
        else:
            hash_str = next(k for k, v in sensitive_data_cache.items() if v == email)
        content = content.replace(email, f"EMAIL[HASH:{hash_str}]")

    # Check for phone numbers
    phones = re.findall(SENSITIVE_PATTERNS["phone"], content)
    for phone in phones:
        if phone not in [v for v in sensitive_data_cache.values()]:
            hash_str = generate_hash(phone)
            sensitive_data_cache[hash_str] = phone
            mappings[hash_str] = phone
        else:
            hash_str = next(k for k, v in sensitive_data_cache.items() if v == phone)
        content = content.replace(phone, f"PHONE[HASH:{hash_str}]")

    # Check for credit card numbers
    cards = re.findall(SENSITIVE_PATTERNS["credit_card"], content)
    for card in cards:
        if card not in [v for v in sensitive_data_cache.values()]:
            hash_str = generate_hash(card)
            sensitive_data_cache[hash_str] = card
            mappings[hash_str] = card
        else:
            hash_str = next(k for k, v in sensitive_data_cache.items() if v == card)
        content = content.replace(card, f"CREDIT_CARD[HASH:{hash_str}]")

    # Check for ID cards
    ids = re.findall(SENSITIVE_PATTERNS["id_card"], content)
    for id in ids:
        if id not in [v for v in sensitive_data_cache.values()]:
            hash_str = generate_hash(id)
            sensitive_data_cache[hash_str] = id
            mappings[hash_str] = id
        else:
            hash_str = next(k for k, v in sensitive_data_cache.items() if v == id)
        content = content.replace(id, f"ID_CARD[HASH:{hash_str}]")

    return content, mappings

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
        all_mappings = {}

        for msg in request.messages:
            masked_content, mappings = mask_sensitive_info(msg.content)
            processed_messages.append({
                "role": msg.role,
                "content": masked_content
            })
            all_mappings.update(mappings)

        # Log the sensitive data mappings for this request
        if all_mappings:
            logging.info(f"Request {request_id} - Sensitive data mappings: {json.dumps(all_mappings)}")

        # Create request data with forced parameters
        request_data = {
            "model": request.model if request.model else DEFAULT_MODEL,
            "messages": processed_messages,
            "stream": False  # Force non-streaming
        }
        logging.info(f"Request {request_id} - Forwarding request data: {json.dumps(request_data)}")

        # Forward the request to the Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json=request_data,
            stream=False
        )

        # Check if the request to Ollama was successful
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM API request failed: {response.text}"
            )

        # Get and log the original LLM response
        original_response = response.json()
        logging.info(f"Request {request_id} - Original LLM response: {json.dumps(original_response)}")

        # Restore sensitive information in the response
        if "message" in original_response and "content" in original_response["message"]:
            original_response["message"]["content"] = restore_sensitive_info(original_response["message"]["content"])

        return original_response

    except requests.exceptions.ConnectionError:
        logging.error(f"Request {request_id} - Unable to connect to LLM API")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to LLM API at: {OLLAMA_API_URL}"
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

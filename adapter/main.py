from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
from typing import List, Dict, Optional

# Create a FastAPI application instance
app = FastAPI(title="LLM Proxy Server")

# Configure Ollama API endpoint - modify this according to your actual Ollama deployment
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Define Pydantic models to validate request structure
# These models match the Ollama API specification
class Message(BaseModel):
    """Represents a single message in the chat conversation"""
    role: str  # Can be "user", "assistant", or "system"
    content: str  # The actual message content

class ChatRequest(BaseModel):
    """Represents a chat completion request"""
    model: str  # Name of the model to use (e.g., "llama2", "mistral")
    messages: List[Message]  # List of messages in the conversation history
    stream: Optional[bool] = False  # Whether to stream the response or return it all at once

@app.post("/api/chat", response_model=Dict)
async def proxy_chat(request: ChatRequest):
    """
    Proxy endpoint that forwards chat requests to the Ollama API.
    This endpoint maintains the same path as the original Ollama API.
    It receives client requests, forwards them to the LLM API,
    waits for the complete response, and returns it to the client.
    """
    try:
        # Forward the request to the Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json=request.dict(),
            stream=request.stream  # Maintain streaming behavior as requested
        )

        # Check if the request to Ollama was successful
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM API request failed: {response.text}"
            )

        # Return the response from Ollama to the client
        return response.json()

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to LLM API at: {OLLAMA_API_URL}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Proxy server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000, accessible from any network interface
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException, Depends
import requests
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import uuid4

from sqlalchemy.orm import Session
from models import (
    SessionInfo, ConversationHistory, engine, Base
)
from pii_engine import PIIMaskEngine, PIIRecoverEngine

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("llm_proxy.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Create a FastAPI application instance
app = FastAPI(title="Enhanced LLM Proxy Server with PII Protection")

# Configure Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Default model configuration
DEFAULT_MODEL = "qwen2:0.5b"

# PII detection method: "Regex", "Presidio", "LLM", "E2E"
PII_DETECTION_METHOD = "E2E"  # Default to E2E for best accuracy


# Dependency to get DB session
def get_db():
    """Get database session"""
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    session_id: Optional[str] = None  # Optional session ID for continuity
    enable_pii_protection: Optional[bool] = True  # Enable/disable PII protection


class ChatResponse(BaseModel):
    """Represents a chat completion response"""
    message: Dict
    session_id: str
    pii_detected: bool
    pii_entities_count: int
    detection_method: str


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced LLM Proxy Server with PII Protection",
        "version": "1.0.0",
        "pii_detection_method": PII_DETECTION_METHOD,
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health",
            "session": "/api/session/{session_id}",
            "entities": "/api/session/{session_id}/entities"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama API
        response = requests.get(f"{OLLAMA_API_URL.rsplit('/', 2)[0]}/api/tags", timeout=2)
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        ollama_status = "unreachable"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ollama_api": ollama_status,
        "pii_detection": PII_DETECTION_METHOD
    }


@app.post("/api/chat", response_model=Dict)
async def proxy_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Enhanced proxy endpoint with PII protection

    Flow:
    1. Extract PII from user messages (Mask)
    2. Replace PII with placeholders
    3. Store original PII and mappings in database
    4. Send masked messages to LLM
    5. Receive LLM response
    6. Restore PII in response if needed (Recover)
    7. Store conversation history
    8. Return response to user
    """
    # Generate request ID for tracking
    request_id = str(uuid4())[:8]
    logger.info(f"[{request_id}] Processing chat request")

    try:
        # Get or create session
        session_id = request.session_id if request.session_id else str(uuid4())

        session = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
        if not session:
            session = SessionInfo(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            logger.info(f"[{request_id}] Created new session: {session_id}")
        else:
            session.last_activity = datetime.utcnow()
            db.commit()
            logger.info(f"[{request_id}] Using existing session: {session_id}")

        # Initialize PII engines
        mask_engine = PIIMaskEngine(db, detection_method=PII_DETECTION_METHOD)
        recover_engine = PIIRecoverEngine(db)

        # Process messages to mask PII
        processed_messages = []
        all_entities_info = []

        for msg in request.messages:
            conversation_id = str(uuid4())

            if request.enable_pii_protection:
                # Mask PII in message content
                masked_content, entities_info = mask_engine.mask_text(
                    text=msg.content,
                    session_id=session_id,
                    conversation_id=conversation_id
                )
                all_entities_info.extend(entities_info)

                logger.info(
                    f"[{request_id}] Masked {len(entities_info)} entities in {msg.role} message"
                )
            else:
                masked_content = msg.content
                logger.info(f"[{request_id}] PII protection disabled, using original content")

            # Store conversation history
            conversation = ConversationHistory(
                conversation_id=conversation_id,
                session_id=session_id,
                role=msg.role,
                original_content=msg.content,
                masked_content=masked_content,
                created_at=datetime.utcnow()
            )
            db.add(conversation)

            processed_messages.append({"role": msg.role, "content": masked_content})

        db.commit()

        # Log the sensitive data mappings for this request
        if all_entities_info:
            logger.info(
                f"[{request_id}] Total PII entities detected: {len(all_entities_info)}"
            )
            logger.debug(
                f"[{request_id}] Entity types: {[e['entity_type'] for e in all_entities_info]}"
            )

        # Create request data
        request_data = {
            "model": request.model if request.model else DEFAULT_MODEL,
            "messages": processed_messages,
            "stream": False,  # Force non-streaming
        }

        logger.info(f"[{request_id}] Forwarding to Ollama API")

        # Forward the request to the Ollama API
        response = requests.post(OLLAMA_API_URL, json=request_data, stream=False, timeout=60)

        # Check if the request to Ollama was successful
        if response.status_code != 200:
            logger.error(f"[{request_id}] LLM API error: {response.status_code}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM API request failed: {response.text}",
            )

        # Get the original LLM response
        original_response = response.json()
        logger.info(f"[{request_id}] Received response from Ollama")

        # Restore PII in the response if enabled
        if request.enable_pii_protection and "message" in original_response and "content" in original_response["message"]:
            original_content = original_response["message"]["content"]
            recovered_content = recover_engine.recover_text(
                masked_text=original_content,
                session_id=session_id
            )
            original_response["message"]["content"] = recovered_content

            logger.info(f"[{request_id}] Recovered PII in response")

            # Store assistant response in conversation history
            assistant_conversation_id = str(uuid4())
            assistant_conversation = ConversationHistory(
                conversation_id=assistant_conversation_id,
                session_id=session_id,
                role="assistant",
                original_content=recovered_content,
                masked_content=original_content,
                created_at=datetime.utcnow()
            )
            db.add(assistant_conversation)
            db.commit()

        # Add metadata to response
        original_response["session_id"] = session_id
        original_response["pii_detected"] = len(all_entities_info) > 0
        original_response["pii_entities_count"] = len(all_entities_info)
        original_response["detection_method"] = PII_DETECTION_METHOD if request.enable_pii_protection else "disabled"

        logger.info(f"[{request_id}] Request completed successfully")

        return original_response

    except requests.exceptions.ConnectionError:
        logger.error(f"[{request_id}] Unable to connect to LLM API")
        raise HTTPException(
            status_code=503, detail=f"Unable to connect to LLM API at: {OLLAMA_API_URL}"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Proxy server error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Proxy server error: {str(e)}")


@app.get("/api/session/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """
    Get session information and conversation history

    Args:
        session_id: Session ID to query

    Returns:
        Session info with conversation history
    """
    # Get session
    session = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get conversation history
    conversations = db.query(ConversationHistory).filter(
        ConversationHistory.session_id == session_id
    ).order_by(ConversationHistory.created_at).all()

    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "conversations": [{
            "conversation_id": c.conversation_id,
            "role": c.role,
            "content": c.original_content,
            "created_at": c.created_at.isoformat()
        } for c in conversations]
    }


@app.get("/api/session/{session_id}/entities")
async def get_session_entities(session_id: str, db: Session = Depends(get_db)):
    """
    Get all PII entities detected in a session

    Args:
        session_id: Session ID to query

    Returns:
        List of detected PII entities
    """
    # Check if session exists
    session = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get entities using recover engine
    recover_engine = PIIRecoverEngine(db)
    entities = recover_engine.get_session_entities(session_id)

    return {
        "session_id": session_id,
        "total_entities": len(entities),
        "entities": entities
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """
    Delete a session and all associated data

    Args:
        session_id: Session ID to delete

    Returns:
        Deletion status
    """
    from models import MaskMapping, SensitiveEntity

    # Check if session exists
    session = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete associated data (cascade)
    db.query(ConversationHistory).filter(ConversationHistory.session_id == session_id).delete()
    db.query(MaskMapping).filter(MaskMapping.session_id == session_id).delete()
    db.query(SensitiveEntity).filter(SensitiveEntity.session_id == session_id).delete()

    # Delete session
    db.delete(session)
    db.commit()

    logger.info(f"Deleted session: {session_id}")

    return {
        "status": "success",
        "message": f"Session {session_id} and all associated data deleted"
    }


if __name__ == "__main__":
    import uvicorn

    # Start the server on port 8000
    logger.info("Starting Enhanced LLM Proxy Server with PII Protection")
    logger.info(f"PII Detection Method: {PII_DETECTION_METHOD}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

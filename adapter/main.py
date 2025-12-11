from fastapi import FastAPI, HTTPException, Depends
import requests
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from uuid import uuid4
import os
import time

from sqlalchemy.orm import Session
from models import (
    SessionInfo, ConversationHistory, engine, Base
)
from pii_engine import PIIMaskEngine, PIIRecoverEngine
from config import PIIConfig, DetectionStrategy

# Import Smart Mode and Text Chunking modules
try:
    from smart_mode import SmartModeEngine, SmartModeSelector, TextDomain, InputSourceType
    SMART_MODE_AVAILABLE = True
except ImportError:
    SMART_MODE_AVAILABLE = False

try:
    from text_chunking import TextChunker, ChunkingConfig, ChunkingStrategy, ChunkedPIIProcessor
    TEXT_CHUNKING_AVAILABLE = True
except ImportError:
    TEXT_CHUNKING_AVAILABLE = False

# Configure logging
log_level = getattr(logging, PIIConfig.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(PIIConfig.LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Create a FastAPI application instance
app = FastAPI(
    title="Enhanced LLM Proxy Server with PII Protection",
    description="PII detection and masking proxy with support for multiple models and strategies",
    version="2.0.0"
)

# Get configuration from environment or use defaults
LLM_BACKEND = PIIConfig.LLM_BACKEND
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", f"{PIIConfig.OLLAMA_API_URL}/api/chat")
OPENAI_API_KEY = PIIConfig.OPENAI_API_KEY
OPENAI_API_BASE = PIIConfig.OPENAI_API_BASE
OPENAI_CHAT_ENDPOINT = PIIConfig.OPENAI_CHAT_ENDPOINT
OPENAI_MODEL = PIIConfig.OPENAI_MODEL
PII_DETECTION_METHOD = PIIConfig.PII_DETECTION_METHOD
DETECTION_STRATEGY = PIIConfig.DETECTION_STRATEGY
DEFAULT_LLM_SIZE = PIIConfig.DEFAULT_LLM_SIZE

# Get default model from config
DEFAULT_MODEL_CONFIG = PIIConfig.get_llm_config(DEFAULT_LLM_SIZE)
DEFAULT_MODEL = DEFAULT_MODEL_CONFIG["model"]

# Determine which backend and endpoint to use
if LLM_BACKEND == "openai":
    LLM_API_URL = OPENAI_CHAT_ENDPOINT
    LLM_MODEL = OPENAI_MODEL
else:
    LLM_API_URL = OLLAMA_API_URL
    LLM_MODEL = DEFAULT_MODEL

logger.info(f"Starting Enhanced LLM Proxy Server")
logger.info(f"LLM Backend: {LLM_BACKEND}")
logger.info(f"LLM API URL: {LLM_API_URL}")
logger.info(f"LLM Model: {LLM_MODEL}")
logger.info(f"PII Detection Method: {PII_DETECTION_METHOD}")
logger.info(f"Detection Strategy: {DETECTION_STRATEGY}")


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
    """Represents an enhanced chat completion request with PII protection options"""
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = False
    session_id: Optional[str] = None  # Optional session ID for continuity
    enable_pii_protection: Optional[bool] = True  # Enable/disable PII protection
    detection_strategy: Optional[str] = None  # "high_recall", "balanced", "high_precision"
    model_size: Optional[str] = None  # "tiny", "small", "medium", "large", "xlarge"
    # New: Smart Mode options
    enable_smart_mode: Optional[bool] = False  # Auto-select strategy based on text
    source_type: Optional[str] = None  # "conversation", "document", "pdf", "email", "form"
    # New: Text chunking options for long text
    enable_chunking: Optional[bool] = None  # Auto-enabled for long text if None
    max_chunk_size: Optional[int] = 2000  # Max characters per chunk
    chunking_strategy: Optional[str] = None  # "sentence", "paragraph", "fixed_size", "adaptive"


class ChatResponse(BaseModel):
    """Represents a chat completion response"""
    message: Dict
    session_id: str
    pii_detected: bool
    pii_entities_count: int
    detection_method: str
    # New: Enhanced response fields
    smart_mode_used: Optional[bool] = None
    recommended_strategy: Optional[str] = None
    chunking_used: Optional[bool] = None
    chunks_processed: Optional[int] = None
    processing_time_ms: Optional[float] = None


# New: Request/Response models for standalone mask/unmask endpoints
class MaskRequest(BaseModel):
    """Request for standalone text masking"""
    text: str
    session_id: Optional[str] = None
    detection_strategy: Optional[str] = None
    model_size: Optional[str] = None
    enable_smart_mode: Optional[bool] = False
    source_type: Optional[str] = None
    enable_chunking: Optional[bool] = None
    max_chunk_size: Optional[int] = 2000


class MaskResponse(BaseModel):
    """Response for standalone text masking"""
    masked_text: str
    session_id: str
    entities: List[Dict[str, Any]]
    entities_count: int
    detection_method: str
    detection_strategy: str
    smart_mode_used: bool
    recommended_strategy: Optional[str] = None
    chunking_used: bool
    chunks_processed: Optional[int] = None
    processing_time_ms: float


class UnmaskRequest(BaseModel):
    """Request for standalone text unmasking"""
    masked_text: str
    session_id: str


class UnmaskResponse(BaseModel):
    """Response for standalone text unmasking"""
    original_text: str
    session_id: str
    entities_recovered: int
    processing_time_ms: float


class AnalyzeRequest(BaseModel):
    """Request for text analysis (Smart Mode)"""
    text: str
    source_type: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response for text analysis"""
    analysis: Dict[str, Any]
    recommendations: Dict[str, Any]
    estimated_pii_counts: Dict[str, int]


@app.get("/")
async def root():
    """Root endpoint with API information and configuration"""
    return {
        "service": "Enhanced LLM Proxy Server with PII Protection",
        "version": "2.1.0",
        "configuration": {
            "pii_detection_method": PII_DETECTION_METHOD,
            "detection_strategy": DETECTION_STRATEGY,
            "default_model": DEFAULT_MODEL,
            "default_model_size": DEFAULT_LLM_SIZE,
            "available_strategies": ["high_recall", "balanced", "high_precision"],
            "available_model_sizes": list(PIIConfig.LLM_MODEL_CONFIGS.keys()),
            "features": {
                "enhanced_regex": PIIConfig.ENABLE_ENHANCED_REGEX,
                "few_shot_learning": PIIConfig.ENABLE_FEW_SHOT,
                "chain_of_thought": PIIConfig.ENABLE_CHAIN_OF_THOUGHT,
                "aggressive_json_cleaning": PIIConfig.ENABLE_AGGRESSIVE_JSON_CLEANING,
                "smart_mode": SMART_MODE_AVAILABLE,
                "text_chunking": TEXT_CHUNKING_AVAILABLE
            }
        },
        "endpoints": {
            "chat": "/api/chat",
            "mask": "/api/mask",
            "unmask": "/api/unmask",
            "analyze": "/api/analyze",
            "health": "/health",
            "session": "/api/session/{session_id}",
            "entities": "/api/session/{session_id}/entities",
            "delete_session": "/api/session/{session_id}"
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

        # Initialize PII engines with user-specified or default configuration
        detection_strategy = request.detection_strategy or DETECTION_STRATEGY
        model_size = request.model_size or DEFAULT_LLM_SIZE

        mask_engine = PIIMaskEngine(
            db,
            detection_method=PII_DETECTION_METHOD,
            strategy=detection_strategy,
            model_size=model_size
        )
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

        # Create request data based on backend type
        if LLM_BACKEND == "openai":
            # OpenAI-compatible API format
            request_data = {
                "model": request.model if request.model else LLM_MODEL,
                "messages": processed_messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            api_url = LLM_API_URL
            logger.info(f"[{request_id}] Forwarding to OpenAI-compatible API: {api_url}")
        else:
            # Ollama API format
            request_data = {
                "model": request.model if request.model else LLM_MODEL,
                "messages": processed_messages,
                "stream": False,
            }
            headers = {"Content-Type": "application/json"}
            api_url = OLLAMA_API_URL
            logger.info(f"[{request_id}] Forwarding to Ollama API: {api_url}")

        # Forward the request to the LLM API
        response = requests.post(
            api_url,
            json=request_data,
            headers=headers,
            stream=False,
            timeout=120  # Increased timeout for remote APIs
        )

        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"[{request_id}] LLM API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM API request failed: {response.text}",
            )

        # Get the LLM response
        original_response = response.json()
        logger.info(f"[{request_id}] Received response from LLM API")

        # Extract content based on response format
        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        # Ollama format: {"message": {"content": "..."}}
        if LLM_BACKEND == "openai":
            if "choices" in original_response and len(original_response["choices"]) > 0:
                llm_content = original_response["choices"][0]["message"]["content"]
                # Convert to Ollama-like format for consistent handling
                original_response["message"] = {"content": llm_content, "role": "assistant"}
            else:
                logger.warning(f"[{request_id}] Unexpected OpenAI response format")
                llm_content = ""
        else:
            llm_content = original_response.get("message", {}).get("content", "")

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


# ==================== New: Standalone Mask/Unmask Endpoints ====================

@app.post("/api/mask", response_model=MaskResponse)
async def mask_text_endpoint(request: MaskRequest, db: Session = Depends(get_db)):
    """
    Standalone endpoint for text masking (PII detection and replacement)

    This endpoint provides direct access to the PII masking functionality
    without requiring LLM proxy. Useful for:
    - Pre-processing text before sending to external LLMs
    - Batch processing documents
    - Testing and debugging PII detection

    Features:
    - Smart Mode: Auto-select detection strategy based on text characteristics
    - Text Chunking: Handle long documents by splitting into manageable chunks
    - Multiple detection methods: Regex, Presidio, LLM, E2E

    Args:
        request: MaskRequest with text and configuration options

    Returns:
        MaskResponse with masked text, detected entities, and metadata
    """
    request_id = str(uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] Processing mask request")

    try:
        # Generate or use provided session ID
        session_id = request.session_id if request.session_id else str(uuid4())

        # Get or create session
        session = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
        if not session:
            session = SessionInfo(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            db.add(session)
            db.commit()

        # Initialize variables for response
        smart_mode_used = False
        recommended_strategy = None
        chunking_used = False
        chunks_processed = None

        # Determine detection strategy
        detection_strategy = request.detection_strategy or DETECTION_STRATEGY
        model_size = request.model_size or DEFAULT_LLM_SIZE

        # Smart Mode: Auto-select strategy based on text analysis
        if request.enable_smart_mode and SMART_MODE_AVAILABLE:
            smart_mode_used = True
            smart_engine = SmartModeEngine()

            # Parse source type if provided
            source_hint = None
            if request.source_type:
                try:
                    source_hint = InputSourceType(request.source_type)
                except ValueError:
                    pass

            # Analyze text and get recommendations
            analysis_result = smart_engine.process_text(request.text, source_hint)
            recommended_strategy = analysis_result["recommendations"]["strategy"]
            detection_strategy = recommended_strategy

            # Also use recommended model size if not explicitly specified
            if not request.model_size:
                model_size = analysis_result["recommendations"]["model_size"]

            logger.info(f"[{request_id}] Smart Mode: strategy={detection_strategy}, model={model_size}")

        # Initialize mask engine
        mask_engine = PIIMaskEngine(
            db,
            detection_method=PII_DETECTION_METHOD,
            strategy=detection_strategy,
            model_size=model_size
        )

        # Determine if chunking should be used
        text_length = len(request.text)
        should_chunk = request.enable_chunking

        # Auto-enable chunking for long text
        if should_chunk is None:
            should_chunk = text_length > request.max_chunk_size and TEXT_CHUNKING_AVAILABLE

        if should_chunk and TEXT_CHUNKING_AVAILABLE:
            chunking_used = True

            # Configure chunking
            chunk_strategy = ChunkingStrategy.ADAPTIVE
            if request.chunking_strategy:
                try:
                    chunk_strategy = ChunkingStrategy(request.chunking_strategy)
                except ValueError:
                    pass

            config = ChunkingConfig(
                strategy=chunk_strategy,
                max_chunk_size=request.max_chunk_size,
                overlap_size=min(100, request.max_chunk_size // 10)
            )
            chunker = TextChunker(config)
            processor = ChunkedPIIProcessor(chunker)

            # Process with chunking
            def extract_func(text):
                return mask_engine.extractor.extract(text)

            all_entities, chunk_metadata = processor.process_text(
                request.text,
                extract_func,
                merge_entities=True
            )
            chunks_processed = chunk_metadata["total_chunks"]

            # Mask using the merged entities
            masked_text, entities_info = mask_engine.mask_text_with_entities(
                text=request.text,
                entities=all_entities,
                session_id=session_id
            )

            logger.info(f"[{request_id}] Chunked processing: {chunks_processed} chunks, {len(entities_info)} entities")
        else:
            # Standard processing without chunking
            masked_text, entities_info = mask_engine.mask_text(
                text=request.text,
                session_id=session_id
            )

        processing_time = (time.time() - start_time) * 1000  # ms

        logger.info(f"[{request_id}] Masked {len(entities_info)} entities in {processing_time:.2f}ms")

        return MaskResponse(
            masked_text=masked_text,
            session_id=session_id,
            entities=entities_info,
            entities_count=len(entities_info),
            detection_method=PII_DETECTION_METHOD,
            detection_strategy=detection_strategy,
            smart_mode_used=smart_mode_used,
            recommended_strategy=recommended_strategy,
            chunking_used=chunking_used,
            chunks_processed=chunks_processed,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"[{request_id}] Mask error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mask error: {str(e)}")


@app.post("/api/unmask", response_model=UnmaskResponse)
async def unmask_text_endpoint(request: UnmaskRequest, db: Session = Depends(get_db)):
    """
    Standalone endpoint for text unmasking (PII recovery)

    Recovers original PII values from masked text using session mappings.
    Use this after receiving responses from external LLMs that contain
    masked placeholders.

    Args:
        request: UnmaskRequest with masked text and session ID

    Returns:
        UnmaskResponse with recovered text and metadata
    """
    request_id = str(uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] Processing unmask request for session: {request.session_id}")

    try:
        # Verify session exists
        session = db.query(SessionInfo).filter(
            SessionInfo.session_id == request.session_id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Initialize recover engine
        recover_engine = PIIRecoverEngine(db)

        # Recover text
        original_text = recover_engine.recover_text(
            masked_text=request.masked_text,
            session_id=request.session_id
        )

        # Count recovered entities
        entities = recover_engine.get_session_entities(request.session_id)

        processing_time = (time.time() - start_time) * 1000  # ms

        logger.info(f"[{request_id}] Recovered {len(entities)} entities in {processing_time:.2f}ms")

        return UnmaskResponse(
            original_text=original_text,
            session_id=request.session_id,
            entities_recovered=len(entities),
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unmask error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unmask error: {str(e)}")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_text_endpoint(request: AnalyzeRequest):
    """
    Analyze text characteristics and get detection recommendations (Smart Mode)

    This endpoint analyzes the input text and provides recommendations for:
    - Detection strategy (high_recall, balanced, high_precision)
    - Model size (medium, large)
    - Estimated PII counts by type

    Useful for:
    - Understanding text complexity before processing
    - Debugging detection issues
    - Optimizing detection configuration

    Args:
        request: AnalyzeRequest with text and optional source type

    Returns:
        AnalyzeResponse with analysis results and recommendations
    """
    if not SMART_MODE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Smart Mode is not available. Please ensure smart_mode.py is installed."
        )

    request_id = str(uuid4())[:8]
    logger.info(f"[{request_id}] Processing analyze request")

    try:
        smart_engine = SmartModeEngine()

        # Parse source type if provided
        source_hint = None
        if request.source_type:
            try:
                source_hint = InputSourceType(request.source_type)
            except ValueError:
                pass

        # Analyze text
        result = smart_engine.process_text(request.text, source_hint)

        logger.info(f"[{request_id}] Analysis complete: domain={result['analysis']['domain']}, "
                   f"strategy={result['recommendations']['strategy']}")

        return AnalyzeResponse(
            analysis=result["analysis"],
            recommendations=result["recommendations"],
            estimated_pii_counts=result["estimated_pii_counts"]
        )

    except Exception as e:
        logger.error(f"[{request_id}] Analyze error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analyze error: {str(e)}")


@app.post("/api/mask-unmask", response_model=Dict)
async def mask_unmask_demo_endpoint(
    text: str,
    session_id: Optional[str] = None,
    detection_strategy: Optional[str] = None,
    enable_smart_mode: Optional[bool] = False,
    db: Session = Depends(get_db)
):
    """
    End-to-end demonstration endpoint: Mask -> (simulated LLM) -> Unmask

    This endpoint demonstrates the complete PII protection workflow:
    1. Detect and mask PII in input text
    2. Simulate sending to external LLM (text is returned as-is)
    3. Recover original PII from masked response

    Useful for testing and understanding the full protection flow.

    Args:
        text: Input text to process
        session_id: Optional session ID
        detection_strategy: Optional detection strategy
        enable_smart_mode: Enable Smart Mode

    Returns:
        Complete workflow results including original, masked, and recovered text
    """
    request_id = str(uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] Processing mask-unmask demo")

    try:
        # Step 1: Mask
        mask_request = MaskRequest(
            text=text,
            session_id=session_id,
            detection_strategy=detection_strategy,
            enable_smart_mode=enable_smart_mode
        )
        mask_response = await mask_text_endpoint(mask_request, db)

        # Step 2: Simulate LLM processing (just pass through)
        simulated_llm_response = f"[LLM Response] Based on your input containing {mask_response.entities_count} entities: {mask_response.masked_text}"

        # Step 3: Unmask
        unmask_request = UnmaskRequest(
            masked_text=simulated_llm_response,
            session_id=mask_response.session_id
        )
        unmask_response = await unmask_text_endpoint(unmask_request, db)

        total_time = (time.time() - start_time) * 1000

        return {
            "workflow": "mask -> llm -> unmask",
            "session_id": mask_response.session_id,
            "step_1_mask": {
                "input": text,
                "output": mask_response.masked_text,
                "entities_detected": mask_response.entities_count,
                "entities": mask_response.entities
            },
            "step_2_llm": {
                "simulated_response": simulated_llm_response
            },
            "step_3_unmask": {
                "input": simulated_llm_response,
                "output": unmask_response.original_text
            },
            "metadata": {
                "detection_strategy": mask_response.detection_strategy,
                "smart_mode_used": mask_response.smart_mode_used,
                "recommended_strategy": mask_response.recommended_strategy,
                "chunking_used": mask_response.chunking_used,
                "total_processing_time_ms": round(total_time, 2)
            }
        }

    except Exception as e:
        logger.error(f"[{request_id}] Mask-unmask demo error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Demo error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Start the server on port 8000
    logger.info("Starting Enhanced LLM Proxy Server with PII Protection")
    logger.info(f"PII Detection Method: {PII_DETECTION_METHOD}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

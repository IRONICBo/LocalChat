from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, BLOB, Boolean, JSON
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///localchat.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Define the Privacy Information Type table
class PrivacyInfoType(Base):
    __tablename__ = "privacy_type"

    type_id = Column(Integer, primary_key=True, autoincrement=True)
    type_name = Column(String(50), nullable=False)
    category = Column(String(20), nullable=False)
    default_strategy = Column(String(20), nullable=False)


# Define the ProcessingStrategy table
class ProcessingStrategy(Base):
    __tablename__ = "processing_strategy"

    strategy_id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), nullable=False)
    strategy_type = Column(String(20), nullable=False, default='mask')  # mask, recover
    method = Column(String(50), nullable=False, default='placeholder')  # placeholder, fake_entity, hash, encrypt
    config_json = Column(JSON)  # Strategy configuration (JSON format)
    reversible = Column(Boolean, default=True)  # Whether the strategy is reversible
    entity_types = Column(JSON)  # Applicable entity types (JSON list)
    description = Column(Text)  # Strategy description


# Define the Session Info table
class SessionInfo(Base):
    __tablename__ = "session_info"

    session_id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(String(100))
    extra_info = Column(JSON)  # Changed from 'metadata' (reserved keyword)


# Define the Conversation History table
class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    conversation_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("session_info.session_id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    original_content = Column(Text, nullable=False)
    masked_content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Define the Mask Mapping table
class MaskMapping(Base):
    __tablename__ = "mask_mapping"

    mapping_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("session_info.session_id"), nullable=False)
    conversation_id = Column(String(36), ForeignKey("conversation_history.conversation_id"))
    entity_id = Column(String(36), ForeignKey("sensitive_entity.entity_id"), nullable=False)
    placeholder = Column(String(100), nullable=False)  # e.g., ${EMAIL_001}
    hash_value = Column(String(64), nullable=False)  # For quick lookup
    # Extended fields for multi-strategy support
    masking_strategy = Column(String(50), nullable=False, default='placeholder')  # placeholder, fake_entity, hash, encrypt
    fake_value = Column(String(200))  # Fake entity value (e.g., "张三", "john@example.com")
    original_hash = Column(String(64))  # Hash of original value for consistency lookup
    created_at = Column(DateTime, default=datetime.utcnow)


# Define the SensitiveEntity table
class SensitiveEntity(Base):
    __tablename__ = "sensitive_entity"

    entity_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("session_info.session_id"), nullable=False)
    conversation_id = Column(String(36), ForeignKey("conversation_history.conversation_id"))
    text = Column(Text, nullable=False)
    start_pos = Column(Integer, nullable=False)
    end_pos = Column(Integer, nullable=False)
    entity_type = Column(String(50), nullable=False)  # EMAIL, PHONE, CREDIT_CARD, etc.
    type_id = Column(Integer, ForeignKey("privacy_type.type_id"))
    sensitivity = Column(Integer, nullable=False, default=5)  # 1-10 scale
    detection_method = Column(String(50))  # Regex, Presidio, LLM, E2E
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


# Define the ProcessedResult table
class ProcessedResult(Base):
    __tablename__ = "processed_result"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String, ForeignKey("sensitive_entity.entity_id"), nullable=False)
    processed_text = Column(Text, nullable=False)
    strategy_id = Column(
        Integer, ForeignKey("processing_strategy.strategy_id"), nullable=False
    )
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Define the KnowledgeBase table
class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    item_id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    type_id = Column(Integer, nullable=False)
    category = Column(String(50))
    embedding = Column(BLOB)


# Define the ProcessingLog table
class ProcessingLog(Base):
    __tablename__ = "processing_log"

    log_id = Column(String(36), primary_key=True)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Define the EvaluationRecord table
class EvaluationRecord(Base):
    __tablename__ = "evaluation_record"

    eval_id = Column(String(36), primary_key=True)
    request_id = Column(String(36), nullable=False)
    privacy_score = Column(Float, nullable=False)
    utility_loss = Column(Float, nullable=False)
    balance_score = Column(Float, nullable=False)
    attack_results = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# Initialize the database
def init_db():
    Base.metadata.create_all(bind=engine)


init_db()

import sqlite3
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, BLOB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///localchat.db"
engine = sqlite3.connect('localchat.db')
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
    strategy_name = Column(String(20), nullable=False)

# Define the SensitiveEntity table
class SensitiveEntity(Base):
    __tablename__ = "sensitive_entity"

    entity_id = Column(String(36), primary_key=True)
    text = Column(Text, nullable=False)
    start_pos = Column(Integer, nullable=False)
    end_pos = Column(Integer, nullable=False)
    type_id = Column(Integer, nullable=False)
    sensitivity = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Define the ProcessedResult table
class ProcessedResult(Base):
    __tablename__ = "processed_result"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String, ForeignKey('sensitive_entity.entity_id'), nullable=False)
    processed_text = Column(Text, nullable=False)
    strategy_id = Column(Integer, ForeignKey('processing_strategy.strategy_id'), nullable=False)
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
from sqlalchemy import (
    DateTime,
    create_engine,
    Column,
    Integer,
    Float,
    String,
    TIMESTAMP,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import event

DATABASE_URL = "sqlite:///localchat.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Define the database model for storing chatbot usage statistics
class ChatbotUsage(Base):
    __tablename__ = "chatbot_usage"

    id = Column(Integer, primary_key=True, index=True)
    model = Column(String, index=True)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    total_token_count = Column(Integer)
    completion_tokens_count = Column(Integer)
    prompt_tokens_count = Column(Integer)
    response_time = Column(Float)
    created_at = Column(DateTime, default=func.now())


# Define the database model for storing document libraries
class DocumentLibrary(Base):
    __tablename__ = "document_libraries"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())


# Define a listener to insert a default record after the table is created
@event.listens_for(DocumentLibrary.__table__, "after_create")
def insert_default_document_library(target, connection, **kwargs):
    """Insert a default record when the database table is created."""
    connection.execute(target.insert().values(name="default"))


Base.metadata.create_all(bind=engine)

from datetime import datetime
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


# Define the database model for storing HTML metadata
class HtmlMetadata(Base):
    __tablename__ = "html_metadata"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)  # URL of the HTML document
    filepath = Column(String, nullable=False)  # Filepath where the document is stored
    created_at = Column(
        DateTime, default=func.now()
    )  # Timestamp when the entry is created


# Define the settings
class LocalChatSettings(Base):
    __tablename__ = "localchat_settings"

    # We only support one setting here, so id must be 1
    id = Column(Integer, primary_key=True, index=True)
    system_prompt = Column(
        String,
        nullable=False,
        default="You are a helpful assistant. Please assist the user with their inquiries.",
    )
    llm = Column(String, nullable=False, default="qwen2:0.5b")
    keep_alive = Column(String, nullable=False, default="1h")
    top_k = Column(Integer, nullable=False, default=40)
    top_p = Column(Float, nullable=False, default=0.9)
    repeat_last_n = Column(Integer, nullable=False, default=64)
    repeat_penalty = Column(Float, nullable=False, default=1.1)
    request_timeout = Column(Float, nullable=False, default=300)
    port = Column(Integer, nullable=False, default=11434)
    context_window = Column(Integer, nullable=False, default=8000)
    temperature = Column(Float, nullable=False, default=0.1)
    chat_token_limit = Column(Integer, nullable=False, default=4000)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<LocalChatRagSettings(model={self.llm}, temperature={self.temperature}, created_at={self.created_at})>"


# Define a listener to insert a default record after the table is created
@event.listens_for(DocumentLibrary.__table__, "after_create")
def insert_default_document_library(target, connection, **kwargs):
    """Insert a default record when the database table is created."""
    connection.execute(target.insert().values(name="default"))


@event.listens_for(LocalChatSettings.__table__, "after_create")
def insert_default_localchat_rag_settings(target, connection, **kwargs):
    """Insert a default record when the database table is created."""
    connection.execute(
        target.insert().values(
            id=1,
            system_prompt="You are a helpful assistant. Please assist the user with their inquiries.",
            llm="qwen2:0.5b",
            keep_alive="1h",
            top_k=40,
            top_p=0.9,
            repeat_last_n=64,
            repeat_penalty=1.1,
            request_timeout=300,
            port=11434,
            context_window=8000,
            temperature=0.1,
            chat_token_limit=4000,
            created_at=datetime.now(),
        )
    )


def check_localchat_settings_is_empty():
    """Check if the localchat_settings table is empty."""
    session = SessionLocal()
    existing_config = (
        session.query(LocalChatSettings).filter(LocalChatSettings.id == 1).first()
    )
    print("existing_config: ", existing_config)
    if not existing_config:
        print("localchat_settings is empty, inserting default record...")
        default_config = LocalChatSettings(
            id=1,
            system_prompt="You are a helpful assistant. Please assist the user with their inquiries.",
            llm="qwen2:0.5b",
            keep_alive="1h",
            top_k=40,
            top_p=0.9,
            repeat_last_n=64,
            repeat_penalty=1.1,
            request_timeout=300,
            port=11434,
            context_window=8000,
            temperature=0.1,
            chat_token_limit=4000,
            created_at=datetime.now(),
        )
        session.add(default_config)
        session.commit()


Base.metadata.create_all(bind=engine)
check_localchat_settings_is_empty()

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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

Base.metadata.create_all(bind=engine)

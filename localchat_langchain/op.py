from sqlalchemy.orm import Session
from models import SessionLocal, ChatbotUsage


def get_usage_by_model_paginated(db: Session, model: str, page: int, page_size: int):
    """
    Get paginated chatbot usage records by model name

    :param db: database session
    :param model: model name
    :param page: current page number
    :param page_size: number of records per page
    :return: list of ChatbotUsage records
    """
    offset = (page - 1) * page_size
    records = (
        db.query(ChatbotUsage)
        .filter(ChatbotUsage.model == model)
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return records

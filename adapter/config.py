# -*- coding: utf-8 -*-
"""
Configuration for PII Detection System

Supports:
- Multiple LLM model sizes (0.5B to 7B+)
- Detection thresholds and strategies
- Flexible parameter tuning
"""

import os
from typing import Dict, List
from enum import Enum


class DetectionStrategy(str, Enum):
    """Detection strategy for different scenarios"""
    HIGH_RECALL = "high_recall"  # Minimize false negatives (FN), allow more false positives
    HIGH_PRECISION = "high_precision"  # Minimize false positives (FP), allow more false negatives
    BALANCED = "balanced"  # Balance between precision and recall


class PIIConfig:
    """Configuration for PII Detection System"""

    # ==================== LLM Backend Configuration ====================

    # LLM Backend type: "ollama" or "openai"
    LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")

    # Ollama API settings (legacy, kept for backward compatibility)
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

    # OpenAI-compatible API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123456")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://0.0.0.0:23333/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "internlm/internlm2-chat-1_8b")

    # Chat completions endpoint (auto-constructed if not provided)
    OPENAI_CHAT_ENDPOINT = os.getenv(
        "OPENAI_CHAT_ENDPOINT",
        f"{OPENAI_API_BASE}/chat/completions" if not OPENAI_API_BASE.endswith("/chat/completions") else OPENAI_API_BASE
    )

    # Model configurations for different scenarios
    # Recommended models by size and use case:
    # - 0.5B: qwen:0.5b (fast but poor accuracy, not recommended)
    # - 1.5B: qwen2:1.5b (faster, acceptable for simple cases)
    # - 4B: qwen2.5:4b (good balance, recommended for most cases)
    # - 7B: qwen2.5:7b (best accuracy, recommended for production)
    # - 14B+: qwen2.5:14b (highest accuracy, for high-value scenarios)

    LLM_MODEL_CONFIGS = {
        "tiny": {
            "model": "qwen:0.5b",
            "timeout": 30,
            "description": "Tiny model, fast but poor accuracy (F1: 1.35%)",
            "recommended": False
        },
        "small": {
            "model": "qwen2:1.5b",
            "timeout": 45,
            "description": "Small model, faster with acceptable accuracy",
            "recommended": False
        },
        "medium": {
            "model": "qwen2.5:4b",
            "timeout": 60,
            "description": "Medium model, good balance (recommended)",
            "recommended": True
        },
        "large": {
            "model": "qwen2.5:7b",
            "timeout": 90,
            "description": "Large model, best accuracy (recommended for production)",
            "recommended": True
        },
        "xlarge": {
            "model": "qwen2.5:14b",
            "timeout": 120,
            "description": "Extra large model, highest accuracy for critical scenarios",
            "recommended": False
        }
    }

    # Default LLM model size
    DEFAULT_LLM_SIZE = os.getenv("LLM_MODEL_SIZE", "medium")

    @classmethod
    def get_llm_config(cls, size: str = None) -> Dict:
        """Get LLM configuration by size"""
        size = size or cls.DEFAULT_LLM_SIZE
        if size not in cls.LLM_MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {size}. Available: {list(cls.LLM_MODEL_CONFIGS.keys())}")
        return cls.LLM_MODEL_CONFIGS[size]

    # ==================== Detection Method Configuration ====================

    # PII detection method: "Regex", "Presidio", "LLM", "E2E"
    PII_DETECTION_METHOD = os.getenv("PII_DETECTION_METHOD", "E2E")

    # Detection strategy
    DETECTION_STRATEGY = os.getenv("DETECTION_STRATEGY", DetectionStrategy.BALANCED)

    # ==================== Confidence Thresholds ====================

    # Confidence thresholds by strategy
    CONFIDENCE_THRESHOLDS = {
        DetectionStrategy.HIGH_RECALL: {
            "presidio_min": 0.3,  # Lower threshold to catch more entities
            "llm_min": 0.5,
            "merge_min": 0.3  # Lower threshold for merging
        },
        DetectionStrategy.BALANCED: {
            "presidio_min": 0.5,  # Balanced threshold
            "llm_min": 0.7,
            "merge_min": 0.5
        },
        DetectionStrategy.HIGH_PRECISION: {
            "presidio_min": 0.7,  # Higher threshold to reduce false positives
            "llm_min": 0.8,
            "merge_min": 0.7
        }
    }

    @classmethod
    def get_thresholds(cls, strategy: str = None) -> Dict:
        """Get confidence thresholds for a strategy"""
        strategy = strategy or cls.DETECTION_STRATEGY
        if isinstance(strategy, str):
            strategy = DetectionStrategy(strategy)
        return cls.CONFIDENCE_THRESHOLDS[strategy]

    # ==================== Entity Merging Configuration ====================

    # Overlap threshold for entity merging (0-1)
    # Higher value = require more overlap to consider entities as duplicates
    OVERLAP_THRESHOLD = 0.5

    # When merging overlapping entities, prefer entities from which source?
    # Options: "longer", "presidio", "llm", "higher_confidence"
    MERGE_PREFERENCE = "longer"

    # ==================== Regex Enhancement ====================

    # Enable enhanced regex patterns for high-frequency FN cases
    ENABLE_ENHANCED_REGEX = True

    # Enhanced regex patterns for complex cases
    ENHANCED_PATTERNS = {
        # Complex phone number formats
        "PHONE_COMPLEX": [
            r"\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
            r"\b\d{3,4}[-.\s]\d{3,4}[-.\s]\d{4}\b",
            r"\(\d{3}\)\s*\d{3}-\d{4}",
        ],
        # Non-standard addresses
        "ADDRESS_PATTERN": [
            r"\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)",
            r"P\.?O\.?\s*Box\s+\d+",
        ],
        # Common name patterns (first + last)
        "PERSON_PATTERN": [
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        ],
        # Additional credit card formats
        "CREDIT_CARD_EXTRA": [
            r"\b[0-9]{4}\s[0-9]{4}\s[0-9]{4}\s[0-9]{4}\b",
            r"\b[0-9]{4}\*{4}\*{4}[0-9]{4}\b",  # Partially masked
        ]
    }

    # ==================== LLM Prompt Configuration ====================

    # Enable few-shot learning in LLM prompts
    ENABLE_FEW_SHOT = True

    # Enable chain-of-thought reasoning in LLM prompts
    ENABLE_CHAIN_OF_THOUGHT = True

    # Number of few-shot examples to include
    NUM_FEW_SHOT_EXAMPLES = 3

    # Few-shot examples for PII extraction
    FEW_SHOT_EXAMPLES = [
        {
            "text": "My name is John Smith and my email is john.smith@example.com",
            "entities": [
                {"entity_type": "PERSON", "entity_value": "John Smith"},
                {"entity_type": "EMAIL_ADDRESS", "entity_value": "john.smith@example.com"}
            ]
        },
        {
            "text": "Contact me at +1-555-123-4567 or visit 123 Main Street, New York",
            "entities": [
                {"entity_type": "PHONE_NUMBER", "entity_value": "+1-555-123-4567"},
                {"entity_type": "LOCATION", "entity_value": "123 Main Street, New York"}
            ]
        },
        {
            "text": "My credit card number is 4532-1111-2222-3333",
            "entities": [
                {"entity_type": "CREDIT_CARD", "entity_value": "4532-1111-2222-3333"}
            ]
        }
    ]

    # ==================== JSON Parsing Configuration ====================

    # Maximum retry attempts for JSON parsing
    MAX_JSON_PARSE_RETRIES = 3

    # Enable aggressive JSON cleaning
    ENABLE_AGGRESSIVE_JSON_CLEANING = True

    # ==================== Database Configuration ====================

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///localchat.db")

    # ==================== Logging Configuration ====================

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "llm_proxy.log")

    # Enable detailed entity logging
    ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"

    # ==================== Performance Configuration ====================

    # Cache LLM responses to avoid repeated API calls
    ENABLE_LLM_CACHE = False

    # Maximum cache size (number of entries)
    MAX_CACHE_SIZE = 1000

    # ==================== Entity Type Mapping ====================

    # Map Presidio entity types to standardized types
    ENTITY_TYPE_MAPPING = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "CREDIT_CARD": "CREDIT_CARD",
        "US_SSN": "SSN",
        "LOCATION": "LOCATION",
        "DATE_TIME": "DATE",
        "ORGANIZATION": "ORGANIZATION",
        "IP_ADDRESS": "IP_ADDRESS",
        "URL": "URL",
        "US_BANK_NUMBER": "BANK_ACCOUNT",
        "US_DRIVER_LICENSE": "DRIVER_LICENSE",
        "US_PASSPORT": "PASSPORT",
    }

    # ==================== Sensitivity Levels ====================

    # Sensitivity scores for different entity types (1-10)
    SENSITIVITY_LEVELS = {
        "CREDIT_CARD": 10,
        "SSN": 10,
        "BANK_ACCOUNT": 10,
        "PASSWORD": 10,
        "PASSPORT": 9,
        "DRIVER_LICENSE": 8,
        "PHONE": 7,
        "ADDRESS": 7,
        "EMAIL": 6,
        "IP_ADDRESS": 6,
        "PERSON": 5,
        "LOCATION": 4,
        "URL": 4,
        "ORGANIZATION": 3,
        "DATE": 3,
    }


# Export configuration instance
config = PIIConfig()

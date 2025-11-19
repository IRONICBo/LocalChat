# -*- coding: utf-8 -*-
"""
Masking Strategies Configuration and Implementation

Defines different masking strategies (placeholder, fake_entity, hash, encrypt)
and their configuration.
"""

from enum import Enum
from typing import Dict, List, Any


class MaskingStrategy(str, Enum):
    """Masking strategy types"""
    PLACEHOLDER = "placeholder"  # ${EMAIL_001} format
    FAKE_ENTITY = "fake_entity"  # Realistic fake values (张三, john@example.com)
    HASH = "hash"              # SHA256 hash (one-way)
    ENCRYPT = "encrypt"        # AES encryption (reversible)
    MASK = "mask"              # Partial masking (****@example.com)
    GENERALIZE = "generalize"  # Generalization/bucketing


# Strategy configurations
STRATEGY_CONFIGS = {
    MaskingStrategy.PLACEHOLDER: {
        "name": "Placeholder",
        "description": "Replace PII with placeholders like ${EMAIL_001}",
        "reversible": True,
        "preserves_format": False,
        "preserves_semantics": False,
        "data_utility": "low",
        "applicable_types": ["ALL"],
        "config": {
            "template": "${{{entity_type}_{index:03d}}}",
            "counter_per_session": True
        }
    },

    MaskingStrategy.FAKE_ENTITY: {
        "name": "Fake Entity",
        "description": "Replace PII with realistic fake values (张三, john@example.com)",
        "reversible": True,
        "preserves_format": True,
        "preserves_semantics": True,
        "data_utility": "high",
        "applicable_types": [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "LOCATION",
            "ADDRESS",
            "CREDIT_CARD",
            "SSN",
            "ORGANIZATION",
            "IP_ADDRESS",
            "URL",
            "DATE",
            "DATE_TIME"
        ],
        "config": {
            "locale": "auto",  # auto-detect or specify (zh_CN, en_US, etc.)
            "consistent": True,  # Same original -> same fake
            "preserve_domain": True,  # For emails, preserve corporate domains
            "preserve_format": True   # Match format of original (dashes, spaces, etc.)
        }
    },

    MaskingStrategy.HASH: {
        "name": "Hash",
        "description": "Replace PII with SHA256 hash (one-way, irreversible)",
        "reversible": False,
        "preserves_format": False,
        "preserves_semantics": False,
        "data_utility": "low",
        "applicable_types": ["ALL"],
        "config": {
            "algorithm": "sha256",
            "salt": None,  # Optional salt
            "truncate": None  # Optional truncation length
        }
    },

    MaskingStrategy.ENCRYPT: {
        "name": "Encrypt",
        "description": "Replace PII with encrypted value (AES-256-GCM, reversible)",
        "reversible": True,
        "preserves_format": False,
        "preserves_semantics": False,
        "data_utility": "low",
        "applicable_types": ["ALL"],
        "config": {
            "algorithm": "AES-256-GCM",
            "key_source": "env",  # env variable, file, or key management service
            "encoding": "base64"   # base64 or hex
        }
    },

    MaskingStrategy.MASK: {
        "name": "Partial Mask",
        "description": "Partially mask PII (e.g., ****@example.com)",
        "reversible": False,
        "preserves_format": True,
        "preserves_semantics": False,
        "data_utility": "medium",
        "applicable_types": [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "SSN"
        ],
        "config": {
            "mask_char": "*",
            "reveal_first": 2,  # Reveal first N characters
            "reveal_last": 4,   # Reveal last N characters
            "mask_domain": False  # For emails, whether to mask domain
        }
    },

    MaskingStrategy.GENERALIZE: {
        "name": "Generalize",
        "description": "Generalize PII to ranges or categories",
        "reversible": False,
        "preserves_format": False,
        "preserves_semantics": False,
        "data_utility": "medium",
        "applicable_types": [
            "DATE",
            "DATE_TIME",
            "LOCATION",
            "AGE"
        ],
        "config": {
            "date_precision": "month",  # year, month, week, day
            "location_precision": "city",  # country, province, city
            "age_bucket_size": 10  # Group ages in buckets of 10
        }
    }
}


class StrategySelector:
    """Select appropriate masking strategy based on entity type and context"""

    @staticmethod
    def get_default_strategy(entity_type: str, scenario: str = "general") -> MaskingStrategy:
        """
        Get default masking strategy for entity type and scenario

        Args:
            entity_type: Type of PII entity
            scenario: Use case scenario (general, llm_training, analytics, display)

        Returns:
            Recommended MaskingStrategy
        """
        # Scenario-based defaults
        scenario_defaults = {
            "general": MaskingStrategy.PLACEHOLDER,
            "llm_training": MaskingStrategy.FAKE_ENTITY,
            "analytics": MaskingStrategy.FAKE_ENTITY,
            "display": MaskingStrategy.MASK,
            "storage": MaskingStrategy.ENCRYPT,
            "deduplication": MaskingStrategy.HASH
        }

        # Entity type specific overrides
        if scenario == "llm_training" or scenario == "analytics":
            # Use fake entities for better data utility
            if entity_type in STRATEGY_CONFIGS[MaskingStrategy.FAKE_ENTITY]["applicable_types"]:
                return MaskingStrategy.FAKE_ENTITY

        # Use scenario default
        return scenario_defaults.get(scenario, MaskingStrategy.PLACEHOLDER)

    @staticmethod
    def is_applicable(strategy: MaskingStrategy, entity_type: str) -> bool:
        """
        Check if strategy is applicable for entity type

        Args:
            strategy: Masking strategy
            entity_type: PII entity type

        Returns:
            True if applicable
        """
        applicable_types = STRATEGY_CONFIGS[strategy]["applicable_types"]

        if "ALL" in applicable_types:
            return True

        return entity_type in applicable_types

    @staticmethod
    def get_strategy_config(strategy: MaskingStrategy) -> Dict[str, Any]:
        """
        Get configuration for strategy

        Args:
            strategy: Masking strategy

        Returns:
            Strategy configuration dict
        """
        return STRATEGY_CONFIGS[strategy]

    @staticmethod
    def list_strategies_for_entity(entity_type: str) -> List[MaskingStrategy]:
        """
        List all applicable strategies for entity type

        Args:
            entity_type: PII entity type

        Returns:
            List of applicable strategies
        """
        applicable = []

        for strategy in MaskingStrategy:
            if StrategySelector.is_applicable(strategy, entity_type):
                applicable.append(strategy)

        return applicable


# Default strategy by entity type for quick reference
DEFAULT_STRATEGIES_BY_TYPE = {
    "PERSON": MaskingStrategy.FAKE_ENTITY,
    "EMAIL_ADDRESS": MaskingStrategy.FAKE_ENTITY,
    "PHONE_NUMBER": MaskingStrategy.FAKE_ENTITY,
    "LOCATION": MaskingStrategy.FAKE_ENTITY,
    "ADDRESS": MaskingStrategy.FAKE_ENTITY,
    "CREDIT_CARD": MaskingStrategy.FAKE_ENTITY,
    "SSN": MaskingStrategy.HASH,  # More sensitive, use hash
    "IP_ADDRESS": MaskingStrategy.FAKE_ENTITY,
    "URL": MaskingStrategy.FAKE_ENTITY,
    "ORGANIZATION": MaskingStrategy.FAKE_ENTITY,
    "DATE": MaskingStrategy.GENERALIZE,
    "DATE_TIME": MaskingStrategy.GENERALIZE,
}


def get_recommended_strategy(
    entity_type: str,
    sensitivity_level: int = 5,
    scenario: str = "general",
    reversible_required: bool = True
) -> MaskingStrategy:
    """
    Get recommended masking strategy based on multiple factors

    Args:
        entity_type: PII entity type
        sensitivity_level: Sensitivity level (1-10)
        scenario: Use case scenario
        reversible_required: Whether reversibility is required

    Returns:
        Recommended masking strategy
    """
    # High sensitivity (8-10) - use strong protection
    if sensitivity_level >= 8:
        if reversible_required:
            return MaskingStrategy.ENCRYPT
        else:
            return MaskingStrategy.HASH

    # Medium-high sensitivity (5-7) - balance security and utility
    if sensitivity_level >= 5:
        if scenario in ["llm_training", "analytics"]:
            return MaskingStrategy.FAKE_ENTITY
        elif reversible_required:
            return MaskingStrategy.PLACEHOLDER
        else:
            return MaskingStrategy.MASK

    # Low sensitivity (1-4) - prioritize utility
    if scenario in ["llm_training", "analytics"]:
        return MaskingStrategy.FAKE_ENTITY
    else:
        return MaskingStrategy.PLACEHOLDER


if __name__ == "__main__":
    # Test the strategy selector
    print("=== Masking Strategies Configuration ===\n")

    # Test 1: List all strategies
    print("Available strategies:")
    for strategy in MaskingStrategy:
        config = STRATEGY_CONFIGS[strategy]
        print(f"  - {strategy.value}: {config['name']}")
        print(f"    Reversible: {config['reversible']}")
        print(f"    Data Utility: {config['data_utility']}\n")

    # Test 2: Get default strategy for entity types
    print("Default strategies by entity type:")
    for entity_type, strategy in DEFAULT_STRATEGIES_BY_TYPE.items():
        print(f"  {entity_type}: {strategy.value}")

    # Test 3: List applicable strategies for PERSON
    print("\nApplicable strategies for PERSON:")
    strategies = StrategySelector.list_strategies_for_entity("PERSON")
    for strategy in strategies:
        print(f"  - {strategy.value}")

    # Test 4: Get recommended strategy
    print("\nRecommended strategies:")
    print(f"  EMAIL (LLM training, sensitivity=5): {get_recommended_strategy('EMAIL_ADDRESS', 5, 'llm_training')}")
    print(f"  SSN (general, sensitivity=10): {get_recommended_strategy('SSN', 10, 'general')}")
    print(f"  PHONE (analytics, sensitivity=3): {get_recommended_strategy('PHONE_NUMBER', 3, 'analytics')}")

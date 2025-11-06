#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PII Engine - Mask and Recover functionality

Tests all four detection methods:
1. Regex
2. Presidio
3. LLM
4. E2E (Presidio + LLM)
"""

import os
import sys
from uuid import uuid4
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import sessionmaker
from models import engine, Base, SessionInfo
from pii_engine import (
    PIIMaskEngine, PIIRecoverEngine,
    RegexExtractor, PresidioExtractor, LLMExtractor, E2EExtractor
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def print_header(title):
    """Print test section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message):
    """Print success message"""
    print(f"✓ {message}")


def print_error(message):
    """Print error message"""
    print(f"✗ {message}")


def print_info(message):
    """Print info message"""
    print(f"ℹ {message}")


def test_extractors():
    """Test individual extractors"""
    print_header("TEST 1: Individual Extractors")

    test_text = """
    My name is John Smith and my email is john.smith@example.com.
    You can reach me at 555-123-4567.
    My credit card number is 4532-1111-2222-3333.
    """

    extractors = [
        ("Regex", RegexExtractor()),
        ("Presidio", PresidioExtractor()),
        ("LLM", LLMExtractor()),
        ("E2E", E2EExtractor()),
    ]

    for name, extractor in extractors:
        print(f"\n--- Testing {name} ---")

        if not extractor.available:
            print_error(f"{name} not available (dependencies missing or service down)")
            continue

        try:
            entities = extractor.extract(test_text)
            print_success(f"{name} extracted {len(entities)} entities")

            for entity in entities:
                print(f"  - {entity['entity_type']}: {entity['entity_value']} "
                      f"[{entity['start']}:{entity['end']}] "
                      f"(confidence: {entity.get('confidence', 1.0):.2f})")

        except Exception as e:
            print_error(f"{name} failed: {str(e)}")


def test_mask_and_recover():
    """Test mask and recover functionality"""
    print_header("TEST 2: Mask and Recover")

    test_text = """
    Hello, my name is Alice Johnson.
    Please contact me at alice.j@company.com or call 555-987-6543.
    My credit card ending in 4532-1111-2222-3333 should be charged.
    """

    detection_methods = ["Regex", "E2E"]  # Test with Regex and E2E

    for method in detection_methods:
        print(f"\n--- Testing {method} Method ---")

        # Create database session
        db = SessionLocal()

        try:
            # Create session
            session_id = str(uuid4())
            session = SessionInfo(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            db.add(session)
            db.commit()

            print_info(f"Created session: {session_id}")

            # Initialize engines
            mask_engine = PIIMaskEngine(db, detection_method=method)
            recover_engine = PIIRecoverEngine(db)

            # Mask text
            print(f"\nOriginal text:\n{test_text}")

            masked_text, entities_info = mask_engine.mask_text(
                text=test_text,
                session_id=session_id
            )

            print(f"\nMasked text:\n{masked_text}")
            print(f"\n{len(entities_info)} entities detected:")
            for entity in entities_info:
                print(f"  - {entity['entity_type']}: {entity['entity_value']} "
                      f"→ {entity['placeholder']}")

            # Recover text
            recovered_text = recover_engine.recover_text(masked_text, session_id)

            print(f"\nRecovered text:\n{recovered_text}")

            # Verify recovery
            if recovered_text.strip() == test_text.strip():
                print_success("Text recovery successful! Original and recovered text match.")
            else:
                print_error("Text recovery failed! Original and recovered text differ.")
                print(f"Difference:\nOriginal: {test_text}\nRecovered: {recovered_text}")

            # Get session entities
            session_entities = recover_engine.get_session_entities(session_id)
            print(f"\nSession entities ({len(session_entities)}):")
            for entity in session_entities:
                print(f"  - {entity['entity_type']}: {entity['entity_value']} "
                      f"(method: {entity['detection_method']}, "
                      f"sensitivity: {entity['sensitivity']})")

        except Exception as e:
            print_error(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            db.close()


def test_multiple_messages():
    """Test masking and recovery across multiple messages"""
    print_header("TEST 3: Multiple Messages (Conversation)")

    messages = [
        "Hi, I'm Bob. My email is bob@example.com",
        "You can also reach me at 555-111-2222",
        "Please send the invoice to bob@example.com"  # Same email should use same placeholder
    ]

    # Create database session
    db = SessionLocal()

    try:
        # Create session
        session_id = str(uuid4())
        session = SessionInfo(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        db.add(session)
        db.commit()

        print_info(f"Created session: {session_id}")

        # Initialize engines
        mask_engine = PIIMaskEngine(db, detection_method="Regex")  # Use Regex for speed
        recover_engine = PIIRecoverEngine(db)

        # Process each message
        masked_messages = []
        all_entities = []

        for i, msg in enumerate(messages):
            print(f"\n--- Message {i + 1} ---")
            print(f"Original: {msg}")

            masked_text, entities_info = mask_engine.mask_text(
                text=msg,
                session_id=session_id
            )

            print(f"Masked:   {masked_text}")
            print(f"Entities: {[(e['entity_type'], e['placeholder']) for e in entities_info]}")

            masked_messages.append(masked_text)
            all_entities.extend(entities_info)

        print(f"\nTotal entities across all messages: {len(all_entities)}")

        # Recover all messages
        print("\n--- Recovery ---")
        for i, masked_msg in enumerate(masked_messages):
            recovered = recover_engine.recover_text(masked_msg, session_id)
            original = messages[i]

            print(f"Message {i + 1}:")
            print(f"  Original:  {original}")
            print(f"  Recovered: {recovered}")

            if recovered == original:
                print_success(f"Message {i + 1} recovered correctly")
            else:
                print_error(f"Message {i + 1} recovery mismatch")

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


def test_placeholder_consistency():
    """Test that same entity values get same placeholders"""
    print_header("TEST 4: Placeholder Consistency")

    # Note: This test checks if the SAME text gets the same placeholder
    # However, current implementation generates NEW placeholders each time
    # This is intentional for security - each occurrence is tracked separately

    test_texts = [
        "Email me at test@example.com",
        "My email is test@example.com",  # Same email
    ]

    db = SessionLocal()

    try:
        session_id = str(uuid4())
        session = SessionInfo(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        db.add(session)
        db.commit()

        mask_engine = PIIMaskEngine(db, detection_method="Regex")

        placeholders = []
        for text in test_texts:
            masked_text, entities_info = mask_engine.mask_text(
                text=text,
                session_id=session_id
            )
            if entities_info:
                placeholders.append(entities_info[0]['placeholder'])

        print(f"Text 1: {test_texts[0]}")
        print(f"  Placeholder: {placeholders[0] if len(placeholders) > 0 else 'None'}")
        print(f"\nText 2: {test_texts[1]}")
        print(f"  Placeholder: {placeholders[1] if len(placeholders) > 1 else 'None'}")

        if len(placeholders) >= 2:
            if placeholders[0] == placeholders[1]:
                print_info("Same entity value got same placeholder (value-based)")
            else:
                print_info("Same entity value got different placeholders (occurrence-based)")
                print_info("This is expected behavior - each occurrence is tracked separately")

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


def main():
    """Run all tests"""
    print_header("PII ENGINE TESTS")
    print("Testing PII detection, masking, and recovery functionality")

    try:
        # Run tests
        test_extractors()
        test_mask_and_recover()
        test_multiple_messages()
        test_placeholder_consistency()

        print_header("ALL TESTS COMPLETED")

    except Exception as e:
        print_error(f"Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

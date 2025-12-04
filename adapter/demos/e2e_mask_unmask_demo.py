# -*- coding: utf-8 -*-
"""
End-to-End PII Mask and Unmask Demo

This demo demonstrates the complete workflow of:
1. PII Detection using multiple methods (Regex, Presidio, LLM, E2E)
2. PII Masking with configurable strategies
3. PII Recovery (Unmasking) to restore original values
4. Strategy selection based on text characteristics (Smart Mode)

Usage:
    python demos/e2e_mask_unmask_demo.py

Features demonstrated:
- Multiple detection methods comparison
- Different masking strategies (placeholder, fake_entity, hash, encrypt, mask)
- Complete mask -> process -> unmask workflow
- Smart mode for automatic strategy selection
- Long text chunking support
- Performance metrics and evaluation
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import core components
from pii_engine import PIIMaskEngine, PIIRecoverEngine, RegexExtractor, PresidioExtractor, LLMExtractor, E2EExtractor
from config import PIIConfig, DetectionStrategy
from models import Base, engine

# Import enhanced generators
from utils.fake_pii_generator import FakePIIGenerator
from utils.credit_card_generator import CreditCardGenerator
from utils.address_generator import AddressGenerator


class PIIMaskUnmaskDemo:
    """
    Comprehensive demo for PII mask and unmask operations.

    This class provides a complete demonstration of the PII protection workflow,
    including detection, masking, and recovery with support for multiple strategies.
    """

    def __init__(
        self,
        detection_method: str = "E2E",
        strategy: str = "balanced",
        model_size: str = "medium",
        enable_smart_mode: bool = False
    ):
        """
        Initialize the demo.

        Args:
            detection_method: PII detection method ("Regex", "Presidio", "LLM", "E2E")
            strategy: Detection strategy ("high_recall", "balanced", "high_precision")
            model_size: LLM model size ("tiny", "small", "medium", "large", "xlarge")
            enable_smart_mode: Enable automatic strategy selection based on text characteristics
        """
        # Create in-memory database for demo
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=self.engine)

        Session = sessionmaker(bind=self.engine)
        self.db = Session()

        self.detection_method = detection_method
        self.strategy = strategy
        self.model_size = model_size
        self.enable_smart_mode = enable_smart_mode

        # Initialize mask and recover engines
        self.mask_engine = PIIMaskEngine(
            self.db,
            detection_method=detection_method,
            strategy=strategy,
            model_size=model_size
        )
        self.recover_engine = PIIRecoverEngine(self.db)

        # Initialize generators for creating test data
        self.pii_gen = FakePIIGenerator(use_enhanced_generators=True)
        self.card_gen = CreditCardGenerator()
        self.addr_gen = AddressGenerator()

        print(f"Demo initialized with:")
        print(f"  Detection Method: {detection_method}")
        print(f"  Strategy: {strategy}")
        print(f"  Model Size: {model_size}")
        print(f"  Smart Mode: {'Enabled' if enable_smart_mode else 'Disabled'}")

    def analyze_text_characteristics(self, text: str) -> Dict:
        """
        Analyze text characteristics for smart mode strategy selection.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing text characteristics
        """
        import re

        # Basic metrics
        text_length = len(text)
        word_count = len(text.split())

        # Detect potential PII density (rough estimate)
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        phone_count = len(re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text))
        cc_count = len(re.findall(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', text))

        # Calculate entity density
        total_entities = email_count + phone_count + cc_count
        entity_density = total_entities / max(word_count, 1)

        # Detect complexity indicators
        has_multilingual = bool(re.search(r'[^\x00-\x7F]+', text))
        has_complex_formatting = bool(re.search(r'[+().\-]{2,}', text))
        sentence_count = text.count('.') + text.count('!') + text.count('?')

        # Domain detection (financial, medical, etc.)
        financial_keywords = ['credit', 'card', 'bank', 'account', 'payment', 'transaction']
        medical_keywords = ['patient', 'diagnosis', 'prescription', 'medical', 'health']

        is_financial = any(kw in text.lower() for kw in financial_keywords)
        is_medical = any(kw in text.lower() for kw in medical_keywords)

        return {
            "text_length": text_length,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "estimated_entities": total_entities,
            "entity_density": entity_density,
            "has_multilingual": has_multilingual,
            "has_complex_formatting": has_complex_formatting,
            "is_financial_domain": is_financial,
            "is_medical_domain": is_medical,
            "is_long_text": text_length > 200,
            "is_high_density": entity_density > 0.3
        }

    def select_smart_strategy(self, characteristics: Dict) -> str:
        """
        Select optimal detection strategy based on text characteristics.

        Args:
            characteristics: Text analysis results

        Returns:
            Recommended strategy name
        """
        # High-risk domains (financial, medical) -> High Precision to reduce false positives
        if characteristics["is_financial_domain"] or characteristics["is_medical_domain"]:
            return "high_precision"

        # Long text with low entity density -> Balanced (efficiency)
        if characteristics["is_long_text"] and characteristics["entity_density"] < 0.1:
            return "balanced"

        # High entity density or complex formatting -> High Recall to catch everything
        if characteristics["is_high_density"] or characteristics["has_complex_formatting"]:
            return "high_recall"

        # Multilingual text -> High Recall (NER may struggle)
        if characteristics["has_multilingual"]:
            return "high_recall"

        # Default: Balanced
        return "balanced"

    def mask_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        use_smart_mode: Optional[bool] = None
    ) -> Tuple[str, List[Dict], Dict]:
        """
        Mask PII in text.

        Args:
            text: Input text containing PII
            session_id: Optional session ID for tracking
            use_smart_mode: Override smart mode setting

        Returns:
            Tuple of (masked_text, entities_info, metadata)
        """
        start_time = time.time()

        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid4())

        # Analyze text characteristics
        characteristics = self.analyze_text_characteristics(text)

        # Determine strategy
        effective_smart_mode = use_smart_mode if use_smart_mode is not None else self.enable_smart_mode

        if effective_smart_mode:
            recommended_strategy = self.select_smart_strategy(characteristics)
            print(f"  Smart Mode: Selected '{recommended_strategy}' strategy based on text analysis")

            # Create new mask engine with recommended strategy if different
            if recommended_strategy != self.strategy:
                self.mask_engine = PIIMaskEngine(
                    self.db,
                    detection_method=self.detection_method,
                    strategy=recommended_strategy,
                    model_size=self.model_size
                )

        # Perform masking
        masked_text, entities_info = self.mask_engine.mask_text(
            text=text,
            session_id=session_id
        )

        processing_time = time.time() - start_time

        metadata = {
            "session_id": session_id,
            "original_length": len(text),
            "masked_length": len(masked_text),
            "entities_detected": len(entities_info),
            "processing_time_ms": round(processing_time * 1000, 2),
            "detection_method": self.detection_method,
            "strategy": recommended_strategy if effective_smart_mode else self.strategy,
            "smart_mode_used": effective_smart_mode,
            "text_characteristics": characteristics
        }

        return masked_text, entities_info, metadata

    def unmask_text(self, masked_text: str, session_id: str) -> Tuple[str, Dict]:
        """
        Recover original PII from masked text.

        Args:
            masked_text: Text with placeholders
            session_id: Session ID used during masking

        Returns:
            Tuple of (recovered_text, metadata)
        """
        start_time = time.time()

        # Perform recovery
        recovered_text = self.recover_engine.recover_text(
            masked_text=masked_text,
            session_id=session_id
        )

        processing_time = time.time() - start_time

        # Get session entities for verification
        entities = self.recover_engine.get_session_entities(session_id)

        metadata = {
            "session_id": session_id,
            "masked_length": len(masked_text),
            "recovered_length": len(recovered_text),
            "entities_recovered": len(entities),
            "processing_time_ms": round(processing_time * 1000, 2)
        }

        return recovered_text, metadata

    def run_complete_workflow(self, text: str, verbose: bool = True) -> Dict:
        """
        Run complete mask -> unmask workflow.

        Args:
            text: Input text containing PII
            verbose: Print detailed output

        Returns:
            Workflow results dictionary
        """
        session_id = str(uuid4())

        if verbose:
            print("\n" + "=" * 80)
            print("COMPLETE MASK -> UNMASK WORKFLOW")
            print("=" * 80)
            print(f"\n[1] ORIGINAL TEXT:")
            print(f"    {text[:200]}..." if len(text) > 200 else f"    {text}")

        # Step 1: Mask
        if verbose:
            print(f"\n[2] MASKING...")

        masked_text, entities_info, mask_metadata = self.mask_text(text, session_id)

        if verbose:
            print(f"    Detected {len(entities_info)} PII entities")
            print(f"    Masked text: {masked_text[:200]}..." if len(masked_text) > 200 else f"    Masked text: {masked_text}")

            if entities_info:
                print(f"\n    Entities detected:")
                for entity in entities_info[:5]:  # Show first 5
                    print(f"      - {entity['entity_type']}: '{entity['entity_value']}' -> '{entity['placeholder']}'")
                if len(entities_info) > 5:
                    print(f"      ... and {len(entities_info) - 5} more")

        # Step 2: Simulate LLM processing (in real scenario, send to LLM)
        if verbose:
            print(f"\n[3] SIMULATING LLM PROCESSING...")
            print(f"    (In production, masked text would be sent to external LLM)")

        # Simulate LLM response that might include placeholders
        llm_response = f"Based on your query about {masked_text[:100]}..., here is my response with the same placeholders preserved."

        # Step 3: Unmask
        if verbose:
            print(f"\n[4] UNMASKING...")

        recovered_text, unmask_metadata = self.unmask_text(masked_text, session_id)

        if verbose:
            print(f"    Recovered text: {recovered_text[:200]}..." if len(recovered_text) > 200 else f"    Recovered text: {recovered_text}")

        # Verify recovery
        text_match = recovered_text == text

        if verbose:
            print(f"\n[5] VERIFICATION:")
            print(f"    Original == Recovered: {text_match}")
            print(f"    Total processing time: {mask_metadata['processing_time_ms'] + unmask_metadata['processing_time_ms']:.2f}ms")

        return {
            "original_text": text,
            "masked_text": masked_text,
            "recovered_text": recovered_text,
            "entities": entities_info,
            "mask_metadata": mask_metadata,
            "unmask_metadata": unmask_metadata,
            "verification": {
                "text_match": text_match,
                "entities_count": len(entities_info)
            }
        }

    def compare_detection_methods(self, text: str) -> Dict:
        """
        Compare different PII detection methods on the same text.

        Args:
            text: Input text to test

        Returns:
            Comparison results
        """
        print("\n" + "=" * 80)
        print("DETECTION METHODS COMPARISON")
        print("=" * 80)
        print(f"\nTest text: {text[:100]}..." if len(text) > 100 else f"\nTest text: {text}")

        results = {}
        methods = [
            ("Regex", RegexExtractor()),
            ("Presidio", PresidioExtractor(strategy=self.strategy)),
        ]

        # Add LLM-based methods if available
        try:
            llm_extractor = LLMExtractor(model_size=self.model_size, strategy=self.strategy)
            if llm_extractor.available:
                methods.append(("LLM", llm_extractor))
                methods.append(("E2E", E2EExtractor(model_size=self.model_size, strategy=self.strategy)))
        except Exception as e:
            print(f"\nNote: LLM-based methods unavailable ({e})")

        for method_name, extractor in methods:
            if hasattr(extractor, 'available') and not extractor.available:
                print(f"\n{method_name}: Not available")
                continue

            start_time = time.time()
            try:
                entities = extractor.extract(text)
                processing_time = time.time() - start_time

                results[method_name] = {
                    "entities_count": len(entities),
                    "entities": entities,
                    "processing_time_ms": round(processing_time * 1000, 2)
                }

                print(f"\n{method_name}:")
                print(f"  Entities found: {len(entities)}")
                print(f"  Processing time: {processing_time * 1000:.2f}ms")
                for entity in entities[:3]:
                    print(f"    - {entity['entity_type']}: '{entity['entity_value']}'")
                if len(entities) > 3:
                    print(f"    ... and {len(entities) - 3} more")

            except Exception as e:
                print(f"\n{method_name}: Error - {e}")
                results[method_name] = {"error": str(e)}

        return results

    def generate_sample_texts(self, count: int = 5) -> List[Dict]:
        """
        Generate sample texts with known PII for testing.

        Args:
            count: Number of samples to generate

        Returns:
            List of sample dictionaries with text and expected entities
        """
        samples = []

        for i in range(count):
            # Generate PII values
            name = self.pii_gen.get_fake_value("person_name", f"user_{i}")
            email = self.pii_gen.get_fake_value("email_address", f"user_{i}")
            phone = self.pii_gen.get_fake_value("phone_number", f"user_{i}")
            card = self.card_gen.generate_card(format_type="space_4")
            address = self.addr_gen.generate_address(country="US")

            # Create sample text
            text = f"Customer {name} can be reached at {email} or {phone}. " \
                   f"Their credit card ending in {card[-4:]} is on file. " \
                   f"Shipping address: {address}"

            # Create full card number sample (more complex)
            text_complex = f"Dear {name},\n\nYour order has been processed. " \
                          f"Payment confirmed with card {card}.\n" \
                          f"Delivery to: {address}\n" \
                          f"For questions, contact {email} or call {phone}.\n\n" \
                          f"Best regards,\nCustomer Service"

            samples.append({
                "id": i,
                "text_simple": text,
                "text_complex": text_complex,
                "expected_entities": {
                    "PERSON": [name],
                    "EMAIL": [email],
                    "PHONE": [phone],
                    "CREDIT_CARD": [card],
                    "LOCATION": [address]
                }
            })

        return samples

    def run_demo_suite(self):
        """Run complete demo suite with multiple test cases."""
        print("\n" + "=" * 80)
        print("PII MASK/UNMASK DEMO SUITE")
        print("=" * 80)

        # Demo 1: Basic mask/unmask workflow
        print("\n\n" + "-" * 60)
        print("DEMO 1: Basic Mask/Unmask Workflow")
        print("-" * 60)

        simple_text = "My name is John Smith and my email is john.smith@example.com. " \
                     "Call me at +1-555-123-4567."

        result1 = self.run_complete_workflow(simple_text)

        # Demo 2: Complex text with multiple PII types
        print("\n\n" + "-" * 60)
        print("DEMO 2: Complex Text with Multiple PII Types")
        print("-" * 60)

        complex_text = """
Dear Customer,

Your order #12345 has been confirmed. Payment processed with credit card 4532 1111 2222 3333.

Billing Address:
Jane Doe
123 Main Street, Apt 4B
New York, NY 10001

Contact Information:
- Email: jane.doe@company.org
- Phone: (555) 987-6543
- Alt Phone: +1.555.123.4567

Your order will be delivered by our partner. If you have questions,
contact support@company.org or visit our office at 456 Commerce Blvd, Suite 100, Los Angeles, CA 90001.

Order placed: 2024-12-15
Expected delivery: 2024-12-20

Thank you for your business!
        """.strip()

        result2 = self.run_complete_workflow(complex_text)

        # Demo 3: Detection methods comparison
        print("\n\n" + "-" * 60)
        print("DEMO 3: Detection Methods Comparison")
        print("-" * 60)

        comparison_result = self.compare_detection_methods(simple_text)

        # Demo 4: Smart mode demonstration
        print("\n\n" + "-" * 60)
        print("DEMO 4: Smart Mode Strategy Selection")
        print("-" * 60)

        # Financial domain text
        financial_text = "Customer Jane Smith needs to update payment method. " \
                        "Current card on file: 4111 1111 1111 1111. " \
                        "Bank account: 123456789. Transaction ID: TXN-2024-001."

        print("\nFinancial Domain Text:")
        chars = self.analyze_text_characteristics(financial_text)
        strategy = self.select_smart_strategy(chars)
        print(f"  Recommended strategy: {strategy}")
        print(f"  Reason: Financial domain detected (is_financial: {chars['is_financial_domain']})")

        # High density text
        high_density_text = "john@a.com, jane@b.com, bob@c.com, alice@d.com, " \
                           "555-1111, 555-2222, 555-3333"

        print("\nHigh Density Text:")
        chars = self.analyze_text_characteristics(high_density_text)
        strategy = self.select_smart_strategy(chars)
        print(f"  Recommended strategy: {strategy}")
        print(f"  Reason: High entity density ({chars['entity_density']:.2%})")

        # Demo 5: Generated samples test
        print("\n\n" + "-" * 60)
        print("DEMO 5: Generated Sample Tests")
        print("-" * 60)

        samples = self.generate_sample_texts(count=3)

        for sample in samples:
            print(f"\nSample {sample['id'] + 1}:")
            result = self.run_complete_workflow(sample['text_simple'], verbose=False)
            print(f"  Entities detected: {result['verification']['entities_count']}")
            print(f"  Recovery verified: {result['verification']['text_match']}")

        print("\n\n" + "=" * 80)
        print("DEMO SUITE COMPLETED")
        print("=" * 80)


def main():
    """Main entry point for the demo."""
    print("\n" + "=" * 80)
    print("End-to-End PII Mask/Unmask Demo")
    print("=" * 80)

    # Create demo instance
    demo = PIIMaskUnmaskDemo(
        detection_method="E2E",
        strategy="balanced",
        model_size="medium",
        enable_smart_mode=True
    )

    # Run demo suite
    demo.run_demo_suite()

    # Interactive mode (optional)
    print("\n\n" + "-" * 60)
    print("INTERACTIVE MODE")
    print("-" * 60)
    print("Enter text to test PII detection (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if user_input.strip():
                result = demo.run_complete_workflow(user_input)
                print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDemo completed. Thank you!")


if __name__ == "__main__":
    main()

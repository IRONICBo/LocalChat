# -*- coding: utf-8 -*-
"""
Smart Mode - Automatic Detection Strategy Selection

This module provides intelligent strategy selection for PII detection based on:
1. Text characteristics (length, density, complexity)
2. Domain detection (financial, medical, etc.)
3. Input source type (conversation, document, PDF)
4. Historical performance data

The Smart Mode analyzes input text and automatically selects the optimal
detection strategy (high_recall, balanced, high_precision) to maximize
accuracy while maintaining acceptable performance.
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

from config import DetectionStrategy

logger = logging.getLogger(__name__)


class TextDomain(str, Enum):
    """Domain classification for input text."""
    FINANCIAL = "financial"
    MEDICAL = "medical"
    LEGAL = "legal"
    GENERAL = "general"
    TECHNICAL = "technical"
    CUSTOMER_SERVICE = "customer_service"


class InputSourceType(str, Enum):
    """Type of input source."""
    CONVERSATION = "conversation"  # Chat/dialog
    DOCUMENT = "document"  # Long-form document
    PDF = "pdf"  # PDF extracted text
    FORM = "form"  # Structured form data
    EMAIL = "email"  # Email content
    UNKNOWN = "unknown"


@dataclass
class TextAnalysisResult:
    """Result of text characteristic analysis."""
    text_length: int
    word_count: int
    sentence_count: int
    estimated_pii_density: float
    has_multilingual: bool
    has_complex_formatting: bool
    domain: TextDomain
    source_type: InputSourceType
    complexity_score: float  # 0-1 scale

    # Detailed PII estimates
    estimated_emails: int
    estimated_phones: int
    estimated_credit_cards: int
    estimated_names: int
    estimated_addresses: int


class SmartModeAnalyzer:
    """
    Analyzes text characteristics for intelligent strategy selection.

    This analyzer examines various aspects of input text to determine
    the most appropriate detection strategy for optimal results.
    """

    # Domain keyword dictionaries
    DOMAIN_KEYWORDS = {
        TextDomain.FINANCIAL: [
            'credit', 'card', 'bank', 'account', 'payment', 'transaction',
            'balance', 'deposit', 'withdraw', 'loan', 'mortgage', 'investment',
            'iban', 'routing', 'wire', 'transfer', 'billing', 'invoice'
        ],
        TextDomain.MEDICAL: [
            'patient', 'diagnosis', 'prescription', 'medical', 'health',
            'doctor', 'hospital', 'treatment', 'symptom', 'medication',
            'insurance', 'clinic', 'nurse', 'surgery', 'lab', 'test'
        ],
        TextDomain.LEGAL: [
            'attorney', 'lawyer', 'court', 'legal', 'lawsuit', 'contract',
            'defendant', 'plaintiff', 'witness', 'testimony', 'evidence',
            'case', 'judge', 'settlement', 'filing', 'jurisdiction'
        ],
        TextDomain.TECHNICAL: [
            'api', 'server', 'database', 'endpoint', 'token', 'key',
            'configuration', 'deployment', 'error', 'log', 'debug',
            'ip', 'port', 'host', 'request', 'response', 'header'
        ],
        TextDomain.CUSTOMER_SERVICE: [
            'customer', 'support', 'ticket', 'order', 'delivery', 'shipping',
            'return', 'refund', 'complaint', 'inquiry', 'service', 'help'
        ]
    }

    # Source type indicators
    SOURCE_INDICATORS = {
        InputSourceType.EMAIL: [
            r'^(from|to|subject|cc|bcc):',
            r'dear\s+\w+',
            r'(best|kind)\s+regards',
            r'sincerely',
            r'sent\s+from'
        ],
        InputSourceType.FORM: [
            r'^\s*(name|email|phone|address|city|state|zip)\s*:',
            r'(required|optional)\s*\*?',
            r'please\s+enter',
            r'(first|last)\s+name'
        ],
        InputSourceType.PDF: [
            r'page\s+\d+\s+of\s+\d+',
            r'\[\d+\]',
            r'figure\s+\d+',
            r'table\s+\d+'
        ]
    }

    def __init__(self):
        """Initialize the analyzer."""
        # Compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile frequently used regex patterns."""
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        self.phone_pattern = re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        )
        self.cc_pattern = re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        )
        self.name_pattern = re.compile(
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        )
        self.address_pattern = re.compile(
            r'\d+\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Dr|Ln|Ct|Way)',
            re.IGNORECASE
        )
        self.multilingual_pattern = re.compile(r'[^\x00-\x7F]+')
        self.complex_format_pattern = re.compile(r'[+().\-]{2,}')

    def analyze(self, text: str, source_hint: Optional[InputSourceType] = None) -> TextAnalysisResult:
        """
        Analyze text characteristics.

        Args:
            text: Input text to analyze
            source_hint: Optional hint about the source type

        Returns:
            TextAnalysisResult with detailed analysis
        """
        # Basic metrics
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')

        # PII estimates
        emails = len(self.email_pattern.findall(text))
        phones = len(self.phone_pattern.findall(text))
        credit_cards = len(self.cc_pattern.findall(text))
        names = len(self.name_pattern.findall(text))
        addresses = len(self.address_pattern.findall(text))

        total_pii = emails + phones + credit_cards + names + addresses
        pii_density = total_pii / max(word_count, 1)

        # Complexity indicators
        has_multilingual = bool(self.multilingual_pattern.search(text))
        has_complex_formatting = bool(self.complex_format_pattern.search(text))

        # Domain detection
        domain = self._detect_domain(text)

        # Source type detection
        source_type = source_hint if source_hint else self._detect_source_type(text)

        # Calculate complexity score (0-1)
        complexity_score = self._calculate_complexity_score(
            text_length=text_length,
            pii_density=pii_density,
            has_multilingual=has_multilingual,
            has_complex_formatting=has_complex_formatting,
            sentence_count=sentence_count
        )

        return TextAnalysisResult(
            text_length=text_length,
            word_count=word_count,
            sentence_count=sentence_count,
            estimated_pii_density=pii_density,
            has_multilingual=has_multilingual,
            has_complex_formatting=has_complex_formatting,
            domain=domain,
            source_type=source_type,
            complexity_score=complexity_score,
            estimated_emails=emails,
            estimated_phones=phones,
            estimated_credit_cards=credit_cards,
            estimated_names=names,
            estimated_addresses=addresses
        )

    def _detect_domain(self, text: str) -> TextDomain:
        """Detect the domain of the text based on keywords."""
        text_lower = text.lower()

        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            domain_scores[domain] = score

        # Get domain with highest score (minimum 2 matches)
        if domain_scores:
            best_domain, best_score = max(domain_scores.items(), key=lambda x: x[1])
            if best_score >= 2:
                return best_domain

        return TextDomain.GENERAL

    def _detect_source_type(self, text: str) -> InputSourceType:
        """Detect the source type of the text."""
        text_lower = text.lower()

        for source_type, patterns in self.SOURCE_INDICATORS.items():
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            if matches >= 2:
                return source_type

        # Heuristics for conversation vs document
        avg_sentence_len = len(text) / max(text.count('.'), 1)

        if avg_sentence_len < 100 and text.count('\n') > text.count('.'):
            return InputSourceType.CONVERSATION

        if len(text) > 500:
            return InputSourceType.DOCUMENT

        return InputSourceType.UNKNOWN

    def _calculate_complexity_score(
        self,
        text_length: int,
        pii_density: float,
        has_multilingual: bool,
        has_complex_formatting: bool,
        sentence_count: int
    ) -> float:
        """Calculate overall complexity score (0-1)."""
        score = 0.0

        # Length factor (longer = more complex)
        if text_length > 500:
            score += 0.2
        if text_length > 1000:
            score += 0.1

        # PII density factor
        if pii_density > 0.1:
            score += 0.2
        if pii_density > 0.3:
            score += 0.1

        # Multilingual factor
        if has_multilingual:
            score += 0.15

        # Complex formatting factor
        if has_complex_formatting:
            score += 0.1

        # Sentence structure
        if sentence_count > 5:
            score += 0.1

        return min(score, 1.0)


class SmartModeSelector:
    """
    Selects optimal detection strategy based on text analysis.

    This selector uses the analysis results to recommend the best
    detection strategy for a given input.
    """

    def __init__(self, analyzer: Optional[SmartModeAnalyzer] = None):
        """
        Initialize the selector.

        Args:
            analyzer: Optional pre-configured analyzer
        """
        self.analyzer = analyzer or SmartModeAnalyzer()

    def select_strategy(
        self,
        text: str,
        source_hint: Optional[InputSourceType] = None,
        override_domain: Optional[TextDomain] = None
    ) -> Tuple[str, Dict]:
        """
        Select the optimal detection strategy for the given text.

        Args:
            text: Input text to analyze
            source_hint: Optional hint about source type
            override_domain: Optional override for domain detection

        Returns:
            Tuple of (strategy_name, selection_metadata)
        """
        # Analyze text
        analysis = self.analyzer.analyze(text, source_hint)

        # Override domain if specified
        if override_domain:
            analysis.domain = override_domain

        # Decision logic
        strategy, reason = self._make_decision(analysis)

        metadata = {
            "analysis": {
                "text_length": analysis.text_length,
                "word_count": analysis.word_count,
                "pii_density": analysis.estimated_pii_density,
                "complexity_score": analysis.complexity_score,
                "domain": analysis.domain.value,
                "source_type": analysis.source_type.value,
                "has_multilingual": analysis.has_multilingual,
                "has_complex_formatting": analysis.has_complex_formatting
            },
            "estimated_pii": {
                "emails": analysis.estimated_emails,
                "phones": analysis.estimated_phones,
                "credit_cards": analysis.estimated_credit_cards,
                "names": analysis.estimated_names,
                "addresses": analysis.estimated_addresses
            },
            "selected_strategy": strategy,
            "selection_reason": reason
        }

        logger.info(f"Smart Mode selected '{strategy}' strategy: {reason}")

        return strategy, metadata

    def _make_decision(self, analysis: TextAnalysisResult) -> Tuple[str, str]:
        """
        Make strategy decision based on analysis.

        Returns:
            Tuple of (strategy_name, reason)
        """
        # Rule 1: High-risk domains -> High Precision
        # Reduces false positives which could cause issues in sensitive contexts
        if analysis.domain in [TextDomain.FINANCIAL, TextDomain.MEDICAL, TextDomain.LEGAL]:
            return (
                "high_precision",
                f"High-risk domain ({analysis.domain.value}) detected - minimizing false positives"
            )

        # Rule 2: High PII density -> High Recall
        # Many PII entities means we want to catch everything
        if analysis.estimated_pii_density > 0.3:
            return (
                "high_recall",
                f"High PII density ({analysis.estimated_pii_density:.2%}) - maximizing detection"
            )

        # Rule 3: Complex formatting -> High Recall
        # Complex patterns may be missed by standard detection
        if analysis.has_complex_formatting:
            return (
                "high_recall",
                "Complex formatting detected - using broader detection patterns"
            )

        # Rule 4: Multilingual text -> High Recall
        # NER models may struggle with non-English text
        if analysis.has_multilingual:
            return (
                "high_recall",
                "Multilingual content detected - using broader detection"
            )

        # Rule 5: Short conversation -> High Precision
        # Less context available, want to avoid over-detection
        if analysis.source_type == InputSourceType.CONVERSATION and analysis.text_length < 200:
            return (
                "high_precision",
                "Short conversation - prioritizing precision"
            )

        # Rule 6: Long document with low density -> Balanced
        # Efficiency matters for long texts
        if analysis.text_length > 500 and analysis.estimated_pii_density < 0.1:
            return (
                "balanced",
                "Long text with low PII density - balanced approach for efficiency"
            )

        # Rule 7: Technical domain -> Balanced
        # May have IP addresses, API keys that need careful handling
        if analysis.domain == TextDomain.TECHNICAL:
            return (
                "balanced",
                "Technical domain - balanced approach for varied PII types"
            )

        # Rule 8: Customer service -> Balanced
        # Mix of sensitive data and general info
        if analysis.domain == TextDomain.CUSTOMER_SERVICE:
            return (
                "balanced",
                "Customer service context - balanced approach"
            )

        # Rule 9: High complexity score -> High Recall
        if analysis.complexity_score > 0.6:
            return (
                "high_recall",
                f"High complexity ({analysis.complexity_score:.2f}) - maximizing detection"
            )

        # Default: Balanced
        return (
            "balanced",
            "Default balanced strategy for general text"
        )

    def suggest_model_size(self, analysis: TextAnalysisResult) -> Tuple[str, str]:
        """
        Suggest appropriate model size based on analysis.

        Args:
            analysis: Text analysis result

        Returns:
            Tuple of (model_size, reason)
        """
        # High-risk domains need best accuracy
        if analysis.domain in [TextDomain.FINANCIAL, TextDomain.MEDICAL, TextDomain.LEGAL]:
            return ("large", "High-risk domain requires maximum accuracy")

        # High complexity needs better model
        if analysis.complexity_score > 0.5:
            return ("large", "Complex text benefits from larger model")

        # Multilingual content
        if analysis.has_multilingual:
            return ("large", "Multilingual content requires larger model")

        # Low density, short text -> smaller model for speed
        if analysis.estimated_pii_density < 0.1 and analysis.text_length < 200:
            return ("medium", "Simple text - medium model sufficient")

        # Default
        return ("medium", "Standard complexity - medium model")


class SmartModeEngine:
    """
    High-level engine for Smart Mode PII detection.

    Combines analyzer and selector with the PII detection engine
    for seamless smart mode operation.
    """

    def __init__(self):
        """Initialize the Smart Mode engine."""
        self.analyzer = SmartModeAnalyzer()
        self.selector = SmartModeSelector(self.analyzer)

    def process_text(
        self,
        text: str,
        source_hint: Optional[InputSourceType] = None,
        override_strategy: Optional[str] = None
    ) -> Dict:
        """
        Process text with smart mode strategy selection.

        Args:
            text: Input text
            source_hint: Optional source type hint
            override_strategy: Optional strategy override

        Returns:
            Processing result with strategy info
        """
        # Analyze text
        analysis = self.analyzer.analyze(text, source_hint)

        # Select strategy (unless overridden)
        if override_strategy:
            strategy = override_strategy
            strategy_metadata = {"override": True, "original": override_strategy}
        else:
            strategy, strategy_metadata = self.selector.select_strategy(text, source_hint)

        # Suggest model size
        model_size, model_reason = self.selector.suggest_model_size(analysis)

        return {
            "analysis": {
                "text_length": analysis.text_length,
                "word_count": analysis.word_count,
                "pii_density": analysis.estimated_pii_density,
                "complexity_score": analysis.complexity_score,
                "domain": analysis.domain.value,
                "source_type": analysis.source_type.value
            },
            "recommendations": {
                "strategy": strategy,
                "strategy_reason": strategy_metadata.get("selection_reason", ""),
                "model_size": model_size,
                "model_reason": model_reason
            },
            "estimated_pii_counts": {
                "emails": analysis.estimated_emails,
                "phones": analysis.estimated_phones,
                "credit_cards": analysis.estimated_credit_cards,
                "names": analysis.estimated_names,
                "addresses": analysis.estimated_addresses,
                "total": (analysis.estimated_emails + analysis.estimated_phones +
                         analysis.estimated_credit_cards + analysis.estimated_names +
                         analysis.estimated_addresses)
            }
        }


# Convenience function for quick analysis
def analyze_and_recommend(text: str) -> Dict:
    """
    Quick analysis and recommendation for a text.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with analysis and recommendations
    """
    engine = SmartModeEngine()
    return engine.process_text(text)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Smart Mode Demo")
    print("=" * 60)

    test_texts = [
        # Financial domain
        "Dear Customer, your credit card 4532-1111-2222-3333 has been charged $150.00. "
        "Please contact our billing department at billing@bank.com for questions.",

        # Medical domain
        "Patient John Smith (DOB: 1980-05-15) was diagnosed with hypertension. "
        "Dr. Jane Doe prescribed medication. Contact: 555-123-4567.",

        # General conversation
        "Hi, my name is Alice and you can reach me at alice@example.com",

        # High density
        "john@a.com, jane@b.com, bob@c.com, alice@d.com, 555-1111, 555-2222",

        # Multilingual
        "Customer 张三 can be reached at zhang.san@company.com"
    ]

    engine = SmartModeEngine()

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text[:80]}...")
        result = engine.process_text(text)
        print(f"Domain: {result['analysis']['domain']}")
        print(f"Recommended Strategy: {result['recommendations']['strategy']}")
        print(f"Reason: {result['recommendations']['strategy_reason']}")
        print(f"Recommended Model: {result['recommendations']['model_size']}")

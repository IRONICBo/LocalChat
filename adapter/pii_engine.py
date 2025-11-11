# -*- coding: utf-8 -*-
"""
PII Detection and Masking Engine (Enhanced)

Integrates four PII detection methods:
1. Regex-based extraction (Enhanced with complex patterns)
2. Presidio (Deep Learning NER)
3. LLM (Ollama) extraction (Improved with few-shot learning)
4. E2E (End-to-End): Presidio + LLM (Optimized merging strategy)

Improvements:
- Support for larger LLM models (4B/7B)
- Enhanced prompt engineering with few-shot examples
- Improved JSON parsing and error handling
- Additional regex patterns for high-frequency FN cases
- Configurable detection thresholds and strategies
- Optimized entity merging logic

Provides masking and recovery functionality with database persistence.
"""

import re
import json
import hashlib
import requests
import logging
from typing import List, Dict, Tuple, Optional
from uuid import uuid4
from datetime import datetime
from collections import defaultdict

from sqlalchemy.orm import Session
from models import (
    SessionInfo, ConversationHistory, MaskMapping,
    SensitiveEntity, ProcessingLog, engine
)
from config import PIIConfig, DetectionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIIExtractor:
    """Base class for PII extractors"""

    def __init__(self, name: str):
        self.name = name
        self.available = True

    def extract(self, text: str) -> List[Dict]:
        """
        Extract PII entities from text

        Returns:
            List[Dict]: [{
                "entity_type": str,
                "start": int,
                "end": int,
                "entity_value": str,
                "confidence": float
            }, ...]
        """
        raise NotImplementedError


class RegexExtractor(PIIExtractor):
    """Enhanced Regex-based PII extractor with additional patterns for high-frequency FN cases"""

    def __init__(self, enable_enhanced: bool = None):
        super().__init__("Regex")

        # Enable enhanced patterns from config if not specified
        if enable_enhanced is None:
            enable_enhanced = PIIConfig.ENABLE_ENHANCED_REGEX

        self.enable_enhanced = enable_enhanced

        # Basic patterns
        self.patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "URL": r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)",
            "DATE": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b",
        }

        # Enhanced patterns for high-frequency FN cases
        if self.enable_enhanced:
            self._add_enhanced_patterns()

    def _add_enhanced_patterns(self):
        """Add enhanced patterns from config to catch more edge cases"""
        # Complex phone number formats
        if "PHONE_COMPLEX" in PIIConfig.ENHANCED_PATTERNS:
            self.patterns["PHONE_COMPLEX"] = PIIConfig.ENHANCED_PATTERNS["PHONE_COMPLEX"]

        # Non-standard addresses
        if "ADDRESS_PATTERN" in PIIConfig.ENHANCED_PATTERNS:
            self.patterns["ADDRESS"] = PIIConfig.ENHANCED_PATTERNS["ADDRESS_PATTERN"]

        # Person name patterns
        if "PERSON_PATTERN" in PIIConfig.ENHANCED_PATTERNS:
            self.patterns["PERSON"] = PIIConfig.ENHANCED_PATTERNS["PERSON_PATTERN"]

        # Additional credit card formats
        if "CREDIT_CARD_EXTRA" in PIIConfig.ENHANCED_PATTERNS:
            extra_patterns = PIIConfig.ENHANCED_PATTERNS["CREDIT_CARD_EXTRA"]
            if isinstance(self.patterns["CREDIT_CARD"], str):
                self.patterns["CREDIT_CARD"] = [self.patterns["CREDIT_CARD"]] + extra_patterns
            else:
                self.patterns["CREDIT_CARD"].extend(extra_patterns)

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using regex patterns (basic + enhanced)"""
        entities = []
        seen_spans = set()  # Track seen spans to avoid duplicates

        for entity_type, patterns in self.patterns.items():
            # Handle both single pattern string and list of patterns
            pattern_list = patterns if isinstance(patterns, list) else [patterns]

            for pattern in pattern_list:
                try:
                    for match in re.finditer(pattern, text):
                        span = (match.start(), match.end())

                        # Skip if we've already found an entity at this span
                        if span in seen_spans:
                            continue

                        # Normalize entity type (remove _COMPLEX, _EXTRA suffixes)
                        normalized_type = entity_type.split('_')[0] if '_' in entity_type else entity_type

                        entities.append({
                            "entity_type": normalized_type,
                            "start": match.start(),
                            "end": match.end(),
                            "entity_value": match.group(),
                            "confidence": 1.0 if not self.enable_enhanced else 0.9  # Slightly lower confidence for enhanced patterns
                        })

                        seen_spans.add(span)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {entity_type}: {e}")
                    continue

        return entities


class PresidioExtractor(PIIExtractor):
    """Presidio-based PII extractor with configurable confidence threshold"""

    def __init__(self, strategy: str = None, min_confidence: float = None):
        super().__init__("Presidio")

        # Get threshold from strategy or use provided value
        self.strategy = strategy or PIIConfig.DETECTION_STRATEGY
        thresholds = PIIConfig.get_thresholds(self.strategy)
        self.min_confidence = min_confidence if min_confidence is not None else thresholds["presidio_min"]

        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Initialize NLP engine
            provider = NlpEngineProvider()
            nlp_engine = provider.create_engine()

            # Initialize analyzer
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.available = True
            logger.info(f"Presidio initialized with min_confidence={self.min_confidence}")
        except Exception as e:
            logger.warning(f"Presidio not available: {e}")
            self.available = False

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using Presidio with confidence filtering"""
        if not self.available:
            return []

        try:
            # Analyze text
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=None,  # Detect all entity types
                score_threshold=self.min_confidence  # Apply threshold at analysis level
            )

            # Convert to standard format
            entities = []
            for result in results:
                # Map entity type to standardized format
                entity_type = PIIConfig.ENTITY_TYPE_MAPPING.get(
                    result.entity_type,
                    result.entity_type
                )

                entities.append({
                    "entity_type": entity_type,
                    "start": result.start,
                    "end": result.end,
                    "entity_value": text[result.start:result.end],
                    "confidence": result.score
                })

            logger.debug(f"Presidio found {len(entities)} entities above threshold {self.min_confidence}")
            return entities
        except Exception as e:
            logger.error(f"Presidio extraction error: {e}")
            return []


class LLMExtractor(PIIExtractor):
    """Enhanced LLM-based PII extractor using Ollama with few-shot learning"""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        model_size: str = None,
        strategy: str = None,
        min_confidence: float = None
    ):
        super().__init__("LLM_Ollama")

        # Get LLM configuration
        if model_size or not model:
            llm_config = PIIConfig.get_llm_config(model_size)
            self.model = model or llm_config["model"]
            self.timeout = llm_config["timeout"]
        else:
            self.model = model
            self.timeout = 60

        self.base_url = base_url or PIIConfig.OLLAMA_API_URL

        # Get threshold from strategy
        self.strategy = strategy or PIIConfig.DETECTION_STRATEGY
        thresholds = PIIConfig.get_thresholds(self.strategy)
        self.min_confidence = min_confidence if min_confidence is not None else thresholds["llm_min"]

        # Prompt settings
        self.enable_few_shot = PIIConfig.ENABLE_FEW_SHOT
        self.enable_cot = PIIConfig.ENABLE_CHAIN_OF_THOUGHT

        self.available = self._check_availability()

        if self.available:
            logger.info(f"LLM initialized: model={self.model}, timeout={self.timeout}s, min_confidence={self.min_confidence}")

    def _check_availability(self) -> bool:
        """Check if LLM API is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            logger.warning("LLM API not available")
            return False

    def _build_prompt(self, text: str) -> str:
        """Build enhanced prompt with few-shot examples and chain-of-thought"""
        prompt_parts = []

        # System instruction
        prompt_parts.append(
            "You are an expert at identifying Personal Identifiable Information (PII) in text. "
            "Your task is to extract ALL PII entities from the given text."
        )

        # Few-shot examples
        if self.enable_few_shot and PIIConfig.FEW_SHOT_EXAMPLES:
            prompt_parts.append("\n\n=== EXAMPLES ===\n")
            for i, example in enumerate(PIIConfig.FEW_SHOT_EXAMPLES[:PIIConfig.NUM_FEW_SHOT_EXAMPLES], 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Text: {example['text']}")
                prompt_parts.append(f"Entities: {json.dumps(example['entities'])}")

        # Chain-of-thought instruction
        if self.enable_cot:
            prompt_parts.append("\n\n=== INSTRUCTIONS ===\n")
            prompt_parts.append(
                "Follow these steps:\n"
                "1. Read the text carefully\n"
                "2. Identify all pieces of personal information\n"
                "3. For each piece, determine its entity type\n"
                "4. Extract the exact value from the text\n"
                "5. Return results as a JSON array"
            )

        # Entity types
        prompt_parts.append("\n\n=== ENTITY TYPES ===")
        prompt_parts.append(
            "Supported types: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, SSN, "
            "LOCATION, DATE, ORGANIZATION, IP_ADDRESS, URL, BANK_ACCOUNT"
        )

        # Output format (strict)
        prompt_parts.append("\n\n=== OUTPUT FORMAT ===")
        prompt_parts.append(
            "Return ONLY a valid JSON array in this EXACT format:\n"
            '[\n'
            '  {"entity_type": "TYPE", "entity_value": "value"},\n'
            '  {"entity_type": "TYPE", "entity_value": "value"}\n'
            ']\n'
            "\nIMPORTANT:\n"
            "- Use double quotes for all strings\n"
            "- Each entity must have both entity_type and entity_value\n"
            "- Return [] if no PII found\n"
            "- Do NOT include any text before or after the JSON array\n"
            "- Do NOT use markdown code blocks"
        )

        # Target text
        prompt_parts.append(f"\n\n=== TEXT TO ANALYZE ===\n{text}")

        # Final instruction
        prompt_parts.append("\n\n=== YOUR RESPONSE (JSON ARRAY ONLY) ===")

        return "".join(prompt_parts)

    def _fix_json_response(self, llm_response: str, retry_count: int = 0) -> str:
        """Enhanced JSON cleaning with multiple strategies"""

        if PIIConfig.ENABLE_AGGRESSIVE_JSON_CLEANING:
            # Strategy 1: Remove all non-JSON content
            llm_response = re.sub(r'```(?:json)?\s*', '', llm_response)
            llm_response = llm_response.strip()

            # Strategy 2: Extract JSON array with more flexible pattern
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', llm_response, re.DOTALL)
            if not json_match:
                # Try to find partial JSON
                json_match = re.search(r'\[.*', llm_response, re.DOTALL)
                if json_match:
                    llm_response = json_match.group(0)
                    # Try to close unclosed brackets
                    if llm_response.count('[') > llm_response.count(']'):
                        llm_response += ']' * (llm_response.count('[') - llm_response.count(']'))
                else:
                    return "[]"
            else:
                llm_response = json_match.group(0)

            # Strategy 3: Fix common formatting issues
            # Fix double underscores
            llm_response = re.sub(r'entity__type', 'entity_type', llm_response)
            llm_response = re.sub(r'entity__value', 'entity_value', llm_response)

            # Strategy 4: Fix unquoted values
            # Match: "entity_value": value (without quotes)
            def quote_value(match):
                key = match.group(1)
                value = match.group(2).strip()

                # Remove trailing commas, brackets
                value = re.sub(r'[,}\]]+$', '', value)

                # If not already quoted, quote it
                if not (value.startswith('"') and value.endswith('"')):
                    # Escape any quotes inside
                    value = value.replace('"', '\\"')
                    return f'"{key}": "{value}"'
                return match.group(0)

            llm_response = re.sub(
                r'"(entity_(?:type|value))":\s*([^,}\]]+)',
                quote_value,
                llm_response
            )

            # Strategy 5: Remove trailing commas
            llm_response = re.sub(r',(\s*[}\]])', r'\1', llm_response)

            # Strategy 6: Fix missing commas between objects
            llm_response = re.sub(r'\}\s*\{', '}, {', llm_response)

        return llm_response

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using LLM with enhanced prompt and JSON parsing"""
        if not self.available:
            return []

        # Build enhanced prompt
        prompt = self._build_prompt(text)

        # Retry logic for JSON parsing
        for retry in range(PIIConfig.MAX_JSON_PARSE_RETRIES):
            try:
                # Call LLM API
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    logger.error(f"LLM API error: {response.status_code}")
                    return []

                llm_response = response.json().get("response", "")

                # Log raw response for debugging
                if PIIConfig.ENABLE_DETAILED_LOGGING:
                    logger.debug(f"LLM raw response (attempt {retry + 1}):\n{llm_response[:500]}...")

                # Fix JSON formatting
                json_str = self._fix_json_response(llm_response, retry)

                # Parse JSON
                try:
                    entities_raw = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"LLM JSON parse error (attempt {retry + 1}/{PIIConfig.MAX_JSON_PARSE_RETRIES}): {e}")
                    if PIIConfig.ENABLE_DETAILED_LOGGING:
                        logger.debug(f"Failed JSON string:\n{json_str}")

                    # If this is not the last retry, continue to next attempt
                    if retry < PIIConfig.MAX_JSON_PARSE_RETRIES - 1:
                        continue
                    else:
                        logger.error(f"Failed to parse LLM response after {PIIConfig.MAX_JSON_PARSE_RETRIES} attempts")
                        return []

                # Successfully parsed JSON
                break

            except requests.exceptions.Timeout:
                logger.error(f"LLM API timeout after {self.timeout}s")
                return []
            except Exception as e:
                logger.error(f"LLM extraction error (attempt {retry + 1}): {e}")
                if retry >= PIIConfig.MAX_JSON_PARSE_RETRIES - 1:
                    return []
                continue

        # Convert to standard format with position information
        entities = []
        for entity in entities_raw:
            entity_value = entity.get("entity_value", "")
            entity_type = entity.get("entity_type", "UNKNOWN")

            if not entity_value:
                continue

            # Map entity type to standardized format
            entity_type = PIIConfig.ENTITY_TYPE_MAPPING.get(entity_type, entity_type)

            # Find position in text (case-insensitive, handle multiple occurrences)
            start = text.find(entity_value)
            if start == -1:
                # Try case-insensitive search
                start = text.lower().find(entity_value.lower())
                if start != -1:
                    # Use the actual text from the original
                    entity_value = text[start:start + len(entity_value)]
                else:
                    # Entity not found in text, skip it
                    logger.warning(f"Entity value '{entity_value}' not found in text")
                    continue

            # Default confidence for LLM
            confidence = 0.8

            # Apply confidence threshold
            if confidence < self.min_confidence:
                logger.debug(f"Skipping entity with low confidence: {entity_value} ({confidence})")
                continue

            entities.append({
                "entity_type": entity_type,
                "start": start,
                "end": start + len(entity_value),
                "entity_value": entity_value,
                "confidence": confidence
            })

        logger.info(f"LLM found {len(entities)} entities")
        return entities


class E2EExtractor(PIIExtractor):
    """Enhanced End-to-End extractor: Presidio + LLM with optimized merging"""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        model_size: str = None,
        strategy: str = None
    ):
        super().__init__("E2E (PII+LLM)")

        # Initialize both extractors with same strategy
        self.presidio_extractor = PresidioExtractor(strategy=strategy)
        self.llm_extractor = LLMExtractor(
            base_url=base_url,
            model=model,
            model_size=model_size,
            strategy=strategy
        )

        self.available = self.presidio_extractor.available or self.llm_extractor.available

        # Get merging configuration
        self.strategy = strategy or PIIConfig.DETECTION_STRATEGY
        self.overlap_threshold = PIIConfig.OVERLAP_THRESHOLD
        self.merge_preference = PIIConfig.MERGE_PREFERENCE

        if self.available:
            logger.info(
                f"E2E initialized: presidio={self.presidio_extractor.available}, "
                f"llm={self.llm_extractor.available}, strategy={self.strategy}"
            )

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using both Presidio and LLM, then intelligently merge"""
        if not self.available:
            return []

        # Extract using both methods (both may be available)
        presidio_entities = []
        llm_entities = []

        if self.presidio_extractor.available:
            presidio_entities = self.presidio_extractor.extract(text)
            logger.debug(f"Presidio found {len(presidio_entities)} entities")

        if self.llm_extractor.available:
            llm_entities = self.llm_extractor.extract(text)
            logger.debug(f"LLM found {len(llm_entities)} entities")

        # Tag entities with their source for merge preference
        for e in presidio_entities:
            e["source"] = "presidio"

        for e in llm_entities:
            e["source"] = "llm"

        # Merge and deduplicate with enhanced strategy
        merged_entities = self._merge_entities(
            presidio_entities + llm_entities,
            text
        )

        logger.info(f"E2E merged to {len(merged_entities)} entities (from {len(presidio_entities)} + {len(llm_entities)})")

        return merged_entities

    def _merge_entities(self, all_entities: List[Dict], text: str) -> List[Dict]:
        """
        Intelligently merge entities with multiple strategies:
        1. Remove exact duplicates
        2. Handle overlapping entities based on preference
        3. Preserve entities with higher confidence when overlapping
        """
        if not all_entities:
            return []

        # Sort by position first, then by confidence (descending)
        all_entities.sort(key=lambda x: (x["start"], -x.get("confidence", 0)))

        merged = []

        for entity in all_entities:
            should_add = True
            entity_to_replace = None

            for i, existing in enumerate(merged):
                overlap_ratio = self._calculate_overlap(entity, existing)

                # Check if entities overlap significantly
                if overlap_ratio > self.overlap_threshold:
                    should_add = False

                    # Decide which entity to keep based on preference
                    keep_new = self._should_keep_new_entity(entity, existing, overlap_ratio)

                    if keep_new:
                        entity_to_replace = i
                        should_add = True

                    break

            # Replace or add entity
            if entity_to_replace is not None:
                merged[entity_to_replace] = entity
            elif should_add:
                merged.append(entity)

        # Remove 'source' tag from final results
        for entity in merged:
            entity.pop("source", None)

        return merged

    def _calculate_overlap(self, e1: Dict, e2: Dict) -> float:
        """
        Calculate overlap ratio between two entities
        Returns value between 0 (no overlap) and 1 (complete overlap)
        """
        # No overlap
        if e1["end"] <= e2["start"] or e2["end"] <= e1["start"]:
            return 0.0

        # Calculate overlap
        overlap_start = max(e1["start"], e2["start"])
        overlap_end = min(e1["end"], e2["end"])
        overlap_len = overlap_end - overlap_start

        # Calculate overlap ratio relative to shorter entity
        min_len = min(e1["end"] - e1["start"], e2["end"] - e2["start"])

        return overlap_len / min_len if min_len > 0 else 0.0

    def _should_keep_new_entity(self, new: Dict, existing: Dict, overlap_ratio: float) -> bool:
        """
        Decide whether to keep new entity or existing entity based on merge preference
        """
        # If exact same span, prefer higher confidence
        if new["start"] == existing["start"] and new["end"] == existing["end"]:
            return new.get("confidence", 0) > existing.get("confidence", 0)

        # Apply merge preference strategy
        if self.merge_preference == "longer":
            # Prefer longer span (lower FN, but may increase FP)
            new_len = new["end"] - new["start"]
            existing_len = existing["end"] - existing["start"]
            return new_len > existing_len

        elif self.merge_preference == "presidio":
            # Always prefer Presidio
            return new.get("source") == "presidio"

        elif self.merge_preference == "llm":
            # Always prefer LLM
            return new.get("source") == "llm"

        elif self.merge_preference == "higher_confidence":
            # Prefer higher confidence (lower FP, but may increase FN)
            return new.get("confidence", 0) > existing.get("confidence", 0)

        else:
            # Default: prefer longer
            new_len = new["end"] - new["start"]
            existing_len = existing["end"] - existing["start"]
            return new_len > existing_len


class PIIMaskEngine:
    """
    Enhanced PII Masking Engine

    Detects PII entities, replaces them with placeholders, and stores mappings in database.
    Now supports configurable models, strategies, and thresholds.
    """

    def __init__(
        self,
        db_session: Session,
        detection_method: str = None,
        placeholder_template: str = "${{{entity_type}_{index:03d}}}",
        strategy: str = None,
        model_size: str = None
    ):
        """
        Initialize PIIMaskEngine with enhanced configuration

        Args:
            db_session: SQLAlchemy database session
            detection_method: One of "Regex", "Presidio", "LLM", "E2E" (default from config)
            placeholder_template: Template for placeholders
            strategy: Detection strategy ("high_recall", "balanced", "high_precision")
            model_size: LLM model size ("tiny", "small", "medium", "large", "xlarge")
        """
        self.db = db_session
        self.placeholder_template = placeholder_template

        # Get detection method from config if not specified
        detection_method = detection_method or PIIConfig.PII_DETECTION_METHOD
        strategy = strategy or PIIConfig.DETECTION_STRATEGY

        # Initialize extractor based on detection method
        if detection_method == "Regex":
            self.extractor = RegexExtractor()
        elif detection_method == "Presidio":
            self.extractor = PresidioExtractor(strategy=strategy)
        elif detection_method == "LLM":
            self.extractor = LLMExtractor(model_size=model_size, strategy=strategy)
        elif detection_method == "E2E":
            self.extractor = E2EExtractor(model_size=model_size, strategy=strategy)
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")

        self.detection_method = detection_method
        self.strategy = strategy

        logger.info(f"PIIMaskEngine initialized: method={detection_method}, strategy={strategy}")

        # Track entity counts for placeholder generation
        self.entity_counters = defaultdict(int)

    def mask_text(
        self,
        text: str,
        session_id: str,
        conversation_id: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Mask PII entities in text

        Args:
            text: Original text
            session_id: Session ID
            conversation_id: Optional conversation ID

        Returns:
            Tuple of (masked_text, entities_info)
        """
        start_time = datetime.utcnow()

        # Extract entities
        entities = self.extractor.extract(text)

        if not entities:
            return text, []

        # Sort entities by position (reverse order for replacement)
        entities.sort(key=lambda x: x["start"], reverse=True)

        # Track entities info
        entities_info = []
        masked_text = text

        # Replace entities with placeholders
        for entity in entities:
            # Generate placeholder
            entity_type = entity["entity_type"]
            self.entity_counters[entity_type] += 1
            placeholder = self.placeholder_template.format(
                entity_type=entity_type,
                index=self.entity_counters[entity_type]
            )

            # Generate unique IDs
            entity_id = str(uuid4())
            mapping_id = str(uuid4())
            hash_value = hashlib.sha256(entity["entity_value"].encode()).hexdigest()

            # Replace in text
            masked_text = (
                masked_text[:entity["start"]] +
                placeholder +
                masked_text[entity["end"]:]
            )

            # Store entity in database
            sensitive_entity = SensitiveEntity(
                entity_id=entity_id,
                session_id=session_id,
                conversation_id=conversation_id,
                text=entity["entity_value"],
                start_pos=entity["start"],
                end_pos=entity["end"],
                entity_type=entity_type,
                sensitivity=self._calculate_sensitivity(entity_type),
                detection_method=self.detection_method,
                confidence=entity.get("confidence", 1.0),
                created_at=datetime.utcnow()
            )
            self.db.add(sensitive_entity)

            # Store mask mapping
            mask_mapping = MaskMapping(
                mapping_id=mapping_id,
                session_id=session_id,
                conversation_id=conversation_id,
                entity_id=entity_id,
                placeholder=placeholder,
                hash_value=hash_value,
                created_at=datetime.utcnow()
            )
            self.db.add(mask_mapping)

            # Track info
            entities_info.append({
                "entity_id": entity_id,
                "entity_type": entity_type,
                "entity_value": entity["entity_value"],
                "placeholder": placeholder,
                "start": entity["start"],
                "end": entity["end"],
                "confidence": entity.get("confidence", 1.0)
            })

        # Commit to database
        self.db.commit()

        # Log processing
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        log = ProcessingLog(
            log_id=str(uuid4()),
            processing_time=processing_time,
            created_at=datetime.utcnow()
        )
        self.db.add(log)
        self.db.commit()

        logger.info(f"Masked {len(entities)} entities in {processing_time:.3f}s using {self.detection_method}")

        return masked_text, entities_info

    def _calculate_sensitivity(self, entity_type: str) -> int:
        """Calculate sensitivity score (1-10) based on entity type"""
        sensitivity_map = {
            "CREDIT_CARD": 10,
            "SSN": 10,
            "PASSWORD": 10,
            "PHONE": 7,
            "EMAIL": 6,
            "PERSON": 5,
            "ADDRESS": 7,
            "DATE": 3,
            "LOCATION": 4,
            "ORGANIZATION": 3,
            "IP_ADDRESS": 6,
            "URL": 4,
        }
        return sensitivity_map.get(entity_type, 5)


class PIIRecoverEngine:
    """
    PII Recovery Engine

    Restores original PII entities from placeholders using database mappings.
    """

    def __init__(self, db_session: Session):
        """
        Initialize PIIRecoverEngine

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session

    def recover_text(self, masked_text: str, session_id: str) -> str:
        """
        Recover original text from masked text

        Args:
            masked_text: Text with placeholders
            session_id: Session ID to lookup mappings

        Returns:
            Recovered text with original PII entities
        """
        # Find all placeholders in text
        placeholder_pattern = r'\$\{([A-Z_]+)_(\d+)\}'
        placeholders = re.findall(placeholder_pattern, masked_text)

        if not placeholders:
            return masked_text

        recovered_text = masked_text

        # Query all mappings for this session
        mappings = self.db.query(MaskMapping).filter(
            MaskMapping.session_id == session_id
        ).all()

        # Create lookup dictionary
        mapping_dict = {m.placeholder: m for m in mappings}

        # Replace placeholders with original values
        for entity_type, index in placeholders:
            placeholder = f"${{{entity_type}_{index}}}"

            if placeholder in mapping_dict:
                mapping = mapping_dict[placeholder]

                # Get original entity
                entity = self.db.query(SensitiveEntity).filter(
                    SensitiveEntity.entity_id == mapping.entity_id
                ).first()

                if entity:
                    recovered_text = recovered_text.replace(placeholder, entity.text)
                    logger.debug(f"Recovered: {placeholder} -> {entity.text}")

        return recovered_text

    def get_session_entities(self, session_id: str) -> List[Dict]:
        """
        Get all entities for a session

        Args:
            session_id: Session ID

        Returns:
            List of entity dictionaries
        """
        entities = self.db.query(SensitiveEntity).filter(
            SensitiveEntity.session_id == session_id
        ).all()

        return [{
            "entity_id": e.entity_id,
            "entity_type": e.entity_type,
            "entity_value": e.text,
            "start_pos": e.start_pos,
            "end_pos": e.end_pos,
            "sensitivity": e.sensitivity,
            "detection_method": e.detection_method,
            "confidence": e.confidence,
            "created_at": e.created_at.isoformat()
        } for e in entities]

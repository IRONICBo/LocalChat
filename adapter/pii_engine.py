# -*- coding: utf-8 -*-
"""
PII Detection and Masking Engine

Integrates four PII detection methods:
1. Regex-based extraction
2. Presidio (Deep Learning NER)
3. LLM (Ollama) extraction
4. E2E (End-to-End): Presidio + LLM

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
    """Regex-based PII extractor"""

    def __init__(self):
        super().__init__("Regex")
        self.patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "URL": r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)",
            "DATE": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b",
        }

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using regex patterns"""
        entities = []
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "entity_value": match.group(),
                    "confidence": 1.0
                })
        return entities


class PresidioExtractor(PIIExtractor):
    """Presidio-based PII extractor"""

    def __init__(self):
        super().__init__("Presidio")
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Initialize NLP engine
            provider = NlpEngineProvider()
            nlp_engine = provider.create_engine()

            # Initialize analyzer
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.available = True
        except Exception as e:
            logger.warning(f"Presidio not available: {e}")
            self.available = False

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using Presidio"""
        if not self.available:
            return []

        try:
            # Analyze text
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=None  # Detect all entity types
            )

            # Convert to standard format
            entities = []
            for result in results:
                entities.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "entity_value": text[result.start:result.end],
                    "confidence": result.score
                })
            return entities
        except Exception as e:
            logger.error(f"Presidio extraction error: {e}")
            return []


class LLMExtractor(PIIExtractor):
    """LLM-based PII extractor using Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen:0.5b"):
        super().__init__("LLM_Ollama")
        self.base_url = base_url
        self.model = model
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if LLM API is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            logger.warning("LLM API not available")
            return False

    def _fix_json_response(self, llm_response: str) -> str:
        """Fix common JSON formatting issues from LLM"""
        # Remove markdown code blocks
        cleaned_response = re.sub(r'```json\s*', '', llm_response)
        cleaned_response = re.sub(r'```\s*', '', cleaned_response)

        # Extract JSON array (non-greedy, first match only)
        json_match = re.search(r'\[.*?\]', cleaned_response, re.DOTALL)
        if not json_match:
            return "[]"

        json_str = json_match.group(0)

        # Fix common LLM JSON errors
        json_str = re.sub(r'entity__type', 'entity_type', json_str)

        # Fix unquoted values in entity_value fields
        def quote_entity_value(match):
            key = match.group(1)
            value = match.group(2).strip()
            value = value.rstrip('"}]')
            if not (value.startswith('"') and value.endswith('"')):
                value = value.replace('"', '')
                return f'"{key}": "{value}"'
            return match.group(0)

        json_str = re.sub(
            r'"(entity_value)":\s*([^,}\]]+?)(?=[,}\]])',
            quote_entity_value,
            json_str
        )

        return json_str

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using LLM"""
        if not self.available:
            return []

        try:
            prompt = f"""Extract all personal identifiable information (PII) from the following text.
Return ONLY a JSON array of entities in this exact format:
[{{"entity_type": "TYPE", "entity_value": "value"}}]

Supported entity types: PERSON, EMAIL, PHONE, CREDIT_CARD, SSN, ADDRESS, DATE, LOCATION, ORGANIZATION

Text: {text}

JSON array:"""

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code}")
                return []

            llm_response = response.json().get("response", "")

            # Fix JSON formatting
            json_str = self._fix_json_response(llm_response)

            # Parse JSON
            entities_raw = json.loads(json_str)

            # Convert to standard format with position information
            entities = []
            for entity in entities_raw:
                entity_value = entity.get("entity_value", "")
                if not entity_value:
                    continue

                # Find position in text
                start = text.find(entity_value)
                if start == -1:
                    continue

                entities.append({
                    "entity_type": entity.get("entity_type", "UNKNOWN"),
                    "start": start,
                    "end": start + len(entity_value),
                    "entity_value": entity_value,
                    "confidence": 0.8  # Default confidence for LLM
                })

            return entities

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return []


class E2EExtractor(PIIExtractor):
    """End-to-End extractor: Presidio + LLM"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen:0.5b"):
        super().__init__("E2E (PII+LLM)")
        self.presidio_extractor = PresidioExtractor()
        self.llm_extractor = LLMExtractor(base_url, model)
        self.available = self.presidio_extractor.available and self.llm_extractor.available

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using both Presidio and LLM, then merge"""
        if not self.available:
            return []

        # Extract using both methods
        presidio_entities = self.presidio_extractor.extract(text)
        llm_entities = self.llm_extractor.extract(text)

        # Merge and deduplicate
        all_entities = presidio_entities + llm_entities

        # Sort by position
        all_entities.sort(key=lambda x: (x["start"], x["end"]))

        # Deduplicate overlapping entities (keep longer span)
        merged_entities = []
        for entity in all_entities:
            # Check if this entity overlaps with any existing entity
            overlap = False
            for existing in merged_entities:
                if self._entities_overlap(entity, existing):
                    overlap = True
                    # Keep the longer one
                    if (entity["end"] - entity["start"]) > (existing["end"] - existing["start"]):
                        merged_entities.remove(existing)
                        merged_entities.append(entity)
                    break

            if not overlap:
                merged_entities.append(entity)

        return merged_entities

    def _entities_overlap(self, e1: Dict, e2: Dict) -> bool:
        """Check if two entities overlap"""
        return not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"])


class PIIMaskEngine:
    """
    PII Masking Engine

    Detects PII entities, replaces them with placeholders, and stores mappings in database.
    """

    def __init__(
        self,
        db_session: Session,
        detection_method: str = "E2E",
        placeholder_template: str = "${{{entity_type}_{index:03d}}}"
    ):
        """
        Initialize PIIMaskEngine

        Args:
            db_session: SQLAlchemy database session
            detection_method: One of "Regex", "Presidio", "LLM", "E2E"
            placeholder_template: Template for placeholders
        """
        self.db = db_session
        self.placeholder_template = placeholder_template

        # Initialize extractor based on detection method
        if detection_method == "Regex":
            self.extractor = RegexExtractor()
        elif detection_method == "Presidio":
            self.extractor = PresidioExtractor()
        elif detection_method == "LLM":
            self.extractor = LLMExtractor()
        elif detection_method == "E2E":
            self.extractor = E2EExtractor()
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")

        self.detection_method = detection_method

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

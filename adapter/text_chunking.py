# -*- coding: utf-8 -*-
"""
Text Chunking Module for Long Text PII Detection

This module provides intelligent text chunking strategies for processing
long documents with PII detection systems. Key features:

1. Multiple chunking strategies (sentence, paragraph, fixed-size, semantic)
2. Overlap handling to prevent entity splitting at chunk boundaries
3. Entity position recalculation for merged results
4. Configurable chunk sizes based on model constraints

Design Considerations:
- LLM context windows are limited (e.g., 4096 tokens)
- Longer prompts may degrade detection accuracy
- Sentence/paragraph boundaries preserve context
- Overlapping windows prevent missing split entities
"""

import re
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    SENTENCE = "sentence"  # Split by sentences
    PARAGRAPH = "paragraph"  # Split by paragraphs
    FIXED_SIZE = "fixed_size"  # Fixed character count
    SEMANTIC = "semantic"  # Semantic boundaries (requires NLP)
    ADAPTIVE = "adaptive"  # Auto-select based on text


@dataclass
class TextChunk:
    """Represents a chunk of text with position information."""
    text: str
    start_offset: int  # Character offset in original text
    end_offset: int
    chunk_index: int
    overlap_start: bool  # Whether this chunk has overlap from previous
    overlap_end: bool  # Whether this chunk has overlap to next


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
    max_chunk_size: int = 2000  # Maximum characters per chunk
    min_chunk_size: int = 100  # Minimum characters per chunk
    overlap_size: int = 100  # Characters to overlap between chunks
    sentence_boundary_buffer: int = 50  # Buffer around sentence boundaries
    preserve_entities: bool = True  # Try to avoid splitting entities


class TextChunker:
    """
    Splits long text into manageable chunks for PII detection.

    This chunker intelligently divides text while preserving context
    and ensuring entities are not split across chunk boundaries.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the text chunker.

        Args:
            config: Optional chunking configuration
        """
        self.config = config or ChunkingConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for chunking."""
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
            r'(?<=[.!?])\s*\n|'  # Sentence end with newline
            r'\n\n+'  # Paragraph break
        )

        # Paragraph boundary pattern
        self.paragraph_pattern = re.compile(r'\n\s*\n')

        # Entity-like patterns (to avoid splitting)
        self.entity_patterns = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),  # Phone
            re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),  # Credit card
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # Names
        ]

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk text according to configured strategy.

        Args:
            text: Input text to chunk

        Returns:
            List of TextChunk objects
        """
        if len(text) <= self.config.max_chunk_size:
            # Text is small enough, no chunking needed
            return [TextChunk(
                text=text,
                start_offset=0,
                end_offset=len(text),
                chunk_index=0,
                overlap_start=False,
                overlap_end=False
            )]

        # Select strategy
        strategy = self.config.strategy
        if strategy == ChunkingStrategy.ADAPTIVE:
            strategy = self._select_adaptive_strategy(text)

        # Apply chunking strategy
        if strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(text)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(text)
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_by_fixed_size(text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_by_semantic(text)
        else:
            chunks = self._chunk_by_fixed_size(text)

        logger.info(f"Chunked text into {len(chunks)} chunks using {strategy.value} strategy")

        return chunks

    def _select_adaptive_strategy(self, text: str) -> ChunkingStrategy:
        """
        Select best chunking strategy based on text characteristics.

        Args:
            text: Input text

        Returns:
            Selected chunking strategy
        """
        # Count paragraphs
        paragraphs = self.paragraph_pattern.split(text)
        avg_paragraph_len = len(text) / max(len(paragraphs), 1)

        # Count sentences
        sentences = self.sentence_pattern.split(text)
        avg_sentence_len = len(text) / max(len(sentences), 1)

        # Decision logic
        if avg_paragraph_len < self.config.max_chunk_size * 0.8:
            # Paragraphs are reasonable size
            return ChunkingStrategy.PARAGRAPH
        elif avg_sentence_len < self.config.max_chunk_size * 0.5:
            # Sentences are reasonable size
            return ChunkingStrategy.SENTENCE
        else:
            # Fall back to fixed size
            return ChunkingStrategy.FIXED_SIZE

    def _chunk_by_sentence(self, text: str) -> List[TextChunk]:
        """
        Chunk text by sentence boundaries.

        Combines sentences until max_chunk_size is reached.
        """
        chunks = []
        sentences = self.sentence_pattern.split(text)

        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence exceeds limit
            if len(current_chunk) + len(sentence) + 1 > self.config.max_chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk,
                        start_offset=current_start,
                        end_offset=current_start + len(current_chunk),
                        chunk_index=chunk_index,
                        overlap_start=chunk_index > 0,
                        overlap_end=True
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, sentence)
                current_chunk = overlap_text + sentence
                current_start = text.find(sentence, current_start)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = text.find(sentence)

        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk,
                start_offset=current_start,
                end_offset=current_start + len(current_chunk),
                chunk_index=chunk_index,
                overlap_start=chunk_index > 0,
                overlap_end=False
            ))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[TextChunk]:
        """
        Chunk text by paragraph boundaries.

        Combines paragraphs until max_chunk_size is reached.
        """
        chunks = []
        paragraphs = self.paragraph_pattern.split(text)

        current_chunk = ""
        current_start = 0
        chunk_index = 0
        search_pos = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph exceeds limit
            if len(current_chunk) + len(paragraph) + 2 > self.config.max_chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk,
                        start_offset=current_start,
                        end_offset=current_start + len(current_chunk),
                        chunk_index=chunk_index,
                        overlap_start=chunk_index > 0,
                        overlap_end=True
                    ))
                    chunk_index += 1

                # Handle very long paragraphs
                if len(paragraph) > self.config.max_chunk_size:
                    # Split paragraph further
                    sub_chunks = self._chunk_by_fixed_size(paragraph)
                    for sub_chunk in sub_chunks:
                        sub_chunk.start_offset += search_pos
                        sub_chunk.end_offset += search_pos
                        sub_chunk.chunk_index = chunk_index
                        chunks.append(sub_chunk)
                        chunk_index += 1
                    current_chunk = ""
                    search_pos = text.find(paragraph, search_pos) + len(paragraph)
                    continue

                # Start new chunk
                current_chunk = paragraph
                current_start = text.find(paragraph, search_pos)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = text.find(paragraph, search_pos)

            search_pos = text.find(paragraph, search_pos) + len(paragraph)

        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk,
                start_offset=current_start,
                end_offset=current_start + len(current_chunk),
                chunk_index=chunk_index,
                overlap_start=chunk_index > 0,
                overlap_end=False
            ))

        return chunks

    def _chunk_by_fixed_size(self, text: str) -> List[TextChunk]:
        """
        Chunk text by fixed character count with overlap.

        Uses smart boundary detection to avoid splitting entities.
        """
        chunks = []
        text_len = len(text)
        chunk_size = self.config.max_chunk_size
        overlap_size = self.config.overlap_size

        current_pos = 0
        chunk_index = 0

        while current_pos < text_len:
            # Calculate chunk boundaries
            end_pos = min(current_pos + chunk_size, text_len)

            # If not at end, try to find a good break point
            if end_pos < text_len:
                end_pos = self._find_safe_boundary(text, end_pos)

            # Extract chunk text
            chunk_text = text[current_pos:end_pos]

            chunks.append(TextChunk(
                text=chunk_text,
                start_offset=current_pos,
                end_offset=end_pos,
                chunk_index=chunk_index,
                overlap_start=chunk_index > 0,
                overlap_end=end_pos < text_len
            ))

            # Move position with overlap
            current_pos = max(end_pos - overlap_size, current_pos + 1)
            chunk_index += 1

        return chunks

    def _chunk_by_semantic(self, text: str) -> List[TextChunk]:
        """
        Chunk text by semantic boundaries.

        Falls back to paragraph chunking if semantic analysis not available.
        """
        # For now, use paragraph chunking as semantic proxy
        # In production, could use NLTK, spaCy, or embedding-based chunking
        return self._chunk_by_paragraph(text)

    def _find_safe_boundary(self, text: str, target_pos: int) -> int:
        """
        Find a safe chunk boundary near the target position.

        Avoids splitting entities or words.

        Args:
            text: Full text
            target_pos: Target position for boundary

        Returns:
            Safe boundary position
        """
        buffer = self.config.sentence_boundary_buffer

        # Search window
        start = max(0, target_pos - buffer)
        end = min(len(text), target_pos + buffer)
        window = text[start:end]

        # Try to find sentence boundary
        sentence_match = re.search(r'[.!?]\s+', window)
        if sentence_match:
            return start + sentence_match.end()

        # Try to find word boundary
        space_pos = window.rfind(' ')
        if space_pos > 0:
            return start + space_pos + 1

        # Check for entity at boundary
        if self.config.preserve_entities:
            for pattern in self.entity_patterns:
                for match in pattern.finditer(text):
                    if match.start() < target_pos < match.end():
                        # Would split entity, move boundary
                        return match.end()

        # Fall back to target position
        return target_pos

    def _get_overlap_text(self, previous_chunk: str, next_text: str) -> str:
        """
        Get overlap text from previous chunk for context.

        Args:
            previous_chunk: Previous chunk text
            next_text: Next chunk starting text

        Returns:
            Overlap text to prepend
        """
        if not previous_chunk or self.config.overlap_size <= 0:
            return ""

        # Get last N characters, trying to start at word boundary
        overlap = previous_chunk[-self.config.overlap_size:]
        space_pos = overlap.find(' ')
        if space_pos > 0:
            overlap = overlap[space_pos + 1:]

        return overlap + " "


class ChunkedPIIProcessor:
    """
    Processes chunked text for PII detection and merges results.

    This processor handles:
    1. Chunk-by-chunk PII extraction
    2. Position recalculation to original text
    3. Deduplication of entities from overlapping chunks
    4. Merging of split entities at chunk boundaries
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        overlap_threshold: float = 0.5
    ):
        """
        Initialize the chunked processor.

        Args:
            chunker: Text chunker instance
            overlap_threshold: Threshold for entity overlap matching
        """
        self.chunker = chunker or TextChunker()
        self.overlap_threshold = overlap_threshold

    def process_text(
        self,
        text: str,
        extractor_func,
        merge_entities: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Process text in chunks and merge PII detection results.

        Args:
            text: Full text to process
            extractor_func: Function that extracts entities from text
                           Should accept text and return List[Dict]
            merge_entities: Whether to merge overlapping entities

        Returns:
            Tuple of (merged_entities, processing_metadata)
        """
        # Chunk the text
        chunks = self.chunker.chunk_text(text)

        all_entities = []
        processing_times = []

        # Process each chunk
        for chunk in chunks:
            import time
            start_time = time.time()

            try:
                # Extract entities from chunk
                chunk_entities = extractor_func(chunk.text)

                # Recalculate positions relative to original text
                adjusted_entities = self._adjust_positions(chunk_entities, chunk)

                all_entities.extend(adjusted_entities)

            except Exception as e:
                logger.warning(f"Error processing chunk {chunk.chunk_index}: {e}")

            processing_times.append(time.time() - start_time)

        # Merge overlapping entities
        if merge_entities:
            merged_entities = self._merge_overlapping_entities(all_entities)
        else:
            merged_entities = all_entities

        metadata = {
            "total_chunks": len(chunks),
            "total_entities_raw": len(all_entities),
            "total_entities_merged": len(merged_entities),
            "entities_removed_by_merge": len(all_entities) - len(merged_entities),
            "chunk_sizes": [len(c.text) for c in chunks],
            "processing_times": processing_times,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0
        }

        return merged_entities, metadata

    def _adjust_positions(
        self,
        entities: List[Dict],
        chunk: TextChunk
    ) -> List[Dict]:
        """
        Adjust entity positions to original text coordinates.

        Args:
            entities: Entities extracted from chunk
            chunk: Chunk information

        Returns:
            Entities with adjusted positions
        """
        adjusted = []

        for entity in entities:
            # Get chunk-relative positions
            chunk_start = entity.get("start", 0)
            chunk_end = entity.get("end", 0)

            # Calculate original text positions
            original_start = chunk.start_offset + chunk_start
            original_end = chunk.start_offset + chunk_end

            # Create adjusted entity
            adjusted_entity = entity.copy()
            adjusted_entity["start"] = original_start
            adjusted_entity["end"] = original_end
            adjusted_entity["_chunk_index"] = chunk.chunk_index
            adjusted_entity["_from_overlap"] = chunk.overlap_start and chunk_start < self.chunker.config.overlap_size

            adjusted.append(adjusted_entity)

        return adjusted

    def _merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge overlapping entities from different chunks.

        Handles duplicates from chunk overlap regions.

        Args:
            entities: All entities from all chunks

        Returns:
            Deduplicated and merged entities
        """
        if not entities:
            return []

        # Sort by position
        sorted_entities = sorted(entities, key=lambda x: (x["start"], -x["end"]))

        merged = []

        for entity in sorted_entities:
            should_add = True
            replace_idx = None

            for i, existing in enumerate(merged):
                overlap_ratio = self._calculate_overlap(entity, existing)

                if overlap_ratio > self.overlap_threshold:
                    # Found overlapping entity
                    should_add = False

                    # Keep the one with higher confidence or from non-overlap region
                    if self._should_replace(entity, existing):
                        replace_idx = i
                        should_add = True

                    break

            if replace_idx is not None:
                merged[replace_idx] = entity
            elif should_add:
                merged.append(entity)

        # Remove internal tracking fields
        for entity in merged:
            entity.pop("_chunk_index", None)
            entity.pop("_from_overlap", None)

        return merged

    def _calculate_overlap(self, e1: Dict, e2: Dict) -> float:
        """Calculate overlap ratio between two entities."""
        start1, end1 = e1["start"], e1["end"]
        start2, end2 = e2["start"], e2["end"]

        # No overlap
        if end1 <= start2 or end2 <= start1:
            return 0.0

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_len = overlap_end - overlap_start

        min_len = min(end1 - start1, end2 - start2)
        return overlap_len / min_len if min_len > 0 else 0.0

    def _should_replace(self, new: Dict, existing: Dict) -> bool:
        """Determine if new entity should replace existing."""
        # Prefer entities not from overlap region
        new_from_overlap = new.get("_from_overlap", False)
        existing_from_overlap = existing.get("_from_overlap", False)

        if existing_from_overlap and not new_from_overlap:
            return True
        if not existing_from_overlap and new_from_overlap:
            return False

        # Prefer higher confidence
        new_conf = new.get("confidence", 0)
        existing_conf = existing.get("confidence", 0)

        if new_conf > existing_conf:
            return True

        # Prefer longer span
        new_len = new["end"] - new["start"]
        existing_len = existing["end"] - existing["start"]

        return new_len > existing_len


# Convenience functions

def chunk_and_process(
    text: str,
    extractor_func,
    max_chunk_size: int = 2000,
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
) -> Tuple[List[Dict], Dict]:
    """
    Convenience function to chunk text and process for PII.

    Args:
        text: Input text
        extractor_func: PII extraction function
        max_chunk_size: Maximum chunk size
        strategy: Chunking strategy

    Returns:
        Tuple of (entities, metadata)
    """
    config = ChunkingConfig(
        strategy=strategy,
        max_chunk_size=max_chunk_size
    )
    chunker = TextChunker(config)
    processor = ChunkedPIIProcessor(chunker)

    return processor.process_text(text, extractor_func)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Text Chunking Demo")
    print("=" * 60)

    # Sample long text
    long_text = """
Dear Customer John Smith,

We are writing to confirm your recent purchase on our platform. Your order #12345 has been successfully processed.

Payment Information:
- Credit Card: 4532 1111 2222 3333
- Billing Address: 123 Main Street, New York, NY 10001
- Amount: $299.99

Contact Details on File:
- Email: john.smith@email.com
- Phone: +1 (555) 123-4567

Your package will be shipped to:
Jane Doe
456 Oak Avenue, Apt 7B
Los Angeles, CA 90001

Expected delivery date: December 20, 2024.

If you have any questions, please contact our customer service at support@company.com or call us at 1-800-555-0123.

Thank you for your business!

Best regards,
Customer Service Team
    """.strip()

    print(f"\nOriginal text length: {len(long_text)} characters")

    # Test chunking
    config = ChunkingConfig(
        strategy=ChunkingStrategy.PARAGRAPH,
        max_chunk_size=500,
        overlap_size=50
    )
    chunker = TextChunker(config)
    chunks = chunker.chunk_text(long_text)

    print(f"\nChunks created: {len(chunks)}")
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_index} ---")
        print(f"  Offset: {chunk.start_offset}-{chunk.end_offset}")
        print(f"  Length: {len(chunk.text)}")
        print(f"  Preview: {chunk.text[:100]}...")

    # Test with mock extractor
    print("\n\n--- Processing with Mock Extractor ---")

    def mock_extractor(text):
        """Mock PII extractor for demo."""
        import re
        entities = []

        # Find emails
        for match in re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            entities.append({
                "entity_type": "EMAIL",
                "start": match.start(),
                "end": match.end(),
                "entity_value": match.group(),
                "confidence": 0.95
            })

        # Find phones
        for match in re.finditer(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text):
            entities.append({
                "entity_type": "PHONE",
                "start": match.start(),
                "end": match.end(),
                "entity_value": match.group(),
                "confidence": 0.90
            })

        return entities

    processor = ChunkedPIIProcessor(chunker)
    entities, metadata = processor.process_text(long_text, mock_extractor)

    print(f"\nTotal entities found: {metadata['total_entities_merged']}")
    print(f"Entities before merge: {metadata['total_entities_raw']}")
    print(f"Duplicates removed: {metadata['entities_removed_by_merge']}")

    for entity in entities:
        print(f"  - {entity['entity_type']}: '{entity['entity_value']}' at {entity['start']}-{entity['end']}")

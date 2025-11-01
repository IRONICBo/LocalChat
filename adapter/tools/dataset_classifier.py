# -*- coding: utf-8 -*-
"""
Dataset Classifier for PII Detection

Classifies samples into "common" vs "boundary" cases to enable:
1. Targeted optimization for high-frequency failure modes
2. Separate evaluation metrics for common vs edge cases
3. Stratified testing strategies
"""

import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter
import pandas as pd


class DatasetClassifier:
    """Classify PII samples into common vs boundary cases"""

    def __init__(self):
        # Define criteria for boundary cases
        self.boundary_criteria = {
            "complex_phone": {
                "patterns": [
                    r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?",  # International format
                    r"\(\d{3}\)\s*\d{3}-\d{4}",  # (555) 123-4567
                    r"\d{3}\.\d{3}\.\d{4}",  # 555.123.4567
                ],
                "entity_type": "PHONE_NUMBER"
            },
            "complex_address": {
                "patterns": [
                    r"P\.?O\.?\s*Box\s+\d+",  # P.O. Box
                    r"\d+\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Dr|Ln|Ct)[,.]",  # Abbreviated street
                    r"Apt\.?\s*\d+|Unit\s*\d+|#\d+",  # Apartment numbers
                ],
                "entity_type": "LOCATION"
            },
            "partial_masked_cc": {
                "patterns": [
                    r"\*{4}[-\s]*\d{4}",  # ****1234
                    r"\d{4}[-\s]*\*{4}",  # 1234****
                    r"XXXX[-\s]*\d{4}",  # XXXX1234
                ],
                "entity_type": "CREDIT_CARD"
            },
            "multilingual_text": {
                # Detect non-ASCII characters (potential multilingual content)
                "patterns": [r"[^\x00-\x7F]+"],
                "entity_type": "ANY"
            },
            "long_text": {
                # Texts longer than 200 characters
                "threshold": 200,
                "entity_type": "ANY"
            },
            "multiple_pii_types": {
                # Samples with 4+ different PII types
                "threshold": 4,
                "entity_type": "ANY"
            },
            "low_confidence_format": {
                # Unusual formats that may have low confidence
                "patterns": [
                    r"\b[A-Z]{2,}\b",  # All caps words (may be confused with names)
                    r"\d{1,2}/\d{1,2}/\d{2,4}",  # Dates (may be confused with other info)
                ],
                "entity_type": "ANY"
            }
        }

    def classify_sample(self, sample: Dict) -> Dict:
        """
        Classify a single sample as common or boundary case

        Args:
            sample: Dict with 'text' and 'entities' keys

        Returns:
            Dict with classification results
        """
        text = sample.get("text", "")
        entities = sample.get("entities", [])

        # Initialize classification
        is_boundary = False
        boundary_reasons = []
        complexity_score = 0

        # Check each boundary criterion
        # 1. Complex phone numbers
        phone_entities = [e for e in entities if e.get("entity_type") == "PHONE_NUMBER"]
        if phone_entities:
            for phone in phone_entities:
                phone_value = phone.get("entity_value", "")
                for pattern in self.boundary_criteria["complex_phone"]["patterns"]:
                    if re.search(pattern, phone_value):
                        is_boundary = True
                        boundary_reasons.append("complex_phone_format")
                        complexity_score += 2
                        break

        # 2. Complex addresses
        location_entities = [e for e in entities if e.get("entity_type") in ["LOCATION", "ADDRESS"]]
        if location_entities:
            for loc in location_entities:
                loc_value = loc.get("entity_value", "")
                for pattern in self.boundary_criteria["complex_address"]["patterns"]:
                    if re.search(pattern, loc_value):
                        is_boundary = True
                        boundary_reasons.append("complex_address_format")
                        complexity_score += 2
                        break

        # 3. Partially masked credit cards
        cc_entities = [e for e in entities if e.get("entity_type") == "CREDIT_CARD"]
        if cc_entities:
            for cc in cc_entities:
                cc_value = cc.get("entity_value", "")
                for pattern in self.boundary_criteria["partial_masked_cc"]["patterns"]:
                    if re.search(pattern, cc_value):
                        is_boundary = True
                        boundary_reasons.append("partial_masked_credit_card")
                        complexity_score += 3
                        break

        # 4. Multilingual text
        if re.search(self.boundary_criteria["multilingual_text"]["patterns"][0], text):
            is_boundary = True
            boundary_reasons.append("multilingual_text")
            complexity_score += 3

        # 5. Long text
        if len(text) > self.boundary_criteria["long_text"]["threshold"]:
            is_boundary = True
            boundary_reasons.append("long_text")
            complexity_score += 1

        # 6. Multiple PII types
        entity_types = set(e.get("entity_type") for e in entities)
        if len(entity_types) >= self.boundary_criteria["multiple_pii_types"]["threshold"]:
            is_boundary = True
            boundary_reasons.append("multiple_pii_types")
            complexity_score += 2

        # 7. Low confidence formats
        for pattern in self.boundary_criteria["low_confidence_format"]["patterns"]:
            if re.search(pattern, text):
                is_boundary = True
                boundary_reasons.append("ambiguous_format")
                complexity_score += 1
                break

        # Additional heuristics
        # Check for nested entities (entities within entities)
        if self._has_nested_entities(entities):
            is_boundary = True
            boundary_reasons.append("nested_entities")
            complexity_score += 2

        # Check for unusual entity density
        entity_density = len(entities) / len(text) if len(text) > 0 else 0
        if entity_density > 0.3:  # More than 30% of text is PII
            is_boundary = True
            boundary_reasons.append("high_entity_density")
            complexity_score += 1

        # Determine final classification
        classification = "boundary" if is_boundary else "common"

        return {
            "text": text,
            "entities": entities,
            "classification": classification,
            "boundary_reasons": list(set(boundary_reasons)),  # Remove duplicates
            "complexity_score": complexity_score,
            "entity_count": len(entities),
            "entity_types": list(entity_types),
            "text_length": len(text)
        }

    def _has_nested_entities(self, entities: List[Dict]) -> bool:
        """Check if entities are nested (one entity contains another)"""
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j:
                    continue

                # Check if e1 is fully contained in e2
                if (e1.get("start", 0) >= e2.get("start", 0) and
                    e1.get("end", 0) <= e2.get("end", 0)):
                    return True

        return False

    def classify_dataset(self, dataset_path: str, output_dir: str = "classified_dataset"):
        """
        Classify entire dataset and split into common/boundary subsets

        Args:
            dataset_path: Path to JSONL dataset file
            output_dir: Output directory for classified subsets
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Load dataset
        samples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        # Classify each sample
        classified_samples = []
        for i, sample in enumerate(samples):
            classified = self.classify_sample(sample)
            classified["sample_id"] = i
            classified_samples.append(classified)

        # Split into common and boundary
        common_samples = [s for s in classified_samples if s["classification"] == "common"]
        boundary_samples = [s for s in classified_samples if s["classification"] == "boundary"]

        # Save subsets
        common_path = output_path / "common_samples.jsonl"
        boundary_path = output_path / "boundary_samples.jsonl"

        with open(common_path, 'w') as f:
            for sample in common_samples:
                f.write(json.dumps({
                    "text": sample["text"],
                    "entities": sample["entities"]
                }) + "\n")

        with open(boundary_path, 'w') as f:
            for sample in boundary_samples:
                f.write(json.dumps({
                    "text": sample["text"],
                    "entities": sample["entities"]
                }) + "\n")

        # Generate statistics
        stats = self._generate_statistics(classified_samples)

        # Print summary
        print("\n" + "="*80)
        print("DATASET CLASSIFICATION SUMMARY")
        print("="*80)

        print(f"\nTotal samples: {len(samples)}")
        print(f"Common samples: {len(common_samples)} ({len(common_samples)/len(samples)*100:.1f}%)")
        print(f"Boundary samples: {len(boundary_samples)} ({len(boundary_samples)/len(samples)*100:.1f}%)")

        print("\n--- Boundary Case Breakdown ---")
        for reason, count in stats["boundary_reasons"].most_common():
            print(f"{reason}: {count} ({count/len(boundary_samples)*100:.1f}%)")

        print("\n--- Complexity Score Distribution ---")
        print(f"Average complexity (all): {stats['avg_complexity_all']:.2f}")
        print(f"Average complexity (common): {stats['avg_complexity_common']:.2f}")
        print(f"Average complexity (boundary): {stats['avg_complexity_boundary']:.2f}")

        print("\n--- Entity Type Distribution ---")
        print("Common samples:")
        for entity_type, count in stats["common_entity_types"].most_common():
            print(f"  {entity_type}: {count}")

        print("Boundary samples:")
        for entity_type, count in stats["boundary_entity_types"].most_common():
            print(f"  {entity_type}: {count}")

        # Export detailed statistics
        stats_path = output_path / "classification_stats.json"
        with open(stats_path, 'w') as f:
            # Convert Counter to dict for JSON
            stats_json = {
                "total_samples": len(samples),
                "common_samples": len(common_samples),
                "boundary_samples": len(boundary_samples),
                "boundary_reasons": dict(stats["boundary_reasons"]),
                "avg_complexity_all": stats["avg_complexity_all"],
                "avg_complexity_common": stats["avg_complexity_common"],
                "avg_complexity_boundary": stats["avg_complexity_boundary"],
                "common_entity_types": dict(stats["common_entity_types"]),
                "boundary_entity_types": dict(stats["boundary_entity_types"])
            }
            json.dump(stats_json, f, indent=2)

        # Export detailed classification results as CSV
        csv_path = output_path / "classification_details.csv"
        df = pd.DataFrame([{
            "sample_id": s["sample_id"],
            "classification": s["classification"],
            "complexity_score": s["complexity_score"],
            "entity_count": s["entity_count"],
            "text_length": s["text_length"],
            "boundary_reasons": ", ".join(s["boundary_reasons"]),
            "entity_types": ", ".join(s["entity_types"])
        } for s in classified_samples])
        df.to_csv(csv_path, index=False)

        print(f"\nClassification complete! Files saved to: {output_dir}/")
        print(f"  - Common samples: {common_path}")
        print(f"  - Boundary samples: {boundary_path}")
        print(f"  - Statistics: {stats_path}")
        print(f"  - Details: {csv_path}")

        return {
            "common_samples": common_samples,
            "boundary_samples": boundary_samples,
            "stats": stats_json
        }

    def _generate_statistics(self, classified_samples: List[Dict]) -> Dict:
        """Generate statistics from classified samples"""
        # Count boundary reasons
        boundary_reasons = Counter()
        for sample in classified_samples:
            if sample["classification"] == "boundary":
                for reason in sample["boundary_reasons"]:
                    boundary_reasons[reason] += 1

        # Calculate average complexity
        all_complexity = [s["complexity_score"] for s in classified_samples]
        common_complexity = [s["complexity_score"] for s in classified_samples if s["classification"] == "common"]
        boundary_complexity = [s["complexity_score"] for s in classified_samples if s["classification"] == "boundary"]

        avg_complexity_all = sum(all_complexity) / len(all_complexity) if all_complexity else 0
        avg_complexity_common = sum(common_complexity) / len(common_complexity) if common_complexity else 0
        avg_complexity_boundary = sum(boundary_complexity) / len(boundary_complexity) if boundary_complexity else 0

        # Count entity types
        common_entity_types = Counter()
        boundary_entity_types = Counter()

        for sample in classified_samples:
            entity_types = sample["entity_types"]
            if sample["classification"] == "common":
                common_entity_types.update(entity_types)
            else:
                boundary_entity_types.update(entity_types)

        return {
            "boundary_reasons": boundary_reasons,
            "avg_complexity_all": avg_complexity_all,
            "avg_complexity_common": avg_complexity_common,
            "avg_complexity_boundary": avg_complexity_boundary,
            "common_entity_types": common_entity_types,
            "boundary_entity_types": boundary_entity_types
        }


def create_stratified_splits(
    common_samples: List[Dict],
    boundary_samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = "stratified_splits"
):
    """
    Create stratified train/val/test splits maintaining common/boundary ratios

    Args:
        common_samples: List of common samples
        boundary_samples: List of boundary samples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        output_dir: Output directory
    """
    import random

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Shuffle samples
    random.shuffle(common_samples)
    random.shuffle(boundary_samples)

    # Calculate split indices
    def get_split_indices(n: int):
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        return train_idx, val_idx

    # Split common samples
    common_train_idx, common_val_idx = get_split_indices(len(common_samples))
    common_train = common_samples[:common_train_idx]
    common_val = common_samples[common_train_idx:common_val_idx]
    common_test = common_samples[common_val_idx:]

    # Split boundary samples
    boundary_train_idx, boundary_val_idx = get_split_indices(len(boundary_samples))
    boundary_train = boundary_samples[:boundary_train_idx]
    boundary_val = boundary_samples[boundary_train_idx:boundary_val_idx]
    boundary_test = boundary_samples[boundary_val_idx:]

    # Combine splits
    train_set = common_train + boundary_train
    val_set = common_val + boundary_val
    test_set = common_test + boundary_test

    # Shuffle combined sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # Save splits
    for name, split in [("train", train_set), ("val", val_set), ("test", test_set)]:
        split_path = output_path / f"{name}.jsonl"
        with open(split_path, 'w') as f:
            for sample in split:
                f.write(json.dumps({
                    "text": sample["text"],
                    "entities": sample["entities"]
                }) + "\n")

    print(f"\nStratified splits created:")
    print(f"  Train: {len(train_set)} samples ({len(common_train)} common, {len(boundary_train)} boundary)")
    print(f"  Val: {len(val_set)} samples ({len(common_val)} common, {len(boundary_val)} boundary)")
    print(f"  Test: {len(test_set)} samples ({len(common_test)} common, {len(boundary_test)} boundary)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_classifier.py <dataset.jsonl> [output_dir]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "classified_dataset"

    classifier = DatasetClassifier()
    result = classifier.classify_dataset(dataset_path, output_dir)

    # Optionally create stratified splits
    print("\nCreate stratified splits? (y/n): ", end="")
    if input().lower() == 'y':
        create_stratified_splits(
            result["common_samples"],
            result["boundary_samples"],
            output_dir=Path(output_dir) / "splits"
        )

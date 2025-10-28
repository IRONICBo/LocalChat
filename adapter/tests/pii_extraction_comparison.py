# -*- coding: utf-8 -*-
"""
PII Extraction Methods Comparison Test

Compare 4 PII extraction methods:
1. BlackWhiteList extraction (database-based)
2. Regular expression extraction
3. Presidio PII extraction
4. LLM (Ollama) extraction

Test accuracy and output results to Excel
"""

import json
import re
import sqlite3
import time
from typing import List, Dict
from pathlib import Path
import sys
import os

# Try to import requests, but make it optional
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests module not installed, LLM extractor will be disabled")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PIIExtractor:
    """PII Extractor base class"""

    def __init__(self, name: str):
        self.name = name

    def extract(self, text: str) -> List[Dict]:
        """
        Extract PII entities

        Returns:
            List[Dict]: [{
                "entity_type": str,
                "start": int,
                "end": int,
                "entity_value": str
            }, ...]
        """
        raise NotImplementedError


class BlackWhiteListExtractor(PIIExtractor):
    """Black/White list extractor (database-based)"""

    def __init__(self, db_path: str = "pii_blackwhitelist.db"):
        super().__init__("BlackWhiteList")
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database and black/white lists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pii_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                pattern TEXT NOT NULL,
                pattern_type TEXT NOT NULL,  -- 'exact' or 'keyword'
                is_whitelist INTEGER DEFAULT 0
            )
        """)

        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM pii_patterns")
        if cursor.fetchone()[0] == 0:
            # Insert common PII keyword patterns
            patterns = [
                # Credit card keywords
                ("CREDIT_CARD", "card", "keyword", 0),
                ("CREDIT_CARD", "credit card", "keyword", 0),
                ("CREDIT_CARD", "card number", "keyword", 0),

                # Email keywords
                ("EMAIL", "email", "keyword", 0),
                ("EMAIL", "e-mail", "keyword", 0),
                ("EMAIL", "@", "keyword", 0),

                # Phone keywords
                ("PHONE", "phone", "keyword", 0),
                ("PHONE", "tel", "keyword", 0),
                ("PHONE", "call", "keyword", 0),

                # Address keywords
                ("ADDRESS", "address", "keyword", 0),
                ("ADDRESS", "street", "keyword", 0),
                ("ADDRESS", "avenue", "keyword", 0),
                ("ADDRESS", "road", "keyword", 0),

                # Person name keywords
                ("PERSON", "name", "keyword", 0),
                ("PERSON", "Mr.", "keyword", 0),
                ("PERSON", "Ms.", "keyword", 0),
                ("PERSON", "Mrs.", "keyword", 0),
                ("PERSON", "Dr.", "keyword", 0),

                # IBAN keywords
                ("IBAN", "IBAN", "keyword", 0),
                ("IBAN", "account", "keyword", 0),

                # Whitelist (common false positives)
                ("NONE", "card game", "exact", 1),
                ("NONE", "business card", "exact", 1),
            ]

            cursor.executemany(
                "INSERT INTO pii_patterns (entity_type, pattern, pattern_type, is_whitelist) VALUES (?, ?, ?, ?)",
                patterns
            )
            conn.commit()

        conn.close()

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using black/white lists"""
        entities = []
        text_lower = text.lower()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check whitelist (if matches whitelist, skip detection)
        cursor.execute("SELECT pattern FROM pii_patterns WHERE is_whitelist = 1")
        whitelist = [row[0] for row in cursor.fetchall()]

        for white_pattern in whitelist:
            if white_pattern.lower() in text_lower:
                conn.close()
                return entities  # Whitelist matched, don't extract

        # Use blacklist for detection
        cursor.execute("SELECT entity_type, pattern, pattern_type FROM pii_patterns WHERE is_whitelist = 0")
        patterns = cursor.fetchall()

        detected_types = set()
        for entity_type, pattern, pattern_type in patterns:
            if pattern_type == "keyword":
                if pattern.lower() in text_lower:
                    detected_types.add(entity_type)

        # If relevant keywords detected, try to extract specific values
        if detected_types:
            # Simplified approach: find possible entities near keywords
            for entity_type in detected_types:
                # Use simple rules for extraction
                if entity_type == "CREDIT_CARD":
                    # Find numeric sequences
                    matches = re.finditer(r'\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}', text)
                    for match in matches:
                        entities.append({
                            "entity_type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "entity_value": match.group()
                        })

                elif entity_type == "EMAIL":
                    matches = re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
                    for match in matches:
                        entities.append({
                            "entity_type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "entity_value": match.group()
                        })

                elif entity_type == "PHONE":
                    matches = re.finditer(r'\+?\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', text)
                    for match in matches:
                        entities.append({
                            "entity_type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "entity_value": match.group()
                        })

        conn.close()
        return entities


class RegexExtractor(PIIExtractor):
    """Regular expression extractor"""

    def __init__(self):
        super().__init__("Regex")
        self.patterns = {
            "CREDIT_CARD": [
                r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # 16-digit card
                r'\b\d{4}[\s\-]?\d{6}[\s\-]?\d{5}\b',  # 15-digit card (Amex)
            ],
            "EMAIL": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            "PHONE": [
                r'\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{2,4}[\s\-]?\d{2,4}',
                r'\d{3}[\s\-]?\d{2}[\s\-]?\d{3}',  # European format
            ],
            "IBAN": [
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
            ],
            "SSN": [
                r'\b\d{3}-\d{2}-\d{4}\b',
            ],
            "IP_ADDRESS": [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',
            ],
            "DATE": [
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{2}/\d{2}/\d{4}\b',
            ],
            "US_DRIVER_LICENSE": [
                r'\b[A-Z]\d{7}\b',
            ],
            "PERSON": [
                r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            ],
        }

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using regular expressions"""
        entities = []

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        "entity_type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "entity_value": match.group()
                    })

        # Deduplicate (keep only one entity per position)
        entities = self._deduplicate(entities)
        return entities

    def _deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """Deduplicate entities"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity["start"], entity["end"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities


class PresidioExtractor(PIIExtractor):
    """Presidio PII extractor"""

    def __init__(self):
        super().__init__("Presidio")
        try:
            from presidio_analyzer import AnalyzerEngine
            self.analyzer = AnalyzerEngine()
            self.available = True
        except ImportError:
            print("Warning: Presidio not installed, PresidioExtractor will be disabled")
            self.available = False

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using Presidio"""
        if not self.available:
            return []

        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=None,  # Detect all entity types
        )

        entities = []
        for result in results:
            entities.append({
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "entity_value": text[result.start:result.end]
            })

        return entities


class LLMExtractor(PIIExtractor):
    """LLM (Ollama) extractor"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2:0.5b"):
        super().__init__("LLM_Ollama")
        self.base_url = base_url
        self.model = model
        self.available = HAS_REQUESTS

    def extract(self, text: str) -> List[Dict]:
        """Extract PII using LLM"""
        if not self.available:
            return []
        prompt = f"""Extract all Personal Identifiable Information (PII) from the following text.

Return the results in JSON format with the following structure:
[
  {{"entity_type": "CREDIT_CARD", "entity_value": "4532 1111 2222 3333"}},
  {{"entity_type": "EMAIL", "entity_value": "john@example.com"}},
  {{"entity_type": "PERSON", "entity_value": "John Smith"}}
]

Supported entity types: CREDIT_CARD, EMAIL, PHONE, PERSON, ADDRESS, IBAN, SSN, IP_ADDRESS, DATE, ORGANIZATION

Text: {text}

Return ONLY the JSON array, no additional text.
"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result.get("response", "").strip()

                # Try to parse JSON
                try:
                    # Extract JSON array
                    json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
                    if json_match:
                        entities_data = json.loads(json_match.group())

                        # Convert to standard format
                        entities = []
                        for entity in entities_data:
                            entity_value = entity.get("entity_value", "")
                            if entity_value and entity_value in text:
                                start = text.find(entity_value)
                                entities.append({
                                    "entity_type": entity.get("entity_type", "UNKNOWN"),
                                    "start": start,
                                    "end": start + len(entity_value),
                                    "entity_value": entity_value
                                })

                        return entities
                except json.JSONDecodeError as e:
                    print(f"LLM JSON parse error: {e}")
                    print(f"Response: {llm_response[:200]}")

            return []

        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []


class PIIEvaluator:
    """PII extraction evaluator"""

    def __init__(self):
        pass

    def evaluate(
        self,
        predicted: List[Dict],
        ground_truth: List[Dict],
        match_type: str = "exact"
    ) -> Dict:
        """
        Evaluate extraction results

        Args:
            predicted: Predicted entity list
            ground_truth: Ground truth entity list
            match_type: Match type - "exact" (exact match) or "overlap" (overlap match)

        Returns:
            Dict: {
                "precision": float,
                "recall": float,
                "f1": float,
                "true_positive": int,
                "false_positive": int,
                "false_negative": int
            }
        """
        if match_type == "exact":
            return self._evaluate_exact(predicted, ground_truth)
        else:
            return self._evaluate_overlap(predicted, ground_truth)

    def _evaluate_exact(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Exact match evaluation"""
        # Convert to set for comparison
        pred_set = {
            (e["entity_type"], e["start"], e["end"])
            for e in predicted
        }

        gt_set = {
            (e["entity_type"], e["start"], e["end"])
            for e in ground_truth
        }

        true_positive = len(pred_set & gt_set)
        false_positive = len(pred_set - gt_set)
        false_negative = len(gt_set - pred_set)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative
        }

    def _evaluate_overlap(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Overlap match evaluation (match if there's overlap)"""
        matched_pred = set()
        matched_gt = set()

        for i, pred in enumerate(predicted):
            for j, gt in enumerate(ground_truth):
                # Check if type and position overlap
                if pred["entity_type"] == gt["entity_type"]:
                    if self._is_overlap(pred, gt):
                        matched_pred.add(i)
                        matched_gt.add(j)

        true_positive = len(matched_gt)
        false_positive = len(predicted) - len(matched_pred)
        false_negative = len(ground_truth) - len(matched_gt)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative
        }

    def _is_overlap(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities overlap"""
        return not (entity1["end"] <= entity2["start"] or entity2["end"] <= entity1["start"])


def load_dataset(file_path: str, limit: int = None) -> List[Dict]:
    """Load dataset"""
    dataset = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            data = json.loads(line)

            # Normalize field names: convert start_position/end_position to start/end
            if "spans" in data:
                for span in data["spans"]:
                    if "start_position" in span:
                        span["start"] = span["start_position"]
                    if "end_position" in span:
                        span["end"] = span["end_position"]

            dataset.append(data)

    return dataset


def run_comparison_test(
    dataset_path: str,
    output_excel: str = "pii_extraction_comparison_results.xlsx",
    sample_limit: int = 100,
    use_llm: bool = True
):
    """Run comparison test"""

    print("=" * 80)
    print("PII Extraction Methods Comparison Test")
    print("=" * 80)
    print()

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, limit=sample_limit)
    print(f"Loaded {len(dataset)} samples")
    print()

    # Initialize extractors
    extractors = [
        BlackWhiteListExtractor(),
        RegexExtractor(),
        PresidioExtractor(),
    ]

    if use_llm and HAS_REQUESTS:
        extractors.append(LLMExtractor())
    elif use_llm and not HAS_REQUESTS:
        print("Warning: LLM extractor requires requests module, disabled")
        print("         Install with: pip install requests")

    # Initialize evaluator
    evaluator = PIIEvaluator()

    # Test results
    results = []
    detailed_results = []

    # Test each extractor
    for extractor in extractors:
        print(f"\n{'=' * 60}")
        print(f"Testing extractor: {extractor.name}")
        print(f"{'=' * 60}")

        if extractor.name == "Presidio" and not extractor.available:
            print("Presidio not available, skipping")
            continue

        start_time = time.time()

        all_metrics_exact = []
        all_metrics_overlap = []

        for i, sample in enumerate(dataset):
            if (i + 1) % 10 == 0:
                print(f"Processing progress: {i + 1}/{len(dataset)}")

            text = sample["full_text"]
            ground_truth = sample.get("spans", [])

            # Extract
            try:
                predicted = extractor.extract(text)
            except Exception as e:
                print(f"Error extracting from sample {i}: {e}")
                predicted = []

            # Evaluate (exact match)
            metrics_exact = evaluator.evaluate(predicted, ground_truth, match_type="exact")
            all_metrics_exact.append(metrics_exact)

            # Evaluate (overlap match)
            metrics_overlap = evaluator.evaluate(predicted, ground_truth, match_type="overlap")
            all_metrics_overlap.append(metrics_overlap)

            # Save detailed results
            detailed_results.append({
                "extractor": extractor.name,
                "sample_id": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "ground_truth_count": len(ground_truth),
                "predicted_count": len(predicted),
                "precision_exact": metrics_exact["precision"],
                "recall_exact": metrics_exact["recall"],
                "f1_exact": metrics_exact["f1"],
                "precision_overlap": metrics_overlap["precision"],
                "recall_overlap": metrics_overlap["recall"],
                "f1_overlap": metrics_overlap["f1"],
            })

        elapsed_time = time.time() - start_time

        # Calculate average metrics
        avg_metrics_exact = {
            "precision": sum(m["precision"] for m in all_metrics_exact) / len(all_metrics_exact),
            "recall": sum(m["recall"] for m in all_metrics_exact) / len(all_metrics_exact),
            "f1": sum(m["f1"] for m in all_metrics_exact) / len(all_metrics_exact),
        }

        avg_metrics_overlap = {
            "precision": sum(m["precision"] for m in all_metrics_overlap) / len(all_metrics_overlap),
            "recall": sum(m["recall"] for m in all_metrics_overlap) / len(all_metrics_overlap),
            "f1": sum(m["f1"] for m in all_metrics_overlap) / len(all_metrics_overlap),
        }

        # Save overall results
        results.append({
            "extractor": extractor.name,
            "samples": len(dataset),
            "avg_precision_exact": avg_metrics_exact["precision"],
            "avg_recall_exact": avg_metrics_exact["recall"],
            "avg_f1_exact": avg_metrics_exact["f1"],
            "avg_precision_overlap": avg_metrics_overlap["precision"],
            "avg_recall_overlap": avg_metrics_overlap["recall"],
            "avg_f1_overlap": avg_metrics_overlap["f1"],
            "total_time_seconds": elapsed_time,
            "avg_time_per_sample": elapsed_time / len(dataset),
        })

        print(f"\nExtractor: {extractor.name}")
        print(f"  Exact match:")
        print(f"    Precision: {avg_metrics_exact['precision']:.4f}")
        print(f"    Recall:    {avg_metrics_exact['recall']:.4f}")
        print(f"    F1:        {avg_metrics_exact['f1']:.4f}")
        print(f"  Overlap match:")
        print(f"    Precision: {avg_metrics_overlap['precision']:.4f}")
        print(f"    Recall:    {avg_metrics_overlap['recall']:.4f}")
        print(f"    F1:        {avg_metrics_overlap['f1']:.4f}")
        print(f"  Processing time: {elapsed_time:.2f}s ({elapsed_time/len(dataset):.3f}s/sample)")

    # Output to Excel
    print(f"\n{'=' * 60}")
    print(f"Saving results to Excel: {output_excel}")
    print(f"{'=' * 60}")

    try:
        import pandas as pd

        # Summary results
        df_summary = pd.DataFrame(results)

        # Detailed results
        df_detailed = pd.DataFrame(detailed_results)

        # Write to Excel
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_detailed.to_excel(writer, sheet_name='Detailed', index=False)

        print(f"Success: Results saved to: {output_excel}")
        print(f"   - Sheet 'Summary': Overall comparison results")
        print(f"   - Sheet 'Detailed': Detailed results for each sample")

    except ImportError:
        print("Warning: pandas or openpyxl not installed, cannot export to Excel")
        print("         Install with: pip install pandas openpyxl")

        # Save as CSV as fallback
        import csv

        csv_summary = output_excel.replace(".xlsx", "_summary.csv")
        csv_detailed = output_excel.replace(".xlsx", "_detailed.csv")

        with open(csv_summary, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        with open(csv_detailed, "w", newline="", encoding="utf-8") as f:
            if detailed_results:
                writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
                writer.writeheader()
                writer.writerows(detailed_results)

        print(f"Success: Results saved as CSV:")
        print(f"   - {csv_summary}")
        print(f"   - {csv_detailed}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    return results, detailed_results


if __name__ == "__main__":
    # Configuration
    # Try both relative paths
    dataset_path = "../data/generated_size_10000_en.jsonl"
    if not Path(dataset_path).exists():
        dataset_path = "/Users/asklv/Projects/AO.space/LocalLLM/LocalChat/adapter/data/generated_size_10000_en.jsonl"

    output_excel = "pii_extraction_comparison_results.xlsx"
    sample_limit = 100  # Test with 100 samples
    use_llm = True  # Whether to use LLM (slower)

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        print("Please run data/dateset_generator.py first to generate the dataset")
        sys.exit(1)

    # Run test
    results, detailed_results = run_comparison_test(
        dataset_path=dataset_path,
        output_excel=output_excel,
        sample_limit=sample_limit,
        use_llm=use_llm
    )

    # Print final comparison
    print("\n" + "=" * 80)
    print("Final Comparison Results (Exact Match)")
    print("=" * 80)
    print(f"{'Extractor':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time(s)':<12}")
    print("-" * 80)

    for result in results:
        print(f"{result['extractor']:<20} "
              f"{result['avg_precision_exact']:<12.4f} "
              f"{result['avg_recall_exact']:<12.4f} "
              f"{result['avg_f1_exact']:<12.4f} "
              f"{result['total_time_seconds']:<12.2f}")

    print("\n" + "=" * 80)
    print("Final Comparison Results (Overlap Match)")
    print("=" * 80)
    print(f"{'Extractor':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time(s)':<12}")
    print("-" * 80)

    for result in results:
        print(f"{result['extractor']:<20} "
              f"{result['avg_precision_overlap']:<12.4f} "
              f"{result['avg_recall_overlap']:<12.4f} "
              f"{result['avg_f1_overlap']:<12.4f} "
              f"{result['total_time_seconds']:<12.2f}")

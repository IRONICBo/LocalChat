# -*- coding: utf-8 -*-
"""
FN/FP Error Analysis Tool for PII Detection System

This tool provides fine-grained analysis of False Negatives (FN) and False Positives (FP)
to identify systematic issues and high-frequency failure patterns.
"""

import json
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd


class ErrorAnalyzer:
    """Analyze FN/FP errors with multi-dimensional classification"""

    def __init__(self):
        # Error categories for classification
        self.fn_categories = defaultdict(list)  # False Negative categories
        self.fp_categories = defaultdict(list)  # False Positive categories

    def analyze_errors(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        text: str,
        sample_id: str = None
    ) -> Dict:
        """
        Analyze errors for a single sample

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities
            text: Original text
            sample_id: Sample identifier

        Returns:
            Dictionary containing FN and FP analysis
        """
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()

        # Track TP, FP, FN
        true_positives = []
        false_positives = []
        false_negatives = []

        # Match predictions to ground truth
        for pred_idx, pred in enumerate(predictions):
            matched = False
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                # Check if entities match (overlap matching)
                if self._entities_match(pred, gt):
                    true_positives.append({
                        "prediction": pred,
                        "ground_truth": gt,
                        "sample_id": sample_id
                    })
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    matched = True
                    break

            if not matched:
                # False Positive: predicted but not in ground truth
                fp_error = self._classify_fp_error(pred, text, sample_id)
                false_positives.append(fp_error)
                self.fp_categories[fp_error["category"]].append(fp_error)

        # Find False Negatives: in ground truth but not predicted
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in matched_gt:
                fn_error = self._classify_fn_error(gt, text, sample_id)
                false_negatives.append(fn_error)
                self.fn_categories[fn_error["category"]].append(fn_error)

        return {
            "sample_id": sample_id,
            "text": text,
            "tp_count": len(true_positives),
            "fp_count": len(false_positives),
            "fn_count": len(false_negatives),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

    def _entities_match(self, entity1: Dict, entity2: Dict, overlap_threshold: float = 0.5) -> bool:
        """Check if two entities match based on type and position overlap"""
        # Type must match (normalize type names)
        type1 = entity1.get("entity_type", "").upper()
        type2 = entity2.get("entity_type", "").upper()

        # Handle common type variations
        type_mapping = {
            "PERSON": "PERSON",
            "EMAIL": "EMAIL_ADDRESS",
            "EMAIL_ADDRESS": "EMAIL_ADDRESS",
            "PHONE": "PHONE_NUMBER",
            "PHONE_NUMBER": "PHONE_NUMBER",
        }

        type1 = type_mapping.get(type1, type1)
        type2 = type_mapping.get(type2, type2)

        if type1 != type2:
            return False

        # Calculate position overlap
        start1, end1 = entity1.get("start", 0), entity1.get("end", 0)
        start2, end2 = entity2.get("start", 0), entity2.get("end", 0)

        # No overlap
        if end1 <= start2 or end2 <= start1:
            return False

        # Calculate overlap ratio
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_len = overlap_end - overlap_start

        min_len = min(end1 - start1, end2 - start2)
        overlap_ratio = overlap_len / min_len if min_len > 0 else 0.0

        return overlap_ratio >= overlap_threshold

    def _classify_fn_error(self, entity: Dict, text: str, sample_id: str) -> Dict:
        """
        Classify False Negative error into categories:
        1. Entity type
        2. Text complexity
        3. Entity format complexity
        4. Text length context
        """
        entity_type = entity.get("entity_type", "UNKNOWN")
        entity_value = entity.get("entity_value", "")
        start, end = entity.get("start", 0), entity.get("end", 0)

        # Extract context
        context_start = max(0, start - 20)
        context_end = min(len(text), end + 20)
        context = text[context_start:context_end]

        # Classify format complexity
        format_complexity = self._assess_format_complexity(entity_value, entity_type)

        # Classify text complexity
        text_complexity = self._assess_text_complexity(context)

        # Determine category
        category = f"FN_{entity_type}_{format_complexity}_{text_complexity}"

        return {
            "error_type": "FN",
            "category": category,
            "entity_type": entity_type,
            "entity_value": entity_value,
            "format_complexity": format_complexity,
            "text_complexity": text_complexity,
            "start": start,
            "end": end,
            "context": context,
            "sample_id": sample_id,
            "text_length": len(text)
        }

    def _classify_fp_error(self, entity: Dict, text: str, sample_id: str) -> Dict:
        """
        Classify False Positive error into categories:
        1. Entity type
        2. Confidence score
        3. Text similarity to real PII
        4. Position in text
        """
        entity_type = entity.get("entity_type", "UNKNOWN")
        entity_value = entity.get("entity_value", "")
        start, end = entity.get("start", 0), entity.get("end", 0)
        confidence = entity.get("confidence", 0.0)

        # Extract context
        context_start = max(0, start - 20)
        context_end = min(len(text), end + 20)
        context = text[context_start:context_end]

        # Classify confidence level
        if confidence >= 0.8:
            confidence_level = "HIGH"
        elif confidence >= 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Determine category
        category = f"FP_{entity_type}_{confidence_level}"

        return {
            "error_type": "FP",
            "category": category,
            "entity_type": entity_type,
            "entity_value": entity_value,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "start": start,
            "end": end,
            "context": context,
            "sample_id": sample_id,
            "text_length": len(text)
        }

    def _assess_format_complexity(self, value: str, entity_type: str) -> str:
        """Assess format complexity of entity value"""
        if entity_type == "PHONE_NUMBER":
            # Check phone number format complexity
            if any(sep in value for sep in ['+', '(', ')', '.', ' ']):
                return "COMPLEX"
            return "SIMPLE"

        elif entity_type == "EMAIL_ADDRESS":
            # Emails are generally standard format
            return "SIMPLE"

        elif entity_type == "LOCATION":
            # Check address complexity
            if len(value.split()) > 4:
                return "COMPLEX"
            return "SIMPLE"

        elif entity_type == "CREDIT_CARD":
            # Check credit card format
            if '-' in value or ' ' in value or '*' in value:
                return "COMPLEX"
            return "SIMPLE"

        else:
            # Default: check length
            if len(value) > 20:
                return "COMPLEX"
            return "SIMPLE"

    def _assess_text_complexity(self, context: str) -> str:
        """Assess text complexity based on context"""
        # Check for multiple sentences
        sentence_count = context.count('.') + context.count('!') + context.count('?')

        # Check for special characters
        special_chars = sum(1 for c in context if not c.isalnum() and c not in ' .,!?')

        if sentence_count > 1 or special_chars > 5:
            return "COMPLEX"
        return "SIMPLE"

    def generate_summary_report(self) -> Dict:
        """Generate summary report of all errors"""
        # Count errors by category
        fn_summary = {
            "total": sum(len(errors) for errors in self.fn_categories.values()),
            "by_category": {cat: len(errors) for cat, errors in self.fn_categories.items()},
            "by_entity_type": Counter(),
            "by_format_complexity": Counter(),
            "by_text_complexity": Counter()
        }

        fp_summary = {
            "total": sum(len(errors) for errors in self.fp_categories.values()),
            "by_category": {cat: len(errors) for cat, errors in self.fp_categories.items()},
            "by_entity_type": Counter(),
            "by_confidence_level": Counter()
        }

        # Aggregate statistics
        for errors in self.fn_categories.values():
            for error in errors:
                fn_summary["by_entity_type"][error["entity_type"]] += 1
                fn_summary["by_format_complexity"][error["format_complexity"]] += 1
                fn_summary["by_text_complexity"][error["text_complexity"]] += 1

        for errors in self.fp_categories.values():
            for error in errors:
                fp_summary["by_entity_type"][error["entity_type"]] += 1
                fp_summary["by_confidence_level"][error["confidence_level"]] += 1

        # Sort categories by frequency
        top_fn_categories = sorted(
            fn_summary["by_category"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        top_fp_categories = sorted(
            fp_summary["by_category"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "fn_summary": fn_summary,
            "fp_summary": fp_summary,
            "top_fn_categories": top_fn_categories,
            "top_fp_categories": top_fp_categories,
            "recommendations": self._generate_recommendations(fn_summary, fp_summary)
        }

    def _generate_recommendations(self, fn_summary: Dict, fp_summary: Dict) -> List[str]:
        """Generate optimization recommendations based on error analysis"""
        recommendations = []

        # Analyze FN patterns
        if fn_summary["by_format_complexity"].get("COMPLEX", 0) > fn_summary["total"] * 0.3:
            recommendations.append(
                "High FN rate for complex formats (>30%). "
                "Recommendation: Add enhanced regex patterns for complex phone numbers, "
                "addresses, and credit cards."
            )

        if fn_summary["by_entity_type"].get("PHONE_NUMBER", 0) > fn_summary["total"] * 0.2:
            recommendations.append(
                "High FN rate for PHONE_NUMBER entities (>20%). "
                "Recommendation: Expand phone number regex patterns to cover international "
                "formats and various separators."
            )

        if fn_summary["by_entity_type"].get("LOCATION", 0) > fn_summary["total"] * 0.2:
            recommendations.append(
                "High FN rate for LOCATION entities (>20%). "
                "Recommendation: Improve address detection with enhanced NER training or "
                "add more few-shot examples for addresses."
            )

        # Analyze FP patterns
        if fp_summary["by_confidence_level"].get("HIGH", 0) > fp_summary["total"] * 0.4:
            recommendations.append(
                "High FP rate with HIGH confidence (>40%). "
                "Recommendation: Review entity type definitions and adjust confidence "
                "thresholds. Consider adding negative examples to few-shot prompts."
            )

        if fp_summary["by_entity_type"].get("PERSON", 0) > fp_summary["total"] * 0.3:
            recommendations.append(
                "High FP rate for PERSON entities (>30%). "
                "Recommendation: Fine-tune NER model to better distinguish person names "
                "from common words. Add context-aware validation."
            )

        # General recommendations
        if fn_summary["total"] > fp_summary["total"] * 1.5:
            recommendations.append(
                "FN count significantly higher than FP (>1.5x). "
                "Recommendation: Switch to HIGH_RECALL strategy to reduce false negatives. "
                "Lower confidence thresholds and use 'longer' merge preference."
            )
        elif fp_summary["total"] > fn_summary["total"] * 1.5:
            recommendations.append(
                "FP count significantly higher than FN (>1.5x). "
                "Recommendation: Switch to HIGH_PRECISION strategy to reduce false positives. "
                "Raise confidence thresholds and use 'higher_confidence' merge preference."
            )

        return recommendations

    def export_detailed_errors(self, output_path: str):
        """Export detailed error analysis to CSV"""
        # Prepare FN data
        fn_records = []
        for category, errors in self.fn_categories.items():
            for error in errors:
                fn_records.append({
                    "error_type": "FN",
                    "category": category,
                    "entity_type": error["entity_type"],
                    "entity_value": error["entity_value"],
                    "format_complexity": error["format_complexity"],
                    "text_complexity": error["text_complexity"],
                    "confidence": "N/A",
                    "context": error["context"][:100],  # Truncate long contexts
                    "sample_id": error["sample_id"]
                })

        # Prepare FP data
        fp_records = []
        for category, errors in self.fp_categories.items():
            for error in errors:
                fp_records.append({
                    "error_type": "FP",
                    "category": category,
                    "entity_type": error["entity_type"],
                    "entity_value": error["entity_value"],
                    "format_complexity": "N/A",
                    "text_complexity": "N/A",
                    "confidence": f"{error['confidence']:.2f}",
                    "context": error["context"][:100],
                    "sample_id": error["sample_id"]
                })

        # Combine and export
        all_records = fn_records + fp_records
        df = pd.DataFrame(all_records)
        df.to_csv(output_path, index=False)

        print(f"Detailed error analysis exported to: {output_path}")


def analyze_evaluation_results(
    predictions_file: str,
    ground_truth_file: str,
    output_dir: str = "error_analysis"
):
    """
    Analyze evaluation results from prediction and ground truth files

    Args:
        predictions_file: Path to predictions JSONL file
        ground_truth_file: Path to ground truth JSONL file
        output_dir: Output directory for analysis reports
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    analyzer = ErrorAnalyzer()

    # Load predictions and ground truth
    with open(predictions_file, 'r') as f:
        predictions = [json.loads(line) for line in f]

    with open(ground_truth_file, 'r') as f:
        ground_truth = [json.loads(line) for line in f]

    # Analyze each sample
    results = []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        result = analyzer.analyze_errors(
            predictions=pred.get("entities", []),
            ground_truth=gt.get("entities", []),
            text=pred.get("text", ""),
            sample_id=f"sample_{i}"
        )
        results.append(result)

    # Generate summary report
    summary = analyzer.generate_summary_report()

    # Print summary
    print("\n" + "="*80)
    print("ERROR ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nFalse Negatives (FN): {summary['fn_summary']['total']}")
    print(f"False Positives (FP): {summary['fp_summary']['total']}")

    print("\n--- Top FN Categories ---")
    for category, count in summary['top_fn_categories']:
        print(f"{category}: {count} ({count/summary['fn_summary']['total']*100:.1f}%)")

    print("\n--- Top FP Categories ---")
    for category, count in summary['top_fp_categories']:
        print(f"{category}: {count} ({count/summary['fp_summary']['total']*100:.1f}%)")

    print("\n--- FN by Entity Type ---")
    for entity_type, count in summary['fn_summary']['by_entity_type'].most_common():
        print(f"{entity_type}: {count} ({count/summary['fn_summary']['total']*100:.1f}%)")

    print("\n--- FP by Entity Type ---")
    for entity_type, count in summary['fp_summary']['by_entity_type'].most_common():
        print(f"{entity_type}: {count} ({count/summary['fp_summary']['total']*100:.1f}%)")

    print("\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"{i}. {rec}")

    # Export detailed errors
    csv_path = output_path / "detailed_errors.csv"
    analyzer.export_detailed_errors(str(csv_path))

    # Export summary as JSON
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        # Convert Counter objects to dicts for JSON serialization
        summary['fn_summary']['by_entity_type'] = dict(summary['fn_summary']['by_entity_type'])
        summary['fn_summary']['by_format_complexity'] = dict(summary['fn_summary']['by_format_complexity'])
        summary['fn_summary']['by_text_complexity'] = dict(summary['fn_summary']['by_text_complexity'])
        summary['fp_summary']['by_entity_type'] = dict(summary['fp_summary']['by_entity_type'])
        summary['fp_summary']['by_confidence_level'] = dict(summary['fp_summary']['by_confidence_level'])

        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Reports saved to: {output_dir}/")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python error_analysis.py <predictions.jsonl> <ground_truth.jsonl> [output_dir]")
        sys.exit(1)

    predictions_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "error_analysis"

    analyze_evaluation_results(predictions_file, ground_truth_file, output_dir)

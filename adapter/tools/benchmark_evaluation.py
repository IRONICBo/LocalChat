# -*- coding: utf-8 -*-
"""
Comprehensive Benchmark Evaluation Script for PII Detection System

This script performs end-to-end evaluation on the full 10,000 sample dataset,
testing multiple model configurations and detection strategies.

Experiment Matrix:
- Models: qwen2.5:4b, qwen2.5:7b
- Strategies: high_recall, balanced, high_precision
- Total: 6 experimental configurations (2 models x 3 strategies)

Metrics Collected:
- Precision, Recall, F1 Score
- TP/FP/FN/TN counts
- Processing time per sample
- Error category breakdown
- Common vs boundary case performance

Usage:
    python tools/benchmark_evaluation.py --dataset data/generated_size_10000_en.jsonl
    python tools/benchmark_evaluation.py --dataset data/generated_size_10000_en.jsonl --sample-limit 1000
    python tools/benchmark_evaluation.py --help
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    # Model configurations to test
    MODEL_CONFIGS = {
        "medium": {
            "model_size": "medium",
            "model_name": "qwen2.5:4b",
            "description": "Medium model (4B parameters)"
        },
        "large": {
            "model_size": "large",
            "model_name": "qwen2.5:7b",
            "description": "Large model (7B parameters)"
        }
    }

    # Detection strategies to test
    STRATEGIES = {
        "high_recall": {
            "name": "high_recall",
            "description": "Minimize false negatives (FN)"
        },
        "balanced": {
            "name": "balanced",
            "description": "Balance precision and recall"
        },
        "high_precision": {
            "name": "high_precision",
            "description": "Minimize false positives (FP)"
        }
    }

    # Detection methods
    DETECTION_METHODS = ["E2E"]  # Focus on E2E which combines Presidio + LLM


class MetricsCollector:
    """Collects and aggregates evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0

        self.processing_times = []
        self.sample_results = []

        # Per entity type metrics
        self.entity_type_metrics = defaultdict(lambda: {
            "tp": 0, "fp": 0, "fn": 0
        })

        # Error categories
        self.fn_categories = Counter()
        self.fp_categories = Counter()

    def add_sample_result(
        self,
        sample_id: int,
        predictions: List[Dict],
        ground_truth: List[Dict],
        processing_time: float,
        text: str = ""
    ):
        """
        Add results for a single sample.

        Args:
            sample_id: Sample identifier
            predictions: Predicted entities
            ground_truth: Ground truth entities
            processing_time: Time taken to process
            text: Original text (for analysis)
        """
        self.processing_times.append(processing_time)

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()

        for pred_idx, pred in enumerate(predictions):
            matched = False
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                if self._entities_match(pred, gt):
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    matched = True

                    # True positive
                    self.true_positives += 1
                    entity_type = gt.get("entity_type", "UNKNOWN")
                    self.entity_type_metrics[entity_type]["tp"] += 1
                    break

            if not matched:
                # False positive
                self.false_positives += 1
                entity_type = pred.get("entity_type", "UNKNOWN")
                self.entity_type_metrics[entity_type]["fp"] += 1

                # Categorize FP
                confidence = pred.get("confidence", 0.5)
                if confidence >= 0.8:
                    conf_level = "HIGH"
                elif confidence >= 0.5:
                    conf_level = "MEDIUM"
                else:
                    conf_level = "LOW"
                self.fp_categories[f"FP_{entity_type}_{conf_level}"] += 1

        # Find false negatives
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in matched_gt:
                self.false_negatives += 1
                entity_type = gt.get("entity_type", "UNKNOWN")
                self.entity_type_metrics[entity_type]["fn"] += 1

                # Categorize FN
                self.fn_categories[f"FN_{entity_type}"] += 1

        # Store sample result
        self.sample_results.append({
            "sample_id": sample_id,
            "predictions_count": len(predictions),
            "ground_truth_count": len(ground_truth),
            "tp": len(matched_gt),
            "fp": len(predictions) - len(matched_pred),
            "fn": len(ground_truth) - len(matched_gt),
            "processing_time": processing_time
        })

    def _entities_match(self, pred: Dict, gt: Dict, overlap_threshold: float = 0.5) -> bool:
        """Check if two entities match based on type and position overlap."""
        # Normalize entity types
        pred_type = pred.get("entity_type", "").upper()
        gt_type = gt.get("entity_type", "").upper()

        # Handle common type variations
        type_mapping = {
            "EMAIL": "EMAIL_ADDRESS",
            "EMAIL_ADDRESS": "EMAIL_ADDRESS",
            "PHONE": "PHONE_NUMBER",
            "PHONE_NUMBER": "PHONE_NUMBER",
            "PERSON": "PERSON",
            "LOCATION": "LOCATION",
            "ADDRESS": "LOCATION",
            "CREDIT_CARD": "CREDIT_CARD",
        }

        pred_type = type_mapping.get(pred_type, pred_type)
        gt_type = type_mapping.get(gt_type, gt_type)

        if pred_type != gt_type:
            return False

        # Calculate position overlap
        pred_start = pred.get("start", pred.get("start_position", 0))
        pred_end = pred.get("end", pred.get("end_position", 0))
        gt_start = gt.get("start", gt.get("start_position", 0))
        gt_end = gt.get("end", gt.get("end_position", 0))

        # No overlap
        if pred_end <= gt_start or gt_end <= pred_start:
            return False

        # Calculate overlap ratio
        overlap_start = max(pred_start, gt_start)
        overlap_end = min(pred_end, gt_end)
        overlap_len = overlap_end - overlap_start

        min_len = min(pred_end - pred_start, gt_end - gt_start)
        overlap_ratio = overlap_len / min_len if min_len > 0 else 0.0

        return overlap_ratio >= overlap_threshold

    def get_metrics(self) -> Dict:
        """Calculate and return all metrics."""
        precision = self.true_positives / (self.true_positives + self.false_positives) \
            if (self.true_positives + self.false_positives) > 0 else 0.0

        recall = self.true_positives / (self.true_positives + self.false_negatives) \
            if (self.true_positives + self.false_negatives) > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0

        avg_time = sum(self.processing_times) / len(self.processing_times) \
            if self.processing_times else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_predictions": self.true_positives + self.false_positives,
            "total_ground_truth": self.true_positives + self.false_negatives,
            "samples_processed": len(self.sample_results),
            "total_time_seconds": sum(self.processing_times),
            "avg_time_per_sample_ms": avg_time * 1000,
            "entity_type_metrics": dict(self.entity_type_metrics),
            "top_fn_categories": self.fn_categories.most_common(10),
            "top_fp_categories": self.fp_categories.most_common(10)
        }


class BenchmarkEvaluator:
    """Main benchmark evaluation class."""

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "benchmark_results",
        sample_limit: Optional[int] = None
    ):
        """
        Initialize the benchmark evaluator.

        Args:
            dataset_path: Path to the dataset JSONL file
            output_dir: Directory to save results
            sample_limit: Optional limit on number of samples to process
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.sample_limit = sample_limit

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        self.dataset = self._load_dataset()

        logger.info(f"Loaded {len(self.dataset)} samples from {dataset_path}")

    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSONL file."""
        dataset = []

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.sample_limit and i >= self.sample_limit:
                    break

                data = json.loads(line)

                # Normalize field names
                if "spans" in data:
                    entities = data["spans"]
                    for entity in entities:
                        if "start_position" in entity:
                            entity["start"] = entity["start_position"]
                        if "end_position" in entity:
                            entity["end"] = entity["end_position"]
                elif "entities" in data:
                    entities = data["entities"]
                else:
                    entities = []

                text = data.get("full_text", data.get("text", ""))

                dataset.append({
                    "text": text,
                    "entities": entities,
                    "sample_id": i
                })

        return dataset

    def run_single_experiment(
        self,
        model_size: str,
        strategy: str,
        detection_method: str = "E2E"
    ) -> Dict:
        """
        Run a single experiment configuration.

        Args:
            model_size: Model size key (e.g., "medium", "large")
            strategy: Detection strategy (e.g., "balanced")
            detection_method: Detection method (default: "E2E")

        Returns:
            Experiment results dictionary
        """
        from pii_engine import PIIMaskEngine
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models import Base

        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: model={model_size}, strategy={strategy}")
        logger.info(f"{'='*60}")

        # Create in-memory database for this experiment
        db_engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=db_engine)
        Session = sessionmaker(bind=db_engine)
        db = Session()

        # Initialize mask engine
        try:
            mask_engine = PIIMaskEngine(
                db,
                detection_method=detection_method,
                strategy=strategy,
                model_size=model_size
            )
        except Exception as e:
            logger.error(f"Failed to initialize mask engine: {e}")
            return {"error": str(e)}

        # Initialize metrics collector
        collector = MetricsCollector()

        # Process each sample
        start_time = time.time()

        for i, sample in enumerate(self.dataset):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processing sample {i + 1}/{len(self.dataset)}")

            sample_start = time.time()

            try:
                # Extract entities
                entities = mask_engine.extractor.extract(sample["text"])

                # Convert to standard format
                predictions = []
                for entity in entities:
                    predictions.append({
                        "entity_type": entity.get("entity_type", "UNKNOWN"),
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0),
                        "entity_value": entity.get("entity_value", ""),
                        "confidence": entity.get("confidence", 0.5)
                    })

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                predictions = []

            sample_time = time.time() - sample_start

            # Add to collector
            collector.add_sample_result(
                sample_id=sample["sample_id"],
                predictions=predictions,
                ground_truth=sample["entities"],
                processing_time=sample_time,
                text=sample["text"]
            )

        total_time = time.time() - start_time

        # Get metrics
        metrics = collector.get_metrics()

        # Add experiment info
        experiment_result = {
            "experiment_id": f"{model_size}_{strategy}_{detection_method}",
            "model_size": model_size,
            "strategy": strategy,
            "detection_method": detection_method,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_samples": len(self.dataset),
            "total_time_seconds": total_time,
            **metrics
        }

        # Log summary
        logger.info(f"\nExperiment Results:")
        logger.info(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"  F1 Score: {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        logger.info(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        logger.info(f"  Total time: {total_time:.2f}s, Avg per sample: {metrics['avg_time_per_sample_ms']:.2f}ms")

        # Cleanup
        db.close()

        return experiment_result

    def run_all_experiments(self) -> Dict:
        """
        Run all experiment configurations.

        Returns:
            Combined results dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK EVALUATION - ALL EXPERIMENTS")
        logger.info("=" * 80)
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Samples: {len(self.dataset)}")
        logger.info(f"Models: {list(BenchmarkConfig.MODEL_CONFIGS.keys())}")
        logger.info(f"Strategies: {list(BenchmarkConfig.STRATEGIES.keys())}")

        all_results = {
            "benchmark_info": {
                "dataset": self.dataset_path,
                "total_samples": len(self.dataset),
                "started_at": datetime.utcnow().isoformat(),
                "models": list(BenchmarkConfig.MODEL_CONFIGS.keys()),
                "strategies": list(BenchmarkConfig.STRATEGIES.keys())
            },
            "experiments": []
        }

        # Run each experiment
        for model_key, model_config in BenchmarkConfig.MODEL_CONFIGS.items():
            for strategy_key, strategy_config in BenchmarkConfig.STRATEGIES.items():
                logger.info(f"\n\n{'#'*60}")
                logger.info(f"# Experiment: {model_key} + {strategy_key}")
                logger.info(f"{'#'*60}")

                result = self.run_single_experiment(
                    model_size=model_config["model_size"],
                    strategy=strategy_config["name"]
                )

                all_results["experiments"].append(result)

        # Add completion info
        all_results["benchmark_info"]["completed_at"] = datetime.utcnow().isoformat()

        # Save results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: Dict):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full JSON results
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {json_path}")

        # Save CSV summary
        csv_path = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        self._save_csv_summary(results, csv_path)
        logger.info(f"Summary saved to: {csv_path}")

    def _save_csv_summary(self, results: Dict, csv_path: Path):
        """Save CSV summary of results."""
        import csv

        headers = [
            "experiment_id", "model_size", "strategy",
            "precision", "recall", "f1",
            "true_positives", "false_positives", "false_negatives",
            "total_time_seconds", "avg_time_per_sample_ms"
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for exp in results["experiments"]:
                row = {h: exp.get(h, "") for h in headers}
                # Format float values
                for key in ["precision", "recall", "f1"]:
                    if key in row and isinstance(row[key], float):
                        row[key] = f"{row[key]:.4f}"
                writer.writerow(row)

    def _print_summary(self, results: Dict):
        """Print formatted summary of all experiments."""
        print("\n\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)

        # Header
        print(f"\n{'Experiment':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time(s)':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 100)

        # Sort by F1 score
        sorted_experiments = sorted(
            results["experiments"],
            key=lambda x: x.get("f1", 0),
            reverse=True
        )

        for exp in sorted_experiments:
            if "error" in exp:
                print(f"{exp.get('experiment_id', 'unknown'):<30} ERROR: {exp['error']}")
                continue

            print(
                f"{exp['experiment_id']:<30} "
                f"{exp['precision']*100:>10.2f}% "
                f"{exp['recall']*100:>10.2f}% "
                f"{exp['f1']*100:>10.2f}% "
                f"{exp['total_time_seconds']:>8.1f}s "
                f"{exp['true_positives']:>6} "
                f"{exp['false_positives']:>6} "
                f"{exp['false_negatives']:>6}"
            )

        # Best result
        if sorted_experiments and "f1" in sorted_experiments[0]:
            best = sorted_experiments[0]
            print(f"\n{'BEST RESULT:':<30}")
            print(f"  Experiment: {best['experiment_id']}")
            print(f"  F1 Score: {best['f1']*100:.2f}%")
            print(f"  Precision: {best['precision']*100:.2f}%")
            print(f"  Recall: {best['recall']*100:.2f}%")

        print("\n" + "=" * 100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for PII detection system"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/generated_size_10000_en.jsonl",
        help="Path to dataset JSONL file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )

    parser.add_argument(
        "--single-experiment",
        action="store_true",
        help="Run single experiment only (medium model, balanced strategy)"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["medium", "large"],
        default="medium",
        help="Model size for single experiment"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["high_recall", "balanced", "high_precision"],
        default="balanced",
        help="Strategy for single experiment"
    )

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(script_dir, dataset_path)

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit
    )

    # Run experiments
    if args.single_experiment:
        result = evaluator.run_single_experiment(
            model_size=args.model_size,
            strategy=args.strategy
        )
        print(json.dumps(result, indent=2, default=str))
    else:
        results = evaluator.run_all_experiments()

    logger.info("\nBenchmark evaluation completed!")


if __name__ == "__main__":
    main()

"""Base classes for benchmarking scientific claim verification.

This module provides a common interface for different benchmark datasets
and evaluation metrics for assessing verification performance.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
)
import numpy as np


class VerificationMethod(Enum):
    """Methods for verifying claims."""
    AGENT = "agent"
    AGENTLESS = "agentless"
    AGENT_WITH_SEARCH = "agent_with_search"
    AGENTLESS_WITH_SEARCH = "agentless_with_search"


@dataclass
class BenchmarkItem:
    """A single benchmark item representing a claim to verify.

    Attributes:
        claim_id: Unique identifier for the claim
        claim: The claim text to verify
        expected_result: Expected verification result (SUPPORTS/REFUTES/INSUFFICIENT_EVIDENCE)
        verification_method: Method to use for verification
        result: Actual verification result (populated after verification)
        metadata: Additional dataset-specific metadata
    """
    claim_id: str
    claim: str
    expected_result: str  # SUPPORTS, REFUTES, or INSUFFICIENT_EVIDENCE
    verification_method: VerificationMethod
    result: Optional[Any] = None  # VerificationResult after verification
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"BenchmarkItem(id={self.claim_id}, "
                f"expected={self.expected_result}, "
                f"claim='{self.claim[:50]}...')")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "expected_result": self.expected_result,
            "verification_method": self.verification_method.value,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
        }


class Benchmark:
    """Base class for benchmark datasets.

    A benchmark consists of a collection of items to verify along with
    their expected results.
    """

    def __init__(self, name: str):
        """Initialize a benchmark.

        Args:
            name: Name of the benchmark dataset
        """
        self.name = name
        self.items: List[BenchmarkItem] = []

    def load(self, max_items: Optional[int] = None) -> List[BenchmarkItem]:
        """Load benchmark items.

        Args:
            max_items: Maximum number of items to load (None for all)

        Returns:
            List of BenchmarkItem objects
        """
        raise NotImplementedError("Subclasses must implement load()")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the benchmark.

        Returns:
            Dictionary with statistics (total items, label distribution, etc.)
        """
        if not self.items:
            return {"total_items": 0, "label_distribution": {}}

        label_counts = {}
        for item in self.items:
            label = item.expected_result
            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "total_items": len(self.items),
            "label_distribution": label_counts,
        }

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Benchmark(name='{self.name}', items={len(self.items)})"


@dataclass
class EvaluationMetrics:
    """Container for benchmark evaluation metrics.

    Attributes:
        accuracy: Overall accuracy (correct / total)
        macro_f1: Macro-averaged F1 score across all classes
        micro_f1: Micro-averaged F1 score (same as accuracy for single-label)
        per_class_metrics: Dict mapping class name to (precision, recall, f1)
        confusion_matrix: Dict of dicts representing confusion matrix
        total_predictions: Total number of predictions made
        correct_predictions: Number of correct predictions
        label_distribution_expected: Distribution of expected labels
        label_distribution_predicted: Distribution of predicted labels
    """
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]  # class -> {precision, recall, f1}
    confusion_matrix: Dict[str, Dict[str, int]]  # actual -> predicted -> count
    total_predictions: int
    correct_predictions: int
    label_distribution_expected: Dict[str, int]
    label_distribution_predicted: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "label_distribution_expected": self.label_distribution_expected,
            "label_distribution_predicted": self.label_distribution_predicted,
        }

    def print_report(self, title: str = "Evaluation Report") -> str:
        """Generate a formatted evaluation report.

        Args:
            title: Title for the report

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 70)
        lines.append(title.center(70))
        lines.append("=" * 70)

        # Overall metrics
        lines.append("\nOVERALL METRICS")
        lines.append("-" * 40)
        lines.append(f"  Accuracy:     {self.accuracy:.4f} ({self.accuracy * 100:.2f}%)")
        lines.append(f"  Macro F1:     {self.macro_f1:.4f}")
        lines.append(f"  Micro F1:     {self.micro_f1:.4f}")
        lines.append(f"  Correct:      {self.correct_predictions}/{self.total_predictions}")

        # Per-class metrics
        lines.append("\nPER-CLASS METRICS")
        lines.append("-" * 40)
        lines.append(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        lines.append("  " + "-" * 55)
        for cls, metrics in sorted(self.per_class_metrics.items()):
            lines.append(
                f"  {cls:<25} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}"
            )

        # Label distribution
        lines.append("\nLABEL DISTRIBUTION")
        lines.append("-" * 40)
        all_labels = sorted(set(self.label_distribution_expected.keys()) |
                          set(self.label_distribution_predicted.keys()))
        lines.append(f"  {'Label':<25} {'Expected':>12} {'Predicted':>12}")
        lines.append("  " + "-" * 49)
        for label in all_labels:
            exp = self.label_distribution_expected.get(label, 0)
            pred = self.label_distribution_predicted.get(label, 0)
            lines.append(f"  {label:<25} {exp:>12} {pred:>12}")

        # Confusion matrix
        lines.append("\nCONFUSION MATRIX")
        lines.append("-" * 40)
        labels = sorted(self.confusion_matrix.keys())

        # Header
        header = "  " + " " * 20 + "Predicted".center(len(labels) * 12)
        lines.append(header)
        col_header = "  " + "Actual".ljust(20) + "".join(f"{lbl[:10]:>12}" for lbl in labels)
        lines.append(col_header)
        lines.append("  " + "-" * (20 + len(labels) * 12))

        # Rows
        for actual in labels:
            row = f"  {actual[:18]:<20}"
            for predicted in labels:
                count = self.confusion_matrix.get(actual, {}).get(predicted, 0)
                row += f"{count:>12}"
            lines.append(row)

        lines.append("=" * 70)

        return "\n".join(lines)


class BenchmarkEvaluator:
    """Evaluator for computing benchmark metrics from results.

    Uses scikit-learn for computing standard NLP evaluation metrics including
    accuracy, precision, recall, F1 scores, and confusion matrices.
    """

    # Standard label set for claim verification
    STANDARD_LABELS = ["SUPPORTS", "REFUTES", "INSUFFICIENT_EVIDENCE"]

    @staticmethod
    def compute_metrics(
        items: List[BenchmarkItem],
        labels: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """Compute evaluation metrics from benchmark items with results.

        Args:
            items: List of BenchmarkItem objects with results populated
            labels: Optional list of label names. If None, auto-detected from data.

        Returns:
            EvaluationMetrics object with all computed metrics

        Raises:
            ValueError: If no items have results
        """
        # Filter items that have results
        evaluated_items = [item for item in items if item.result is not None]

        if not evaluated_items:
            raise ValueError("No items have results. Run verification first.")

        # Extract expected and predicted labels
        y_true = [item.expected_result for item in evaluated_items]
        y_pred = [item.result.verdict if hasattr(item.result, 'verdict')
                  else str(item.result) for item in evaluated_items]

        return BenchmarkEvaluator.compute_from_predictions(y_true, y_pred, labels)

    @staticmethod
    def compute_from_predictions(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """Compute metrics directly from prediction lists using sklearn.

        Args:
            y_true: List of ground truth labels
            y_pred: List of predicted labels
            labels: Optional list of all possible labels

        Returns:
            EvaluationMetrics object
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")

        # Auto-detect labels if not provided
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
        correct = int(accuracy * len(y_true))
        total = len(y_true)

        # Compute precision, recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        # Build per-class metrics dict
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Compute macro and micro F1
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0))

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion_dict = {}
        for i, actual in enumerate(labels):
            confusion_dict[actual] = {}
            for j, predicted in enumerate(labels):
                confusion_dict[actual][predicted] = int(cm[i, j])

        # Label distributions
        label_dist_expected = {label: int(np.sum(np.array(y_true) == label)) for label in labels}
        label_dist_predicted = {label: int(np.sum(np.array(y_pred) == label)) for label in labels}

        return EvaluationMetrics(
            accuracy=accuracy,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_dict,
            total_predictions=total,
            correct_predictions=correct,
            label_distribution_expected=label_dist_expected,
            label_distribution_predicted=label_dist_predicted,
        )

    @staticmethod
    def get_classification_report(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> str:
        """Get sklearn's formatted classification report.

        Convenience method for quick text-based report.

        Args:
            y_true: List of ground truth labels
            y_pred: List of predicted labels
            labels: Optional list of all possible labels

        Returns:
            Formatted classification report string
        """
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        return classification_report(y_true, y_pred, labels=labels, zero_division=0)

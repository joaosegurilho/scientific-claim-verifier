#!/usr/bin/env python3
"""Generate label-only evaluation reports (SUPPORTS/REFUTES only, following SciFact protocol).

This script:
1. Finds all summary.json files in benchmark_results
2. Computes precision, recall, F1 for SUPPORTS and REFUTES classes
3. Properly accounts for false positives from INSUFFICIENT_EVIDENCE predictions
4. Saves new "label_only_report.txt" files alongside existing reports

This matches the SciFact "label-only" evaluation protocol used in the literature.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class LabelOnlyMetrics:
    """Metrics for SciFact label-only evaluation."""
    # Per-class metrics
    supports_precision: float
    supports_recall: float
    supports_f1: float
    refutes_precision: float
    refutes_recall: float
    refutes_f1: float
    
    # Aggregate metrics
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    
    # Counts
    supports_tp: int
    supports_fp: int
    supports_fn: int
    refutes_tp: int
    refutes_fp: int
    refutes_fn: int
    
    def print_report(self, title: str) -> str:
        """Generate a formatted report string."""
        lines = [
            "=" * 70,
            title,
            "=" * 70,
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"  Micro Precision: {self.micro_precision:.4f} ({self.micro_precision:.2%})",
            f"  Micro Recall:    {self.micro_recall:.4f} ({self.micro_recall:.2%})",
            f"  Micro F1:        {self.micro_f1:.4f} ({self.micro_f1:.2%})",
            f"  Macro F1:        {self.macro_f1:.4f} ({self.macro_f1:.2%})",
            "",
            "PER-CLASS METRICS",
            "-" * 40,
            f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            f"  {'-' * 44}",
            f"  {'SUPPORTS':<12} {self.supports_precision:>10.4f} {self.supports_recall:>10.4f} {self.supports_f1:>10.4f}",
            f"  {'REFUTES':<12} {self.refutes_precision:>10.4f} {self.refutes_recall:>10.4f} {self.refutes_f1:>10.4f}",
            "",
            "CONFUSION MATRIX COMPONENTS",
            "-" * 40,
            f"  {'Class':<12} {'TP':>8} {'FP':>8} {'FN':>8}",
            f"  {'-' * 38}",
            f"  {'SUPPORTS':<12} {self.supports_tp:>8} {self.supports_fp:>8} {self.supports_fn:>8}",
            f"  {'REFUTES':<12} {self.refutes_tp:>8} {self.refutes_fp:>8} {self.refutes_fn:>8}",
            "=" * 70,
        ]
        return "\n".join(lines)


def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Load a summary.json file."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_scifact_label_only_metrics(results: List[Dict]) -> Tuple[LabelOnlyMetrics, Dict[str, int]]:
    """Compute SciFact label-only metrics correctly.
    
    SciFact's label-only evaluation:
    - Only SUPPORTS and REFUTES are "evidence" labels
    - INSUFFICIENT_EVIDENCE (NEI/NOINFO) means "no evidence found"
    
    For each evidence class (SUPPORTS, REFUTES):
    - True Positive: Gold = X, Predicted = X
    - False Positive: Gold != X, Predicted = X (includes NEI -> X)
    - False Negative: Gold = X, Predicted != X (includes X -> NEI)
    
    Args:
        results: List of result dictionaries with 'expected' and 'predicted' fields
        
    Returns:
        Tuple of (LabelOnlyMetrics, stats_dict)
    """
    # Initialize counters
    tp = {"SUPPORTS": 0, "REFUTES": 0}
    fp = {"SUPPORTS": 0, "REFUTES": 0}
    fn = {"SUPPORTS": 0, "REFUTES": 0}
    
    # Statistics
    total_claims = len(results)
    error_count = 0
    nei_gold_count = 0
    nei_pred_count = 0
    
    for r in results:
        expected = r.get("expected")
        predicted = r.get("predicted")
        
        # Skip ERROR predictions entirely
        if predicted == "ERROR":
            error_count += 1
            continue
        
        # Track NEI statistics
        if expected == "INSUFFICIENT_EVIDENCE":
            nei_gold_count += 1
        if predicted == "INSUFFICIENT_EVIDENCE":
            nei_pred_count += 1
        
        # Compute TP, FP, FN for each evidence class
        for label in ["SUPPORTS", "REFUTES"]:
            if expected == label and predicted == label:
                # Correct prediction of this evidence class
                tp[label] += 1
            elif expected != label and predicted == label:
                # Incorrectly predicted this class (false alarm)
                # This includes: other evidence class -> this class
                #                NEI -> this class
                fp[label] += 1
            elif expected == label and predicted != label:
                # Missed this evidence class (miss)
                # This includes: this class -> other evidence class
                #                this class -> NEI
                fn[label] += 1
            # Note: expected != label and predicted != label -> no impact on this class
    
    # Compute per-class metrics
    def safe_div(num, denom):
        return num / denom if denom > 0 else 0.0
    
    def f1_score(precision, recall):
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    supports_precision = safe_div(tp["SUPPORTS"], tp["SUPPORTS"] + fp["SUPPORTS"])
    supports_recall = safe_div(tp["SUPPORTS"], tp["SUPPORTS"] + fn["SUPPORTS"])
    supports_f1 = f1_score(supports_precision, supports_recall)
    
    refutes_precision = safe_div(tp["REFUTES"], tp["REFUTES"] + fp["REFUTES"])
    refutes_recall = safe_div(tp["REFUTES"], tp["REFUTES"] + fn["REFUTES"])
    refutes_f1 = f1_score(refutes_precision, refutes_recall)
    
    # Compute micro-averaged metrics (pooled across classes)
    total_tp = tp["SUPPORTS"] + tp["REFUTES"]
    total_fp = fp["SUPPORTS"] + fp["REFUTES"]
    total_fn = fn["SUPPORTS"] + fn["REFUTES"]
    
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = f1_score(micro_precision, micro_recall)
    
    # Compute macro-averaged F1 (simple average of per-class F1)
    macro_f1 = (supports_f1 + refutes_f1) / 2
    
    metrics = LabelOnlyMetrics(
        supports_precision=supports_precision,
        supports_recall=supports_recall,
        supports_f1=supports_f1,
        refutes_precision=refutes_precision,
        refutes_recall=refutes_recall,
        refutes_f1=refutes_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        supports_tp=tp["SUPPORTS"],
        supports_fp=fp["SUPPORTS"],
        supports_fn=fn["SUPPORTS"],
        refutes_tp=tp["REFUTES"],
        refutes_fp=fp["REFUTES"],
        refutes_fn=fn["REFUTES"],
    )
    
    stats = {
        "total_claims": total_claims,
        "error_count": error_count,
        "nei_gold_count": nei_gold_count,
        "nei_pred_count": nei_pred_count,
        "evaluated_pairs": total_claims - error_count,
        "evidence_gold_count": (tp["SUPPORTS"] + fn["SUPPORTS"] + 
                                tp["REFUTES"] + fn["REFUTES"]),
        "evidence_pred_count": (tp["SUPPORTS"] + fp["SUPPORTS"] + 
                                tp["REFUTES"] + fp["REFUTES"]),
    }
    
    return metrics, stats


def generate_label_only_report(summary_path: Path) -> bool:
    """Generate label-only evaluation report for a single benchmark run.

    Args:
        summary_path: Path to summary.json file

    Returns:
        True if report was generated, False if insufficient data
    """
    try:
        summary = load_summary(summary_path)
        results = summary.get("results", [])

        if not results:
            return False

        # Compute metrics
        metrics, stats = compute_scifact_label_only_metrics(results)
        
        if stats["evidence_gold_count"] == 0:
            print(f"  No SUPPORTS/REFUTES gold labels found in {summary_path.parent.name}")
            return False

        # Generate report
        benchmark_name = summary.get("benchmark", "Unknown")
        method = summary.get("method", "Unknown")
        config_name = summary_path.parent.name

        report = metrics.print_report(
            f"Label-Only Evaluation: {benchmark_name} ({method})\n"
            f"Configuration: {config_name}\n"
            f"(SciFact protocol: SUPPORTS/REFUTES only, with proper FP accounting)"
        )

        # Add filtering statistics
        report += f"\n\nDATASET STATISTICS\n"
        report += f"{'-' * 40}\n"
        report += f"  Total claims in benchmark:       {stats['total_claims']}\n"
        report += f"  ERROR predictions (excluded):    {stats['error_count']}\n"
        report += f"  Evaluated pairs:                 {stats['evaluated_pairs']}\n"
        report += f"  Gold SUPPORTS/REFUTES:           {stats['evidence_gold_count']}\n"
        report += f"  Gold INSUFFICIENT_EVIDENCE:      {stats['nei_gold_count']}\n"
        report += f"  Predicted SUPPORTS/REFUTES:      {stats['evidence_pred_count']}\n"
        report += f"  Predicted INSUFFICIENT_EVIDENCE: {stats['nei_pred_count']}\n"

        # Save report
        output_path = summary_path.parent / "label_only_report.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"  ✓ Saved label-only report: {output_path.relative_to(summary_path.parent.parent.parent)}")
        print(f"    Micro F1: {metrics.micro_f1:.2%}, Macro F1: {metrics.macro_f1:.2%}")
        print(f"    SUPPORTS F1: {metrics.supports_f1:.2%}, REFUTES F1: {metrics.refutes_f1:.2%}")

        return True

    except Exception as e:
        print(f"  ✗ Error processing {summary_path.parent.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Find benchmark_results directory
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "benchmark_results"

    if not results_dir.exists():
        print(f"Error: benchmark_results directory not found at {results_dir}")
        return 1

    # Find all summary.json files
    summary_files = list(results_dir.rglob("summary.json"))

    if not summary_files:
        print(f"No summary.json files found in {results_dir}")
        return 1

    print(f"Found {len(summary_files)} benchmark result summaries")
    print(f"{'=' * 80}\n")

    # Generate label-only reports for each
    success_count = 0
    for summary_path in sorted(summary_files):
        config_name = summary_path.parent.name
        print(f"Processing: {config_name}")

        if generate_label_only_report(summary_path):
            success_count += 1
        print()

    print(f"{'=' * 80}")
    print(f"Successfully generated {success_count}/{len(summary_files)} label-only reports")
    print(f"{'=' * 80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# ### The Core Concepts

# **Precision, Recall, and F1** are metrics for evaluating classification tasks:

# | Metric | Question it answers | Formula |
# |--------|---------------------|---------|
# | **Precision** | "Of all the items I predicted as X, how many actually were X?" | TP / (TP + FP) |
# | **Recall** | "Of all the items that actually are X, how many did I find?" | TP / (TP + FN) |
# | **F1** | Harmonic mean balancing both | 2 * P * R / (P + R) |

# Where:
# - **TP (True Positive)**: Gold = X, Predicted = X (correct detection)
# - **FP (False Positive)**: Gold ≠ X, Predicted = X (false alarm)
# - **FN (False Negative)**: Gold = X, Predicted ≠ X (missed detection)

# ---

# ### SciFact's Evaluation Logic

# SciFact has three classes: **SUPPORTS**, **REFUTES**, and **NEI** (Not Enough Info / INSUFFICIENT_EVIDENCE).

# However, the task is framed as: **"Find abstracts that support or refute the claim."**

# This means:
# - SUPPORTS and REFUTES are **"evidence" labels** (positive classes)
# - NEI means **"this abstract doesn't contain evidence"** (negative/null class)

# **The key insight**: SciFact evaluates your ability to identify evidence abstracts, not your ability to classify NEI.

# ---

# ### How SciFact Computes F1

# For each evidence class separately:

# **SUPPORTS:**
# ```
# TP = predicted SUPPORTS when gold is SUPPORTS
# FP = predicted SUPPORTS when gold is REFUTES or NEI    ← THIS IS CRUCIAL
# FN = predicted REFUTES or NEI when gold is SUPPORTS
# ```

# **REFUTES:**
# ```
# TP = predicted REFUTES when gold is REFUTES
# FP = predicted REFUTES when gold is SUPPORTS or NEI    ← THIS IS CRUCIAL
# FN = predicted SUPPORTS or NEI when gold is REFUTES
#!/usr/bin/env python3
"""Analyze confidence scores from benchmark results.

This script:
1. Finds all summary.json files in benchmark_results
2. Extracts confidence scores and their relationship to correctness
3. Generates statistical breakdowns per benchmark/configuration
4. Creates visualization plots saved to each run's figures folder
5. Outputs results for thesis analysis
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Load a summary.json file."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_confidence_accuracy(results: List[Dict]) -> Dict[str, Any]:
    """Analyze relationship between confidence and accuracy.

    Args:
        results: List of result dictionaries with 'confidence', 'correct', 'predicted', 'expected'

    Returns:
        Dictionary with confidence analysis statistics
    """
    # Filter out ERROR predictions
    valid_results = [r for r in results if r.get("predicted") != "ERROR"]

    if not valid_results:
        return None

    # Overall confidence stats
    all_confidences = [r["confidence"] for r in valid_results]
    correct_results = [r for r in valid_results if r["correct"]]
    incorrect_results = [r for r in valid_results if not r["correct"]]

    correct_confidences = [r["confidence"] for r in correct_results]
    incorrect_confidences = [r["confidence"] for r in incorrect_results]

    # Confidence by verdict type
    by_verdict = defaultdict(lambda: {"correct": [], "incorrect": []})
    for r in valid_results:
        verdict = r["predicted"]
        if r["correct"]:
            by_verdict[verdict]["correct"].append(r["confidence"])
        else:
            by_verdict[verdict]["incorrect"].append(r["confidence"])

    # Confidence thresholds analysis
    thresholds = [5, 6, 7, 8, 9]
    threshold_analysis = {}
    for threshold in thresholds:
        high_conf = [r for r in valid_results if r["confidence"] >= threshold]
        if high_conf:
            threshold_analysis[f"conf>={threshold}"] = {
                "count": len(high_conf),
                "accuracy": sum(1 for r in high_conf if r["correct"]) / len(high_conf),
                "avg_confidence": statistics.mean([r["confidence"] for r in high_conf])
            }

    return {
        "overall": {
            "avg_confidence": statistics.mean(all_confidences),
            "median_confidence": statistics.median(all_confidences),
            "std_confidence": statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0,
            "min_confidence": min(all_confidences),
            "max_confidence": max(all_confidences),
        },
        "by_correctness": {
            "correct": {
                "count": len(correct_results),
                "avg_confidence": statistics.mean(correct_confidences) if correct_confidences else 0,
                "median_confidence": statistics.median(correct_confidences) if correct_confidences else 0,
            },
            "incorrect": {
                "count": len(incorrect_results),
                "avg_confidence": statistics.mean(incorrect_confidences) if incorrect_confidences else 0,
                "median_confidence": statistics.median(incorrect_confidences) if incorrect_confidences else 0,
            }
        },
        "by_verdict": {
            verdict: {
                "correct_avg": statistics.mean(data["correct"]) if data["correct"] else None,
                "incorrect_avg": statistics.mean(data["incorrect"]) if data["incorrect"] else None,
                "correct_count": len(data["correct"]),
                "incorrect_count": len(data["incorrect"]),
            }
            for verdict, data in by_verdict.items()
        },
        "threshold_analysis": threshold_analysis,
        "total_predictions": len(valid_results),
    }


def format_confidence_report(config_name: str, analysis: Dict[str, Any]) -> str:
    """Format confidence analysis into readable report."""
    if not analysis:
        return f"\n{config_name}: No valid results\n"

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"{config_name}")
    lines.append(f"{'='*80}")

    # Overall statistics
    overall = analysis["overall"]
    lines.append(f"\nOVERALL CONFIDENCE STATISTICS:")
    lines.append(f"  Mean: {overall['avg_confidence']:.2f}")
    lines.append(f"  Median: {overall['median_confidence']:.2f}")
    lines.append(f"  Std Dev: {overall['std_confidence']:.2f}")
    lines.append(f"  Range: [{overall['min_confidence']:.1f}, {overall['max_confidence']:.1f}]")

    # Confidence by correctness
    by_correct = analysis["by_correctness"]
    lines.append(f"\nCONFIDENCE BY CORRECTNESS:")
    lines.append(f"  Correct predictions ({by_correct['correct']['count']}): "
                f"avg={by_correct['correct']['avg_confidence']:.2f}, "
                f"median={by_correct['correct']['median_confidence']:.2f}")
    lines.append(f"  Incorrect predictions ({by_correct['incorrect']['count']}): "
                f"avg={by_correct['incorrect']['avg_confidence']:.2f}, "
                f"median={by_correct['incorrect']['median_confidence']:.2f}")

    # Calculate difference
    if by_correct['correct']['count'] > 0 and by_correct['incorrect']['count'] > 0:
        diff = by_correct['correct']['avg_confidence'] - by_correct['incorrect']['avg_confidence']
        lines.append(f"  Difference: {diff:+.2f} (correct - incorrect)")

    # Confidence by verdict
    lines.append(f"\nCONFIDENCE BY VERDICT TYPE:")
    for verdict, data in analysis["by_verdict"].items():
        lines.append(f"  {verdict}:")
        if data["correct_avg"] is not None:
            lines.append(f"    Correct ({data['correct_count']}): avg={data['correct_avg']:.2f}")
        if data["incorrect_avg"] is not None:
            lines.append(f"    Incorrect ({data['incorrect_count']}): avg={data['incorrect_avg']:.2f}")

    # Threshold analysis
    if analysis["threshold_analysis"]:
        lines.append(f"\nACCURACY AT CONFIDENCE THRESHOLDS:")
        for threshold, data in sorted(analysis["threshold_analysis"].items()):
            lines.append(f"  {threshold}: {data['count']} predictions, "
                        f"accuracy={data['accuracy']:.2%}, "
                        f"avg_conf={data['avg_confidence']:.2f}")

    return "\n".join(lines)


def create_confidence_visualizations(results: List[Dict], figures_dir: Path, config_name: str):
    """Create and save confidence score visualizations.

    Args:
        results: List of result dictionaries
        figures_dir: Directory to save figures
        config_name: Configuration name for plot titles
    """
    # Filter out ERROR predictions
    valid_results = [r for r in results if r.get("predicted") != "ERROR"]

    if not valid_results:
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. Confidence distribution by correctness (Box plot)
    fig, ax = plt.subplots(figsize=(10, 6))

    correct_conf = [r["confidence"] for r in valid_results if r["correct"]]
    incorrect_conf = [r["confidence"] for r in valid_results if not r["correct"]]

    data_to_plot = [correct_conf, incorrect_conf]
    labels = ['Correct', 'Incorrect']

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # Color boxes
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Confidence Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction Correctness', fontsize=14, fontweight='bold')
    ax.set_title(f'Confidence Score Distribution\n{config_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "confidence_by_correctness.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Accuracy vs Confidence Threshold
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = list(range(1, 11))
    accuracies = []
    counts = []

    for threshold in thresholds:
        high_conf = [r for r in valid_results if r["confidence"] >= threshold]
        if high_conf:
            acc = sum(1 for r in high_conf if r["correct"]) / len(high_conf)
            accuracies.append(acc * 100)
            counts.append(len(high_conf))
        else:
            accuracies.append(0)
            counts.append(0)

    ax.plot(thresholds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Minimum Confidence Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy at Different Confidence Thresholds',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Add count annotations
    for i, (threshold, count) in enumerate(zip(thresholds, counts)):
        if count > 0:
            ax.annotate(f'n={count}', xy=(threshold, accuracies[i]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_vs_confidence_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Confidence by Verdict Type
    fig, ax = plt.subplots(figsize=(12, 6))

    verdicts = ['SUPPORTS', 'REFUTES', 'INSUFFICIENT_EVIDENCE']
    verdict_data = {v: {'correct': [], 'incorrect': []} for v in verdicts}

    for r in valid_results:
        verdict = r["predicted"]
        if verdict in verdict_data:
            if r["correct"]:
                verdict_data[verdict]['correct'].append(r["confidence"])
            else:
                verdict_data[verdict]['incorrect'].append(r["confidence"])

    x_pos = np.arange(len(verdicts))
    width = 0.35

    correct_means = [np.mean(verdict_data[v]['correct']) if verdict_data[v]['correct'] else 0
                     for v in verdicts]
    incorrect_means = [np.mean(verdict_data[v]['incorrect']) if verdict_data[v]['incorrect'] else 0
                       for v in verdicts]

    bars1 = ax.bar(x_pos - width/2, correct_means, width, label='Correct', color='#06A77D')
    bars2 = ax.bar(x_pos + width/2, incorrect_means, width, label='Incorrect', color='#D62246')

    ax.set_xlabel('Verdict Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Confidence Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Average Confidence by Verdict Type\n{config_name}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['SUPPORTS', 'REFUTES', 'INSUF. EVID.'], fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "confidence_by_verdict.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved 3 confidence visualizations to {figures_dir}")


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
    print(f"{'='*80}\n")

    # Analyze each summary
    all_analyses = {}

    for summary_path in sorted(summary_files):
        # Extract configuration name from path
        config_name = summary_path.parent.name
        run_dir = summary_path.parent
        figures_dir = run_dir / "figures"

        try:
            summary = load_summary(summary_path)
            analysis = analyze_confidence_accuracy(summary["results"])

            if analysis:
                all_analyses[config_name] = analysis
                print(format_confidence_report(config_name, analysis))

                # Create visualizations
                if figures_dir.exists():
                    create_confidence_visualizations(summary["results"], figures_dir, config_name)

        except Exception as e:
            print(f"\nError processing {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save detailed results to JSON
    output_file = results_dir / "confidence_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Detailed analysis saved to: {output_file}")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

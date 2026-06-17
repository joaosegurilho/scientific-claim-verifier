"""Generate ablation study plots comparing experiments with different feature sets.

Usage:
    python experiments/analysis/ablation_plot.py [--output ablation.png] [--dataset scifact]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_experiment_metrics(exp_dir: Path) -> list[dict]:
    results = []

    config_file = exp_dir / "config.yaml"
    if not config_file.exists():
        return results
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        config = {}
    exp_name = (config or {}).get("experiment", {}).get("name", exp_dir.name)
    features = frozenset((config or {}).get("features", []))

    for metrics_path in sorted(exp_dir.glob("*/summary.json")):
        dataset_method = metrics_path.parent.name
        dataset = dataset_method.split("_", 1)[0]
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        eval_metrics = metrics.get("evaluation_metrics") or {}
        accuracy = eval_metrics.get("accuracy", metrics.get("accuracy"))
        macro_f1 = eval_metrics.get("macro_f1")
        if accuracy is None:
            continue
        results.append(
            {
                "name": exp_name,
                "dataset": dataset,
                "features": features,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
    return results


def find_baseline(experiments: list[dict]) -> int:
    for i, r in enumerate(experiments):
        if not r["features"]:
            return i
    return 0


def _compute_deltas(experiments: list[dict]):
    baseline_idx = find_baseline(experiments)
    baseline = experiments[baseline_idx]
    baseline_acc = baseline["accuracy"]
    baseline_f1 = baseline["macro_f1"]

    others = [r for i, r in enumerate(experiments) if i != baseline_idx]
    if not others:
        return None, None, None

    labels = []
    acc_deltas = []
    f1_deltas = []
    for r in others:
        feat_label = ", ".join(sorted(r["features"])) if r["features"] else "none"
        labels.append(f"{r['name']}\n({feat_label})")
        acc_delta = (
            (r["accuracy"] - baseline_acc) * 100 if r["accuracy"] is not None and baseline_acc is not None else 0
        )
        f1_delta = (r["macro_f1"] - baseline_f1) * 100 if r["macro_f1"] is not None and baseline_f1 is not None else 0
        acc_deltas.append(acc_delta)
        f1_deltas.append(f1_delta)

    return labels, acc_deltas, f1_deltas


def _add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4 if height >= 0 else -4),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def _make_single_plot(ax, experiments: list[dict], dataset: str, baseline_name: str):
    labels, acc_deltas, f1_deltas = _compute_deltas(experiments)
    if labels is None:
        ax.text(0.5, 0.5, "No non-baseline experiments", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, acc_deltas, width, label="Accuracy Δ (pp)", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1_deltas, width, label="Macro F1 Δ (pp)", color="coral")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Delta vs baseline (percentage points)")
    baseline_acc = [r["accuracy"] for r in experiments if not r["features"]]
    baseline_label = f"baseline: {baseline_name}"
    if baseline_acc:
        baseline_label += f" (acc={baseline_acc[0]:.2%})"
    ax.set_title(f"{dataset} — {baseline_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _add_labels(ax, bars1)
    _add_labels(ax, bars2)
    ax.legend(fontsize=8)


def make_ablation_plot(dataset_groups: dict[str, list[dict]], output_file: Path):
    if not dataset_groups:
        print("No results to plot.")
        return

    if len(dataset_groups) == 1:
        dataset, experiments = next(iter(dataset_groups.items()))
        fig, ax = plt.subplots(figsize=(10, 6))
        _make_single_plot(ax, experiments, dataset, experiments[0]["name"])
    else:
        n_datasets = len(dataset_groups)
        ncols = 2
        nrows = (n_datasets + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten() if n_datasets > 1 else [axes]
        for ax, (dataset, experiments) in zip(axes, sorted(dataset_groups.items())):
            _make_single_plot(ax, experiments, dataset, experiments[0]["name"])
        for ax in axes[n_datasets:]:
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ablation study results comparing experiments with different feature sets."
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Path to output image file.")
    parser.add_argument("--dataset", type=str, default=None, help="Only plot results for this dataset (e.g. scifact).")
    args = parser.parse_args()

    results_dir = Path("experiments/results")
    if not results_dir.exists():
        print("No experiments/results/ directory found.")
        return

    exp_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # Collect all results
    all_experiments = []
    for exp_dir in exp_dirs:
        all_experiments.extend(load_experiment_metrics(exp_dir))

    if not all_experiments:
        print("No experiment results found.")
        return

    # Group by dataset, then by experiment name
    # Within each dataset group, collect one experiment per name+features combination
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for exp in all_experiments:
        dataset = exp["dataset"]
        by_dataset[dataset].append(exp)

    if args.dataset:
        if args.dataset not in by_dataset:
            print(f"Dataset '{args.dataset}' not found. Available: {', '.join(sorted(by_dataset))}")
            return
        by_dataset = {args.dataset: by_dataset[args.dataset]}

    if args.output:
        output_file = args.output
    else:
        output_file = Path(f"experiments/analysis/ablation_plot_{datetime.now().strftime('%Y%m%d_%H%M')}.png")

    make_ablation_plot(dict(by_dataset), output_file)


if __name__ == "__main__":
    main()

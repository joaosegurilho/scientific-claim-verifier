"""Generate ablation/comparison plots across features, models, methods, and datasets.

Generates 4 figures:
  1. feature_ablation  — vary features, fix (dataset, method, llm_model)
  2. model_comparison  — vary llm_model, fix (dataset, method, features)
  3. method_comparison — vary method, fix (dataset, llm_model, features)
  4. dataset_comparison — vary dataset, fix (method, llm_model, features)

Usage:
    python experiments/analysis/ablation_plot.py [--output base_name.png] [--dataset scifact]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def format_features(features: frozenset) -> str:
    if not features:
        return "no-features"
    return "+".join(sorted(features))


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
        meta = metrics.get("model_metadata") or {}
        accuracy = eval_metrics.get("accuracy", metrics.get("accuracy"))
        macro_f1 = eval_metrics.get("macro_f1")
        if accuracy is None:
            continue
        results.append(
            {
                "name": exp_name,
                "dataset": dataset,
                "method": metrics.get("method", dataset_method.split("_", 1)[1] if "_" in dataset_method else "N/A"),
                "llm_model": meta.get("llm_model", "N/A"),
                "features": features,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
    return results


def group_experiments(experiments: list[dict], keys: tuple[str, ...]) -> dict:
    groups = defaultdict(list)
    for exp in experiments:
        key = tuple(exp[k] for k in keys)
        groups[key].append(exp)
    return dict(groups)


def find_baseline_idx(experiments: list[dict], varying_prop: str) -> int | None:
    if varying_prop == "features":
        for i, r in enumerate(experiments):
            if not r["features"]:
                return i
        return None
    return 0


def _key_sort_value(varying_prop: str):
    """Return a sort key function for experiments by the varying property."""
    if varying_prop == "features":
        return lambda e: format_features(e["features"])
    return lambda e: str(e[varying_prop])


def compute_deltas(experiments: list[dict], baseline_idx: int, varying_prop: str):
    baseline = experiments[baseline_idx]
    baseline_acc = baseline["accuracy"]
    baseline_f1 = baseline["macro_f1"]

    others = [r for i, r in enumerate(experiments) if i != baseline_idx]
    if not others:
        return None, None, None, None

    others.sort(key=_key_sort_value(varying_prop))
    labels = []
    acc_deltas = []
    f1_deltas = []
    for r in others:
        labels.append(format_features(r["features"]) if varying_prop == "features" else str(r[varying_prop]))
        acc_delta = (
            (r["accuracy"] - baseline_acc) * 100 if r["accuracy"] is not None and baseline_acc is not None else 0
        )
        f1_delta = (r["macro_f1"] - baseline_f1) * 100 if r["macro_f1"] is not None and baseline_f1 is not None else 0
        acc_deltas.append(acc_delta)
        f1_deltas.append(f1_delta)

    baseline_desc = format_features(baseline["features"]) if varying_prop == "features" else str(baseline[varying_prop])
    return labels, acc_deltas, f1_deltas, baseline_desc


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


def _display_key(key: tuple) -> str:
    parts = []
    for v in key:
        if isinstance(v, frozenset):
            parts.append(format_features(v))
        else:
            parts.append(str(v))
    return " / ".join(parts)


def plot_comparison(ax, experiments, baseline_idx, varying_prop, title):
    labels, acc_deltas, f1_deltas, baseline_desc = compute_deltas(experiments, baseline_idx, varying_prop)
    if labels is None:
        ax.text(
            0.5,
            0.5,
            "No non-baseline experiments",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, acc_deltas, width, label="Accuracy \u0394 (pp)", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1_deltas, width, label="Macro F1 \u0394 (pp)", color="coral")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Delta vs baseline (percentage points)")

    baseline_acc = experiments[baseline_idx]["accuracy"]
    baseline_label = f"baseline: {baseline_desc} (acc={baseline_acc:.2%})"
    ax.set_title(f"{title}\n{baseline_label}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    _add_labels(ax, bars1)
    _add_labels(ax, bars2)
    ax.legend(fontsize=8)


def generate_figure(all_experiments, group_keys, varying_prop, filter_dataset, output_path):
    groups = group_experiments(all_experiments, group_keys)

    if filter_dataset:
        groups = {k: [e for e in v if e["dataset"] == filter_dataset] for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}

    valid_groups = {}
    for key, exps in groups.items():
        if len(exps) < 2:
            continue
        values = {_key_sort_value(varying_prop)(e) for e in exps}
        if len(values) < 2:
            continue
        baseline_idx = find_baseline_idx(exps, varying_prop)
        if baseline_idx is None:
            continue
        valid_groups[key] = (exps, baseline_idx)

    if not valid_groups:
        print(f"No valid groups for {varying_prop} comparison → skipping {output_path}")
        return

    n = len(valid_groups)
    if n == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

    for ax, (key, (exps, baseline_idx)) in zip(axes, sorted(valid_groups.items())):
        plot_comparison(ax, exps, baseline_idx, varying_prop, _display_key(key))

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation/comparison plots across features, models, methods, and datasets."
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Base path for output images.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only plot results for this dataset (e.g. scifact).",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Alternative experiments folder path.",
    )
    args = parser.parse_args()

    results_dir = Path(args.exp_dir) if args.exp_dir else Path("experiments/results")
    if not results_dir.exists():
        print("No experiments/results/ directory found.")
        return

    exp_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    all_experiments = []
    for exp_dir in exp_dirs:
        all_experiments.extend(load_experiment_metrics(exp_dir))

    if not all_experiments:
        print("No experiment results found.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if args.output:
        base_stem = args.output.stem
        base_suffix = args.output.suffix
        parent_dir = args.output.parent
    else:
        base_stem = "ablation"
        base_suffix = ".png"
        parent_dir = Path("experiments/analysis")

    modes = [
        ("features", ("dataset", "method", "llm_model"), "features"),
        ("model", ("dataset", "method", "features"), "llm_model"),
        ("method", ("dataset", "llm_model", "features"), "method"),
        ("dataset", ("method", "llm_model", "features"), "dataset"),
    ]

    for mode_name, group_keys, varying_prop in modes:
        output_path = parent_dir / f"{base_stem}_{mode_name}_{timestamp}{base_suffix}"
        generate_figure(all_experiments, group_keys, varying_prop, args.dataset, output_path)


if __name__ == "__main__":
    main()

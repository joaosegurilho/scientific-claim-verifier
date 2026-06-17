"""Compare multiple experiment results and print a summary table.

Usage:
    python experiments/analysis/compare_experiments.py [--csv output.csv] [--filter name1,name2]
"""

import argparse
import json
from pathlib import Path

import yaml


def load_experiment_results(exp_dir: Path) -> list[dict]:
    config_file = exp_dir / "config.yaml"
    if not config_file.exists():
        return []
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        config = {}
    exp_name = (config or {}).get("experiment", {}).get("name", exp_dir.name)
    features = (config or {}).get("features", [])

    results = []
    for metrics_path in sorted(exp_dir.glob("*/summary.json")):
        dataset_method = metrics_path.parent.name
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        eval_metrics = metrics.get("evaluation_metrics") or {}
        results.append(
            {
                "path": str(metrics_path.parent),
                "name": f"{exp_name} ({dataset_method})",
                "dataset": dataset_method.split("_", 1)[0],
                "features": features,
                "accuracy": metrics.get("accuracy", "N/A"),
                "macro_f1": eval_metrics.get("macro_f1", "N/A"),
                "total_items": metrics.get("total_items", "N/A"),
                "successful": metrics.get("successful", "N/A"),
                "failed": metrics.get("failed", "N/A"),
                "execution_time": metrics.get("execution_time", {}).get("total_formatted", "N/A"),
            }
        )
    return results


def print_comparison_table(results: list[dict]):
    if not results:
        print("No experiment results found.")
        return
    header = f"{'Name':<35} {'Features':<20} {'Accuracy':<10} {'Macro F1':<10} {'Items':<8} {'Time':<16}"
    print("=" * len(header))
    print("Experiment Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        features = ", ".join(r["features"]) if r["features"] else "none"
        print(
            f"{r['name']:<35} {features:<20} {str(r['accuracy']):<10} {str(r['macro_f1']):<10} "
            f"{str(r['total_items']):<8} {str(r['execution_time']):<16}"
        )
    print("=" * len(header))


def export_csv(results: list[dict], path: Path):
    import csv

    fieldnames = ["name", "dataset", "features", "accuracy", "macro_f1", "total_items", "failed", "execution_time"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "name": r["name"],
                    "dataset": r["dataset"],
                    "features": ", ".join(r["features"]),
                    "accuracy": r["accuracy"],
                    "macro_f1": r["macro_f1"],
                    "total_items": r["total_items"],
                    "failed": r["failed"],
                    "execution_time": r["execution_time"],
                }
            )
    print(f"\nCSV exported to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiment configurations.")
    parser.add_argument("--csv", type=Path, help="Path to output CSV file.")
    parser.add_argument(
        "--filter",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help="Filter experiments by name (comma-separated).",
    )
    args = parser.parse_args()

    results_dir = Path("experiments/results")
    if not results_dir.exists():
        print("No experiments/results/ directory found.")
        return

    exp_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    all_results = []
    for exp_dir in exp_dirs:
        all_results.extend(load_experiment_results(exp_dir))

    if args.filter:
        filter_set = set(f.lower() for f in args.filter)
        all_results = [r for r in all_results if any(f in r["name"].lower().split(" (")[0] for f in filter_set)]

    print_comparison_table(all_results)

    if args.csv:
        export_csv(all_results, args.csv)


if __name__ == "__main__":
    main()

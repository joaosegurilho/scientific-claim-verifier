#!/usr/bin/env python3
"""Recompute benchmark metrics from per-claim log JSON files.

This is useful when a run directory's summary/evaluation files were overwritten,
corrupted, or generated from a subset of claims.

Usage (PowerShell):
  python scripts/recompute_run_metrics.py <path-to-run-dir>
  python scripts/recompute_run_metrics.py <path-to-run-dir> --overwrite

Outputs:
  - evaluation_report_recomputed.txt (or evaluation_report.txt with --overwrite)
  - summary_recomputed.json (or summary.json with --overwrite)
  - figures/confusion_matrix_recomputed.png (or figures/confusion_matrix.png with --overwrite)

The script expects per-claim JSONs under <run-dir>/logs/*.json produced by the
benchmark runner (they contain expected_result and result.verdict).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from scverifier.core.benchmarking.base import BenchmarkEvaluator


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_expected(data: dict[str, Any]) -> Optional[str]:
    for key in ("expected_result", "expected", "label", "gold"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_predicted(data: dict[str, Any]) -> str:
    result = data.get("result")
    if isinstance(result, dict):
        verdict = result.get("verdict")
        if isinstance(verdict, str) and verdict.strip():
            return verdict.strip()
    return "ERROR"


def _extract_claim_id(data: dict[str, Any], fallback: str) -> str:
    cid = data.get("claim_id") or data.get("id")
    if cid is None:
        return fallback
    return str(cid)


def _compute_from_logs(logs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for json_path in sorted(logs_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        data = _read_json(json_path)
        if not isinstance(data, dict):
            errors.append(json_path.name)
            continue

        claim_id = _extract_claim_id(data, json_path.stem)
        claim = data.get("claim") if isinstance(data.get("claim"), str) else ""
        expected = _extract_expected(data)
        predicted = _extract_predicted(data)

        if expected is None:
            # If expected is missing we cannot evaluate; keep traceability.
            errors.append(json_path.name)
            continue

        rows.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "expected": expected,
                "predicted": predicted,
                "correct": (predicted != "ERROR" and predicted == expected),
            }
        )

    meta = {
        "log_json_files": len(list(logs_dir.glob("*.json"))),
        "parsed_items": len(rows),
        "skipped_files": len(errors),
        "skipped_examples": errors[:10],
    }

    return rows, meta


def _save_confusion_matrix_png(figures_dir: Path, metrics, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(metrics.confusion_matrix.keys())

    label_mapping = {
        "INSUFFICIENT_EVIDENCE": "INSUF. EVID.",
        "REFUTES": "REFUTES",
        "SUPPORTS": "SUPPORTS",
    }
    display_labels = [label_mapping.get(l, l) for l in labels]

    cm_array = np.array([[metrics.confusion_matrix[a].get(p, 0) for p in labels] for a in labels])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        annot_kws={"size": 40},
        cbar_kws={"shrink": 0.8},
    )
    plt.xlabel("Predicted", fontsize=20, fontweight="bold")
    plt.ylabel("Actual", fontsize=20, fontweight="bold")
    plt.title(title, fontsize=22, fontweight="bold", pad=20)
    plt.xticks(fontsize=16, rotation=45, ha="right")
    plt.yticks(fontsize=16, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute benchmark metrics from logs/*.json")
    parser.add_argument("run_dir", type=str, help="Path to a benchmark run directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite summary.json/evaluation_report.txt/confusion_matrix.png instead of writing *_recomputed.* files",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not logs_dir.exists():
        raise SystemExit(f"logs directory not found: {logs_dir}")

    rows, meta = _compute_from_logs(logs_dir)

    # Best-effort title inference to keep figures consistent with BenchmarkRunner output.
    run_name = run_dir.name.lower()
    if "scifact" in run_name:
        benchmark_name = "SciFact"
    elif "msvec" in run_name:
        benchmark_name = "MSVEC"
    elif "healthver" in run_name:
        benchmark_name = "HealthVer"
    elif "coverbench" in run_name:
        benchmark_name = "CoverBench"
    else:
        benchmark_name = "Benchmark"

    sample_json = next(iter(sorted(logs_dir.glob("*.json"))), None)
    sample_data = _read_json(sample_json) if sample_json else None
    method = "unknown"
    if isinstance(sample_data, dict):
        vm = sample_data.get("verification_method")
        if isinstance(vm, str) and vm.strip():
            method = vm.strip()

    cm_title = f"Confusion Matrix - {benchmark_name} ({method})"

    if not rows:
        raise SystemExit("No evaluable items found in logs/*.json")

    # Treat ERROR as INSUFFICIENT_EVIDENCE for metric computation, matching runner behavior.
    y_true = [r["expected"] for r in rows]
    y_pred = ["INSUFFICIENT_EVIDENCE" if r["predicted"] == "ERROR" else r["predicted"] for r in rows]

    metrics = BenchmarkEvaluator.compute_from_predictions(y_true, y_pred)

    # Prepare summary structure (similar to BenchmarkRunner._save_summary)
    error_count = sum(1 for r in rows if r["predicted"] == "ERROR")
    correct = sum(1 for r in rows if r["predicted"] != "ERROR" and r["predicted"] == r["expected"])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_items": len(rows),
        "successful": len(rows) - error_count,
        "failed": error_count,
        "error_count": error_count,
        "accuracy": correct / len(rows),
        "results": rows,
        "evaluation_metrics": metrics.to_dict(),
        "recomputed_from_logs": True,
        "recompute_meta": meta,
        "label_distribution_expected": dict(Counter(y_true)),
        "label_distribution_predicted": dict(Counter(y_pred)),
    }

    if args.overwrite:
        report_path = run_dir / "evaluation_report.txt"
        summary_path = run_dir / "summary.json"
        cm_path = figures_dir / "confusion_matrix.png"
    else:
        report_path = run_dir / "evaluation_report_recomputed.txt"
        summary_path = run_dir / "summary_recomputed.json"
        cm_path = figures_dir / "confusion_matrix_recomputed.png"

    report_title = "SciFact (agent) [recomputed from logs]"
    report_path.write_text(metrics.print_report(report_title), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _save_confusion_matrix_png(figures_dir, metrics, cm_path, title=cm_title)

    print(f"Wrote: {report_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {cm_path}")

    # Quick sanity diagnostics
    print("\nSANITY")
    print(f"- log_json_files: {meta['log_json_files']}")
    print(f"- parsed_items:   {meta['parsed_items']}")
    print(f"- skipped_files:  {meta['skipped_files']}")
    print(f"- accuracy:       {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
    print(f"- correct:        {metrics.correct_predictions}/{metrics.total_predictions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

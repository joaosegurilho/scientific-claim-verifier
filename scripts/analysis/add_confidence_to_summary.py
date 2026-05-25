#!/usr/bin/env python3
"""Add confidence scores from individual log files to summary.json.

This script reads confidence scores from individual claim logs (*.json files in logs/)
and adds them to the results in summary.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def add_confidence_to_summary(benchmark_dir: Path) -> bool:
    """Add confidence scores from logs to summary.json.

    Args:
        benchmark_dir: Path to benchmark result directory

    Returns:
        True if successful, False otherwise
    """
    summary_path = benchmark_dir / "summary.json"
    logs_dir = benchmark_dir / "logs"

    if not summary_path.exists():
        print(f"Error: summary.json not found in {benchmark_dir}")
        return False

    if not logs_dir.exists():
        print(f"Error: logs directory not found in {benchmark_dir}")
        return False

    # Load summary
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # Build a map of claim_id -> log data
    log_data = {}
    for log_file in logs_dir.glob("*.json"):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log = json.load(f)
                claim_id = log.get("claim_id")
                if claim_id and "result" in log and log["result"]:
                    confidence = log["result"].get("confidence")
                    if confidence is not None:
                        log_data[claim_id] = {"confidence": confidence}
        except Exception as e:
            print(f"Warning: Failed to read {log_file}: {e}")
            continue

    if not log_data:
        print(f"No confidence data found in logs")
        return False

    # Update summary results with confidence scores
    updated_count = 0
    for result in summary.get("results", []):
        claim_id = result.get("claim_id")
        if claim_id in log_data and "confidence" not in result:
            result["confidence"] = log_data[claim_id]["confidence"]
            updated_count += 1

    # Save updated summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated_count} results with confidence scores in {summary_path}")
    return True


def main():
    if len(sys.argv) > 1:
        # Process specific directory
        benchmark_dir = Path(sys.argv[1])
        if not benchmark_dir.exists():
            print(f"Error: Directory not found: {benchmark_dir}")
            return 1
        success = add_confidence_to_summary(benchmark_dir)
        return 0 if success else 1
    else:
        # Find and process all benchmark directories
        base_dir = Path(__file__).parent.parent.parent
        results_dir = base_dir / "benchmark_results"

        if not results_dir.exists():
            print(f"Error: benchmark_results directory not found at {results_dir}")
            return 1

        # Find all directories with summary.json
        summary_files = list(results_dir.rglob("summary.json"))

        if not summary_files:
            print(f"No summary.json files found in {results_dir}")
            return 1

        print(f"Found {len(summary_files)} benchmark results")
        print(f"{'=' * 80}\n")

        success_count = 0
        for summary_path in sorted(summary_files):
            benchmark_dir = summary_path.parent
            config_name = benchmark_dir.name

            print(f"Processing: {config_name}")
            if add_confidence_to_summary(benchmark_dir):
                success_count += 1
            print()

        print(f"{'=' * 80}")
        print(f"Successfully updated {success_count}/{len(summary_files)} summaries")
        print(f"{'=' * 80}")

        return 0


if __name__ == "__main__":
    sys.exit(main())

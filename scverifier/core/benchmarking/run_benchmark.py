#!/usr/bin/env python3
"""Main script to run benchmarks for scientific claim verification.

Usage:
    python -m scverifier.core.benchmarking.run_benchmark scifact --method agentless
    python -m scverifier.core.benchmarking.run_benchmark scifact --method agent
    python -m scverifier.core.benchmarking.run_benchmark msvec --method agentless
    python -m scverifier.core.benchmarking.run_benchmark healthver --split validation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import io
import contextlib

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scverifier.core.benchmarking.base import Benchmark, VerificationMethod, BenchmarkEvaluator
from scverifier.core.benchmarking.scifact_benchmark import SciFact
from scverifier.core.benchmarking.coverbench_benchmark import CoverBench
from scverifier.core.benchmarking.healthver_benchmark import HealthVer
from scverifier.core.benchmarking.msvec_benchmark import MSVEC
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.agents.autonomous_agent import AutonomousClaimAgent
from scverifier.pipelines.verification_pipeline import VerificationPipeline
from scverifier.config.settings import Config


def detect_claim_status(run_dir: Path) -> tuple[set, set]:
    """Detect completed and error claims from run directory.

    Args:
        run_dir: Path to benchmark run directory

    Returns:
        Tuple of (completed_ids, error_ids) - Sets of claim IDs as strings
    """
    completed = set()
    errors = set()

    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return completed, errors

    for json_file in logs_dir.glob("*.json"):
        claim_id = json_file.stem
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                if data.get("result") is None:
                    errors.add(claim_id)
                else:
                    completed.add(claim_id)
        except (json.JSONDecodeError, KeyError):
            # Treat corrupted files as errors
            errors.add(claim_id)

    return completed, errors


class BenchmarkRunner:
    """Runner for executing benchmark evaluations with multiple verification methods."""

    def __init__(self, benchmark: Benchmark, results_dir: Path, method: VerificationMethod, resume_dir: Path = None):
        self.benchmark = benchmark
        self.method = method
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.resume_dir = resume_dir

        if resume_dir:
            # Reuse existing directory
            self.run_dir = resume_dir
        else:
            # Create new timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method_name = method.value
            self.run_dir = self.results_dir / f"{benchmark.name.lower()}_{method_name}_{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def run(self, max_papers: int = 10):
        """Run the benchmark evaluation."""
        # Start timing
        start_time = time.time()

        use_search = self.method in (VerificationMethod.AGENT_WITH_SEARCH, VerificationMethod.AGENTLESS_WITH_SEARCH)
        use_agent = self.method in (VerificationMethod.AGENT, VerificationMethod.AGENT_WITH_SEARCH)

        # Collect model metadata
        self.model_metadata = {
            "agent_model": Config.AGENT_MODEL,
            "llm_model": Config.LLM_MODEL,
            "fallback_model": Config.LLM_FALLBACK_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
            "temperature": Config.AGENT_TEMPERATURE if use_agent else Config.LLM_TEMPERATURE,
            "agent_max_output_tokens": Config.AGENT_MAX_OUTPUT_TOKENS,
            "agent_recursion_limit": Config.RECURSION_LIMIT,
        }

        print(f"\n{'='*70}")
        print(f"Running {self.benchmark.name} Benchmark")
        print(f"{'='*70}")
        print(f"Method: {self.method.value}")
        print(f"Model: {self.model_metadata['agent_model'] if use_agent else self.model_metadata['llm_model']}")
        print(f"Total items: {len(self.benchmark.items)}")
        print(f"Results directory: {self.run_dir}")
        print(f"Online search: {use_search}")
        if use_search:
            print(f"Max papers per claim: {max_papers}")
        print(f"{'='*70}\n")

        # Initialize KB
        kb = KnowledgeBase()
        try:
            kb.load()
            print("Knowledge base loaded.\n")
        except Exception:
            print("Starting with fresh knowledge base.\n")

        # Initialize verifier based on method
        pipeline = VerificationPipeline(kb=kb) if not use_agent else None
        agent = AutonomousClaimAgent(kb=kb, allow_online_search=use_search) if use_agent else None

        # Run benchmark
        results = []
        successful, failed = 0, 0

        for idx, item in enumerate(self.benchmark.items, 1):
            print(f"\n[{idx}/{len(self.benchmark.items)}] Claim {item.claim_id}")
            print(f"Claim: {item.claim[:100]}...")
            print(f"Expected: {item.expected_result}")

            log_buffer = io.StringIO()
            error_occurred = False
            error_message = None

            # Retry logic for rate limiting (429 errors)
            max_retries = 3
            retry_delay = 60  # 1 minute

            for attempt in range(max_retries + 1):
                try:
                    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                        if use_agent:
                            result = agent.verify_claim(item.claim, thread_id=f"bench_{item.claim_id}")
                        elif use_search:
                            result = pipeline.verify_claim_with_search(item.claim, max_papers=max_papers)
                        else:
                            # quality_claims=False because batch processing skips proposition evaluation
                            result = pipeline.verify_claim_from_kb(item.claim, quality_claims=False)

                    item.result = result
                    successful += 1
                    print(f"Verdict: {result.verdict} | Confidence: {result.confidence} | Correct: {result.verdict == item.expected_result}")
                    break  # Success, exit retry loop

                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower()

                    if is_rate_limit and attempt < max_retries:
                        print(f"Rate limit error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"Waiting {retry_delay} seconds before retry...")
                        log_buffer.write(f"\nRate limit error (attempt {attempt + 1}): {e}\n")
                        log_buffer.write(f"Waiting {retry_delay} seconds before retry...\n")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final failure (either not a rate limit error, or exhausted retries)
                        failed += 1
                        error_occurred = True
                        error_message = str(e)
                        print(f"ERROR: {e}")
                        log_buffer.write(f"\nERROR: {e}\n")
                        item.result = None  # No result due to error
                        break

            # Save logs
            self._save_claim_logs(item, use_agent, log_buffer.getvalue())

            # Collect results - record ERROR for traceability, but metrics will treat as INSUFFICIENT_EVIDENCE
            if error_occurred:
                results.append({
                    "claim_id": item.claim_id,
                    "claim": item.claim,
                    "expected": item.expected_result,
                    "predicted": "ERROR",
                    "confidence": 0.0,
                    "correct": False,
                    "error_message": error_message,
                })
            else:
                result_dict = {
                    "claim_id": item.claim_id,
                    "claim": item.claim,
                    "expected": item.expected_result,
                    "predicted": item.result.verdict,
                    "confidence": item.result.confidence,
                    "correct": item.result.verdict == item.expected_result,
                }
                # Add token usage if available
                if item.result.token_usage:
                    result_dict["token_usage"] = item.result.token_usage
                results.append(result_dict)

            # Save KB periodically (only if using online search)
            if use_search and idx % 10 == 0:
                kb.save()

        # Final save (only if using online search)
        if use_search:
            kb.save()

        # Merge with original results if resuming
        if self.resume_dir:
            results = self._merge_results(results)
            # Recompute successful/failed counts from merged results
            successful = sum(1 for r in results if r["predicted"] != "ERROR")
            failed = sum(1 for r in results if r["predicted"] == "ERROR")

        # Calculate execution time
        end_time = time.time()
        total_execution_time = end_time - start_time
        avg_execution_time = total_execution_time / len(results) if results else 0

        # Compute metrics and save results
        metrics = self._compute_metrics(results, successful, failed)
        self._save_summary(results, successful, failed, metrics, total_execution_time, avg_execution_time)
        self._save_visualizations(metrics)

    def _merge_results(self, new_results: list) -> list:
        """Merge new results with original run results.

        Args:
            new_results: List of result dictionaries from current run

        Returns:
            Combined list of all results
        """
        if not self.resume_dir:
            return new_results

        original_summary = self.resume_dir / "summary.json"
        if not original_summary.exists():
            return new_results  # No original to merge

        try:
            with open(original_summary, encoding="utf-8") as f:
                original_data = json.load(f)

            # Create dict for easy lookup by claim_id
            merged = {r["claim_id"]: r for r in original_data.get("results", [])}

            # Overwrite with new results (errors replaced + new claims)
            for result in new_results:
                merged[result["claim_id"]] = result

            return list(merged.values())

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not merge with original results: {e}")
            return new_results

    def _save_claim_logs(self, item, use_agent: bool, log_content: str):
        """Save individual claim logs."""
        log_file = self.logs_dir / f"{item.claim_id}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Claim ID: {item.claim_id}\n")
            f.write(f"Claim: {item.claim}\n")
            f.write(f"Expected: {item.expected_result}\n")
            f.write(f"Method: {self.method.value}\n")
            f.write(f"Model: {self.model_metadata['agent_model'] if use_agent else self.model_metadata['llm_model']}\n")
            f.write(f"Temperature: {self.model_metadata['temperature']}\n")
            f.write(f"{'='*70}\n\n")
            if log_content:
                f.write(log_content)
            if item.result:
                f.write(f"\n{'='*70}\nRESULT:\n")
                f.write(f"Verdict: {item.result.verdict}\n")
                f.write(f"Confidence: {item.result.confidence}\n")
                f.write(f"Reasoning: {item.result.reasoning}\n")

        # Save JSON
        json_file = self.logs_dir / f"{item.claim_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(item.to_dict(), f, indent=2)

    def _compute_metrics(self, results, successful, failed):
        """Compute evaluation metrics.

        ERROR predictions are treated as INSUFFICIENT_EVIDENCE for metric computation,
        but remain as ERROR in the results for traceability.
        """
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Successful: {successful}, Failed: {failed}")

        metrics = None
        total = successful + failed
        if total > 0:
            try:
                # Extract predictions from results, treating ERROR as INSUFFICIENT_EVIDENCE
                y_true = [r["expected"] for r in results]
                y_pred = [
                    "INSUFFICIENT_EVIDENCE" if r["predicted"] == "ERROR" else r["predicted"]
                    for r in results
                ]

                metrics = BenchmarkEvaluator.compute_from_predictions(y_true, y_pred)
                print(metrics.print_report(f"{self.benchmark.name} ({self.method.value})"))

                if failed > 0:
                    print(f"\nNote: {failed} ERROR predictions were treated as INSUFFICIENT_EVIDENCE for metrics.")
            except ValueError as e:
                print(f"Could not compute metrics: {e}")
                correct = sum(1 for r in results if r["correct"])
                print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

        return metrics

    def _save_summary(self, results, successful, failed, metrics, total_execution_time, avg_execution_time):
        """Save summary JSON."""
        total = successful + failed
        # Count errors in results
        error_count = sum(1 for r in results if r["predicted"] == "ERROR")

        # Format execution time for display
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.2f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes}m {secs:.1f}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours}h {minutes}m {secs:.1f}s"

        # Calculate token usage statistics
        token_stats = self._calculate_token_stats(results)

        summary = {
            "benchmark": self.benchmark.name,
            "method": self.method.value,
            "model_metadata": self.model_metadata,
            "timestamp": datetime.now().isoformat(),
            "total_items": len(self.benchmark.items),
            "successful": successful,
            "failed": failed,
            "error_count": error_count,
            "accuracy": sum(1 for r in results if r["correct"]) / total if total > 0 else 0,
            "execution_time": {
                "total_seconds": round(total_execution_time, 2),
                "total_formatted": format_time(total_execution_time),
                "avg_per_claim_seconds": round(avg_execution_time, 2),
                "avg_per_claim_formatted": format_time(avg_execution_time),
            },
            "token_usage": token_stats,
            "results": results,
        }
        if metrics:
            summary["evaluation_metrics"] = metrics.to_dict()

        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if metrics:
            with open(self.run_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
                f.write(metrics.print_report(f"{self.benchmark.name} ({self.method.value})"))

        print(f"\nResults saved to: {self.run_dir}")
        print(f"Total execution time: {format_time(total_execution_time)}")
        print(f"Average time per claim: {format_time(avg_execution_time)}")
        if token_stats:
            print(f"Total tokens used: {token_stats['total_tokens']:,} (input: {token_stats['total_input_tokens']:,}, output: {token_stats['total_output_tokens']:,})")
            print(f"Average tokens per claim: {token_stats['avg_tokens_per_claim']:.0f}")

    def _calculate_token_stats(self, results):
        """Calculate token usage statistics from results."""
        results_with_tokens = [r for r in results if "token_usage" in r and r["token_usage"]]

        if not results_with_tokens:
            return None

        total_input = sum(r["token_usage"]["input_tokens"] for r in results_with_tokens)
        total_output = sum(r["token_usage"]["output_tokens"] for r in results_with_tokens)
        total_tokens = sum(r["token_usage"]["total_tokens"] for r in results_with_tokens)

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "avg_input_tokens_per_claim": round(total_input / len(results_with_tokens), 2) if results_with_tokens else 0,
            "avg_output_tokens_per_claim": round(total_output / len(results_with_tokens), 2) if results_with_tokens else 0,
            "avg_tokens_per_claim": round(total_tokens / len(results_with_tokens), 2) if results_with_tokens else 0,
            "claims_with_token_data": len(results_with_tokens),
        }

    def _save_visualizations(self, metrics):
        """Generate and save visualization figures."""
        if not metrics:
            return

        # Confusion Matrix
        labels = sorted(metrics.confusion_matrix.keys())
        
        # Abbreviated label names for better display
        label_mapping = {
            "INSUFFICIENT_EVIDENCE": "INSUF. EVID.",
            "REFUTES": "REFUTES",
            "SUPPORTS": "SUPPORTS"
        }
        display_labels = [label_mapping.get(l, l) for l in labels]
        
        cm_array = np.array([[metrics.confusion_matrix[a].get(p, 0) for p in labels] for a in labels])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                    xticklabels=display_labels, yticklabels=display_labels,
                    annot_kws={'size': 40},
                    cbar_kws={'shrink': 0.8})
        plt.xlabel('Predicted', fontsize=20, fontweight='bold')
        plt.ylabel('Actual', fontsize=20, fontweight='bold')
        plt.title(f'Confusion Matrix - {self.benchmark.name} ({self.method.value})', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(fontsize=16, rotation=45, ha='right')
        plt.yticks(fontsize=16, rotation=0)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Label Distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Expected distribution
        expected = metrics.label_distribution_expected
        axes[0].bar(expected.keys(), expected.values(), color='steelblue')
        axes[0].set_title('Expected Labels')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)

        # Predicted distribution
        predicted = metrics.label_distribution_predicted
        axes[1].bar(predicted.keys(), predicted.values(), color='coral')
        axes[1].set_title('Predicted Labels')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)

        plt.suptitle(f'Label Distribution - {self.benchmark.name} ({self.method.value})')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "label_distribution.png", dpi=150)
        plt.close()

        print(f"Figures saved to: {self.figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")

    parser.add_argument("dataset", choices=["scifact", "coverbench", "healthver", "msvec"])
    parser.add_argument("--method", choices=["agent", "agentless", "agent_with_search", "agentless_with_search"],
                        default="agentless")
    parser.add_argument("--max-items", type=int, help="Max items to evaluate")
    parser.add_argument("--max-papers", type=int, default=10, help="Max papers per claim (for search methods)")
    parser.add_argument("--results-dir", default="./benchmark_results")
    parser.add_argument("--claims-file", type=str, help="Specific claims file (SciFact)")
    parser.add_argument("--scifact-split", choices=["train", "dev", "test"], help="SciFact split")
    parser.add_argument("--split", default="validation", help="Dataset split (HealthVer)")
    parser.add_argument("--data-dir", type=str, help="Data directory (MSVEC)")
    parser.add_argument("--resume", type=str, help="Resume incomplete run from directory (e.g., benchmark_results/scifact_agent_20251214_221046)")

    args = parser.parse_args()

    method_map = {
        "agent": VerificationMethod.AGENT,
        "agentless": VerificationMethod.AGENTLESS,
        "agent_with_search": VerificationMethod.AGENT_WITH_SEARCH,
        "agentless_with_search": VerificationMethod.AGENTLESS_WITH_SEARCH,
    }
    method = method_map[args.method]

    # Load benchmark
    try:
        if args.dataset == "scifact":
            benchmark = SciFact(claims_file=args.claims_file,
                                split=args.scifact_split if not args.claims_file else None,
                                verification_method=method)
            benchmark.load(max_items=args.max_items)

        elif args.dataset == "coverbench":
            benchmark = CoverBench(verification_method=method)
            benchmark.load(max_items=args.max_items)

        elif args.dataset == "healthver":
            benchmark = HealthVer(verification_method=method)
            benchmark.load(max_items=args.max_items, split=args.split)

        elif args.dataset == "msvec":
            benchmark = MSVEC(data_dir=args.data_dir, verification_method=method)
            benchmark.load(max_items=args.max_items)

        else:
            print(f"Unknown dataset: {args.dataset}")
            sys.exit(1)

    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"Error loading benchmark: {e}")
        sys.exit(1)

    # Print stats
    stats = benchmark.get_statistics()
    print(f"\n{benchmark.name} - {stats['total_items']} items")
    print(f"Labels: {stats['label_distribution']}")

    # Handle resume if specified
    resume_dir = None
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"Error: Resume directory not found: {resume_dir}")
            sys.exit(1)

        # Detect completed and error claims
        completed, errors = detect_claim_status(resume_dir)
        all_ids = {str(item.claim_id) for item in benchmark.items}
        never_attempted = all_ids - (completed | errors)
        to_process = never_attempted | errors

        if len(to_process) == 0:
            print("\nAll claims already completed successfully!")
            print(f"Results are in: {resume_dir}")
            sys.exit(0)

        # Filter benchmark items to only process incomplete/error claims
        benchmark.items = [item for item in benchmark.items if str(item.claim_id) in to_process]

        print(f"\n{'='*70}")
        print(f"RESUMING FROM: {resume_dir.name}")
        print(f"{'='*70}")
        print(f"Completed claims: {len(completed)}")
        print(f"Error claims to retry: {len(errors)}")
        print(f"Never attempted: {len(never_attempted)}")
        print(f"Total to process: {len(to_process)}")
        print(f"{'='*70}\n")

    # Run
    runner = BenchmarkRunner(benchmark, Path(args.results_dir), method, resume_dir=resume_dir)
    runner.run(max_papers=args.max_papers)


if __name__ == "__main__":
    main()

"""Test script for the BenchmarkRunner.

This script tests the full BenchmarkRunner infrastructure with a single claim.
It's closer to actual usage than test_benchmark_system.py and tests:
- BenchmarkRunner orchestration
- Complete logging and output capture
- Knowledge base persistence
- Statistics generation
- Summary file creation

This creates output that looks exactly like a real benchmark run.
"""

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

from scverifier.core.benchmarking import SciFact, VerificationMethod


def create_test_claims_file(num_claims=1):
    """Create a temporary SciFact-style claims file with test claims.

    Args:
        num_claims: Number of test claims to create (default: 1)

    Returns:
        Path to temporary claims file
    """
    # Define test claims with different labels
    test_claims = [
        {
            "id": 1001,
            "claim": "Regular physical exercise reduces blood pressure in adults.",
            "evidence": {
                "123": [{"label": "SUPPORT", "sentences": [0, 1, 2]}]
            },
            "cited_doc_ids": [123]
        },
        {
            "id": 1002,
            "claim": "Vitamin C supplements cure the common cold.",
            "evidence": {
                "456": [{"label": "CONTRADICT", "sentences": [3, 4, 5]}]
            },
            "cited_doc_ids": [456]
        },
        {
            "id": 1003,
            "claim": "Green tea consumption prevents all forms of cancer.",
            "evidence": {},
            "cited_doc_ids": []
        },
    ]

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)

    for i, claim in enumerate(test_claims[:num_claims]):
        json.dump(claim, temp_file)
        temp_file.write('\n')

    temp_file.flush()
    temp_file.close()

    return temp_file.name


def test_benchmark_runner(use_search=False, num_claims=1):
    """Test the BenchmarkRunner with a small number of claims.

    Args:
        use_search: Whether to use online paper search
        num_claims: Number of claims to test (1-3)
    """
    # Import here to avoid circular imports and to show it works
    from scverifier.core.benchmarking.run_benchmark import BenchmarkRunner

    print("\n" + "="*70)
    print("BENCHMARK RUNNER TEST")
    print("="*70)
    print(f"Number of claims: {num_claims}")
    print(f"Use search: {use_search}")
    print("="*70 + "\n")

    # Create test directory
    test_results_dir = Path("tests/test_benchmark_runner_output")
    test_results_dir.mkdir(exist_ok=True)

    # Create test claims file
    print("Creating test claims file...")
    claims_file = create_test_claims_file(num_claims=num_claims)
    print(f"Test claims file: {claims_file}\n")

    try:
        # Load benchmark
        print("Loading benchmark...")
        benchmark = SciFact(
            claims_file=claims_file,
            verification_method=VerificationMethod.AGENTLESS
        )
        benchmark.load(max_items=num_claims)
        print(f"Loaded {len(benchmark.items)} item(s)\n")

        # Print claims
        print("Test claims:")
        for item in benchmark.items:
            print(f"  {item.claim_id}: {item.claim}")
            print(f"    Expected: {item.expected_result}")
        print()

        # Create runner
        print("Creating BenchmarkRunner...")
        runner = BenchmarkRunner(benchmark, test_results_dir)
        print(f"Results will be saved to: {runner.run_dir}\n")

        # Run benchmark
        print("="*70)
        print("RUNNING BENCHMARK")
        print("="*70 + "\n")

        runner.run(use_search=use_search, max_papers=5)

        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {runner.run_dir}")

        # Print key results
        summary_file = runner.run_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

            print("\nKey results:")
            print(f"  Accuracy: {summary['accuracy']:.2%}")
            print(f"  Successful: {summary['successful']}/{summary['total_items']}")
            print(f"  Failed: {summary['failed']}/{summary['total_items']}")

            print("\nPer-claim results:")
            for result in summary['results']:
                status = "✓" if result['correct'] else "✗"
                print(f"  {status} {result['claim_id']}: "
                      f"{result['predicted']} (expected: {result['expected']})")

        print("\n" + "="*70)

        # Cleanup
        Path(claims_file).unlink()
        print("\nCleaned up temporary claims file")

        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        traceback.print_exc()

        # Cleanup
        if Path(claims_file).exists():
            Path(claims_file).unlink()

        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the BenchmarkRunner with a small number of claims"
    )
    parser.add_argument(
        "--use-search",
        action="store_true",
        help="Use online paper search (otherwise uses existing KB only)"
    )
    parser.add_argument(
        "--num-claims",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of test claims to run (1-3, default: 1)"
    )

    args = parser.parse_args()

    success = test_benchmark_runner(
        use_search=args.use_search,
        num_claims=args.num_claims
    )

    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

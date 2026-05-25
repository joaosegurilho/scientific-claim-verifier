"""Test script for the benchmarking system.

This script tests the new benchmark infrastructure by running a single claim
through the benchmark system, verifying that:
- Benchmark loading works correctly
- Verification pipeline integrates properly
- Logs are saved correctly
- Knowledge base is persisted
- Results are formatted properly

This is faster than running full benchmarks and useful for development/debugging.
"""

import argparse
import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from scverifier.core.benchmarking import SciFact, VerificationMethod
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.pipelines.verification_pipeline import VerificationPipeline


def create_test_scifact_claim():
    """Create a temporary SciFact-style claims file with one claim for testing."""
    # Create a test claim in SciFact format
    test_claim = {
        "id": 999,
        "claim": "Regular physical exercise reduces blood pressure in adults.",
        "evidence": {
            "123": [
                {"label": "SUPPORT", "sentences": [0, 1, 2]}
            ]
        },
        "cited_doc_ids": [123]
    }

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    json.dump(test_claim, temp_file)
    temp_file.write('\n')
    temp_file.flush()
    temp_file.close()

    return temp_file.name


def test_benchmark_system(use_search=False):
    """Test the benchmark system with a single claim.

    Args:
        use_search: Whether to use online paper search (False = use existing KB only)
    """
    print("\n" + "="*70)
    print("BENCHMARK SYSTEM TEST")
    print("="*70)
    print(f"Test mode: {'With online search' if use_search else 'Knowledge base only'}")
    print("="*70 + "\n")

    # Create test directories
    test_dir = Path("tests/test_benchmark_output")
    test_dir.mkdir(exist_ok=True)

    logs_dir = test_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    kb_dir = test_dir / "knowledge_base"
    kb_dir.mkdir(exist_ok=True)

    print(f"Test output directory: {test_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Knowledge base directory: {kb_dir}\n")

    # Create test claims file
    print("Creating test claim...")
    claims_file = create_test_scifact_claim()
    print(f"Test claims file: {claims_file}\n")

    try:
        # Load benchmark
        print("Loading SciFact benchmark...")
        benchmark = SciFact(
            claims_file=claims_file,
            verification_method=VerificationMethod.AGENTLESS
        )
        benchmark.load(max_items=1)
        print(f"Loaded {len(benchmark.items)} item(s)\n")

        # Print benchmark statistics
        stats = benchmark.get_statistics()
        print("Benchmark statistics:")
        print(f"  Total items: {stats['total_items']}")
        print(f"  Label distribution: {stats['label_distribution']}\n")

        # Initialize knowledge base
        print("Initializing knowledge base...")
        kb = KnowledgeBase()
        try:
            kb.load()
            print(f"Loaded existing KB: {len(kb.papers)} papers, {len(kb.propositions)} propositions\n")
        except Exception:
            print("Starting with fresh knowledge base\n")

        # Initialize pipeline
        print("Initializing verification pipeline...")
        pipeline = VerificationPipeline(kb=kb)
        print("Pipeline ready\n")

        # Process the single test claim
        item = benchmark.items[0]

        print("="*70)
        print(f"Processing claim: {item.claim_id}")
        print("="*70)
        print(f"Claim: {item.claim}")
        print(f"Expected result: {item.expected_result}")
        print(f"Verification method: {item.verification_method.value}")
        print("="*70 + "\n")

        # Run verification
        print("Running verification...\n")
        try:
            if use_search:
                result = pipeline.verify_claim_with_search(item.claim, max_papers=5)
            else:
                result = pipeline.verify_claim_from_kb(item.claim)

            item.result = result

            print("\n" + "="*70)
            print("VERIFICATION COMPLETE")
            print("="*70)
            print(f"Verdict: {result.verdict}")
            print(f"Confidence: {result.confidence}/10")
            print(f"Expected: {item.expected_result}")
            print(f"Correct: {result.verdict == item.expected_result}")
            print(f"\nReasoning:\n{result.reasoning}")
            print(f"\nEvidence propositions: {len(result.evidence)}")
            print(f"Unique papers: {len(result.get_papers_used())}")

        except Exception as e:
            print(f"\nERROR during verification: {e}")
            traceback.print_exc()
            item.result = None

        # Save logs
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)

        log_file = logs_dir / f"{item.claim_id}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Claim ID: {item.claim_id}\n")
            f.write(f"Claim: {item.claim}\n")
            f.write(f"Expected Result: {item.expected_result}\n")
            f.write(f"{'='*70}\n\n")

            if item.result:
                f.write("VERIFICATION RESULT:\n")
                f.write(f"{'='*70}\n")
                f.write(f"Verdict: {item.result.verdict}\n")
                f.write(f"Confidence: {item.result.confidence}\n")
                f.write(f"Reasoning: {item.result.reasoning}\n")
                f.write(f"\nEvidence Propositions ({len(item.result.evidence)}):\n")
                for i, prop in enumerate(item.result.evidence, 1):
                    f.write(f"\n{i}. {prop.text}\n")
                    f.write(f"   Paper: {prop.paper_id}\n")
                    quality_score = prop.evaluation.average_score() if prop.evaluation else None
                    f.write(f"   Quality Score: {quality_score}\n")
            else:
                f.write("ERROR: No result available\n")

        print(f"Log saved to: {log_file}")

        # Save JSON result
        result_file = logs_dir / f"{item.claim_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(item.to_dict(), f, indent=2)
        print(f"JSON result saved to: {result_file}")

        # Save knowledge base (to default location)
        print("\nSaving knowledge base to default location...")
        kb.save()
        stats = kb.get_statistics()
        print(f"Knowledge base saved: {stats['papers']} papers, "
              f"{stats['chunks']} chunks, {stats['propositions_total']} propositions")

        # Save summary
        summary = {
            "test_type": "benchmark_system_test",
            "timestamp": datetime.now().isoformat(),
            "benchmark": benchmark.name,
            "total_items": 1,
            "use_search": use_search,
            "claim_id": item.claim_id,
            "claim": item.claim,
            "expected": item.expected_result,
            "predicted": item.result.verdict if item.result else "ERROR",
            "confidence": item.result.confidence if item.result else 0.0,
            "reasoning": item.result.reasoning if item.result else "No result available",
            "correct": item.result.verdict == item.expected_result if item.result else False,
        }

        summary_file = test_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")

        # Print final summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        if item.result:
            print(f"Expected: {item.expected_result}")
            print(f"Predicted: {item.result.verdict}")
            print(f"Status: {'PASS' if item.result.verdict == item.expected_result else 'FAIL'}")
            print(f"Confidence: {item.result.confidence}/10")
        else:
            print("Status: ERROR - No result available")
        print("="*70 + "\n")

        # Cleanup
        Path(claims_file).unlink()
        print("Cleaned up temporary claims file")

        print("\nTest complete! Check the output in: " + str(test_dir))

        return item.result.verdict == item.expected_result if item.result else False

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
        description="Test the benchmark system with a single claim"
    )
    parser.add_argument(
        "--use-search",
        action="store_true",
        help="Use online paper search (otherwise uses existing KB only)"
    )

    args = parser.parse_args()

    success = test_benchmark_system(use_search=args.use_search)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

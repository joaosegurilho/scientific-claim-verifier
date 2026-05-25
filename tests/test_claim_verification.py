"""Test script for claim verification pipeline.

This script runs a series of claim verification tests to evaluate the pipeline's accuracy.

Key features:
- Tests multiple scientific claims with expected verdicts
- Uses the VerificationPipeline with KnowledgeBase
- Displays enhanced paper metadata (study type, methodology, credibility ratings)
- Saves detailed test results to files
- Tracks success rates across test cases

The pipeline now includes LLM-based metadata extraction for papers, providing:
- Study type classification
- Methodology characteristics (sample size, blinding, randomization)
- Enhanced credibility scoring based on paper quality
"""

import os
import sys
import traceback
from datetime import datetime
from io import StringIO

from scverifier.config.settings import Config
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.pipelines.verification_pipeline import VerificationPipeline

# Test cases: (claim, expected_verdict, max_papers)
TEST_CASES = [
    ("Vitamin D prevents COVID-19", "INSUFFICIENT_EVIDENCE", 12),
    ("Regular physical exercise lowers blood pressure", "SUPPORTS", 12),
    ("Antibiotics are effective against viral infections", "REFUTES", 12),
    ("Drinking coffee causes dehydration", "REFUTES", 12),
    ("Omega-3 fatty acids reduce the risk of cardiovascular disease", "INSUFFICIENT_EVIDENCE", 12),
    ("Artificial sweeteners cause cancer", "REFUTES", 12),
    ("mRNA vaccines alter human DNA", "REFUTES", 12),
    ("Telomerase activity is higher in cancer cells than in normal cells", "SUPPORTS", 12),
    ("Exercise improves cardiovascular health", "SUPPORTS", 12),
    ("5G radiation causes cancer", "REFUTES", 12),
]


class OutputCapture:
    """Captures stdout to both console and a string buffer."""

    def __init__(self):
        self.buffer = StringIO()
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def get_output(self):
        return self.buffer.getvalue()


def save_test_output(test_result: dict, output_dir: str):
    """Save test output to a text file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_{test_result['test_num']}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"TEST {test_result['test_num']}: CLAIM VERIFICATION TEST\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Claim: {test_result['claim']}\n")
        f.write(f"Expected Verdict: {test_result['expected_verdict']}\n")
        f.write(f"Actual Verdict: {test_result['actual_verdict']}\n")
        f.write(f"Confidence: {test_result['confidence']}/10\n")
        f.write(f"Test Result: {'PASS' if test_result['passed'] else 'FAIL'}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("CAPTURED OUTPUT:\n")
        f.write("=" * 70 + "\n\n")

        f.write(test_result["captured_output"])

        if "error" in test_result:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("ERROR:\n")
            f.write("=" * 70 + "\n")
            f.write(test_result["error"])

    print(f"\nSaved test output to: {filepath}")


def run_single_test(
    pipeline: VerificationPipeline,
    kb: KnowledgeBase,
    claim: str,
    expected_verdict: str,
    max_papers: int,
    test_num: int,
    output_dir: str,
    kb_path: str,
) -> dict:
    """Run a single test case."""

    print("\n" + "=" * 70)
    print(f"TEST {test_num}: {claim}")
    print("=" * 70 + "\n")

    # Capture output
    output_capture = OutputCapture()
    sys.stdout = output_capture

    try:
        # Run the full verification with search (updated method name)
        result = pipeline.verify_claim_with_search(claim, max_papers=max_papers)

        # Save knowledge base after each test to avoid wasting LLM calls
        print(f"\nSaving knowledge base to {kb_path}...")
        kb.save(kb_path)

        # Display results - VerificationResult is now a domain object
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"\nClaim: {result.claim}")
        print(f"\nVerdict: {result.verdict}")
        print(f"Confidence: {result.confidence}/10")

        print(f"\nReasoning:\n{result.reasoning}")

        print("\nEvidence Statistics:")
        print(f"   - Sources used in verdict: {len(result.evidence)}")
        print(f"   - Unique papers: {len(result.get_papers_used())}")

        print("\nSources:")
        for i, prop in enumerate(result.evidence, 1):
            print(f"\n   {i}. {prop.source}")
            print(f"      {prop.text}")

            # Get paper metadata from KB if available
            paper = kb.get_paper(prop.paper_id)
            if paper and paper.credibility:
                cred = paper.credibility
                year = paper.year if paper.year else "Unknown"
                stars = "★" * int(cred.rating) + "☆" * (5 - int(cred.rating))
                print(
                    f"      Rating: {stars} ({cred.rating:.1f}/5) | "
                    f"Study: {cred.study_type.replace('_', ' ').title()} | "
                    f"Type: {cred.evidence_type} | "
                    f"Year: {year}"
                )

                # Show methodology details if available
                methodology = []
                if cred.sample_size:
                    methodology.append(f"n={cred.sample_size}")
                if cred.randomized:
                    methodology.append("Randomized")
                if cred.blinding:
                    methodology.append(f"{cred.blinding.title()}-blind")
                if methodology:
                    print(f"      Methodology: {', '.join(methodology)}")

        # Restore stdout
        sys.stdout = output_capture.terminal
        captured_output = output_capture.get_output()

        # Check if test passed
        actual_verdict = result.verdict
        confidence = result.confidence
        passed = actual_verdict == expected_verdict

        # Print results
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Expected: {expected_verdict}")
        print(f"Actual:   {actual_verdict}")
        print(f"Status:   {'PASS' if passed else 'FAIL'}")

        test_result = {
            "test_num": test_num,
            "claim": claim,
            "expected_verdict": expected_verdict,
            "actual_verdict": actual_verdict,
            "confidence": confidence,
            "passed": passed,
            "result": result.to_dict(),
            "captured_output": captured_output,
        }

        # Save output to file
        save_test_output(test_result, output_dir)

        return test_result

    except Exception as e:
        # Restore stdout
        sys.stdout = output_capture.terminal
        captured_output = output_capture.get_output()

        print(f"\nERROR: {e}")
        traceback.print_exc()

        test_result = {
            "test_num": test_num,
            "claim": claim,
            "expected_verdict": expected_verdict,
            "actual_verdict": "ERROR",
            "confidence": "ERROR",
            "passed": False,
            "error": str(e),
            "captured_output": captured_output,
        }

        # Save output to file
        save_test_output(test_result, output_dir)

        return test_result


def main():
    """Run all tests."""

    # Create output directories
    output_dir = "test_results"
    kb_path = Config.DB_NAME
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("CLAIM VERIFICATION PIPELINE TEST SUITE")
    print("=" * 70)
    print(f"Running {len(TEST_CASES)} test(s)...\n")

    # Initialize knowledge base
    kb = KnowledgeBase(kb_path)

    # Try to load existing KB state
    if os.path.exists(kb_path):
        try:
            print(f"Found existing knowledge base, loading from {kb_path}...")
            kb.load(kb_path)
            print(f"Loaded {len(kb.papers)} papers with {len(kb.propositions)} propositions!\n")
        except Exception as e:
            print(f"Could not load KB: {e}")
            print("Starting fresh...\n")
    else:
        print(f"No existing KB found at {kb_path}, starting fresh...\n")

    # Initialize pipeline with KB
    pipeline = VerificationPipeline(kb)

    print(f"Knowledge base will be saved to: {kb_path}")
    print(f"Test outputs will be saved to: {output_dir}\n")

    results = []

    # Run each test with the same pipeline instance
    for i, (claim, expected_verdict, max_papers) in enumerate(TEST_CASES, 1):
        test_result = run_single_test(pipeline, kb, claim, expected_verdict, max_papers, i, output_dir, kb_path)
        results.append(test_result)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\nTotal Tests:   {total}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {total - passed}")
    print(f"Success Rate:  {(passed/total)*100:.1f}%\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  Test {r['test_num']}: {status} - {r['claim']}")


if __name__ == "__main__":
    main()

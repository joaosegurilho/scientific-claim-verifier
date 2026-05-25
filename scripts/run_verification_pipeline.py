#!/usr/bin/env python3
"""
Verification Pipeline Script

Verify a scientific claim by searching for papers, extracting evidence, and evaluating.

Usage:
    python run_verification_pipeline.py "claim to verify"
    python run_verification_pipeline.py "claim to verify" --max-papers 10
    python run_verification_pipeline.py "claim to verify" --kb-only
    python run_verification_pipeline.py "claim to verify" --skip-extraction-eval
    python run_verification_pipeline.py "claim to verify" --kb-only --use-all-propositions
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scverifier.pipelines.verification_pipeline import VerificationPipeline
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.data.models import VerificationResult
from scverifier.config.settings import Config
from scverifier.core.verification.confidence_interpreter import get_confidence_interpretation


def format_verdict(verdict: str) -> str:
    """Format verdict with emoji.

    Args:
        verdict: Verdict string

    Returns:
        Formatted verdict with emoji
    """
    emoji_map = {"SUPPORTS": "", "REFUTES": "", "INSUFFICIENT_EVIDENCE": "❓"}
    return f"{emoji_map.get(verdict, '')} {verdict}"


def format_confidence(confidence: float) -> str:
    """Format confidence with visual bar.

    Args:
        confidence: Confidence value (0-10)

    Returns:
        Formatted confidence string
    """
    bars = int(confidence)
    empty = 10 - bars
    bar_str = "█" * bars + "░" * empty
    return f"{confidence:.1f}/10 [{bar_str}]"


def print_results(result: VerificationResult, kb: KnowledgeBase):
    """Print verification results in a nice format.

    Args:
        result: VerificationResult object
        kb: KnowledgeBase instance for accessing paper data
    """
    print("\n" + "=" * 70)
    print(" VERIFICATION RESULTS")
    print("=" * 70)

    print("\n Claim:")
    print(f"   {result.claim}")

    print(f"\n{format_verdict(result.verdict)}")

    print("\n Confidence:")
    print(f"   {format_confidence(result.confidence)}")
    interpretation = get_confidence_interpretation(result.verdict, int(round(result.confidence)))
    print(f"   {interpretation}")

    print("\n Reasoning:")
    # Word wrap reasoning
    reasoning = result.reasoning
    words = reasoning.split()
    lines = []
    current_line = "   "

    for word in words:
        if len(current_line) + len(word) + 1 <= 70:
            current_line += word + " "
        else:
            lines.append(current_line.rstrip())
            current_line = "   " + word + " "

    if current_line.strip():
        lines.append(current_line.rstrip())

    print("\n".join(lines))

    print("\n Evidence Summary:")
    print(f"   Evidence used: {len(result.evidence)} propositions")
    print(f"   Sources: {len(result.get_papers_used())} papers")

    # Show ALL evidence sources with condensed format
    if result.evidence:
        print("\n All Evidence Sources:")
        print("=" * 70)

        for i, prop in enumerate(result.evidence, 1):
            # Get paper and credibility info
            paper = kb.get_paper(prop.paper_id)
            if paper and paper.credibility:
                evidence_emoji = "" if paper.credibility.evidence_type == "full_text" else ""
                paper_info = (
                    f"{prop.source} "
                    f"({paper.credibility.study_type}, {evidence_emoji} {paper.credibility.evidence_type}, "
                    f"{paper.credibility.rating}/5★)"
                )
            else:
                paper_info = f"{prop.source} (Not scored)"

            # Proposition quality
            if prop.evaluation:
                avg_score = prop.evaluation.average_score()
                prop_quality = f"{prop.text.strip()}  (avg: {avg_score:.1f}/10)"
            else:
                prop_quality = f"{prop.text.strip()}  (not evaluated)"

            # Get URL (prefer PDF URL if available)
            url = ""
            if paper:
                url = paper.pdf_url if paper.pdf_url else paper.url

            # Print condensed 4-line summary
            print(f"\n   {i}. {paper_info}")
            if url and paper:
                print(f"      {paper.id} |  {url}")
            print(f"      {prop_quality}")
            print()  # blank line for spacing


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Verify a scientific claim using the verification pipeline")
    parser.add_argument("claim", help="Scientific claim to verify")
    parser.add_argument(
        "--max-papers", type=int, default=30, help="Maximum number of papers to search for (default: 30)"
    )
    parser.add_argument(
        "--kb-only",
        action="store_true",
        help="Use only existing knowledge base data (don't search for new papers)",
    )
    parser.add_argument(
        "--skip-extraction-eval",
        action="store_true",
        help="Skip quality evaluation during proposition extraction (faster, accepts all propositions). Only applies when searching for new papers.",
    )
    parser.add_argument(
        "--use-all-propositions",
        action="store_true",
        help="Use all propositions instead of only quality ones during claim verification. Useful with --kb-only when quality evaluation was skipped during extraction.",
    )
    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print(" CLAIM VERIFICATION PIPELINE")
    print("=" * 70)
    print("\n Configuration:")
    print(f"   Claim: {args.claim}")
    if not args.kb_only:
        print(f"   Max papers: {args.max_papers}")
        print("   Batch size: 5 papers (with incremental saving)")
        print(f"   Skip extraction evaluation: {args.skip_extraction_eval}")
    print(f"   Mode: {'KB-only' if args.kb_only else 'Search + KB'}")
    print(f"   Use all propositions: {args.use_all_propositions}")

    # Initialize knowledge base
    print("\n Initializing knowledge base...")
    kb = KnowledgeBase()

    # Load existing KB
    print("\n Loading existing knowledge base...")
    try:
        kb.load()
        print(f"   Knowledge base loaded from {Config.DB_NAME}")
    except FileNotFoundError:
        if args.kb_only:
            print(f"    Error: No knowledge base found at {Config.DB_NAME}")
            print("   Cannot use --kb-only mode without an existing knowledge base.")
            print("   Run without --kb-only to search for papers first.")
            sys.exit(1)
        else:
            print(f"     No existing knowledge base found at {Config.DB_NAME}")
            print("   Starting with fresh knowledge base.")
    except Exception as e:
        print(f"     Error loading knowledge base: {e}")
        if args.kb_only:
            print("   Cannot proceed in --kb-only mode.")
            sys.exit(1)
        print("   Starting with fresh knowledge base.")

    # Initialize pipeline
    print("\n Initializing pipeline...")
    pipeline = VerificationPipeline(kb)

    # Configure extraction evaluation (only applies to new paper extraction during search)
    if args.skip_extraction_eval:
        pipeline.extractor.skip_evaluation = True
        print("   Extraction evaluation: DISABLED (faster, all propositions accepted)")
    else:
        pipeline.extractor.skip_evaluation = False
        print("   Extraction evaluation: ENABLED (quality filtering)")

    # Run verification
    print("\n Starting verification...\n")

    try:
        # quality_claims controls whether to filter to quality propositions during verification
        quality_claims = not args.use_all_propositions

        if args.kb_only:
            # Use only KB data
            result = pipeline.verify_claim_from_kb(args.claim, quality_claims=quality_claims)
        else:
            # Search for papers and verify
            # Note: The extraction evaluation is already configured via pipeline.extractor.skip_evaluation
            result = pipeline.verify_claim_with_search(args.claim, max_papers=args.max_papers, quality_claims=quality_claims)

        # Print results
        print_results(result, kb)

        # Note: KB already saved incrementally during search
        # This final save is just a safety check
        if not args.kb_only:
            print(f"\n{'='*70}")
            print("  Note: Knowledge base was saved incrementally during processing")

        print(f"\n{'='*70}")
        print(" Verification complete!")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n  Verification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

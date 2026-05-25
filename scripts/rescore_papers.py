"""Re-score all papers in the database with updated credibility scoring."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.verification.paper_scorer import PaperScorer


def rescore_all_papers():
    """Re-calculate credibility scores for all papers in the database."""

    print("=" * 60)
    print("PAPER RE-SCORING SCRIPT")
    print("=" * 60)

    # Initialize
    print("\n1. Loading knowledge base...")
    kb = KnowledgeBase()
    kb.load()

    print("\n2. Initializing scorer...")
    # Use shorter timeout for batch processing (60 seconds per paper instead of 280)
    timeout = 10
    scorer = PaperScorer(llm_timeout=timeout)
    print(f"   Using {timeout}s timeout per paper (to prevent hanging)")

    # Get all papers
    all_papers = kb.list_papers()
    print(f"\n3. Found {len(all_papers)} papers in database")

    if len(all_papers) == 0:
        print("   No papers to score. Exiting.")
        return

    # Ask for confirmation
    response = input(f"\n   Re-score all {len(all_papers)} papers? (y/n): ")
    if response.lower() != "y":
        print("   Cancelled.")
        return

    # Re-score all papers
    print("\n4. Re-scoring papers...")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for i, paper in enumerate(all_papers, 1):
        import time

        paper_start_time = time.time()

        try:
            # Show which paper we're starting
            print(f"\n   [{i}/{len(all_papers)}] Processing: {paper.title[:60]}...")
            print(f"      ID: {paper.id}")

            # Score the paper (modifies in place)
            credibility = scorer.score_paper(paper)

            # Calculate elapsed time
            paper_elapsed = time.time() - paper_start_time

            # Show the result
            print(
                f"      ✓ Rating: {credibility.rating}/5 (Type: {credibility.study_type}, Evidence: {credibility.evidence_type})"
            )
            print(
                f"        Bonuses: Citation={credibility.citation_bonus}, Open Access={credibility.open_access_bonus}, Full Text={credibility.fulltext_content_bonus}"
            )
            print(f"        Time: {paper_elapsed:.1f}s")

            success_count += 1

        except Exception as e:
            print(f"      ✗ ERROR: {str(e)}")
            print("         Paper details:")
            print(f"         - Title: {paper.title[:100] if paper.title else 'None'}")
            print(f"         - Abstract length: {len(paper.abstract) if paper.abstract else 0}")
            print(f"         - Has full text: {bool(paper.full_text)}")
            error_count += 1

            # Ask if user wants to continue
            response = input("\n   Continue with remaining papers? (y/n): ")
            if response.lower() != "y":
                print("   Stopping re-scoring...")
                break

    # Save updated papers
    print("\n5. Saving updated papers to database...")
    kb.save()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Total papers: {len(all_papers)}")
    print(f"   Successfully scored: {success_count}")
    print(f"   Errors: {error_count}")
    print("\nDone!")


if __name__ == "__main__":
    rescore_all_papers()

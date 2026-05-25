#!/usr/bin/env python
"""Test script to verify deletion performance improvement."""

import sys
import time
import os

# Add the package to path
sys.path.insert(0, '/home/pc-hp-dev/Desktop/Masters_Thesis/scientific-claim-verifier')

from scverifier.core.knowledge.knowledge_base import KnowledgeBase

def test_deletion_performance():
    """Test the deletion performance of the knowledge base."""

    print("=" * 80)
    print("DELETION PERFORMANCE TEST")
    print("=" * 80)

    # Load existing knowledge base
    kb = KnowledgeBase()

    db_path = '/home/pc-hp-dev/Desktop/Masters_Thesis/scientific-claim-verifier/data/output'

    if not os.path.exists(db_path):
        print(f"Error: Knowledge base not found at {db_path}")
        print("Please ensure you have papers in your knowledge base first.")
        return

    print("\nLoading knowledge base...")
    t0 = time.time()
    kb.load(db_path)
    t1 = time.time()
    print(f"Knowledge base loaded in {t1-t0:.2f}s")

    # Get list of papers
    papers = kb.list_papers()

    if len(papers) == 0:
        print("\nNo papers found in knowledge base. Cannot test deletion.")
        return

    print(f"\nFound {len(papers)} papers in knowledge base")

    # Select a paper to delete (use the first one)
    paper_to_delete = papers[0]
    paper_id = paper_to_delete.id
    paper_title = paper_to_delete.title[:50]

    print(f"\nTesting deletion of paper:")
    print(f"  ID: {paper_id}")
    print(f"  Title: {paper_title}...")
    print(f"  Chunks: {len(paper_to_delete.chunks)}")
    print(f"  Propositions: {len(paper_to_delete.propositions)}")

    # Time the deletion
    print(f"\nDeleting paper (using efficient FAISS delete)...")
    t_delete_start = time.time()
    success = kb.delete_paper(paper_id, verbose=True)
    t_delete_end = time.time()

    deletion_time = t_delete_end - t_delete_start

    if success:
        print(f"\n{'='*80}")
        print(f"SUCCESS! Paper deleted in {deletion_time:.4f} seconds")
        print(f"{'='*80}")
        print("\nThis should be much faster than the old method which rebuilt")
        print("the entire vectorstore (re-embedding all documents).")
        print("\nFor large knowledge bases, this could be 100-1000x faster!")
    else:
        print(f"\nERROR: Failed to delete paper {paper_id}")

    # Don't save - this is just a test
    print("\n(Not saving changes - this was just a performance test)")

if __name__ == "__main__":
    test_deletion_performance()

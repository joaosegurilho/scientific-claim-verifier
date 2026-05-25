#!/usr/bin/env python3
"""
Enhanced Knowledge Base Query Script

Query the knowledge base interactively or from command line with comprehensive query options.

Usage:
    python query_knowledge_base.py                          # Interactive mode
    python query_knowledge_base.py "search query"           # Single query
    python query_knowledge_base.py --stats                  # Show statistics only
    python query_knowledge_base.py --list-papers            # List all papers
    python query_knowledge_base.py --paper PAPER_ID         # Show paper details
    python query_knowledge_base.py --all-propositions       # Show all propositions
    python query_knowledge_base.py --quality-propositions   # Show only quality propositions
    python query_knowledge_base.py --chunks PAPER_ID        # Show chunks for paper
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.data.models import Paper, Proposition, Chunk
from scverifier.config.settings import Config


def print_search_results(results, query: str, result_type: str = "propositions"):
    """Print search results in a nice format.

    Args:
        results: List of Proposition or Chunk objects
        query: The search query
        result_type: Type of results ("propositions" or "chunks")
    """
    print(f"\n{'='*70}")
    if result_type == "propositions":
        print(f" Search Results for: '{query}'")
    else:
        print(f" Chunk Search Results for: '{query}'")
    print("=" * 70)

    if not results:
        print("\n   No results found.")
        return

    print(f"\n   Found {len(results)} relevant {result_type}:\n")

    for i, item in enumerate(results, 1):
        if result_type == "propositions":
            _print_proposition(i, item)
        else:
            _print_chunk(i, item)


def _print_proposition(index: int, prop: Proposition):
    """Print a single proposition with details."""
    # Updated field access for Proposition
    quality_score = prop.evaluation.average_score() if hasattr(prop, 'evaluation') and prop.evaluation else 0
    quality_bar = "★" * int(quality_score / 2)
    quality_status = "QUALITY" if hasattr(prop, 'is_high_quality') and prop.is_high_quality() else "✗ LOW QUALITY"

    print(f"   {index}. {getattr(prop, 'text', '[no text]')}")
    print(f"      Proposition ID: {getattr(prop, 'prop_id', '[no prop_id]')}")
    print(f"      Chunk ID: {getattr(prop, 'chunk_id', '[no chunk_id]')}")
    print(f"      Paper ID: {getattr(prop, 'paper_id', '[no paper_id]')}")
    print(f"      Source: {getattr(prop, 'source', '[no source]')}")
    if hasattr(prop, 'page') and prop.page:
        print(f"      Page: {prop.page}")
    if hasattr(prop, 'evaluation') and prop.evaluation:
        print(f"      Quality: {quality_score:.1f}/10 {quality_bar} ({quality_status})")
        print(
            f"      Scores: A{prop.evaluation.accuracy} C{prop.evaluation.clarity} "
            f"Co{prop.evaluation.completeness} Cn{prop.evaluation.conciseness}"
        )
    print()


def _print_chunk(index: int, chunk: Chunk):
    """Print a single chunk with details."""
    # Updated field access for Chunk
    print(f"   {index}. {getattr(chunk, 'text', '[no text]')[:200]}{'...' if len(getattr(chunk, 'text', '')) > 200 else ''}")
    print(f"      Chunk ID: {getattr(chunk, 'chunk_id', '[no chunk_id]')}")
    print(f"      Paper ID: {getattr(chunk, 'paper_id', '[no paper_id]')}")
    print(f"      Source: {getattr(chunk, 'source', '[no source]')}")
    if hasattr(chunk, 'page') and chunk.page:
        print(f"      Page: {chunk.page}")
    print(f"      Length: {len(getattr(chunk, 'text', ''))} characters")
    print()


def print_paper_details(paper: Paper, kb: KnowledgeBase):
    """Print detailed information about a paper.

    Args:
        paper: Paper object
        kb: KnowledgeBase instance for additional queries
    """
    stats = paper.get_statistics()

    print(f"\n{'='*70}")
    print(f" PAPER DETAILS: {paper.title}")
    print("=" * 70)

    print("\n Metadata:")
    print(f"   ID: {paper.id}")
    print(f"   Year: {paper.year or 'Unknown'}")
    print(f"   Authors: {', '.join(paper.authors[:5]) if paper.authors else 'Unknown'}")
    if len(paper.authors) > 5:
        print(f"           (and {len(paper.authors) - 5} more)")
    print(f"   Citations: {paper.citations}")
    print(f"   Source: {paper.source}")
    if paper.doi:
        print(f"   DOI: {paper.doi}")
    if paper.pmc_id:
        print(f"   PMC ID: {paper.pmc_id}")
    if paper.url:
        print(f"   URL: {paper.url}")

    print("\n Statistics:")
    print(f"   Chunks: {stats['chunks']}")
    print(f"   Propositions: {stats['propositions_quality']} quality / {stats['propositions_total']} total")
    print(f"   Success rate: {stats['success_rate']*100:.1f}%")
    if hasattr(paper, 'credibility') and paper.credibility:
        # Extract credibility rating
        cred_value = paper.credibility.rating
        stars = "★" * int(round(cred_value))
        print(f"   Credibility: {cred_value:.1f}/5 {stars}")

    # Show abstract preview
    if paper.abstract:
        abstract_preview = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
        print(f"\n Abstract Preview:\n   {abstract_preview}")


def list_papers(kb: KnowledgeBase, show_details: bool = False):
    """List all papers in knowledge base.

    Args:
        kb: KnowledgeBase instance
        show_details: Whether to show detailed information
    """
    papers = kb.list_papers()

    print(f"\n{'='*70}")
    print(f" Papers in Knowledge Base ({len(papers)} total)")
    print("=" * 70 + "\n")

    if not papers:
        print("   No papers in knowledge base.")
        return

    for i, paper in enumerate(papers, 1):
        stats = paper.get_statistics()

        if show_details:
            print(f"{i}. {paper.title}")
            print(f"   ID: {paper.id}")
            print(f"   Year: {paper.year or 'Unknown'}")
            print(f"   Authors: {', '.join(paper.authors[:3]) if paper.authors else 'Unknown'}")
            if len(paper.authors) > 3:
                print(f"           (and {len(paper.authors) - 3} more)")
            print(f"   Propositions: {stats['propositions_quality']} quality / {stats['propositions_total']} total")
            print(f"   Success rate: {stats['success_rate']*100:.1f}%")
            if hasattr(paper, 'credibility') and paper.credibility:
                cred_value = paper.credibility.rating
                stars = "★" * int(round(cred_value))
                print(f"   Credibility: {cred_value:.1f}/5 {stars}")
            print()
        else:
            print(
                f"{i}. {paper.title} ({paper.year or 'Unknown'}) - "
                f"{stats['propositions_quality']} quality propositions"
            )


def show_all_propositions(kb: KnowledgeBase, quality_only: bool = False):
    """Show all propositions in the knowledge base.

    Args:
        kb: KnowledgeBase instance
        quality_only: If True, only show quality propositions
    """
    all_propositions = []

    for paper in kb.list_papers():
        if quality_only:
            propositions = paper.get_quality_propositions()
        else:
            propositions = paper.propositions
        all_propositions.extend(propositions)

    title = "All Quality Propositions" if quality_only else "All Propositions"
    print(f"\n{'='*70}")
    print(f" {title} ({len(all_propositions)} total)")
    print("=" * 70 + "\n")

    if not all_propositions:
        print("   No propositions found.")
        return

    # Group by paper
    papers_dict: dict[str, list] = {}
    for prop in all_propositions:
        if prop.paper_id not in papers_dict:
            papers_dict[prop.paper_id] = []
        papers_dict[prop.paper_id].append(prop)

    for paper_id, propositions in papers_dict.items():
        paper = kb.get_paper(paper_id)  # type: ignore[assignment]
        title = paper.title if paper is not None and hasattr(paper, 'title') else paper_id
        print(f"\n {title}:")
        print(f"   {len(propositions)} propositions\n")

        for i, prop in enumerate(propositions, 1):
            quality_score = prop.evaluation.average_score() if hasattr(prop, 'evaluation') and prop.evaluation else 0
            quality_status = "✓" if hasattr(prop, 'is_high_quality') and prop.is_high_quality() else "✗"

            print(f"   {i}. {quality_status} {getattr(prop, 'text', '[no text]')}")
            print(f"      Proposition ID: {getattr(prop, 'prop_id', '[no prop_id]')}")
            print(f"      Chunk ID: {getattr(prop, 'chunk_id', '[no chunk_id]')}")
            print(f"      Paper ID: {getattr(prop, 'paper_id', '[no paper_id]')}")
            if hasattr(prop, 'evaluation') and prop.evaluation:
                print(
                    f"      Scores: A{prop.evaluation.accuracy} C{prop.evaluation.clarity} "
                    f"Co{prop.evaluation.completeness} Cn{prop.evaluation.conciseness} "
                    f"({quality_score:.1f}/10)"
                )
            print()


def show_chunks_for_paper(kb: KnowledgeBase, paper_id: str):
    """Show all chunks for a specific paper.

    Args:
        kb: KnowledgeBase instance
        paper_id: Paper ID to show chunks for
    """
    paper = kb.get_paper(paper_id)
    if not paper:
        print(f"\n Paper with ID '{paper_id}' not found.")
        return

    chunks = paper.chunks

    print(f"\n{'='*70}")
    print(f" Chunks for: {paper.title}")
    print(f"   Paper ID: {paper_id}")
    print("=" * 70 + "\n")

    if not chunks:
        print("   No chunks found for this paper.")
        return

    print(f"   Found {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. Chunk ID: {getattr(chunk, 'chunk_id', '[no chunk_id]')}")
        if hasattr(chunk, 'page') and chunk.page:
            print(f"      Page: {chunk.page}")
        print(f"      Text: {getattr(chunk, 'text', '[no text]')[:200]}{'...' if len(getattr(chunk, 'text', '')) > 200 else ''}")
        print(f"      Length: {len(getattr(chunk, 'text', ''))} characters")
        print(f"      Paper ID: {getattr(chunk, 'paper_id', '[no paper_id]')}")
        print(f"      Source: {getattr(chunk, 'source', '[no source]')}")
        print()

        # Show propositions from this chunk
        chunk_propositions = [p for p in paper.propositions if getattr(p, 'chunk_id', None) == getattr(chunk, 'chunk_id', None)]
        if chunk_propositions:
            print(f"       Propositions from this chunk ({len(chunk_propositions)}):")
            for j, prop in enumerate(chunk_propositions, 1):
                quality_status = "✓" if hasattr(prop, 'is_high_quality') and prop.is_high_quality() else "✗"
                print(f"         {j}. {quality_status} {getattr(prop, 'text', '[no text]')}")
                print(f"            Proposition ID: {getattr(prop, 'prop_id', '[no prop_id]')}")
            print()


def show_proposition_context(kb: KnowledgeBase, chunk_id: str):
    """Show the chunk context for a specific proposition.

    Args:
        kb: KnowledgeBase instance
        chunk_id: Chunk ID to show context for
    """
    # Find the paper and chunk
    for paper in kb.list_papers():
        for chunk in paper.chunks:
            if getattr(chunk, 'chunk_id', None) == chunk_id:
                print(f"\n{'='*70}")
                print(f" Context for Chunk: {chunk_id}")
                print(f"   Paper: {getattr(paper, 'title', '[no title]')}")
                print("=" * 70 + "\n")

                print(" Full Chunk Text:")
                print(f"   {getattr(chunk, 'text', '[no text]')}")
                print(f"\n   Length: {len(getattr(chunk, 'text', ''))} characters")
                if hasattr(chunk, 'page') and chunk.page:
                    print(f"   Page: {chunk.page}")

                # Show propositions from this chunk
                chunk_propositions = [p for p in paper.propositions if getattr(p, 'chunk_id', None) == chunk_id]
                if chunk_propositions:
                    print(f"\n Propositions extracted from this chunk ({len(chunk_propositions)}):")
                    for i, prop in enumerate(chunk_propositions, 1):
                        quality_score = prop.evaluation.average_score() if hasattr(prop, 'evaluation') and prop.evaluation else 0
                        quality_status = "QUALITY" if hasattr(prop, 'is_high_quality') and prop.is_high_quality() else "✗ LOW QUALITY"

                        print(f"\n   {i}. {getattr(prop, 'text', '[no text]')}")
                        print(f"      Proposition ID: {getattr(prop, 'prop_id', '[no prop_id]')}")
                        print(f"      {quality_status} (Score: {quality_score:.1f}/10)")
                        if hasattr(prop, 'evaluation') and prop.evaluation:
                            print(
                                f"      Accuracy: {prop.evaluation.accuracy}/10, "
                                f"Clarity: {prop.evaluation.clarity}/10, "
                                f"Completeness: {prop.evaluation.completeness}/10, "
                                f"Conciseness: {prop.evaluation.conciseness}/10"
                            )
                else:
                    print("\n No propositions extracted from this chunk.")

                return

    print(f"\n Chunk with ID '{chunk_id}' not found.")


def interactive_mode(kb: KnowledgeBase):
    """Run interactive query mode.

    Args:
        kb: KnowledgeBase instance
    """
    print(f"\n{'='*70}")
    print(" ENHANCED INTERACTIVE QUERY MODE")
    print("=" * 70)
    print("\nCommands:")
    print("   • Type your search query and press Enter")
    print("   'stats'       - Show knowledge base statistics")
    print("   'papers'      - List all papers")
    print("   'paper ID'    - Show details for specific paper")
    print("   'all'         - Show all propositions")
    print("   'quality'     - Show only quality propositions")
    print("   'chunks ID'   - Show chunks for paper")
    print("   'context ID'  - Show chunk context for proposition")
    print("   'help'        - Show this help message")
    print("   'quit'        - Exit interactive mode")
    print()

    while True:
        try:
            # Get query
            query = input("\n Query> ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ["quit", "exit", "q"]:
                print("\n Goodbye!")
                break

            elif query.lower() == "help":
                print("\nCommands:")
                print("   • Type your search query and press Enter")
                print("   'stats'       - Show knowledge base statistics")
                print("   'papers'      - List all papers")
                print("   'paper ID'    - Show details for specific paper")
                print("   'all'         - Show all propositions")
                print("   'quality'     - Show only quality propositions")
                print("   'chunks ID'   - Show chunks for paper")
                print("   'context ID'  - Show chunk context for proposition")
                print("   'help'        - Show this help message")
                print("   'quit'        - Exit interactive mode")
                continue

            elif query.lower() == "stats":
                kb.print_statistics()
                continue

            elif query.lower() == "papers":
                list_papers(kb, show_details=True)
                continue

            elif query.lower() == "all":
                show_all_propositions(kb, quality_only=False)
                continue

            elif query.lower() == "quality":
                show_all_propositions(kb, quality_only=True)
                continue

            elif query.lower().startswith("paper "):
                paper_id = query[6:].strip()
                paper = kb.get_paper(paper_id)
                if paper:
                    print_paper_details(paper, kb)
                else:
                    print(f"\n Paper with ID '{paper_id}' not found.")
                continue

            elif query.lower().startswith("chunks "):
                paper_id = query[7:].strip()
                show_chunks_for_paper(kb, paper_id)
                continue

            elif query.lower().startswith("context "):
                chunk_id = query[8:].strip()
                show_proposition_context(kb, chunk_id)
                continue

            # Regular search
            print("\n Searching propositions...")
            results = kb.search_propositions(query)
            print_search_results(results, query, "propositions")

            # Also show chunk search
            print("\n Searching chunks...")
            chunk_results = kb.search_chunks(query)
            print_search_results(chunk_results, query, "chunks")

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Query the knowledge base with enhanced capabilities")
    parser.add_argument("query", nargs="?", help="Search query (optional - leave empty for interactive mode)")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--list-papers", action="store_true", help="List all papers")
    parser.add_argument("--paper", type=str, help="Show details for specific paper ID")
    parser.add_argument("--all-propositions", action="store_true", help="Show all propositions")
    parser.add_argument("--quality-propositions", action="store_true", help="Show only quality propositions")
    parser.add_argument("--chunks", type=str, help="Show chunks for specific paper ID")
    parser.add_argument("--context", type=str, help="Show chunk context for specific chunk ID")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum number of results to show (default: 10)")

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print(" ENHANCED KNOWLEDGE BASE QUERY")
    print("=" * 70)

    # Load knowledge base
    print(f"\n Loading knowledge base from {Config.DB_NAME}...")

    try:
        kb = KnowledgeBase()
        kb.load()
        print("   Knowledge base loaded successfully")
    except FileNotFoundError:
        print(f"\n Error: No knowledge base found at {Config.DB_NAME}")
        print("   Run the extraction pipeline first to create a knowledge base.")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error loading knowledge base: {e}")
        sys.exit(1)

    # Handle different modes
    if args.stats:
        kb.print_statistics()

    elif args.list_papers:
        list_papers(kb, show_details=True)

    elif args.paper:
        paper = kb.get_paper(args.paper)
        if paper:
            print_paper_details(paper, kb)
        else:
            print(f"\n Paper with ID '{args.paper}' not found.")

    elif args.all_propositions:
        show_all_propositions(kb, quality_only=False)

    elif args.quality_propositions:
        show_all_propositions(kb, quality_only=True)

    elif args.chunks:
        show_chunks_for_paper(kb, args.chunks)

    elif args.context:
        show_proposition_context(kb, args.context)

    elif args.query:
        # Single query mode
        print(f"\n Searching for: '{args.query}'")

        print("\n Searching propositions...")
        results = kb.search_propositions(args.query)
        print_search_results(results[: args.max_results], args.query, "propositions")

        print("\n Searching chunks...")
        chunk_results = kb.search_chunks(args.query)
        print_search_results(chunk_results[: args.max_results], args.query, "chunks")

        print(f"\n{'='*70}")
        print(" Query complete!")
        print("=" * 70 + "\n")

    else:
        # Interactive mode
        interactive_mode(kb)


if __name__ == "__main__":
    main()

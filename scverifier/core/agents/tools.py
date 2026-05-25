"""Tools for the claim verification agent system.

These tools wrap existing functionality to make it accessible to LangGraph agents.
Each tool has a clear description to help the LLM understand when to use it.
"""

from langchain_core.tools import tool
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.knowledge.literature_search import LiteratureSearch
from scverifier.core.extraction.proposition_extractor import PropositionExtractor
from scverifier.core.verification.paper_scorer import PaperScorer

# Global knowledge base reference (set by agent initialization)
_kb: KnowledgeBase | None = None
_lit_search: LiteratureSearch | None = None
_quality_only: bool = False


def set_knowledge_base(kb: KnowledgeBase):
    """Set the global knowledge base reference for tools."""
    global _kb
    _kb = kb


def set_literature_search(lit_search: LiteratureSearch):
    """Set the global literature search reference for tools."""
    global _lit_search
    _lit_search = lit_search


def set_quality_filter(quality_only: bool):
    """Set whether to filter for quality propositions only."""
    global _quality_only
    _quality_only = quality_only


# ======================== KB SEARCH TOOLS ========================


@tool
def search_similar_propositions(query: str, top_k: int = 20) -> str:
    """Search the knowledge base for propositions similar to a query.

    Use this tool when you need to find existing scientific claims or evidence
    that are semantically similar to the claim you're verifying.

    Args:
        query: The search query (usually the claim or a related concept)
        top_k: Number of propositions to retrieve (default: 20)

    Returns:
        Formatted string with propositions and their sources
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        props = _kb.search_propositions(query, top_k=top_k * 2 if _quality_only else top_k)
        
        # Filter for quality propositions if quality_only is set
        if _quality_only:
            props = [p for p in props if p.is_high_quality()][:top_k]

        if not props:
            return f"No propositions found for query: '{query}'"

        result = f"Found {len(props)} propositions:\n\n"
        for i, prop in enumerate(props, 1):
            paper = _kb.get_paper(prop.paper_id)

            result += f"{i}. {prop.text}\n"
            result += f"   Proposition ID: {prop.prop_id}\n"

            if paper:
                result += f"   Paper ID: {prop.paper_id}\n"
                result += f"   Title: {paper.title}\n"
                result += f"   Year: {paper.year or 'Unknown'}\n"
                result += f"   Citations: {paper.citations or 0}\n"

                if paper.credibility:
                    cred = paper.credibility
                    result += f"   Credibility Rating: {cred.rating:.1f}/5\n"
                    result += f"   Study Type: {cred.study_type}\n"
                    result += f"   Evidence Type: {cred.evidence_type}\n"
                    if cred.sample_size:
                        result += f"   Sample Size: {cred.sample_size}\n"
                    if cred.study_duration:
                        result += f"   Study Duration: {cred.study_duration}\n"
                    if cred.randomized is not None:
                        result += f"   Randomized: {'Yes' if cred.randomized else 'No'}\n"
                    if cred.blinding:
                        result += f"   Blinding: {cred.blinding}\n"
                    if cred.population_type:
                        result += f"   Population Type: {cred.population_type}\n"
            else:
                result += f"   Paper ID: {prop.paper_id} (details not available)\n"

            result += "\n"

        return result
    except Exception as e:
        return f"Error searching propositions: {str(e)}"


@tool
def search_similar_chunks(query: str, top_k: int = 10) -> str:
    """Search the knowledge base for document chunks similar to a query.

    Use this when you need broader context than individual propositions,
    or when looking for more detailed explanations from the literature.

    Args:
        query: The search query
        top_k: Number of chunks to retrieve (default: 10)

    Returns:
        Formatted string with chunks and their sources
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        chunks = _kb.search_chunks(query, top_k=top_k)

        if not chunks:
            return f"No chunks found for query: '{query}'"

        result = f"Found {len(chunks)} relevant chunks:\n\n"
        for i, chunk in enumerate(chunks, 1):
            paper = _kb.get_paper(chunk.paper_id)
            paper_info = "Unknown Paper"
            if paper:
                paper_info = f"{paper.title} ({paper.year or 'N/A'})"

            # Truncate long chunks
            text = chunk.text if len(chunk.text) <= 500 else chunk.text[:497] + "..."

            result += f"{i}. {text}\n"
            result += f"   Chunk ID: {chunk.chunk_id}\n"
            result += f"   Source: {paper_info}\n"
            result += f"   Section: {chunk.section or 'Unknown'}\n"
            result += f"   Paper ID: {chunk.paper_id}\n\n"

        return result
    except Exception as e:
        return f"Error searching chunks: {str(e)}"


@tool
def search_propositions_in_paper(query: str, paper_id: str, top_k: int = 10) -> str:
    """Search for propositions within a specific paper.

    Use this when you've found a relevant paper and want to search for
    specific aspects or topics within just that paper's content.

    Args:
        query: What to search for within the paper
        paper_id: ID of the paper to search within
        top_k: Number of propositions to retrieve (default: 10)

    Returns:
        Formatted string with matching propositions from the paper
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        props = _kb.search_propositions_by_paper(query, paper_id)
        
        # Filter for quality propositions if quality_only is set
        if _quality_only:
            props = [p for p in props if p.is_high_quality()]

        if not props:
            return f"No propositions found for query '{query}' in paper '{paper_id}'"

        # Limit to top_k results
        props = props[:top_k]

        paper = _kb.get_paper(paper_id)
        paper_title = paper.title if paper else "Unknown Paper"

        result = f"Found {len(props)} propositions in '{paper_title}':\n\n"
        for i, prop in enumerate(props, 1):
            result += f"{i}. {prop.text}\n"
            result += f"   Proposition ID: {prop.prop_id}\n"
            if prop.evaluation:
                result += f"   Quality Score: {prop.evaluation.average_score():.1f}/10\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error searching propositions in paper: {str(e)}"

@tool
def find_similar_papers(paper_id: str, top_k: int = 5) -> str:
    """Find papers in the knowledge base that are similar to a given paper.

    Use this to find related research after identifying a relevant paper.

    Args:
        paper_id: ID of the paper to find similar papers for
        top_k: Number of similar papers to return (default: 5)

    Returns:
        Formatted string with similar papers and their similarity scores
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        similar = _kb.find_similar_papers(paper_id, top_k=top_k)

        if not similar:
            return f"No similar papers found for paper ID: '{paper_id}'"

        result = f"Found {len(similar)} similar papers:\n\n"
        for i, (paper, score) in enumerate(similar, 1):
            result += f"{i}. {paper.title} (Similarity: {score:.2f})\n"
            result += f"   Paper ID: {paper.id}\n"
            result += f"   Authors: {', '.join(paper.authors[:3])}"
            if len(paper.authors) > 3:
                result += " et al."
            result += f"\n   Year: {paper.year or 'Unknown'}\n"
            result += f"   Citations: {paper.citations or 0}\n"

            if paper.credibility:
                cred = paper.credibility
                result += f"   Credibility Rating: {cred.rating:.1f}/5\n"
                result += f"   Study Type: {cred.study_type}\n"
                result += f"   Evidence Type: {cred.evidence_type}\n"
                if cred.sample_size:
                    result += f"   Sample Size: {cred.sample_size}\n"
                if cred.study_duration:
                    result += f"   Duration: {cred.study_duration}\n"
                if cred.randomized is not None:
                    result += f"   Randomized: {'Yes' if cred.randomized else 'No'}\n"

            result += "\n"

        return result
    except Exception as e:
        return f"Error finding similar papers: {str(e)}"


# ======================== PAPER INFO TOOLS ========================


@tool
def get_paper_details(paper_id: str) -> str:
    """Get detailed information about a specific paper.

    Use this after finding a paper ID from search results to get full details
    including abstract, metadata, and credibility information.

    Args:
        paper_id: The paper ID to look up

    Returns:
        Formatted string with paper details
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        paper = _kb.get_paper(paper_id)

        if not paper:
            return f"Paper not found: '{paper_id}'"

        result = f"Paper: {paper.title}\n\n"
        result += f"Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}\n"
        result += f"Year: {paper.year or 'Unknown'}\n"
        result += f"Citations: {paper.citations or 0}\n"
        result += f"DOI: {paper.doi or 'Not available'}\n"
        result += f"Source: {paper.source}\n\n"

        if paper.credibility:
            result += f"Credibility Rating: {paper.credibility.rating:.1f}/5\n"
            result += f"Study Type: {paper.credibility.study_type}\n"
            result += f"Evidence Type: {paper.credibility.evidence_type}\n"
            if paper.credibility.sample_size:
                result += f"Sample Size: {paper.credibility.sample_size}\n"
            if paper.credibility.study_duration:
                result += f"Study Duration: {paper.credibility.study_duration}\n"
            result += "\n"

        result += f"Abstract:\n{paper.abstract}\n\n"
        result += f"Has Full Text: {'Yes' if paper.full_text else 'No'}\n"
        result += f"Total Propositions: {len(paper.propositions)}\n"
        result += f"Quality Propositions: {len(paper.get_quality_propositions())}\n"

        return result
    except Exception as e:
        return f"Error getting paper details: {str(e)}"


@tool
def get_kb_statistics() -> str:
    """Get statistics about the knowledge base.

    Use this to understand what's available in the knowledge base before searching.

    Returns:
        Formatted string with knowledge base statistics
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        stats = _kb.get_statistics()

        result = "Knowledge Base Statistics:\n\n"
        result += f"Total Papers: {stats['papers']}\n"
        result += f"Total Chunks: {stats['chunks']}\n"
        result += f"Total Propositions: {stats['propositions_total']}\n"
        result += f"Quality Propositions: {stats['propositions_quality']}\n"
        result += f"Success Rate: {stats['overall_success_rate']*100:.1f}%\n"
        result += f"Avg Propositions/Paper: {stats['avg_propositions_per_paper']:.1f}\n"

        return result
    except Exception as e:
        return f"Error getting KB statistics: {str(e)}"


@tool
def get_proposition_source_chunk(proposition_id: str) -> str:
    """Get the source chunk that a proposition was extracted from.

    Use this when you need more context about where a proposition came from,
    or to read the original text that a claim was extracted from.

    Args:
        proposition_id: The ID of the proposition

    Returns:
        Formatted string with the source chunk text and metadata
    """
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        # Get the proposition
        proposition = _kb.get_proposition(proposition_id)
        if not proposition:
            return f"Proposition not found: '{proposition_id}'"

        # Get the source chunk
        chunk = _kb.get_chunk(proposition.chunk_id)
        if not chunk:
            return f"Source chunk not found for proposition '{proposition_id}' (chunk_id: {proposition.chunk_id})"

        # Get paper info for context
        paper = _kb.get_paper(chunk.paper_id)
        paper_info = "Unknown Paper"
        if paper:
            paper_info = f"{paper.title} ({paper.year or 'N/A'})"

        result = f"Proposition: {proposition.text}\n"
        result += f"Proposition ID: {proposition_id}\n\n"
        result += f"Source Chunk:\n{chunk.text}\n"
        result += f"Chunk ID: {chunk.chunk_id}\n\n"
        result += f"Paper: {paper_info}\n"
        result += f"Paper ID: {chunk.paper_id}\n"
        result += f"Section: {chunk.section or 'Unknown'}\n"
        if chunk.page is not None:
            result += f"Page: {chunk.page}\n"

        return result
    except Exception as e:
        return f"Error getting proposition source chunk: {str(e)}"


# ======================== ONLINE SEARCH TOOLS ========================


@tool
def search_online_papers(query: str, max_papers: int = 10) -> str:
    """Search online databases for scientific papers.

    Use this when the knowledge base doesn't have sufficient information
    about the claim. This searches PubMed, Semantic Scholar, and CORE.
    
    Papers are automatically extracted and added to the knowledge base,
    so you can then search their propositions using other tools.

    WARNING: This is slow (30+ seconds) and should only be used when necessary.

    Args:
        query: Search query for papers
        max_papers: Maximum number of papers to retrieve (default: 10)

    Returns:
        Formatted string with found papers and their processing status
    """
    if _lit_search is None:
        return "Error: Literature search not initialized"
    
    if _kb is None:
        return "Error: Knowledge base not initialized"

    try:
        print(f"\n[SEARCH ONLINE] Starting online search for: {query[:80]}{'...' if len(query) > 80 else ''}", flush=True)
        print(f"[SEARCH ONLINE] Max papers: {max_papers}", flush=True)

        # Generate search queries
        search_queries = _lit_search.generate_search_queries(query)
        print(f"[SEARCH ONLINE] Generated {len(search_queries)} search queries", flush=True)

        # Search for papers
        papers = _lit_search.search_papers(query, search_queries=search_queries, max_papers=max_papers, verbose=False)

        if not papers:
            print(f"[SEARCH ONLINE] No papers found\n", flush=True)
            return f"No papers found online for query: '{query}'"

        print(f"[SEARCH ONLINE] Found {len(papers)} papers to process", flush=True)

        # Initialize extractor and scorer
        extractor = PropositionExtractor()
        extractor.skip_evaluation = False  # Keep evaluation for quality
        scorer = PaperScorer()

        result = f"Found {len(papers)} papers online. Processing and adding to knowledge base...\n\n"

        processed_count = 0
        skipped_count = 0

        for i, paper in enumerate(papers, 1):
            print(f"[PAPER {i}/{len(papers)}] {paper.title[:80]}{'...' if len(paper.title) > 80 else ''}", flush=True)
            print(f"  ID: {paper.id}, Year: {paper.year or 'N/A'}, Citations: {paper.citations or 0}", flush=True)

            # Check if paper already exists in KB
            existing = _kb.get_paper(paper.id)
            if existing and existing.propositions:
                skipped_count += 1
                print(f"  Status: Already in KB ({len(existing.propositions)} propositions) - Skipped", flush=True)
                result += f"{i}. {paper.title}\n"
                result += f"   Paper ID: {paper.id}\n"
                result += f"   Status: Already in KB ({len(existing.propositions)} propositions)\n\n"
                continue

            # Extract propositions from abstract
            result += f"{i}. {paper.title}\n"
            result += f"   Paper ID: {paper.id}\n"
            result += f"   Authors: {', '.join(paper.authors[:3])}"
            if len(paper.authors) > 3:
                result += " et al."
            result += f"\n   Year: {paper.year or 'Unknown'}\n"
            result += f"   Citations: {paper.citations or 0}\n"

            print(f"  Extracting propositions from abstract...", flush=True)
            # Extract propositions (from abstract only for online papers)
            extractor.extract_from_paper(paper, show_steps=False, use_full_text=False)

            print(f"  Scoring paper credibility...", flush=True)
            # Score paper credibility
            scorer.score_paper(paper)

            # Add to knowledge base and save immediately
            _kb.add_paper(paper, verbose=False)
            _kb.save()

            processed_count += 1

            quality_props = len(paper.get_quality_propositions())
            total_props = len(paper.propositions)

            print(f"  Status: Saved ({quality_props} quality / {total_props} total props, credibility: {paper.credibility.rating:.1f}/5 - {paper.credibility.study_type if paper.credibility else 'N/A'})", flush=True)

            result += f"   Status: Extracted and saved ({quality_props} quality / {total_props} total propositions)\n"

            if paper.credibility:
                result += f"   Credibility: {paper.credibility.rating:.1f}/5 ({paper.credibility.study_type})\n"

            result += "\n"

        print(f"\n[SEARCH ONLINE] Complete: {processed_count} new papers processed, {skipped_count} skipped\n", flush=True)

        result += f"\nSummary: Processed {processed_count} new papers, skipped {skipped_count} already in KB.\n"
        result += f"You can now search propositions from these papers using search_similar_propositions or other KB tools."

        return result
    except Exception as e:
        return f"Error searching online papers: {str(e)}"


# ======================== TOOL LISTS ========================


def get_all_tools():
    """Get all available tools for the agent."""
    return [
        search_similar_propositions,
        search_similar_chunks,
        search_propositions_in_paper,
        find_similar_papers,
        get_paper_details,
        get_kb_statistics,
        get_proposition_source_chunk,
        search_online_papers,
    ]


def get_kb_only_tools():
    """Get tools that only use the knowledge base (no online search)."""
    return [
        search_similar_propositions,
        search_similar_chunks,
        search_propositions_in_paper,
        find_similar_papers,
        get_paper_details,
        get_kb_statistics,
        get_proposition_source_chunk,
    ]

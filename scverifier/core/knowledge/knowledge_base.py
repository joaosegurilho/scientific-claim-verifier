"""Knowledge base for managing papers, propositions, and vector stores.

Works entirely with domain objects (Paper, Proposition, Chunk).
RetrievalSystem handles the conversion to/from LangChain Documents internally.
"""

import os
from typing import List, Dict, Any, Optional
import time

from scverifier.data.models import Paper, Chunk, Proposition
from scverifier.data.database import PaperDatabase
from scverifier.utils import id_generator
from scverifier.core.retrieval.retrieval_system import RetrievalSystem
from scverifier.config.settings import Config


class KnowledgeBase:
    """Central repository for managing scientific papers and their propositions.

    The KnowledgeBase is the single source of truth for:
    - Papers and their metadata
    - Extracted propositions
    - Vector stores for semantic search (via RetrievalSystem)
    - Persistence (save/load)

    Works entirely with domain objects - RetrievalSystem handles LangChain Documents.
    """

    def __init__(self):
        """Initialize an empty knowledge base."""
        self.papers: Dict[str, Paper] = {}  # paper_id -> Paper (in-memory cache)
        self.retrieval_system = RetrievalSystem()
        self.db: Optional[PaperDatabase] = None  # Initialized on save/load

    # ======================== ID COUNTER INITIALIZATION ========================

    def initialize_id_counters(self):
        """Scan all loaded papers and set the global chunk and proposition ID counters to the highest used values."""
        max_chunk = 0
        max_prop = 0
        for paper in self.papers.values():
            # Chunks
            for chunk in getattr(paper, "chunks", []):
                if (
                    hasattr(chunk, "chunk_id")
                    and isinstance(chunk.chunk_id, str)
                    and chunk.chunk_id.startswith("chunk_")
                ):
                    try:
                        n = int(chunk.chunk_id.split("_")[1])
                        if n > max_chunk:
                            max_chunk = n
                    except Exception:
                        pass
            # Propositions
            for prop in getattr(paper, "propositions", []):
                if hasattr(prop, "prop_id") and isinstance(prop.prop_id, str) and prop.prop_id.startswith("prop_"):
                    try:
                        n = int(prop.prop_id.split("_")[1])
                        if n > max_prop:
                            max_prop = n
                    except Exception:
                        pass
        id_generator.set_counters(chunk=max_chunk, prop=max_prop)
        print(f"   Initialized ID counters: chunk={max_chunk}, prop={max_prop}")

    # ======================== CHUNK MANAGEMENT ========================

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk object or None if not found
        """
        for paper in self.papers.values():
            for chunk in getattr(paper, "chunks", []):
                if chunk.chunk_id == chunk_id:
                    return chunk
        return None

    def get_proposition(self, prop_id: str) -> Optional[Proposition]:
        """Get a proposition by its ID.

        Args:
            prop_id: Proposition identifier

        Returns:
            Proposition object or None if not found
        """
        for paper in self.papers.values():
            for prop in getattr(paper, "propositions", []):
                if prop.prop_id == prop_id:
                    return prop
        return None

    # ======================== PAPER MANAGEMENT ========================

    def add_paper(self, paper: Paper, verbose: bool = False) -> None:
        """Add or update a paper in the knowledge base and update vector stores.

        This method:
        1. Stores/updates the paper object
        2. Adds its chunks to the chunk vector store
        3. Adds its propositions to the proposition vector store

        Args:
            paper: Paper object with extracted propositions
            verbose: Whether to print progress messages
        """
        if verbose:
            start_time = time.time()
            print(f"\n{'='*80}")
            print(f"Adding paper: {paper.title[:70]}")
            print(f"Paper ID: {paper.id}")
            print(f"Chunks: {len(paper.chunks)}, Propositions: {len(paper.propositions)}")
            print(f"{'='*80}")

        # Store/update paper
        t0 = time.time()
        self.papers[paper.id] = paper
        if verbose:
            print(f"  [1/3] Stored paper object ({time.time() - t0:.3f}s)")

        # Add to vector stores - RetrievalSystem handles Document conversion
        if paper.chunks:
            t1 = time.time()
            self.add_chunks_to_vectorstore(paper.chunks, verbose=verbose)
            if verbose:
                print(f"  [2/3] Added {len(paper.chunks)} chunks to vectorstore ({time.time() - t1:.3f}s)")
        elif verbose:
            print(f"  [2/3] No chunks to add (0.000s)")

        if paper.propositions:
            t2 = time.time()
            self.add_propositions_to_vectorstore(paper.propositions, verbose=verbose)
            if verbose:
                print(f"  [3/3] Added {len(paper.propositions)} propositions to vectorstore ({time.time() - t2:.3f}s)")
        elif verbose:
            print(f"  [3/3] No propositions to add (0.000s)")

        if verbose:
            total_time = time.time() - start_time
            print(f"\n  Total time for paper: {total_time:.3f}s")
            print(f"{'='*80}\n")

    def add_papers(self, papers: List[Paper], verbose: bool = False) -> None:
        """Add multiple papers to the knowledge base.

        Args:
            papers: List of Paper objects
            verbose: Whether to print progress messages
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"ADDING {len(papers)} PAPERS TO KNOWLEDGE BASE")
            print(f"{'='*80}\n")
            start_time = time.time()

        for idx, paper in enumerate(papers, 1):
            if verbose:
                print(f"[{idx}/{len(papers)}] Processing paper...")
            self.add_paper(paper, verbose=verbose)

        if verbose:
            total_time = time.time() - start_time
            total_props = sum(len(p.propositions) for p in papers)
            total_chunks = sum(len(p.chunks) for p in papers)
            avg_time_per_paper = total_time / len(papers) if papers else 0

            print(f"\n{'='*80}")
            print(f"BATCH COMPLETE")
            print(f"{'='*80}")
            print(f"Papers added: {len(papers)}")
            print(f"Total chunks: {total_chunks}")
            print(f"Total propositions: {total_props}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per paper: {avg_time_per_paper:.3f}s")
            print(f"{'='*80}\n")

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by ID.

        Args:
            paper_id: Paper identifier

        Returns:
            Paper object or None if not found
        """
        return self.papers.get(paper_id)

    def has_paper(self, paper_id: str) -> bool:
        """Check if a paper exists in the knowledge base.

        Args:
            paper_id: Paper identifier

        Returns:
            True if paper exists, False otherwise
        """
        return paper_id in self.papers

    def list_papers(self) -> List[Paper]:
        """Get all papers in the knowledge base.

        Returns:
            List of all Paper objects
        """
        return list(self.papers.values())

    def list_paper_ids(self) -> List[str]:
        """Get all paper IDs in the knowledge base.

        Returns:
            List of paper IDs
        """
        return list(self.papers.keys())

    def delete_paper(self, paper_id: str, verbose: bool = False) -> bool:
        """Delete a paper efficiently using FAISS's native delete capability.

        This removes the paper and its associated chunks/propositions
        from both the knowledge base and the vector stores without rebuilding.

        Args:
            paper_id: Paper identifier
            verbose: Whether to print progress messages

        Returns:
            True if paper was deleted, False if not found
        """
        if paper_id not in self.papers:
            return False

        # Remove paper from dict
        del self.papers[paper_id]

        # Efficiently delete from vectorstores using native FAISS delete
        self.retrieval_system.delete_paper_from_vectorstores(paper_id, verbose=verbose)

        if verbose:
            print(f"   Deleted paper '{paper_id}' from knowledge base")

        return True

    def add_chunks_to_vectorstore(self, chunks: List[Chunk], verbose: bool = False) -> None:
        """Add document chunks to the vectorstore.

        Args:
            chunks: List of Chunk objects to add
            verbose: Whether to print progress messages
        """
        if chunks:
            self.retrieval_system.add_to_chunk_vectorstore(chunks, verbose=verbose)

    def add_propositions_to_vectorstore(self, propositions: List[Proposition], verbose: bool = False) -> None:
        """Add propositions to the vectorstore.

        Args:
            propositions: List of Proposition objects to add
            verbose: Whether to print progress messages
        """
        if propositions:
            self.retrieval_system.add_to_proposition_vectorstore(propositions, verbose=verbose)

    # ======================== SEARCH & QUERY ========================

    def search_propositions(self, query: str, top_k: Optional[int] = None) -> List[Proposition]:
        """Search for relevant propositions.

        Args:
            query: Search query string

        Returns:
            List of Proposition objects matching the query
        """
        # RetrievalSystem now returns domain objects directly
        return self.retrieval_system.query_propositions(query, top_k)

    def search_chunks(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Search for relevant document chunks.

        Args:
            query: Search query string

        Returns:
            List of Chunk objects matching the query
        """
        # RetrievalSystem now returns domain objects directly
        return self.retrieval_system.query_chunks(query, top_k)

    def search_propositions_by_paper(self, query: str, paper_id: str) -> List[Proposition]:
        """Search for propositions within a specific paper.

        Args:
            query: Search query string
            paper_id: Paper to search within

        Returns:
            List of matching Proposition objects from the paper
        """
        # RetrievalSystem now returns domain objects directly
        return self.retrieval_system.query_propositions_by_source(query, paper_id)

    def search_chunks_by_paper(self, query: str, paper_id: str) -> List[Chunk]:
        """Search for chunks within a specific paper.

        Args:
            query: Search query string
            paper_id: Paper to search within

        Returns:
            List of matching Chunk objects from the paper
        """
        # RetrievalSystem now returns domain objects directly
        return self.retrieval_system.query_chunks_by_source(query, paper_id)

    def get_papers_for_query(self, query: str) -> Dict[str, List[Proposition]]:
        """Get query results grouped by paper.

        Args:
            query: Search query string

        Returns:
            Dictionary mapping paper_id to list of matching propositions
        """
        results = self.retrieval_system.query_propositions(query)  # returns domain objects directly
        papers_dict = {}

        for prop in results:
            paper_id = prop.paper_id
            if paper_id not in papers_dict:
                papers_dict[paper_id] = []
            papers_dict[paper_id].append(prop)

        return papers_dict

    def find_similar_papers(self, paper_id: str, top_k: int = 5) -> List[tuple[Paper, float]]:
        """Find papers with similar propositions to the given paper.

        Args:
            paper_id: ID of the paper to find similar papers for
            top_k: Number of similar papers to return

        Returns:
            List of tuples (Paper, similarity_score) sorted by similarity
        """
        paper = self.get_paper(paper_id)
        if paper is None:
            return []

        # Get all propositions from this paper
        all_props = paper.propositions
        if not all_props:
            return []

        # Aggregate similarity scores by paper
        similar_paper_scores = {}

        for prop in all_props:
            # Query for similar propositions
            results = self.retrieval_system.query_propositions_with_scores(
                prop.text, top_k=20
            )

            for similar_prop, score in results:
                # Exclude the current paper
                if similar_prop.paper_id != paper_id:
                    if similar_prop.paper_id not in similar_paper_scores:
                        similar_paper_scores[similar_prop.paper_id] = []
                    similar_paper_scores[similar_prop.paper_id].append(score)

        # Calculate average similarity score for each paper
        avg_scores = {
            pid: sum(scores) / len(scores)
            for pid, scores in similar_paper_scores.items()
        }

        # Sort by score and get top-k
        sorted_papers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Return Paper objects with their scores
        return [(self.get_paper(pid), score) for pid, score in sorted_papers]

    def find_similar_propositions(
        self, proposition_id: str, top_k: int = 5
    ) -> List[tuple[Proposition, float]]:
        """Find propositions similar to the given proposition.

        Args:
            proposition_id: ID of the proposition to find similar propositions for
            top_k: Number of similar propositions to return

        Returns:
            List of tuples (Proposition, similarity_score) sorted by similarity
        """
        proposition = self.get_proposition(proposition_id)
        if proposition is None:
            return []

        # Query for similar propositions
        results = self.retrieval_system.query_propositions_with_scores(
            proposition.text, top_k=top_k + 1
        )

        # Filter out the proposition itself and limit to top_k
        similar_props = [
            (prop, score)
            for prop, score in results
            if prop.prop_id != proposition_id
        ][:top_k]

        return similar_props

    def find_similar_chunks(
        self, chunk_id: str, top_k: int = 5
    ) -> List[tuple[Chunk, float]]:
        """Find chunks similar to the given chunk.

        Args:
            chunk_id: ID of the chunk to find similar chunks for
            top_k: Number of similar chunks to return

        Returns:
            List of tuples (Chunk, similarity_score) sorted by similarity
        """
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return []

        # Query for similar chunks
        results = self.retrieval_system.query_chunks_with_scores(
            chunk.text, top_k=top_k + 1
        )

        # Filter out the chunk itself and limit to top_k
        similar_chunks = [
            (c, score)
            for c, score in results
            if c.chunk_id != chunk_id
        ][:top_k]

        return similar_chunks

    # ======================== STATISTICS ========================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base.

        Returns:
            Dictionary with counts and statistics
        """
        total_papers = len(self.papers)
        total_chunks = sum(len(p.chunks) for p in self.papers.values())
        total_propositions = sum(len(p.propositions) for p in self.papers.values())
        total_quality = sum(len(p.get_quality_propositions()) for p in self.papers.values())

        avg_props_per_paper = total_propositions / total_papers if total_papers > 0 else 0
        avg_quality_per_paper = total_quality / total_papers if total_papers > 0 else 0
        overall_success_rate = total_quality / total_propositions if total_propositions > 0 else 0

        vectorstore_stats = self.retrieval_system.get_vectorstore_stats()

        return {
            "papers": total_papers,
            "chunks": total_chunks,
            "propositions_total": total_propositions,
            "propositions_quality": total_quality,
            "avg_propositions_per_paper": avg_props_per_paper,
            "avg_quality_per_paper": avg_quality_per_paper,
            "overall_success_rate": overall_success_rate,
            "vectorstores": vectorstore_stats,
        }

    def get_paper_statistics(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific paper.

        Args:
            paper_id: Paper identifier

        Returns:
            Dictionary with paper statistics or None if not found
        """
        paper = self.get_paper(paper_id)
        if paper is None:
            return None
        return paper.get_statistics()

    def print_statistics(self):
        """Print a formatted summary of knowledge base statistics."""
        stats = self.get_statistics()

        print("\n Knowledge Base Statistics")
        print("=" * 50)
        print(f"Papers: {stats['papers']}")
        print(f"Total Chunks: {stats['chunks']}")
        print(f"Total Propositions: {stats['propositions_total']}")
        print(f"Quality Propositions: {stats['propositions_quality']}")
        print(f"Success Rate: {stats['overall_success_rate']*100:.1f}%")
        print(f"Avg Propositions/Paper: {stats['avg_propositions_per_paper']:.1f}")
        print(f"Avg Quality Props/Paper: {stats['avg_quality_per_paper']:.1f}")
        print("\nVector Stores:")
        print(f"  Propositions: {stats['vectorstores']['propositions']['count']}")
        print(f"  Chunks: {stats['vectorstores']['chunks']['count']}")

    # ======================== PERSISTENCE ========================

    def save(self, path: str = None):
        """Save the knowledge base to disk.

        This saves:
        1. All papers and their metadata (to SQLite database)
        2. Vector stores (as FAISS indices)

        Args:
            path: Directory to save to (defaults to Config.DB_NAME)
        """

        if path is None:
            path = Config.DB_NAME

        os.makedirs(path, exist_ok=True)

        # Save papers to SQLite database
        print("   Saving papers to SQLite...")
        t0 = time.time()
        db_path = os.path.join(path, "papers.db")

        print(f"     Writing {len(self.papers)} papers to database...")
        db = PaperDatabase(db_path)

        # Delete papers from database that are no longer in memory
        db_papers = db.get_all_papers()
        papers_to_delete = set(db_papers.keys()) - set(self.papers.keys())
        for paper_id in papers_to_delete:
            db.delete_paper(paper_id)
            print(f"     Deleted paper '{paper_id}' from database")

        # Save current papers
        db.save_papers(list(self.papers.values()))
        db.close()

        t1 = time.time()
        print(f"   Saved {len(self.papers)} papers to {db_path} (Total: {t1-t0:.2f}s)")

        # Save vector stores
        print("   Saving vector stores...")
        t2 = time.time()
        prop_path = os.path.join(path, "propositions")
        chunk_path = os.path.join(path, "chunks")
        self.retrieval_system.save_vectorstores(prop_path, chunk_path)
        t3 = time.time()
        print(f"   Vector stores saved ({t3-t2:.2f}s)")

        print(f" Knowledge base saved to {path} (Total: {t3-t0:.2f}s)")

    def load(self, path: str = None):
        """Load the knowledge base from disk.

        If the database doesn't exist, initializes an empty knowledge base.
        This allows the system to start fresh and build up papers over time.

        Args:
            path: Directory to load from (defaults to Config.DB_NAME)
        """
        if path is None:
            path = Config.DB_NAME

        # Create directory if it doesn't exist
        if not os.path.exists(path):
            print(f"   Knowledge base directory not found at {path}")
            print(f"   Creating directory and initializing empty knowledge base...")
            os.makedirs(path, exist_ok=True)
            self.papers = {}
            self.initialize_id_counters()
            return

        # Load papers from SQLite database if it exists
        db_path = os.path.join(path, "papers.db")
        if not os.path.exists(db_path):
            print(f"   Database not found at {db_path}")
            print(f"   Initializing empty knowledge base (database will be created on first save)")
            self.papers = {}
            self.initialize_id_counters()
            return

        db = PaperDatabase(db_path)
        self.papers = db.get_all_papers()
        db.close()
        print(f"   Loaded {len(self.papers)} papers from {db_path}")

        # Load vector stores with error handling
        prop_path = os.path.join(path, "propositions")
        chunk_path = os.path.join(path, "chunks")

        try:
            self.retrieval_system.load_vectorstores(prop_path, chunk_path)
            # Reconstruct chunks and propositions from FAISS and attach to papers
            self._populate_papers_from_vectorstores()
            print(f" Knowledge base loaded from {path}")
            self.print_statistics()
        except Exception as e:
            print(f"  Warning: Could not load vector stores: {e}")
            print("   Rebuilding vector stores from papers...")
            self._rebuild_vector_stores()
            print("   Vector stores rebuilt successfully")
            # Save the rebuilt state
            self.save(path)

        self.initialize_id_counters()

    def _populate_papers_from_vectorstores(self):
        """Reconstruct chunks and propositions from FAISS vector stores and attach to papers.

        This is needed after loading from SQLite, since the database only stores
        Paper metadata, while chunks and propositions live in FAISS.
        """
        # Clear existing chunks and propositions
        for paper in self.papers.values():
            paper.chunks = []
            paper.propositions = []

        # Reconstruct chunks from FAISS
        if self.retrieval_system.chunk_vectorstore:
            for doc_id, doc in self.retrieval_system.chunk_vectorstore.docstore._dict.items():
                chunk = Chunk.from_langchain_document(doc)
                paper_id = chunk.paper_id
                if paper_id in self.papers:
                    self.papers[paper_id].chunks.append(chunk)

        # Reconstruct propositions from FAISS
        if self.retrieval_system.proposition_vectorstore:
            for doc_id, doc in self.retrieval_system.proposition_vectorstore.docstore._dict.items():
                prop = Proposition.from_langchain_document(doc)
                paper_id = prop.paper_id
                if paper_id in self.papers:
                    self.papers[paper_id].propositions.append(prop)

        print(f"   Reconstructed chunks and propositions from vector stores")

    def _rebuild_vector_stores(self):
        """Rebuild vector stores from existing papers."""
        all_chunks = []
        all_propositions = []

        for paper in self.papers.values():
            # Collect chunks
            if paper.chunks:
                all_chunks.extend(paper.chunks)

            # Collect quality propositions
            quality_props = paper.get_quality_propositions()
            if quality_props:
                all_propositions.extend(quality_props)

        # Rebuild vector stores - RetrievalSystem handles Document conversion
        if all_propositions:
            self.retrieval_system.create_proposition_vectorstore(all_propositions, verbose=True)

        if all_chunks:
            self.retrieval_system.create_chunk_vectorstore(all_chunks, verbose=True)

    def clear(self):
        """Clear all data from the knowledge base."""
        self.papers = {}
        self.retrieval_system.proposition_vectorstore = None
        self.retrieval_system.chunk_vectorstore = None
        self.retrieval_system.proposition_retriever = None
        self.retrieval_system.chunk_retriever = None
        print(" Knowledge base cleared")

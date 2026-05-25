"""Retrieval system for propositions and chunks using vector stores.

This is the BOUNDARY LAYER between domain objects and LangChain.
- Public methods work with Proposition and Chunk domain objects
- Internal methods work with LangChain Documents
- Conversion happens at the boundary
"""

import os
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from scverifier.config.settings import Config
from scverifier.data.models import Proposition, Chunk


class RetrievalSystem:
    """Manages vector stores and retrieval for both propositions and chunks.

    This system handles:
    - Creating and managing FAISS vector stores (internal LangChain Documents)
    - Similarity-based document retrieval
    - Multi-source query support
    - Vector store persistence

    PUBLIC API: All public methods accept and return domain objects (Proposition, Chunk)
    INTERNAL: Vector stores use LangChain Documents (implementation detail)
    """

    def __init__(self):
        self.embedding_model = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)

        # Vector stores (internal - work with Documents)
        self.proposition_vectorstore = None
        self.chunk_vectorstore = None

        # Retrievers (internal - work with Documents)
        self.proposition_retriever = None
        self.chunk_retriever = None

        # Track document IDs by paper for efficient deletion
        # Format: paper_id -> list of document IDs
        self.paper_to_prop_ids: Dict[str, List[str]] = {}
        self.paper_to_chunk_ids: Dict[str, List[str]] = {}

    # ======================== VECTOR STORE CREATION ========================

    def create_proposition_vectorstore(self, propositions: List[Proposition], verbose: bool = True):
        """Create a new vector store from propositions.

        Args:
            propositions: List of Proposition domain objects to index
            verbose: Whether to print progress messages

        Raises:
            ValueError: If no propositions are provided
        """
        if not propositions:
            raise ValueError("No propositions provided for vector store creation")

        # Convert to Documents at the boundary
        docs = [p.to_langchain_document() for p in propositions]

        # Internal work with Documents
        self.proposition_vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.proposition_retriever = self.proposition_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": Config.PROPOSITION_RETRIEVAL_K}
        )

        # Track document IDs by paper for efficient deletion
        self.paper_to_prop_ids = {}
        for prop in propositions:
            paper_id = prop.paper_id
            if paper_id not in self.paper_to_prop_ids:
                self.paper_to_prop_ids[paper_id] = []
            # FAISS stores documents with IDs - we need to find the ID for this prop
            # The document ID in FAISS corresponds to the index in the docstore
            for doc_id, doc in self.proposition_vectorstore.docstore._dict.items():
                if doc.metadata.get("prop_id") == prop.prop_id:
                    self.paper_to_prop_ids[paper_id].append(doc_id)
                    break

        if verbose:
            print(f"   Created proposition vector store with {len(propositions)} propositions")

    def create_chunk_vectorstore(self, chunks: List[Chunk], verbose: bool = True):
        """Create a new vector store from document chunks.

        Args:
            chunks: List of Chunk domain objects to index
            verbose: Whether to print progress messages

        Raises:
            ValueError: If no chunks are provided
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")

        # Convert to Documents at the boundary
        docs = [c.to_langchain_document() for c in chunks]

        # Internal work with Documents
        self.chunk_vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.chunk_retriever = self.chunk_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": Config.CHUNK_RETRIEVAL_K}
        )

        # Track document IDs by paper for efficient deletion
        self.paper_to_chunk_ids = {}
        for chunk in chunks:
            paper_id = chunk.paper_id
            if paper_id not in self.paper_to_chunk_ids:
                self.paper_to_chunk_ids[paper_id] = []
            # FAISS stores documents with IDs - we need to find the ID for this chunk
            # The document ID in FAISS corresponds to the index in the docstore
            for doc_id, doc in self.chunk_vectorstore.docstore._dict.items():
                if doc.metadata.get("chunk_id") == chunk.chunk_id:
                    self.paper_to_chunk_ids[paper_id].append(doc_id)
                    break

        if verbose:
            print(f"   Created chunk vector store with {len(chunks)} chunks")

    # ======================== INCREMENTAL UPDATES ========================

    def add_to_proposition_vectorstore(self, propositions: List[Proposition], verbose: bool = True):
        """Add propositions to existing vectorstore or create new one if none exists.

        Args:
            propositions: List of Proposition domain objects to add
            verbose: Whether to print progress messages
        """
        if not propositions:
            return

        if self.proposition_vectorstore is None:
            # First time - create new vectorstore
            self.create_proposition_vectorstore(propositions, verbose=verbose)
        else:
            # Convert to Documents at the boundary
            docs = [p.to_langchain_document() for p in propositions]

            # Merge with existing vectorstore
            new_vectorstore = FAISS.from_documents(docs, self.embedding_model)
            self.proposition_vectorstore.merge_from(new_vectorstore)

            # Update retriever with new configuration
            self.proposition_retriever = self.proposition_vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": Config.PROPOSITION_RETRIEVAL_K}
            )

            # Track document IDs for the newly added propositions
            for prop in propositions:
                paper_id = prop.paper_id
                if paper_id not in self.paper_to_prop_ids:
                    self.paper_to_prop_ids[paper_id] = []
                # Find the document ID for this proposition
                for doc_id, doc in self.proposition_vectorstore.docstore._dict.items():
                    if doc.metadata.get("prop_id") == prop.prop_id and doc_id not in self.paper_to_prop_ids[paper_id]:
                        self.paper_to_prop_ids[paper_id].append(doc_id)
                        break

            if verbose:
                print(f"   Added {len(propositions)} propositions to existing vectorstore")

    def add_to_chunk_vectorstore(self, chunks: List[Chunk], verbose: bool = True):
        """Add chunks to existing vectorstore or create new one if none exists.

        Args:
            chunks: List of Chunk domain objects to add
            verbose: Whether to print progress messages
        """
        if not chunks:
            return

        if self.chunk_vectorstore is None:
            # First time - create new vectorstore
            self.create_chunk_vectorstore(chunks, verbose=verbose)
        else:
            # Convert to Documents at the boundary
            docs = [c.to_langchain_document() for c in chunks]

            # Merge with existing vectorstore
            new_vectorstore = FAISS.from_documents(docs, self.embedding_model)
            self.chunk_vectorstore.merge_from(new_vectorstore)

            # Update retriever with new configuration
            self.chunk_retriever = self.chunk_vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": Config.CHUNK_RETRIEVAL_K}
            )

            # Track document IDs for the newly added chunks
            for chunk in chunks:
                paper_id = chunk.paper_id
                if paper_id not in self.paper_to_chunk_ids:
                    self.paper_to_chunk_ids[paper_id] = []
                # Find the document ID for this chunk
                for doc_id, doc in self.chunk_vectorstore.docstore._dict.items():
                    if doc.metadata.get("chunk_id") == chunk.chunk_id and doc_id not in self.paper_to_chunk_ids[paper_id]:
                        self.paper_to_chunk_ids[paper_id].append(doc_id)
                        break

            if verbose:
                print(f"   Added {len(chunks)} chunks to existing vectorstore")

    # ======================== DELETION ========================

    def delete_paper_from_vectorstores(self, paper_id: str, verbose: bool = False) -> bool:
        """Delete all documents for a specific paper from vectorstores efficiently.

        This uses FAISS's native delete capability instead of rebuilding entire vectorstores.

        Args:
            paper_id: Paper identifier
            verbose: Whether to print progress messages

        Returns:
            True if paper was found and deleted, False otherwise
        """
        deleted = False

        # Delete propositions
        if paper_id in self.paper_to_prop_ids and self.proposition_vectorstore:
            prop_ids = self.paper_to_prop_ids[paper_id]
            if prop_ids:
                # Validate IDs exist in vectorstore before deleting
                existing_ids = set(self.proposition_vectorstore.docstore._dict.keys())
                valid_ids = [pid for pid in prop_ids if pid in existing_ids]
                invalid_ids = [pid for pid in prop_ids if pid not in existing_ids]

                if invalid_ids and verbose:
                    print(f"   Warning: {len(invalid_ids)} proposition IDs not found in vectorstore (stale mapping)")

                if valid_ids:
                    try:
                        self.proposition_vectorstore.delete(valid_ids)
                        if verbose:
                            print(f"   Deleted {len(valid_ids)} propositions from vectorstore")
                    except Exception as e:
                        if verbose:
                            print(f"   Warning: Deletion failed: {e}")

                # Clean up mapping regardless
                del self.paper_to_prop_ids[paper_id]
                deleted = True

                # Handle empty vectorstore
                if len(self.proposition_vectorstore.docstore._dict) == 0:
                    self.proposition_vectorstore = None
                    self.proposition_retriever = None
                    if verbose:
                        print("   Proposition vectorstore is now empty")

        # Delete chunks
        if paper_id in self.paper_to_chunk_ids and self.chunk_vectorstore:
            chunk_ids = self.paper_to_chunk_ids[paper_id]
            if chunk_ids:
                # Validate IDs exist in vectorstore before deleting
                existing_ids = set(self.chunk_vectorstore.docstore._dict.keys())
                valid_ids = [cid for cid in chunk_ids if cid in existing_ids]
                invalid_ids = [cid for cid in chunk_ids if cid not in existing_ids]

                if invalid_ids and verbose:
                    print(f"   Warning: {len(invalid_ids)} chunk IDs not found in vectorstore (stale mapping)")

                if valid_ids:
                    try:
                        self.chunk_vectorstore.delete(valid_ids)
                        if verbose:
                            print(f"   Deleted {len(valid_ids)} chunks from vectorstore")
                    except Exception as e:
                        if verbose:
                            print(f"   Warning: Deletion failed: {e}")

                # Clean up mapping regardless
                del self.paper_to_chunk_ids[paper_id]
                deleted = True

                # Handle empty vectorstore
                if len(self.chunk_vectorstore.docstore._dict) == 0:
                    self.chunk_vectorstore = None
                    self.chunk_retriever = None
                    if verbose:
                        print("   Chunk vectorstore is now empty")

        return deleted

    # ======================== QUERYING (PUBLIC API - RETURNS DOMAIN OBJECTS) ========================

    def query_propositions(self, query: str, top_k: Optional[int] = None) -> List[Proposition]:
        """Query the proposition-based vector store for similar documents.

        Args:
            query: Search query string
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of Proposition domain objects (converted at boundary)

        Raises:
            ValueError: If proposition vector store is not initialized
        """
        if self.proposition_retriever is None:
            raise ValueError(
                "Proposition vector store not initialized. "
                "Call create_proposition_vectorstore or add_to_proposition_vectorstore first."
            )

        # Use provided top_k or fall back to config default
        k = top_k if top_k is not None else Config.PROPOSITION_RETRIEVAL_K

        # Create temporary retriever with custom k value if needed
        if top_k is not None:
            retriever = self.proposition_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        else:
            retriever = self.proposition_retriever

        # Internal retrieval returns Documents
        docs = Config.retry_llm_call(lambda: retriever.invoke(query))

        # Convert to domain objects at the boundary
        return [Proposition.from_langchain_document(doc) for doc in docs]

    def query_propositions_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Proposition, float]]:
        """Query propositions and return with similarity scores.

        Args:
            query: Search query string
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of (Proposition, similarity_score) tuples where similarity_score is 0-1

        Raises:
            ValueError: If proposition vector store is not initialized
        """
        if self.proposition_vectorstore is None:
            raise ValueError(
                "Proposition vector store not initialized. "
                "Call create_proposition_vectorstore or add_to_proposition_vectorstore first."
            )

        # Use provided top_k or fall back to config default
        k = top_k if top_k is not None else Config.PROPOSITION_RETRIEVAL_K

        # FAISS similarity_search_with_score returns (Document, distance) tuples
        # Lower distance = more similar
        docs_with_scores = self.proposition_vectorstore.similarity_search_with_score(query, k=k)

        # Convert to domain objects with normalized similarity scores
        results = []
        for doc, distance in docs_with_scores:
            prop = Proposition.from_langchain_document(doc)
            # Convert L2 distance to similarity score (0-1 range)
            # Using inverse distance formula: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)
            results.append((prop, similarity))

        return results

    def query_chunks(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Query the chunk-based vector store for similar documents.

        Args:
            query: Search query string
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of Chunk domain objects (converted at boundary)

        Raises:
            ValueError: If chunk vector store is not initialized
        """
        if self.chunk_retriever is None:
            raise ValueError(
                "Chunk vector store not initialized. "
                "Call create_chunk_vectorstore or add_to_chunk_vectorstore first."
            )

        # Use provided top_k or fall back to config default
        k = top_k if top_k is not None else Config.CHUNK_RETRIEVAL_K

        # Create temporary retriever with custom k value if needed
        if top_k is not None:
            retriever = self.chunk_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        else:
            retriever = self.chunk_retriever

        # Internal retrieval returns Documents
        docs = Config.retry_llm_call(lambda: retriever.invoke(query))

        # Convert to domain objects at the boundary
        return [Chunk.from_langchain_document(doc) for doc in docs]

    def query_chunks_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """Query chunks and return with similarity scores.

        Args:
            query: Search query string
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of (Chunk, similarity_score) tuples where similarity_score is 0-1

        Raises:
            ValueError: If chunk vector store is not initialized
        """
        if self.chunk_vectorstore is None:
            raise ValueError(
                "Chunk vector store not initialized. "
                "Call create_chunk_vectorstore or add_to_chunk_vectorstore first."
            )

        # Use provided top_k or fall back to config default
        k = top_k if top_k is not None else Config.CHUNK_RETRIEVAL_K

        # FAISS similarity_search_with_score returns (Document, distance) tuples
        # Lower distance = more similar
        docs_with_scores = self.chunk_vectorstore.similarity_search_with_score(query, k=k)

        # Convert to domain objects with normalized similarity scores
        results = []
        for doc, distance in docs_with_scores:
            chunk = Chunk.from_langchain_document(doc)
            # Convert L2 distance to similarity score (0-1 range)
            # Using inverse distance formula: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)
            results.append((chunk, similarity))

        return results

    def query_chunks(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Query the chunk-based vector store for similar documents.

        Args:
            query: Search query string
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of Chunk domain objects (converted at boundary)

        Raises:
            ValueError: If chunk vector store is not initialized
        """
        # Use query_chunks_with_scores and extract just the chunks
        results_with_scores = self.query_chunks_with_scores(query, top_k=top_k)
        return [chunk for chunk, _ in results_with_scores]

    def query_propositions_by_source(self, query: str, source: str, top_k: Optional[int] = None) -> List[Proposition]:
        """Query proposition documents from a specific paper source.

        Args:
            query: Search query string
            source: Paper source to filter by
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of Proposition domain objects from the specified source
        """
        all_results = self.query_propositions(query, top_k=top_k)
        return [prop for prop in all_results if prop.source == source]

    def query_chunks_by_source(self, query: str, source: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Query chunk documents from a specific paper source.

        Args:
            query: Search query string
            source: Paper source to filter by
            top_k: Optional number of results to return (overrides config default)

        Returns:
            List of Chunk domain objects from the specified source
        """
        all_results = self.query_chunks(query, top_k=top_k)
        return [chunk for chunk in all_results if chunk.source == source]

    def compare_retrieval(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Proposition], List[Chunk]]:
        """Compare retrieval results from both proposition and chunk stores.

        Args:
            query: Search query string
            top_k: Optional number of results to return for both queries

        Returns:
            Tuple of (proposition_results, chunk_results) as domain objects
        """
        prop_results = self.query_propositions(query, top_k=top_k)
        chunk_results = self.query_chunks(query, top_k=top_k)

        return prop_results, chunk_results

    # ======================== UTILITIES ========================

    def get_all_sources(self) -> List[str]:
        """Get sorted list of all paper sources in the vectorstores.

        Returns:
            Sorted list of unique source names
        """
        sources = set()

        if self.proposition_vectorstore:
            for doc_id in self.proposition_vectorstore.docstore._dict:
                doc = self.proposition_vectorstore.docstore._dict[doc_id]
                source = doc.metadata.get("source")
                if source:
                    sources.add(source)

        return sorted(list(sources))

    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector stores.

        Returns:
            Dictionary with counts and status for both vectorstores
        """
        stats = {}

        if self.proposition_vectorstore:
            stats["propositions"] = {"count": len(self.proposition_vectorstore.docstore._dict), "initialized": True}
        else:
            stats["propositions"] = {"count": 0, "initialized": False}

        if self.chunk_vectorstore:
            stats["chunks"] = {"count": len(self.chunk_vectorstore.docstore._dict), "initialized": True}
        else:
            stats["chunks"] = {"count": 0, "initialized": False}

        return stats

    # ======================== DISPLAY METHODS ========================

    def print_retrieval_results(self, query: str, results: List, retrieval_type: str):
        """Print formatted retrieval results.

        Args:
            query: The query that was used
            results: Retrieved domain objects (Proposition or Chunk)
            retrieval_type: Description of retrieval method (e.g., "Proposition-based")
        """
        print(f"\n=== {retrieval_type} Retrieval Results for: '{query}' ===")

        for i, item in enumerate(results, 1):
            if isinstance(item, Proposition):
                print(f"{i}) Content: {item.text}")
                print(f"   Chunk ID: {item.chunk_id} | Page: {item.page} | Source: {item.source}")
            elif isinstance(item, Chunk):
                print(f"{i}) Content: {item.text}")
                print(f"   Chunk ID: {item.chunk_id} | Page: {item.page} | Source: {item.source}")
            print()

    def compare_and_print_results(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[List[Proposition], List[Chunk]]:
        """Compare retrieval results and print both side by side.

        Args:
            query: Search query string
            top_k: Optional number of results to return for both queries

        Returns:
            Tuple of (proposition_results, chunk_results)
        """
        prop_results, chunk_results = self.compare_retrieval(query, top_k=top_k)

        self.print_retrieval_results(query, prop_results, "Proposition-based")
        self.print_retrieval_results(query, chunk_results, "Chunk-based")

        return prop_results, chunk_results

    # ======================== PERSISTENCE ========================

    def save_vectorstores(self, proposition_path: str = None, chunk_path: str = None):
        """Save vector stores to disk.

        Args:
            proposition_path: Path to save proposition vectorstore
            chunk_path: Path to save chunk vectorstore
        """
        if proposition_path and self.proposition_vectorstore:
            prop_count = len(self.proposition_vectorstore.docstore._dict)
            print(f"     Saving proposition vectorstore ({prop_count} items)...")
            t0 = time.time()
            self.proposition_vectorstore.save_local(proposition_path)

            # Save ID mapping
            mapping_path = proposition_path + "_id_mapping.pkl"
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.paper_to_prop_ids, f)

            t1 = time.time()
            print(f"     Proposition vectorstore saved: {t1-t0:.2f}s")

        if chunk_path and self.chunk_vectorstore:
            chunk_count = len(self.chunk_vectorstore.docstore._dict)
            print(f"     Saving chunk vectorstore ({chunk_count} items)...")
            t0 = time.time()
            self.chunk_vectorstore.save_local(chunk_path)

            # Save ID mapping
            mapping_path = chunk_path + "_id_mapping.pkl"
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.paper_to_chunk_ids, f)

            t1 = time.time()
            print(f"     Chunk vectorstore saved: {t1-t0:.2f}s")

    def load_vectorstores(self, proposition_path: str = None, chunk_path: str = None):
        """Load vector stores from disk.

        Args:
            proposition_path: Path to load proposition vectorstore from
            chunk_path: Path to load chunk vectorstore from
        """
        if proposition_path:
            self.proposition_vectorstore = FAISS.load_local(
                proposition_path, self.embedding_model, allow_dangerous_deserialization=True
            )
            self.proposition_retriever = self.proposition_vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": Config.PROPOSITION_RETRIEVAL_K}
            )

            # Load ID mapping if it exists
            mapping_path = proposition_path + "_id_mapping.pkl"
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.paper_to_prop_ids = pickle.load(f)
            else:
                # Fallback: rebuild mapping from vectorstore
                self.paper_to_prop_ids = {}
                for doc_id, doc in self.proposition_vectorstore.docstore._dict.items():
                    paper_id = doc.metadata.get("paper_id")
                    if paper_id:
                        if paper_id not in self.paper_to_prop_ids:
                            self.paper_to_prop_ids[paper_id] = []
                        self.paper_to_prop_ids[paper_id].append(doc_id)

            print(f"   Proposition vector store loaded from {proposition_path}")

        if chunk_path:
            self.chunk_vectorstore = FAISS.load_local(
                chunk_path, self.embedding_model, allow_dangerous_deserialization=True
            )
            self.chunk_retriever = self.chunk_vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": Config.CHUNK_RETRIEVAL_K}
            )

            # Load ID mapping if it exists
            mapping_path = chunk_path + "_id_mapping.pkl"
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.paper_to_chunk_ids = pickle.load(f)
            else:
                # Fallback: rebuild mapping from vectorstore
                self.paper_to_chunk_ids = {}
                for doc_id, doc in self.chunk_vectorstore.docstore._dict.items():
                    paper_id = doc.metadata.get("paper_id")
                    if paper_id:
                        if paper_id not in self.paper_to_chunk_ids:
                            self.paper_to_chunk_ids[paper_id] = []
                        self.paper_to_chunk_ids[paper_id].append(doc_id)

            print(f"   Chunk vector store loaded from {chunk_path}")

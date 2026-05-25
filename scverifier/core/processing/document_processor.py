"""Document processing utilities for chunking and preparing text."""

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from scverifier.config.settings import Config
from scverifier.data.models import Chunk
from scverifier.utils.id_generator import get_next_chunk_id


class DocumentProcessor:
    """Handles document chunking and preprocessing for the extraction pipeline.

    This processor:
    - Splits documents into semantic chunks using tiktoken
    - Preserves metadata (page numbers, sources, paper_id, etc.)
    - Assigns unique chunk IDs using global counter (format: "chunk_1", "chunk_2", etc.)
    - Returns Chunk domain objects (LangChain Documents are internal implementation detail)
    """

    def __init__(self):
        """Initialize the document processor with configured chunking strategy."""
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
        )

    # ======================== CHUNKING ========================

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk raw text directly into Chunk objects.

        Args:
            text: Raw text string to chunk
            metadata: Metadata dict containing:
                - paper_id: Paper identifier
                - source: Paper title or source name
                - section: Section name (e.g., "abstract", "introduction", "results")
                - page: Optional page number

        Returns:
            List of Chunk domain objects with section tracking
        """
        # Ensure required metadata exists
        paper_id = metadata.get("paper_id", "doc")
        section = metadata.get("section", "")

        # Create a temporary Document for the text splitter (internal implementation detail)
        temp_doc = Document(page_content=text, metadata=metadata)

        # Split using LangChain
        doc_splits = self.text_splitter.split_documents([temp_doc])

        # Convert to Chunk objects with proper IDs and section tracking
        chunks = []
        for doc in doc_splits:
            chunk = Chunk(
                text=doc.page_content,
                chunk_id=get_next_chunk_id(),
                source=doc.metadata.get("source", "Unknown"),
                paper_id=paper_id,
                page=doc.metadata.get("page"),
                section=section,
            )
            chunks.append(chunk)

        return chunks

    # ======================== CHUNK UTILITIES ========================

    def get_chunk_by_id(self, chunks: List[Chunk], chunk_id: str) -> Chunk:
        """Retrieve a specific chunk by its ID.

        Args:
            chunks: List of Chunk objects to search
            chunk_id: Unique chunk ID (e.g., "chunk_1", "chunk_2")

        Returns:
            The Chunk object with matching ID

        Raises:
            ValueError: If chunk with given ID is not found
        """
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk

        available_ids = [c.chunk_id for c in chunks]
        raise ValueError(f"Chunk with ID {chunk_id} not found. Available chunk IDs: {available_ids}")

    def get_chunks_by_page(self, chunks: List[Chunk], page_num: int) -> List[Chunk]:
        """Get all chunks that originated from a specific page.

        Args:
            chunks: List of Chunk objects to filter
            page_num: Page number to filter by

        Returns:
            List of chunks from the specified page
        """
        return [chunk for chunk in chunks if chunk.page == page_num]

    def get_chunks_by_source(self, chunks: List[Chunk], source: str) -> List[Chunk]:
        """Get all chunks from a specific source.

        Args:
            chunks: List of Chunk objects to filter
            source: Source name to filter by

        Returns:
            List of chunks from the specified source
        """
        return [chunk for chunk in chunks if chunk.source == source]

    def get_chunks_by_paper(self, chunks: List[Chunk], paper_id: str) -> List[Chunk]:
        """Get all chunks from a specific paper.

        Args:
            chunks: List of Chunk objects to filter
            paper_id: Paper ID to filter by

        Returns:
            List of chunks from the specified paper
        """
        return [chunk for chunk in chunks if chunk.paper_id == paper_id]

    # ======================== STATISTICS & DISPLAY ========================

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get comprehensive statistics about the chunks.

        Args:
            chunks: List of Chunk objects to analyze

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "pages": {},
                "sources": {},
                "papers": {},
                "avg_chunk_length": 0,
            }

        # Group by page
        pages = {}
        for chunk in chunks:
            page = chunk.page or "Unknown"
            pages[page] = pages.get(page, 0) + 1

        # Group by source
        sources = {}
        for chunk in chunks:
            sources[chunk.source] = sources.get(chunk.source, 0) + 1

        # Group by paper_id
        papers = {}
        for chunk in chunks:
            papers[chunk.paper_id] = papers.get(chunk.paper_id, 0) + 1

        # Calculate average chunk length
        total_length = sum(len(chunk.text) for chunk in chunks)
        avg_length = total_length / len(chunks) if chunks else 0

        return {
            "total_chunks": len(chunks),
            "pages": dict(sorted(pages.items())),
            "sources": dict(sorted(sources.items())),
            "papers": dict(sorted(papers.items())),
            "avg_chunk_length": round(avg_length, 2),
            "chunk_ids": [c.chunk_id for c in chunks],
        }

    def print_chunk_summary(self, chunks: List[Chunk]):
        """Print a formatted summary of chunks.

        Args:
            chunks: List of Chunk objects to summarize
        """
        print("\n=== Chunk Summary ===")
        print(f"Total chunks: {len(chunks)}")

        if not chunks:
            return

        stats = self.get_chunk_statistics(chunks)

        # Print page distribution
        if len(stats["pages"]) > 1:
            print("Chunks per page:")
            for page, count in sorted(stats["pages"].items()):
                print(f"  Page {page}: {count} chunks")

        # Print source distribution
        if len(stats["sources"]) > 1:
            print("Chunks per source:")
            for source, count in sorted(stats["sources"].items()):
                print(f"  {source}: {count} chunks")

        print(f"Average chunk length: {stats['avg_chunk_length']:.0f} characters")

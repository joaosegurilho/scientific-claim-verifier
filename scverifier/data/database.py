"""SQLite database for managing Paper objects.

This module provides a simple interface for storing and retrieving Paper objects
using SQLite instead of pickle serialization.
"""

import sqlite3
import json
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scverifier.data.models import Paper, CredibilityScores


class PaperDatabase:
    """SQLite database for Paper storage and retrieval."""

    def __init__(self, db_path: str):
        """Initialize database connection and create tables if needed.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

    def _create_tables(self):
        """Create papers table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                doi TEXT,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,
                year INTEGER,
                citations INTEGER,
                url TEXT,
                pdf_url TEXT,
                source TEXT,
                has_pdf INTEGER,
                pmc_id TEXT,
                full_text TEXT,
                extracted_from TEXT,
                credibility TEXT
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")

        self.conn.commit()

    def save_paper(self, paper: "Paper") -> None:
        """Save or update a paper in the database.

        Args:
            paper: Paper object to save
        """
        cursor = self.conn.cursor()

        # Serialize complex fields to JSON
        authors_json = json.dumps(paper.authors)
        full_text_json = json.dumps(paper.full_text)
        credibility_json = json.dumps(paper.credibility.to_dict()) if paper.credibility else None

        cursor.execute("""
            INSERT OR REPLACE INTO papers (
                id, doi, title, abstract, authors, year, citations,
                url, pdf_url, source, has_pdf, pmc_id, full_text,
                extracted_from, credibility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper.id,
            paper.doi,
            paper.title,
            paper.abstract,
            authors_json,
            paper.year,
            paper.citations,
            paper.url,
            paper.pdf_url,
            paper.source,
            1 if paper.has_pdf else 0,
            paper.pmc_id,
            full_text_json,
            paper.extracted_from,
            credibility_json
        ))

        self.conn.commit()

    def save_papers(self, papers: List["Paper"]) -> None:
        """Save multiple papers in a single transaction.

        Args:
            papers: List of Paper objects to save
        """
        cursor = self.conn.cursor()

        for paper in papers:
            authors_json = json.dumps(paper.authors)
            full_text_json = json.dumps(paper.full_text)
            credibility_json = json.dumps(paper.credibility.to_dict()) if paper.credibility else None

            cursor.execute("""
                INSERT OR REPLACE INTO papers (
                    id, doi, title, abstract, authors, year, citations,
                    url, pdf_url, source, has_pdf, pmc_id, full_text,
                    extracted_from, credibility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.id,
                paper.doi,
                paper.title,
                paper.abstract,
                authors_json,
                paper.year,
                paper.citations,
                paper.url,
                paper.pdf_url,
                paper.source,
                1 if paper.has_pdf else 0,
                paper.pmc_id,
                full_text_json,
                paper.extracted_from,
                credibility_json
            ))

        self.conn.commit()

    def get_paper(self, paper_id: str) -> Optional["Paper"]:
        """Retrieve a paper by ID.

        Args:
            paper_id: Paper identifier

        Returns:
            Paper object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_paper(row)

    def get_all_papers(self) -> Dict[str, "Paper"]:
        """Retrieve all papers from the database.

        Returns:
            Dictionary mapping paper_id to Paper object
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers")
        rows = cursor.fetchall()

        papers = {}
        for row in rows:
            paper = self._row_to_paper(row)
            papers[paper.id] = paper

        return papers

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from the database.

        Args:
            paper_id: Paper identifier

        Returns:
            True if paper was deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        self.conn.commit()

        return cursor.rowcount > 0

    def count_papers(self) -> int:
        """Get total number of papers in database.

        Returns:
            Number of papers
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        return cursor.fetchone()[0]

    def _row_to_paper(self, row: sqlite3.Row) -> "Paper":
        """Convert a database row to a Paper object.

        Args:
            row: SQLite row object

        Returns:
            Paper object
        """
        # Import at runtime to avoid circular imports
        from scverifier.data.models import Paper, CredibilityScores

        # Deserialize JSON fields
        authors = json.loads(row['authors']) if row['authors'] else []
        full_text = json.loads(row['full_text']) if row['full_text'] else []

        credibility = None
        if row['credibility']:
            credibility_dict = json.loads(row['credibility'])
            credibility = CredibilityScores.from_dict(credibility_dict)

        # Note: chunks and propositions are NOT stored in the database
        # They live in FAISS vector stores and will be empty here
        paper = Paper(
            id=row['id'],
            doi=row['doi'],
            title=row['title'],
            abstract=row['abstract'],
            authors=authors,
            year=row['year'],
            citations=row['citations'] if row['citations'] else 0,
            url=row['url'] if row['url'] else "",
            pdf_url=row['pdf_url'] if row['pdf_url'] else "",
            source=row['source'] if row['source'] else "",
            has_pdf=bool(row['has_pdf']),
            pmc_id=row['pmc_id'],
            full_text=full_text,
            extracted_from=row['extracted_from'] if row['extracted_from'] else "abstract",
            credibility=credibility,
            chunks=[],  # Will be populated from FAISS vector store
            propositions=[]  # Will be populated from FAISS vector store
        )

        return paper

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

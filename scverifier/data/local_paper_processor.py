import os
import re
from typing import List, Optional

from scverifier.data.models import Paper
from scverifier.data.simple_pdf_extractors import (
    extract_with_pymupdf,
    extract_with_pdfplumber,
    extract_with_pypdf,
    extract_with_marker,
)


class LocalPaperProcessor:
    """Process local paper files and return domain `Paper` objects.

    This wraps the previous helper functions and provides a small, well-defined
    API used elsewhere in the codebase: `extract_from_file` and
    `create_demo_paper`.
    """

    def __init__(self, extraction_method: str = "pymupdf") -> None:
        """Initialize processor with extraction method.

        Args:
            extraction_method: One of "pymupdf", "marker", "pdfplumber", "pypdf"
                - pymupdf: Fast (~0.12s), excellent quality (RECOMMENDED, DEFAULT)
                - marker: Slow (~11s) but highest quality
                - pdfplumber: Fast (~0.10s), great for tables
                - pypdf: Ultra fast (<0.05s), basic text
        """
        self.extraction_method = extraction_method

    def extract_from_file(
        self,
        pdf_path: str,
        text_content: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> Paper:
        """Extract metadata and sections from a local file and return a `Paper`.

        Args:
            pdf_path: path to file (pdf, txt, md)
            text_content: optional pre-loaded text (used for non-pdf files or when
                FileLoader already provided the text). If omitted for PDFs,
                the configured extraction method will be used.
            original_filename: optional original filename to use for paper ID
        """
        markdown_content = ""
        pdf_metadata = {}
        pages_data = None  # For page-by-page extraction

        ext = os.path.splitext(pdf_path)[1].lower()
        if ext == ".pdf" and text_content is None:
            # Always use page-aware extraction for PDFs
            if self.extraction_method == "pymupdf":
                pages_data, pdf_metadata = extract_with_pymupdf(pdf_path)
            elif self.extraction_method == "marker":
                pages_data, pdf_metadata = extract_with_marker(pdf_path)
            elif self.extraction_method == "pdfplumber":
                pages_data, pdf_metadata = extract_with_pdfplumber(pdf_path)
            elif self.extraction_method == "pypdf":
                pages_data, pdf_metadata = extract_with_pypdf(pdf_path)
            else:
                raise ValueError(f"Unknown extraction method: {self.extraction_method}")

            # Join pages for metadata extraction
            markdown_content = "\n\n".join([p["text"] for p in pages_data])
        else:
            # Use provided text_content (fall back to file read)
            if text_content:
                markdown_content = text_content
            else:
                try:
                    with open(pdf_path, "r", encoding="utf-8") as fh:
                        markdown_content = fh.read()
                except Exception:
                    markdown_content = ""

        # Store content with page information for PDFs
        if pages_data:
            # Store each page separately with page numbers for better retrieval
            sections = [(f"page_{p['page']}", p["text"]) for p in pages_data]
        else:
            # Store all content as a single full_text entry (for non-PDF files)
            sections = [("full_text", markdown_content.strip())] if markdown_content.strip() else []

        # Extract title, authors, year, id, doi (skip abstract extraction)
        title, authors_list, year, paper_id, doi = _extract_metadata_from_text(
            markdown_content, pdf_metadata, pdf_path, original_filename
        )

        # Save raw markdown for manual inspection
        _save_raw_markdown(paper_id, markdown_content)

        paper = Paper(
            id=paper_id,
            doi=doi,
            title=title,
            abstract="",  # No abstract extraction for local papers
            authors=authors_list,
            year=int(year) if year and str(year).isdigit() else None,
            source="local",
            has_pdf=(ext == ".pdf"),
            full_text=sections,
        )

        return paper

    def create_demo_paper(self) -> Paper:
        """Return a small demo Paper instance for testing/demo runs."""
        return Paper(
            id="demo_paper",
            doi=None,
            title="Demo Paper: On Example Extraction",
            abstract="This is a demo abstract used for pipeline testing.",
            authors=["Jane Doe", "John Smith"],
            year=2024,
            source="local",
            has_pdf=False,
            full_text=[("abstract", "This is a demo abstract used for pipeline testing.")],
        )


def _save_raw_markdown(paper_id: str, markdown_content: str) -> None:
    """Save raw markdown content to file for manual inspection."""
    try:
        # Create markdown directory if it doesn't exist
        markdown_dir = os.path.join("data", "output", "markdown")
        os.makedirs(markdown_dir, exist_ok=True)

        # Save markdown file
        markdown_path = os.path.join(markdown_dir, f"{paper_id}.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Saved raw markdown to: {markdown_path}")
    except Exception as e:
        print(f"Warning: Could not save raw markdown for {paper_id}: {e}")


def _extract_metadata_from_text(
    markdown_text: str, pdf_metadata: dict, pdf_path: str, original_filename: Optional[str] = None
):
    """
    Simplified metadata extraction logic:
    - Title = first non-empty line
    - Authors = from PDF metadata
    - Year = first 4-digit year found or metadata fallback
    - DOI = attempt to detect common DOI patterns
    - ID = sanitized filename without extension

    Note: Abstract extraction is skipped; all text is stored in full_text field.
    """
    lines = [line.strip() for line in markdown_text.split("\n") if line.strip()]
    title = lines[0] if lines else ""
    authors: List[str] = []
    year: Optional[str] = None
    doi: Optional[str] = None

    # Extract authors from PDF metadata
    meta_authors = pdf_metadata.get("authors", "")
    if isinstance(meta_authors, (list, tuple)):
        authors = list(meta_authors)
    elif isinstance(meta_authors, str) and meta_authors:
        authors = [a.strip() for a in re.split(r",|;", meta_authors) if a.strip()]
    else:
        authors = []

    # Extract publication year
    match = re.search(r"\b(19|20)\d{2}\b", markdown_text)
    if match:
        year = match.group(0)
    else:
        year = str(pdf_metadata.get("year", "")) if pdf_metadata.get("year") else None

    # Extract DOI: common patterns like doi:10.1234/abc or https://doi.org/... or dx.doi.org
    doi_match = re.search(r"doi:\s*(10\.\d{4,9}/[^\s\n\)\"]+)", markdown_text, re.I)
    if not doi_match:
        doi_match = re.search(r"https?://doi\.org/(10\.\d{4,9}/[^\s\n\)\"]+)", markdown_text, re.I)
    if not doi_match:
        doi_match = re.search(r"dx\.doi\.org/(10\.\d{4,9}/[^\s\n\)\"]+)", markdown_text, re.I)

    if doi_match:
        doi = doi_match.group(1).rstrip(".\n")
    else:
        # fallback to metadata
        meta_doi = pdf_metadata.get("doi") or pdf_metadata.get("DOI")
        doi = meta_doi if meta_doi else None

    # Use filename as ID - prefer original filename if provided, otherwise use pdf_path
    if original_filename:
        filename_for_id = original_filename
    else:
        filename_for_id = os.path.basename(pdf_path)

    # Remove extension and sanitize: replace spaces with underscores, remove special chars
    paper_id = os.path.splitext(filename_for_id)[0]
    paper_id = re.sub(r"[^\w\s-]", "", paper_id)  # Remove special chars except spaces, hyphens, underscores
    paper_id = re.sub(r"\s+", "_", paper_id)  # Replace spaces with underscores
    paper_id = re.sub(r"_+", "_", paper_id)  # Replace multiple underscores with single
    paper_id = paper_id.strip("_")  # Remove leading/trailing underscores

    return title, authors, year, paper_id, doi

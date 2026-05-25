"""Simple PDF extraction functions - one function per method.

Each function extracts text from a PDF and returns page-by-page data.
No complex abstractions, just straightforward extraction.
"""

from typing import Tuple, List, Dict


def extract_with_pymupdf(pdf_path: str) -> Tuple[List[Dict], dict]:
    """Extract PDF page-by-page using PyMuPDF, preserving page numbers.

    This preserves page information for each chunk and proposition for better retrieval.

    Speed: ~0.12s per document
    Install: pip install pymupdf

    Returns:
        Tuple of:
        - List of dicts with 'text' and 'page' keys for each page
        - Metadata dict with title, author, num_pages
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()  # Extract text from this page

        if text.strip():  # Only include pages with content
            pages_data.append({
                "text": text.strip(),
                "page": page_num + 1  # 1-indexed for users
            })

    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "num_pages": len(doc),
    }

    doc.close()

    return pages_data, metadata


def extract_with_pdfplumber(pdf_path: str) -> Tuple[List[Dict], dict]:
    """Extract PDF page-by-page using pdfplumber, preserving page numbers.

    This preserves page information for each chunk and proposition for better retrieval.

    Speed: ~0.10s per document
    Install: pip install pdfplumber

    Returns:
        Tuple of:
        - List of dicts with 'text' and 'page' keys for each page
        - Metadata dict with num_pages
    """
    import pdfplumber

    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        metadata = {
            "num_pages": len(pdf.pages),
        }

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():  # Only include pages with content
                pages_data.append({
                    "text": text.strip(),
                    "page": page_num + 1  # 1-indexed for users
                })

    return pages_data, metadata


def extract_with_pypdf(pdf_path: str) -> Tuple[List[Dict], dict]:
    """Extract PDF page-by-page using pypdf, preserving page numbers.

    This preserves page information for each chunk and proposition for better retrieval.

    Speed: <0.05s per document
    Install: pip install pypdf

    Returns:
        Tuple of:
        - List of dicts with 'text' and 'page' keys for each page
        - Metadata dict with num_pages
    """
    import pypdf

    pages_data = []

    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)

        metadata = {
            "num_pages": len(reader.pages),
        }

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # Only include pages with content
                pages_data.append({
                    "text": text.strip(),
                    "page": page_num + 1  # 1-indexed for users
                })

    return pages_data, metadata


def extract_with_marker(pdf_path: str) -> Tuple[List[Dict], dict]:
    """Extract PDF using marker as a single page.

    Marker doesn't provide native page-by-page extraction, so we return
    the full text as a single page to maintain consistency with other extractors.

    Speed: ~11.3s per document
    Install: pip install marker-pdf

    Returns:
        Tuple of:
        - List with a single dict containing 'text' and 'page' keys
        - Metadata dict
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    artifact_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=artifact_dict)

    rendered = converter(pdf_path)
    markdown = rendered.markdown
    metadata = getattr(rendered, "metadata", {}) or {}

    # Return as a single "page" since marker doesn't provide page splits
    pages_data = [{
        "text": markdown,
        "page": 1
    }]

    return pages_data, metadata

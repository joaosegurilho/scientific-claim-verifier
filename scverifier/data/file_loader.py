"""File loading utilities for various document formats."""

import os
from typing import List
from langchain_core.documents import Document
import PyPDF2
import fitz  # PyMuPDF - alternative for better text extraction


class FileLoader:
    """Handles loading and processing of various file formats."""

    @staticmethod
    def load_pdf_with_pages(file_path: str) -> List[Document]:
        """Load PDF and create documents with page metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        documents = []
        filename = os.path.basename(file_path)

        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()

                if text.strip():  # Only add pages with content
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,  # 1-indexed page numbers
                        "total_pages": len(doc),
                        "file_path": file_path,
                    }

                    documents.append(Document(page_content=text, metadata=metadata))

            doc.close()

        except ImportError:
            # Fallback to PyPDF2 if PyMuPDF not available
            print("PyMuPDF not available, falling back to PyPDF2...")

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()

                    if text.strip():  # Only add pages with content
                        metadata = {
                            "source": filename,
                            "page": page_num + 1,  # 1-indexed page numbers
                            "total_pages": total_pages,
                            "file_path": file_path,
                        }

                        documents.append(Document(page_content=text, metadata=metadata))

        if not documents:
            raise ValueError(f"No readable text found in {file_path}")

        return documents

    @staticmethod
    def load_text_file(file_path: str) -> Document:
        """Load a plain text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": "text",
        }

        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """Detect file type based on extension."""
        _, ext = os.path.splitext(file_path.lower())

        if ext == ".pdf":
            return "pdf"
        elif ext in [".txt", ".md"]:
            return "text"
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """Auto-detect file type and load appropriately."""
        file_type = FileLoader.detect_file_type(file_path)

        if file_type == "pdf":
            return FileLoader.load_pdf_with_pages(file_path)
        elif file_type == "text":
            return [FileLoader.load_text_file(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

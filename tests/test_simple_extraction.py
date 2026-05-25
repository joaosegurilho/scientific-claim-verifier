"""Simple tests for PDF extraction methods.

Tests each extraction method, times it, and saves output.
Usage: python tests/test_simple_extraction.py
"""

import sys
import time
from pathlib import Path

from scverifier.data.simple_pdf_extractors import (
    extract_with_pymupdf,
    extract_with_pdfplumber,
    extract_with_pypdf,
    extract_with_marker,
)

# Test PDF
TEST_PDF = Path(__file__).parent.parent.parent / "literature" / "papers" / "TEST.pdf"
OUTPUT_DIR = Path(__file__).parent / "data" / "output" / "simple_extraction_tests"


def test_pymupdf():
    """Test PyMuPDF extraction."""
    print("\n" + "=" * 80)
    print("Testing: pymupdf")
    print("=" * 80)

    try:
        start = time.time()
        markdown, metadata = extract_with_pymupdf(str(TEST_PDF))
        elapsed = time.time() - start

        # Save output
        output_file = OUTPUT_DIR / "pymupdf_output.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")

        print("Status: SUCCESS")
        print(f"Time: {elapsed:.3f}s")
        print(f"Characters: {len(markdown):,}")
        print(f"Output: {output_file}")

        return elapsed

    except ImportError as e:
        print(f"Status: SKIPPED - {e}")
        print("Install with: pip install pymupdf4llm")
        return None

    except Exception as e:
        print(f"Status: FAILED - {e}")
        return None


def test_pdfplumber():
    """Test pdfplumber extraction."""
    print("\n" + "=" * 80)
    print("Testing: pdfplumber")
    print("=" * 80)

    try:
        start = time.time()
        markdown, metadata = extract_with_pdfplumber(str(TEST_PDF))
        elapsed = time.time() - start

        # Save output
        output_file = OUTPUT_DIR / "pdfplumber_output.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")

        print("Status: SUCCESS")
        print(f"Time: {elapsed:.3f}s")
        print(f"Characters: {len(markdown):,}")
        print(f"Output: {output_file}")

        return elapsed

    except ImportError as e:
        print(f"Status: SKIPPED - {e}")
        print("Install with: pip install pdfplumber")
        return None

    except Exception as e:
        print(f"Status: FAILED - {e}")
        return None


def test_pypdf():
    """Test pypdf extraction."""
    print("\n" + "=" * 80)
    print("Testing: pypdf")
    print("=" * 80)

    try:
        start = time.time()
        markdown, metadata = extract_with_pypdf(str(TEST_PDF))
        elapsed = time.time() - start

        # Save output
        output_file = OUTPUT_DIR / "pypdf_output.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")

        print("Status: SUCCESS")
        print(f"Time: {elapsed:.3f}s")
        print(f"Characters: {len(markdown):,}")
        print(f"Output: {output_file}")

        return elapsed

    except ImportError as e:
        print(f"Status: SKIPPED - {e}")
        print("Install with: pip install pypdf")
        return None

    except Exception as e:
        print(f"Status: FAILED - {e}")
        return None


def test_marker():
    """Test marker extraction."""
    print("\n" + "=" * 80)
    print("Testing: marker")
    print("=" * 80)

    try:
        start = time.time()
        markdown, metadata = extract_with_marker(str(TEST_PDF))
        elapsed = time.time() - start

        # Save output
        output_file = OUTPUT_DIR / "marker_output.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")

        print("Status: SUCCESS")
        print(f"Time: {elapsed:.3f}s")
        print(f"Characters: {len(markdown):,}")
        print(f"Output: {output_file}")

        return elapsed

    except ImportError as e:
        print(f"Status: SKIPPED - {e}")
        print("Install with: pip install marker-pdf")
        return None

    except Exception as e:
        print(f"Status: FAILED - {e}")
        return None


def main():
    """Run all tests."""
    print("=" * 80)
    print("PDF EXTRACTION METHOD TESTS")
    print("=" * 80)
    print(f"\nTest PDF: {TEST_PDF}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not TEST_PDF.exists():
        print(f"\nERROR: Test PDF not found at {TEST_PDF}")
        sys.exit(1)

    # Run all tests
    results = {}
    results["pymupdf"] = test_pymupdf()
    results["pdfplumber"] = test_pdfplumber()
    results["pypdf"] = test_pypdf()
    results["marker"] = test_marker()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [(name, time) for name, time in results.items() if time is not None]

    if successful:
        # Sort by time
        successful.sort(key=lambda x: x[1])

        print(f"\n{'Method':<15} {'Time':<12} {'Status'}")
        print("-" * 80)

        for name, elapsed in successful:
            print(f"{name:<15} {elapsed:.3f}s      SUCCESS")

        fastest = successful[0]
        slowest = successful[-1]

        print(f"\nFastest: {fastest[0]} ({fastest[1]:.3f}s)")
        if len(successful) > 1:
            print(f"Slowest: {slowest[0]} ({slowest[1]:.3f}s)")
            print(f"Speed difference: {slowest[1] / fastest[1]:.1f}x")

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()

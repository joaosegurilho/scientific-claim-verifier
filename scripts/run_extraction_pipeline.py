#!/usr/bin/env python3
"""
Extraction Pipeline Script

Extract propositions from local files or folders and store them in the knowledge base.

Usage:
    python run_extraction_pipeline.py                           # Process demo paper
    python run_extraction_pipeline.py data/demo_paper.pdf       # Process single file
    python run_extraction_pipeline.py folder/                   # Process all files in folder
    python run_extraction_pipeline.py file1.pdf file2.pdf       # Process multiple files
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scverifier.core.extraction.proposition_extractor import PropositionExtractor
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.data.file_loader import FileLoader
from scverifier.data.local_paper_processor import LocalPaperProcessor
from scverifier.config.settings import Config


def process_file(
    file_path: str, extractor: PropositionExtractor, local_paper_processor: LocalPaperProcessor, kb: KnowledgeBase
):
    """Process a single file.

    Args:
        file_path: Path to the file to process
        extractor: PropositionExtractor instance
        local_paper_processor: LocalPaperProcessor instance
        kb: KnowledgeBase instance
    """
    try:
        print(f"\n{'='*70}")
        print(f" Processing: {file_path}")
        print("=" * 70)

        # Load file content
        loader = FileLoader()
        documents = loader.load_file(file_path)

        # Combine all document content
        content = "\n\n".join(doc.page_content for doc in documents)

        # Extract metadata and create Paper object
        paper = local_paper_processor.extract_from_file(file_path, content)

        print(" Paper Info:")
        print(f"   Title: {paper.title}")
        print(f"   ID: {paper.id}")
        print(f"   Year: {paper.year or 'Unknown'}")
        print(f"   DOI: {paper.doi or 'None'}")
        print(f"   Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}")

        # Check if already processed with propositions
        existing_paper = kb.get_paper(paper.id)
        if existing_paper and existing_paper.propositions:
            print(f"Paper already processed with {len(existing_paper.propositions)} propositions")
            quality_props = len(existing_paper.get_quality_propositions())
            print(f"   Quality propositions: {quality_props}")
            print("   Skipping re-extraction to preserve existing data...")
            return

        # If paper exists but has no propositions, reprocess it
        if existing_paper:
            print(" Paper exists but has no propositions. Reprocessing...")
            kb.delete_paper(paper.id)

        # Extract propositions (use full text if abstract is not available)
        print("\n Extracting propositions...")
        use_full_text = not paper.abstract or len(paper.abstract.strip()) == 0
        if use_full_text and paper.full_text:
            print("   Using full text for extraction (no abstract available)")
        extractor.extract_from_paper(paper, show_steps=True, use_full_text=use_full_text)

        # Add to knowledge base
        print("\n Adding to knowledge base...")
        kb.add_paper(paper, verbose=True)

        # Show statistics
        stats = paper.get_statistics()
        print("\n Extraction complete!")
        print(f"   Total propositions: {stats['propositions_total']}")
        print(f"   Quality propositions: {stats['propositions_quality']}")
        print(f"   Success rate: {stats['success_rate']*100:.1f}%")

    except Exception as e:
        print(f" Error processing {file_path}: {e}")
        traceback.print_exc()


def get_demo_paper_path() -> str:
    """Get the path to the demo paper.

    Returns:
        Path to the demo paper PDF
    """
    # Get the project root directory (parent of scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    demo_path = project_root / "data" / "demo_paper.pdf"

    return str(demo_path)


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print(" EXTRACTION PIPELINE")
    print("=" * 70)

    # Initialize components
    print("\n Initializing components...")
    extractor = PropositionExtractor()
    local_paper_processor = LocalPaperProcessor()
    kb = KnowledgeBase()

    # ALWAYS load existing knowledge base first to preserve data
    print("\n Loading existing knowledge base...")
    try:
        kb.load()
        print("   Loaded existing knowledge base")
    except FileNotFoundError:
        print("     No existing knowledge base found. Starting fresh.")
    except Exception as e:
        print(f"     Error loading knowledge base: {e}")
        print("   Starting with empty knowledge base.")

    # Get files to process
    if len(sys.argv) == 1:
        # No arguments - process demo paper
        demo_path = get_demo_paper_path()

        if not os.path.exists(demo_path):
            print(f"\n Error: Demo paper not found at {demo_path}")
            print("   Please ensure data/demo_paper.pdf exists.")
            print("   Or provide a file path as an argument.")
            return

        print(f"\n No files specified. Processing demo paper: {demo_path}")
        process_file(demo_path, extractor, local_paper_processor, kb)
    else:
        # Process provided files/folders
        paths = sys.argv[1:]
        files_to_process = []

        for path in paths:
            if os.path.isfile(path):
                files_to_process.append(path)
            elif os.path.isdir(path):
                # Get all PDF and TXT files in directory
                for ext in ["*.pdf", "*.txt", "*.md"]:
                    files_to_process.extend(Path(path).glob(ext))
            else:
                print(f"  Path not found: {path}")

        if not files_to_process:
            print(" No valid files found to process.")
            return

        print(f"\n Found {len(files_to_process)} file(s) to process")

        # Process each file
        for file_path in files_to_process:
            process_file(str(file_path), extractor, local_paper_processor, kb)

    # Save knowledge base
    print(f"\n{'='*70}")
    print(" Saving knowledge base...")
    kb.save()

    # Show final statistics
    print(f"\n{'='*70}")
    print(" FINAL STATISTICS")
    print("=" * 70)
    kb.print_statistics()

    print(f"\n{'='*70}")
    print(" Extraction pipeline complete!")
    print(f" Knowledge base saved to: {Config.DB_NAME}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

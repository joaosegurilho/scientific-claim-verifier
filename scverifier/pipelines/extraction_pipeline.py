"""Stateless extraction pipeline"""

import logging
import traceback
from pathlib import Path

from scverifier.config.settings import Config
from scverifier.core.extraction.proposition_extractor import PropositionExtractor
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.data.file_loader import FileLoader
from scverifier.data.local_paper_processor import LocalPaperProcessor

logger = logging.getLogger()


# TODO: change FileLoader to handle Path objects
class ExtractionPipeline:
    def __init__(self, arg_path: Path):
        """Initialize the extraction pipeline.

        Args:
            arg_path: Path to file or folder to process
        """
        self._path = arg_path

    def __call__(self):
        """Call the instance to run the pipeline."""
        print("\n" + "=" * 70)
        print(" EXTRACTION PIPELINE")
        print("=" * 70)

        print("\n Initializing components...")
        self._init_components()
        print("\n Loading existing knowledge base...")
        self._start_kb()

        if self._path.is_dir():
            self.process_folder(self._path)
        else:
            self.process_file(self._path)

        print(f"\n{'=' * 70}")
        print(" Saving knowledge base...")
        self.kb.save()

        print(f"\n{'=' * 70}")
        print(" FINAL STATISTICS")
        print("=" * 70)
        self.kb.print_statistics()

        print(f"\n{'=' * 70}")
        print(" Extraction pipeline complete!")
        print(f" Knowledge base saved to: {Config.DB_NAME}")
        print("=" * 70 + "\n")

    def _init_components(self):
        self.kb = KnowledgeBase()
        self.extractor = PropositionExtractor()
        self.local_paper_processor = LocalPaperProcessor()

    def process_file(self, file_path: str | Path):
        file_path = str(file_path.resolve()) if isinstance(file_path, Path) else file_path

        try:
            # Load file
            loader = FileLoader()
            documents = loader.load_file(file_path)

            # Combine all doc content
            content = "\n\n".join(doc.page_content for doc in documents)

            # Extract metadata and create Paper object
            paper = self.local_paper_processor.extract_from_file(file_path, content)

            print(" Paper Info:")
            print(f"\tTitle: {paper.title}")
            print(f"\tID: {paper.id}")
            print(f"\tYear: {paper.year or 'Unknown'}")
            print(f"\tDOI: {paper.doi or 'None'}")
            print(f"\tAuthors: {', '.join(paper.authors) if paper.authors else 'Unknown'}")

            # Check if already processed with propositions
            existing_paper = self.kb.get_paper(paper.id)
            if existing_paper and existing_paper.propositions:
                print(f"Paper already processed with {len(existing_paper.propositions)} propositions")
                quality_props = len(existing_paper.get_quality_propositions())
                print(f"\tQuality propositions: {quality_props}")
                print("\tSkipping re-extraction to preserve existing data...")
                return

            # If paper exists but has no propositions, reprocess it
            if existing_paper:
                print(" Paper exists but has no propositions. Reprocessing...")
                self.kb.delete_paper(paper.id)

            # Extract propositions (use full text if abstract is not available)
            print("\n Extracting propositions...")
            use_full_text = not paper.abstract or len(paper.abstract.strip()) == 0
            if use_full_text and paper.full_text:
                print("\tUsing full text for extraction (no abstract available)")
            self.extractor.extract_from_paper(paper, show_steps=True, use_full_text=use_full_text)

            # Add to knowledge base
            print("\n Adding to knowledge base...")
            self.kb.add_paper(paper, verbose=True)

            # Show statistics
            stats = paper.get_statistics()
            print("\n Extraction complete!")
            print(f"\tTotal propositions: {stats['propositions_total']}")
            print(f"\tQuality propositions: {stats['propositions_quality']}")
            print(f"\tSuccess rate: {stats['success_rate'] * 100:.1f}%")

        except Exception as e:
            print(f" Error processing {file_path}: {e}")
            traceback.print_exc()

    def process_folder(self, folder_path: Path):
        files_to_process = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            files_to_process.extend(folder_path.glob(ext))

        if not files_to_process:
            print(" No valid files found to process.")
            return

        print(f"\n Found {len(files_to_process)} file(s) to process")

        for file in files_to_process:
            self.process_file(file)

    def _start_kb(self):
        try:
            self.kb.load()
            print("   Loaded existing knowledge base")
        except FileNotFoundError:
            print("     No existing knowledge base found. Starting fresh.")
        except Exception as e:
            print(f"     Error loading knowledge base: {e}")
            print("   Starting with empty knowledge base.")

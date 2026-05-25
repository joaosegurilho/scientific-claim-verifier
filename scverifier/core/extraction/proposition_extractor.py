"""Simplified proposition extraction component - stateless and modular."""

from typing import Dict, Any, List, Optional

from scverifier.core.processing.document_processor import DocumentProcessor
from scverifier.core.extraction.proposition_generator import PropositionGenerator
from scverifier.core.extraction.proposition_evaluator import PropositionEvaluator
from scverifier.data.models import Paper


class PropositionExtractor:
    """Stateless extractor for converting text into evaluated propositions.

    This component focuses solely on the extraction process and doesn't manage
    papers, vector stores, or persistence. It takes text as input and returns
    structured extraction results.

    Works entirely with domain objects (Chunk, Proposition, Paper) - no LangChain Documents.
    """

    def __init__(self, skip_evaluation: bool = False):
        """Initialize the extractor.

        Args:
            skip_evaluation: If True, skip LLM evaluation and accept all propositions
        """
        self.document_processor = DocumentProcessor()
        self.proposition_generator = PropositionGenerator()
        self.proposition_evaluator = PropositionEvaluator()
        self.skip_evaluation = skip_evaluation

    def extract(self, text: str, metadata: Optional[Dict[str, Any]] = None, show_steps: bool = False) -> Dict[str, Any]:
        """Extract propositions from a single document text.

        This is the main interface for extraction. It performs:
        1. Document chunking
        2. Proposition generation from chunks
        3. Quality evaluation (unless skip_evaluation=True)
        4. Filtering to quality propositions

        Args:
            text: Document text to process
            metadata: Optional metadata (paper_id, source, title, etc.)
            show_steps: Whether to print progress messages

        Returns:
            Dictionary with:
                - chunks: List of Chunk objects
                - propositions_all: All generated Proposition objects
                - propositions_quality: Only propositions that passed quality checks
        """
        if show_steps:
            print(" Extracting propositions...")

        # Ensure metadata has required fields
        if metadata is None:
            metadata = {}

        # Ensure paper_id exists (needed for chunk_id format)
        if "paper_id" not in metadata:
            # Generate a paper_id from source or use generic
            if "source" in metadata:
                # Use first 20 chars of source as paper_id
                metadata["paper_id"] = metadata["source"][:20].replace(" ", "_")
            else:
                metadata["paper_id"] = "unknown"

        # Step 1: Chunk the text into Chunk objects
        if show_steps:
            print("    Step 1: Chunking document...")

        chunks = self.document_processor.chunk(text, metadata)

        if show_steps:
            print(f"      Created {len(chunks)} chunks")

        # Step 2: Generate propositions from chunks
        if show_steps:
            print("    Step 2: Generating propositions...")

        raw_propositions = self.proposition_generator.generate_propositions_from_chunks(chunks)

        if show_steps:
            print(f"      Generated {len(raw_propositions)} propositions")

        # Step 3: Evaluate propositions (or skip if disabled)
        if self.skip_evaluation:
            if show_steps:
                print("    Step 3: Skipping evaluation (all propositions accepted)")

            propositions_all = raw_propositions
            propositions_quality = raw_propositions
        else:
            if show_steps:
                print("     Step 3: Evaluating proposition quality...")

            # Evaluate propositions using Chunk objects
            evaluated_propositions = self.proposition_evaluator.evaluate_propositions(raw_propositions, chunks)

            if show_steps:
                passed = sum(1 for p in evaluated_propositions if p.is_high_quality())
                print(f"      {passed}/{len(evaluated_propositions)} propositions passed quality checks")

            propositions_all = evaluated_propositions
            propositions_quality = [p for p in evaluated_propositions if p.is_high_quality()]

        return {
            "chunks": chunks,
            "propositions_all": propositions_all,
            "propositions_quality": propositions_quality,
        }

    def extract_from_paper(self, paper: Paper, show_steps: bool = False, use_full_text: bool = False) -> Paper:
        """Extract propositions from a Paper object's full_text sections (or abstract if no full text).

        This is a convenience method that updates the paper object in-place
        with extracted chunks and propositions from all sections.

        Args:
            paper: Paper object with full_text or abstract to process
            show_steps: Whether to print progress messages
            use_full_text: If True, extract from full_text. If False, extract from abstract only. Default False.

        Returns:
            The same Paper object, now with chunks and propositions populated
        """
        all_chunks = []
        all_propositions = []

        # Announce what we're about to extract (full text vs abstract)
        if show_steps:
            intended = "full text" if use_full_text else "abstract"
            print(
                f" Extracting propositions from paper '{paper.title[:60]}' (id={paper.id}) — intended source: {intended}"
            )

        # Determine which text to process
        if use_full_text and paper.full_text:
            # Process each section of full_text
            if show_steps:
                print(f"  Using full text for extraction: found {len(paper.full_text)} sections")

            for section_name, section_text in paper.full_text:
                if not section_text.strip():
                    continue

                if show_steps:
                    print(f"    Processing section: {section_name} (paper id={paper.id})")

                # Extract page number if section name is "page_N"
                page_num = None
                if section_name.startswith("page_"):
                    try:
                        page_num = int(section_name.split("_")[1])
                    except (IndexError, ValueError):
                        pass

                # Prepare metadata with section and page information
                metadata = {
                    "paper_id": paper.id,
                    "source": paper.title,
                    "section": section_name,
                    "page": page_num,
                    "year": paper.year,
                    "citations": paper.citations,
                }

                # Chunk the section text (chunks get unique IDs from global counter)
                section_chunks = self.document_processor.chunk(section_text, metadata)

                all_chunks.extend(section_chunks)

                # Generate propositions from these chunks
                raw_propositions = self.proposition_generator.generate_propositions_from_chunks(section_chunks)
                
                # Evaluate propositions (or skip if disabled) - consistent with abstract extraction
                if self.skip_evaluation:
                    section_propositions = raw_propositions
                else:
                    section_propositions = self.proposition_evaluator.evaluate_propositions(raw_propositions, section_chunks)
                
                all_propositions.extend(section_propositions)

                if show_steps:
                    if self.skip_evaluation:
                        print(f"      Generated {len(section_propositions)} propositions from {len(section_chunks)} chunks (evaluation skipped)")
                    else:
                        passed = sum(1 for p in section_propositions if p.is_high_quality())
                        print(f"      Generated {len(section_propositions)} propositions from {len(section_chunks)} chunks ({passed} passed quality checks)")

        else:
            # Fallback to abstract if no full text available
            if show_steps:
                if use_full_text and not paper.full_text:
                    print("  Requested full-text extraction, but no full text available — falling back to abstract.")
                else:
                    print("  Using abstract for extraction...")

            metadata = {
                "paper_id": paper.id,
                "source": paper.title,
                "section": "abstract",
                "year": paper.year,
                "citations": paper.citations,
            }

            # Extract from abstract only - no fallback to full_text
            text_to_extract = paper.abstract
            if not text_to_extract or not text_to_extract.strip():
                # No abstract available - skip extraction
                if show_steps:
                    print(f"  Warning: Paper '{paper.id}' has no abstract - skipping proposition extraction")
                    print(f"    (Use use_full_text=True to extract from full text instead)")
                all_chunks = []
                all_propositions = []
            else:
                result = self.extract(text_to_extract, metadata, show_steps=False)
                all_chunks = result["chunks"]
                all_propositions = result["propositions_all"]

        # Update paper object
        paper.chunks = all_chunks
        paper.propositions = all_propositions

        # Track what was used for extraction
        if use_full_text and paper.full_text:
            paper.extracted_from = "full_text"
        else:
            paper.extracted_from = "abstract"

        return paper

    def extract_from_papers(
        self, papers: List[Paper], show_steps: bool = False, use_full_text: bool = False
    ) -> List[Paper]:
        """Extract propositions from multiple papers.

        Args:
            papers: List of Paper objects
            show_steps: Whether to print progress messages
            use_full_text: If True, extract from full_text. If False, extract from abstract only. Default False.

        Returns:
            List of updated Paper objects with propositions
        """
        if show_steps:
            print(f"\n Extracting from {len(papers)} papers...")

        for i, paper in enumerate(papers):
            if show_steps:
                source_msg = "full_text" if use_full_text and paper.full_text else "abstract"
                print(f"\n   Paper {i+1}/{len(papers)}: {paper.title[:60]}... (extracting from {source_msg})")

            # Call extractor; let extract_from_paper control its own detailed prints
            self.extract_from_paper(paper, show_steps=show_steps, use_full_text=use_full_text)

            if show_steps:
                quality = len(paper.get_quality_propositions())
                print(f"      Extracted {len(paper.propositions)} total propositions")
                print(f"      Extracted {quality} quality propositions")

        return papers

    def _combine_full_text_sections(self, full_text: List, max_chars: int = 2000) -> str:
        """Combine full_text sections into a single string up to max_chars.

        Args:
            full_text: List of (section_name, content) tuples
            max_chars: Maximum characters to return (default 2000)

        Returns:
            Combined text from sections, truncated to max_chars
        """
        if not full_text:
            return ""

        combined = []
        total_chars = 0

        for section_name, content in full_text:
            if not content:
                continue

            # Add section content
            if total_chars + len(content) > max_chars:
                # Truncate to fit within max_chars
                remaining = max_chars - total_chars
                combined.append(content[:remaining])
                break
            else:
                combined.append(content)
                total_chars += len(content)

        return "\n\n".join(combined)

"""Paper credibility scoring system with simple 1-5 rating."""

from typing import List, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from scverifier.data.models import Paper, CredibilityScores, PaperMetadata
from scverifier.config.settings import Config


class PaperScorer:
    """Scores paper credibility based on source quality, study type, and other factors.

    This scorer evaluates papers using:
    - Study type (meta-analysis, RCT, case report, etc.) - extracted via LLM
    - Methodology characteristics (sample size, blinding, randomization, etc.)
    - Citation counts (highly cited papers get bonus points)
    - Open access availability (+0.5 if PDF URL available)
    - Full-text content (+0.5 if full text extracted)
    - Recency (very old papers get penalties)

    Outputs a simple 1-5 rating that's easy to understand and use.
    """

    # Study types: (score, description)
    # Higher quality study designs receive higher scores (1-5 stars)
    STUDY_TYPES = {
        "meta_analysis": (4, "Systematic review that combines results from multiple studies statistically"),
        "systematic_review": (4, "Comprehensive review following systematic methodology"),
        "rct": (3, "Randomized controlled trial"),
        "computational": (2, "Computational or in silico study"),
        "cohort": (2, "Cohort study (follows groups over time)"),
        "case_control": (2, "Case-control study (compares cases with controls)"),
        "review": (2, "General review article (not systematic)"),
        "animal_study": (2, "Animal research"),
        "in_vitro": (1, "Laboratory/cell culture study"),
        "observational": (1, "Observational study (not experimental)"),
        "case_report": (1, "Case report or case series"),
        "editorial": (1, "Opinion piece or editorial"),
        "unknown": (1, "Cannot determine from abstract"),
    }

    def __init__(self, llm_timeout: int | None = None):
        """Initialize the paper scorer with LLM for metadata extraction.

        Args:
            llm_timeout: Optional custom timeout in seconds. If None, uses Config.LLM_TIMEOUT
        """
        Config.setup_environment()

        # Use custom timeout or default
        self.timeout = llm_timeout if llm_timeout is not None else Config.LLM_TIMEOUT

        # Initialize LLM for metadata extraction
        self.llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0, timeout=self.timeout)
        self.structured_llm = self.llm.with_structured_output(PaperMetadata)

        # Create metadata extraction prompt
        self.prompt = self._create_metadata_extraction_prompt()
        self.metadata_extractor = self.prompt | self.structured_llm

    def _create_metadata_extraction_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for extracting paper metadata from abstract."""

        # Dynamically generate study type list from STUDY_TYPES
        study_types_list = "\n".join(
            f"- {study_type}: {description}"
            for study_type, (score, description) in self.STUDY_TYPES.items()
        )

        system_message = f"""You are a scientific paper metadata extraction system. Your task is to analyze a paper's title and abstract and extract key information about the study design and methodology.

EXTRACTION GUIDELINES:
- Only extract information that is CLEARLY STATED in the title or abstract
- Leave fields as None/null if the information is not explicitly mentioned or is unclear
- Do not make assumptions or infer beyond what is directly stated
- Be conservative in your extraction - when in doubt, leave it as None

STUDY TYPE CLASSIFICATION:
Classify the study into one of these types based on the abstract:
{study_types_list}

METHODOLOGY EXTRACTION:
Extract the following ONLY if clearly stated:
- sample_size: Number of participants/subjects (e.g., 500, 1200)
- study_duration: How long the study lasted (e.g., "2 years", "6 months")
- control_type: Type of control used (placebo, active, none)
- blinding: Blinding approach (single, double, triple, open)
- randomized: Whether randomization was used (true/false)
- population_type: Type of subjects (human, animal, cell_culture, computational)

CONFIDENCE:
Rate your confidence (1-10) in the study_type classification:
- 9-10: Extremely clear from abstract
- 7-8: Very clear with strong indicators
- 5-6: Reasonably clear but some ambiguity
- 3-4: Somewhat unclear, best guess
- 1-2: Very unclear, mostly guessing"""

        user_message = """Paper Title: {title}

Paper Abstract: {abstract}

Extract the metadata:"""

        return ChatPromptTemplate.from_messages([("system", system_message), ("human", user_message)])

    # ======================== CORE SCORING ========================

    def score_paper(self, paper: Paper, verbose: bool = True) -> CredibilityScores:
        """Calculate credibility scores for a paper.

        Returns a CredibilityScores object with 1-5 rating and metadata.

        Args:
            paper: Paper object (title, abstract, year, citations, etc.)
            verbose: Whether to print progress messages

        Returns:
            CredibilityScores object with rating and metadata
        """
        # 1. Extract metadata using LLM (includes study type and methodology)
        metadata = self._extract_metadata_with_llm(paper)
        study_type = metadata.study_type
        study_score = self.STUDY_TYPES.get(study_type, (1, ""))[0]

        # 2. Citation bonus
        citation_bonus = self._get_citation_bonus(paper)

        # 3. Open access availability bonus (+0.5 if PDF URL exists)
        open_access_bonus = 0.5 if (paper.pdf_url or paper.has_pdf) else 0

        # 4. Full text content bonus (+0.5 if full_text extracted)
        has_fulltext_content = bool(paper.full_text)
        fulltext_content_bonus = 0.5 if has_fulltext_content else 0

        # Determine evidence type for display
        if has_fulltext_content:
            evidence_type = "full_text"
        elif paper.pdf_url or paper.has_pdf:
            evidence_type = "open_access"
        else:
            evidence_type = "abstract"

        # 5. Recency penalty (very old papers)
        recency_penalty = self._get_recency_penalty(paper.year)

        # 6. Methodology bonuses (based on extracted metadata)
        methodology_bonus = self._get_methodology_bonus(metadata)

        # Calculate final rating (1-5 scale)
        raw_score = (
            study_score
            + citation_bonus
            + open_access_bonus
            + fulltext_content_bonus
            + methodology_bonus
            - recency_penalty
        )
        rating = max(1, min(5, raw_score))  # Clamp between 1 and 5
        
        # Simplified logging: just 2 lines per paper with tab indentation
        if verbose:
            print(f"\t  Study: {study_type} | Citations: {paper.citations or 0}")
            print(f"\t  Score: {rating}/5 (base: {study_score}, cit: +{citation_bonus}, access: +{open_access_bonus}, full: +{fulltext_content_bonus}, method: +{methodology_bonus}, recency: -{recency_penalty})")

        # Create CredibilityScores object with all metadata
        scores = CredibilityScores(
            rating=rating,
            study_type=study_type,
            evidence_type=evidence_type,
            citation_bonus=citation_bonus,
            recency_penalty=recency_penalty,
            open_access_bonus=open_access_bonus,
            fulltext_content_bonus=fulltext_content_bonus,
            sample_size=metadata.sample_size,
            study_duration=metadata.study_duration,
            control_type=metadata.control_type,
            blinding=metadata.blinding,
            randomized=metadata.randomized,
            population_type=metadata.population_type,
            metadata_confidence=metadata.confidence,
        )

        # Update paper with scores
        paper.credibility = scores

        return scores

    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """Rank papers by credibility scores.

        Args:
            papers: List of Paper objects to rank

        Returns:
            Sorted list of papers (highest credibility first)
        """
        # Score each paper if not already scored
        for paper in papers:
            if paper.credibility is None:
                self.score_paper(paper)

        # Sort by rating (descending)
        return sorted(papers, key=lambda x: x.credibility.rating if x.credibility else 1, reverse=True)

    # ======================== DETECTION & SCORING METHODS ========================

    def _extract_metadata_with_llm(self, paper: Paper) -> PaperMetadata:
        """Extract metadata from paper using LLM.

        Args:
            paper: Paper object with title and abstract (or full_text)

        Returns:
            PaperMetadata object with study type and methodology information
        """
        # Get title
        paper_title = paper.title if paper.title is not None else ""
        title = paper_title.strip() if paper_title.strip() else "No title available"

        # Truncate title if too long (max 500 chars)
        if len(title) > 500:
            title = title[:500] + "..."
            print(f"    Warning: Paper '{paper.id}' title truncated (was {len(paper_title)} chars)")

        # Get abstract, or fallback to first 2000 characters of full_text
        paper_abstract = paper.abstract if paper.abstract is not None else ""
        abstract = paper_abstract.strip()

        if not abstract and paper.full_text:
            # Use first 3000 characters from full_text as proxy for abstract
            # Typical abstract length is 150-400 words (1000-3000 chars)
            full_text_combined = self._combine_full_text_sections(paper.full_text, max_chars=3000)
            if full_text_combined:
                abstract = full_text_combined
                print(f"    Info: Paper '{paper.id}' has no abstract - using first {len(abstract)} chars of full text")

        if not abstract:
            abstract = "No abstract available"

        # CRITICAL: Truncate abstract if too long (max 3000 chars for LLM processing)
        # Very long abstracts cause timeouts
        if len(abstract) > 3000:
            original_len = len(abstract)
            abstract = abstract[:3000] + "..."
            print(f"      Warning: Paper '{paper.id}' abstract truncated ({original_len} → 3000 chars)")
        # Debug logging for missing data
        if title == "No title available" or abstract == "No abstract available":
            print(
                f"      Warning: Paper '{paper.id}' missing data - Title: '{paper_title[:50] if paper_title else 'None'}', Abstract available: {bool(paper.abstract)}, Full text sections: {len(paper.full_text) if paper.full_text else 0}"
            )

        def extract():
            import time

            start_time = time.time()
            print(f"    Extracting metadata for paper '{paper.id}'...", flush=True)
            try:
                print(f"      Calling LLM (timeout: {self.timeout}s)...", flush=True)
                result = self.metadata_extractor.invoke({"title": title, "abstract": abstract})
                elapsed = time.time() - start_time
                print(f"      Done in {elapsed:.1f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                # Add more context to the error
                print(f"      LLM extraction failed after {elapsed:.1f}s")
                print(f"      Error: {type(e).__name__}: {str(e)[:200]}")
                print(f"      Title length: {len(title)}, Abstract length: {len(abstract)}")
                raise

        # Use retry wrapper for robustness, with fallback to alternate model on failure
        try:
            return Config.retry_llm_call(extract)
        except Exception:
            # Fallback to alternate model
            print(f"         ⚠ Primary model failed, falling back to {Config.LLM_FALLBACK_MODEL}...")
            self.llm = ChatGoogleGenerativeAI(
                model=Config.LLM_FALLBACK_MODEL,
                temperature=0,
                timeout=self.timeout
            )
            self.structured_llm = self.llm.with_structured_output(PaperMetadata)
            self.metadata_extractor = self.prompt | self.structured_llm
            # Retry with fallback model
            return Config.retry_llm_call(extract)

    def _combine_full_text_sections(self, full_text: List, max_chars: int = 3000) -> str:
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

    def _get_methodology_bonus(self, metadata: PaperMetadata) -> float:
        """Calculate bonus points based on methodology quality.

        Args:
            metadata: Extracted paper metadata

        Returns:
            Bonus points (0 to 1.0)
        """
        bonus = 0.0

        # Large sample size bonus
        if metadata.sample_size and metadata.sample_size >= 500:
            bonus += 0.3
        elif metadata.sample_size and metadata.sample_size >= 100:
            bonus += 0.15

        # RCT quality indicators
        if metadata.randomized:
            bonus += 0.2

        if metadata.blinding and metadata.blinding.lower() in ["double", "triple"]:
            bonus += 0.2
        elif metadata.blinding and metadata.blinding.lower() == "single":
            bonus += 0.1

        return min(bonus, 1.0)  # Cap at 1.0 bonus point

    def _get_citation_bonus(self, paper: Paper) -> float:
        """Calculate bonus based on citation count.

        Highly cited papers are generally more influential and trustworthy.

        Args:
            paper: Paper object

        Returns:
            Bonus points (0, 0.25, or 0.5)
        """
        citations = paper.citations or 0

        # Highly cited papers get a bonus
        if citations >= 100:
            return 0.5
        elif citations >= 50:
            return 0.25

        return 0

    def _get_recency_penalty(self, year: Optional[int]) -> float:
        """Calculate penalty for very old papers.

        Science evolves, so very old papers may be outdated.

        Args:
            year: Publication year

        Returns:
            Penalty points (0, 0.5, or 1)
        """
        if not year:
            return 0  # No penalty if year unknown

        current_year = datetime.now().year
        age = current_year - year

        # Older papers get penalized
        if age > 20:
            return 1
        elif age > 10:
            return 0.5
        else:
            return 0

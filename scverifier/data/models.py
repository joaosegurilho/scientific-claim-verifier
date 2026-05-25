"""Domain models for the scientific claim verification system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from scverifier.config.settings import Config


# ============================================================================
# PYDANTIC MODELS (for LLM structured outputs)
# ============================================================================


class GeneratePropositions(BaseModel):
    """List of all the propositions in a given document."""

    propositions: List[str] = Field(
        description="List of specific, factual propositions (not meta-commentary or vague statements)"
    )


class GradePropositions(BaseModel):
    """Grade a given proposition on accuracy, clarity, completeness, and conciseness."""

    accuracy: int = Field(description="Rate from 1-10: How well does the proposition reflect the original text?")

    clarity: int = Field(description="Rate from 1-10: How easy is it to understand without additional context?")

    completeness: int = Field(
        description="Rate from 1-10: Does it include necessary details like dates, measurements, specific entities?"
    )

    conciseness: int = Field(description="Rate from 1-10: Is it concise without losing important information?")


class ClaimVerification(BaseModel):
    """Structured output for scientific claim verification."""

    verdict: str = Field(
        description="One of: SUPPORTS, REFUTES, INSUFFICIENT_EVIDENCE - based on the evidence provided"
    )

    confidence: int = Field(
        description="Confidence level 1-10: How certain are you about this verdict based on available evidence?"
    )

    reasoning: str = Field(description="Clear explanation of why this verdict was reached, citing specific evidence")


class PaperMetadata(BaseModel):
    """Structured output for extracting paper metadata from abstract."""

    study_type: str = Field(
        description="Study type: meta_analysis, systematic_review, rct, cohort, case_control, observational, case_report, in_vitro, animal_study, review, editorial, computational, or unknown"
    )

    sample_size: Optional[int] = Field(
        default=None, description="Number of participants/subjects in the study, if clearly stated"
    )

    study_duration: Optional[str] = Field(
        default=None, description="Duration of the study (e.g., '2 years', '6 months'), if clearly stated"
    )

    control_type: Optional[str] = Field(
        default=None, description="Type of control: placebo, active, none, or other, if clearly stated"
    )

    blinding: Optional[str] = Field(
        default=None, description="Blinding type: single, double, triple, open, if clearly stated"
    )

    randomized: Optional[bool] = Field(default=None, description="Whether the study is randomized, if clearly stated")

    population_type: Optional[str] = Field(
        default=None,
        description="Population type: human, animal, cell_culture, computational, other, if clearly stated",
    )

    confidence: int = Field(description="Confidence level 1-10 in the study_type classification based on the abstract")


# ============================================================================
# DOMAIN CLASSES (core data structures)
# ============================================================================


@dataclass
class PropositionEvaluation:
    """Quality evaluation scores for a proposition."""

    accuracy: int  # 1-10: How well it reflects the original text
    clarity: int  # 1-10: How easy to understand without context
    completeness: int  # 1-10: Includes necessary details
    conciseness: int  # 1-10: Concise without losing information

    def passes_quality_check(self, thresholds: Optional[Dict[str, int]] = None) -> bool:
        """Check if all scores meet quality thresholds.

        Args:
            thresholds: Optional custom thresholds, otherwise uses Config defaults
        """
        if thresholds is None:
            thresholds = Config.QUALITY_THRESHOLDS

        scores = {
            "accuracy": self.accuracy,
            "clarity": self.clarity,
            "completeness": self.completeness,
            "conciseness": self.conciseness,
        }
        for category, score in scores.items():
            if score < thresholds.get(category, 7):
                return False
        return True

    def average_score(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.accuracy + self.clarity + self.completeness + self.conciseness) / 4

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "clarity": self.clarity,
            "completeness": self.completeness,
            "conciseness": self.conciseness,
        }

    @classmethod
    def from_dict(cls, scores: Dict[str, int]) -> "PropositionEvaluation":
        """Create from dictionary."""
        return cls(
            accuracy=scores["accuracy"],
            clarity=scores["clarity"],
            completeness=scores["completeness"],
            conciseness=scores["conciseness"],
        )


@dataclass
class Chunk:
    """A document chunk with metadata."""

    text: str
    chunk_id: str  # Unique ID from global counter (format: "chunk_1", "chunk_2", etc.)
    source: str  # Paper title or source name
    paper_id: str  # Reference to parent paper
    page: Optional[int] = None
    section: str = ""  # Section name where chunk originated (e.g., "abstract", "introduction", "results")

    def to_langchain_document(self):
        """Convert to LangChain Document format for vector stores."""
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "source": self.source,
                "paper_id": self.paper_id,
                "page": self.page,
                "section": self.section,
            },
        )

    @classmethod
    def from_langchain_document(cls, doc) -> "Chunk":
        """Create Chunk from LangChain Document."""
        return cls(
            text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_id", "unknown"),
            source=doc.metadata.get("source", "unknown"),
            paper_id=doc.metadata.get("paper_id", "unknown"),
            page=doc.metadata.get("page"),
            section=doc.metadata.get("section", ""),
        )


@dataclass
class Proposition:
    """An atomic factual claim extracted from a document."""

    text: str
    chunk_id: str  # Which chunk it came from
    source: str  # Paper title or source name
    paper_id: str  # Reference to parent paper
    prop_id: str = ""  # Unique proposition ID from global counter (format: "prop_1", "prop_2", etc.)
    evaluation: Optional[PropositionEvaluation] = None
    page: Optional[int] = None

    def is_high_quality(self) -> bool:
        """Check if proposition passed quality evaluation."""
        if self.evaluation is None:
            return False
        return self.evaluation.passes_quality_check()

    def to_langchain_document(self):
        """Convert to LangChain Document format for vector stores."""
        metadata = {
            "prop_id": self.prop_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "paper_id": self.paper_id,
            "page": self.page,
        }

        # Add evaluation scores if available
        if self.evaluation:
            metadata.update(self.evaluation.to_dict())
            metadata["passed_quality"] = self.evaluation.passes_quality_check()

        return Document(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_document(cls, doc) -> "Proposition":
        """Create Proposition from LangChain Document."""
        evaluation = None
        if all(k in doc.metadata for k in ["accuracy", "clarity", "completeness", "conciseness"]):
            evaluation = PropositionEvaluation(
                accuracy=doc.metadata["accuracy"],
                clarity=doc.metadata["clarity"],
                completeness=doc.metadata["completeness"],
                conciseness=doc.metadata["conciseness"],
            )

        return cls(
            text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_id", "unknown"),
            source=doc.metadata.get("source", "unknown"),
            paper_id=doc.metadata.get("paper_id", "unknown"),
            prop_id=doc.metadata.get("prop_id", ""),
            evaluation=evaluation,
            page=doc.metadata.get("page"),
        )


@dataclass
class CredibilityScores:
    """Credibility scoring information for a paper."""

    rating: float  # Overall 1-5 rating
    study_type: str  # e.g., "meta_analysis", "rct", "case_report"
    evidence_type: str  # "full_text", "open_access", or "abstract"
    citation_bonus: float = 0.0  # Bonus points from citations
    recency_penalty: float = 0.0  # Penalty for old papers
    open_access_bonus: float = 0.0  # Bonus for having PDF URL available
    fulltext_content_bonus: float = 0.0  # Bonus for having extracted full text

    # Methodology metadata (extracted from abstract via LLM)
    sample_size: Optional[int] = None
    study_duration: Optional[str] = None
    control_type: Optional[str] = None
    blinding: Optional[str] = None
    randomized: Optional[bool] = None
    population_type: Optional[str] = None
    metadata_confidence: Optional[int] = None  # LLM confidence in metadata extraction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "rating": self.rating,
            "study_type": self.study_type,
            "evidence_type": self.evidence_type,
            "citation_bonus": self.citation_bonus,
            "recency_penalty": self.recency_penalty,
            "open_access_bonus": self.open_access_bonus,
            "fulltext_content_bonus": self.fulltext_content_bonus,
            "sample_size": self.sample_size,
            "study_duration": self.study_duration,
            "control_type": self.control_type,
            "blinding": self.blinding,
            "randomized": self.randomized,
            "population_type": self.population_type,
            "metadata_confidence": self.metadata_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CredibilityScores":
        """Create from dictionary."""
        return cls(
            rating=data["rating"],
            study_type=data["study_type"],
            evidence_type=data["evidence_type"],
            citation_bonus=data.get("citation_bonus", 0.0),
            recency_penalty=data.get("recency_penalty", 0.0),
            open_access_bonus=data.get("open_access_bonus", 0.0),
            fulltext_content_bonus=data.get("fulltext_content_bonus", 0.0),
            sample_size=data.get("sample_size"),
            study_duration=data.get("study_duration"),
            control_type=data.get("control_type"),
            blinding=data.get("blinding"),
            randomized=data.get("randomized"),
            population_type=data.get("population_type"),
            metadata_confidence=data.get("metadata_confidence"),
        )


@dataclass
class Paper:
    """Represents a scientific paper."""

    id: str  # Unique identifier (PMC ID, PubMed ID, or generated)
    doi: str | None
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    citations: int = 0
    url: str = ""
    pdf_url: str = ""
    source: str = ""  # "pubmed", "pmc", "semantic_scholar", "local"
    has_pdf: bool = False
    pmc_id: Optional[str] = None

    # Full text storage: List of (section_name, section_text) tuples
    # e.g., [("introduction", "..."), ("methodology", "...")]
    # For unsectioned papers: [("full_text", "...")]
    full_text: List[tuple[str, str]] = field(default_factory=list)

    # Extracted content (populated during processing)
    chunks: List[Chunk] = field(default_factory=list)
    propositions: List[Proposition] = field(default_factory=list)

    # Credibility scoring
    credibility: Optional[CredibilityScores] = None

    # Track what was used for proposition extraction
    extracted_from: str = "abstract"  # "abstract", "full_text", or "both"

    def get_quality_propositions(self) -> List[Proposition]:
        """Get only propositions that passed quality checks."""
        return [p for p in self.propositions if p.is_high_quality()]

    def get_proposition(self, prop_id: str) -> Optional[Proposition]:
        """Get a specific proposition by its prop_id."""
        for prop in self.propositions:
            if prop.prop_id == prop_id:
                return prop
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this paper's extraction."""
        quality_props = self.get_quality_propositions()
        return {
            "id": self.id,
            "doi": self.doi,
            "title": self.title,
            "chunks": len(self.chunks),
            "propositions_total": len(self.propositions),
            "propositions_quality": len(quality_props),
            "success_rate": len(quality_props) / len(self.propositions) if self.propositions else 0,
            "year": self.year,
            "citations": self.citations,
            "credibility": self.credibility.to_dict() if self.credibility else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "citations": self.citations,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "source": self.source,
            "has_pdf": self.has_pdf,
            "pmc_id": self.pmc_id,
            "full_text": self.full_text,
            "credibility": self.credibility.to_dict() if self.credibility else None,
            "extracted_from": self.extracted_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        """Create Paper from dictionary."""
        credibility = None
        if data.get("credibility"):
            credibility = CredibilityScores.from_dict(data["credibility"])

        return cls(
            id=data["id"],
            doi=data["doi"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data.get("authors", []),
            year=data.get("year"),
            citations=data.get("citations", 0),
            url=data.get("url", ""),
            pdf_url=data.get("pdf_url", ""),
            source=data.get("source", ""),
            has_pdf=data.get("has_pdf", False),
            pmc_id=data.get("pmc_id"),
            full_text=data.get("full_text", []),
            credibility=credibility,
            extracted_from=data.get("extracted_from", "abstract"),
        )


@dataclass
class VerificationResult:
    """Result of claim verification."""

    claim: str
    verdict: str  # "SUPPORTS", "REFUTES", "INSUFFICIENT_EVIDENCE"
    confidence: float  # 1-10
    reasoning: str
    evidence: List[Proposition]  # All relevant propositions used for verification
    token_usage: Optional[Dict[str, int]] = None  # Token usage: {input_tokens, output_tokens, total_tokens}

    def get_papers_used(self) -> List[str]:
        """Get unique paper IDs from evidence."""
        return list(set(prop.paper_id for prop in self.evidence))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": [
                {
                    "text": p.text,
                    "source": p.source,
                    "paper_id": p.paper_id,
                    "page": p.page,
                    "quality_score": p.evaluation.average_score() if p.evaluation else None,
                }
                for p in self.evidence
            ],
            "papers_used": self.get_papers_used(),
            "num_evidence": len(self.evidence),
        }
        if self.token_usage:
            result["token_usage"] = self.token_usage
        return result

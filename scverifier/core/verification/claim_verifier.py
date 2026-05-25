"""Scientific claim verification using retrieved evidence with credibility scoring."""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from scverifier.config.settings import Config
from scverifier.data.models import ClaimVerification, Proposition, VerificationResult
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.verification.confidence_interpreter import CONFIDENCE_INTERPRETATIONS


class ClaimVerifier:
    """Verifies scientific claims using retrieved evidence with credibility assessment.

    This verifier:
    - Analyzes claims against retrieved scientific evidence
    - Provides verdicts: SUPPORTS, REFUTES, or INSUFFICIENT_EVIDENCE
    - Assigns confidence scores (1-10)
    - Looks up paper credibility from KB automatically
    - Generates detailed reasoning for verdicts
    - Returns VerificationResult domain objects

    Works entirely with Proposition domain objects - no LangChain Documents.
    """

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        """Initialize the claim verifier.

        Args:
            kb: Optional KnowledgeBase reference for looking up paper credibility
        """
        Config.setup_environment()

        # Initialize LLM with structured output
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE, timeout=Config.LLM_TIMEOUT
        )
        self.structured_llm = self.llm.with_structured_output(ClaimVerification, include_raw=True)

        # Create prompt template and chain
        self.prompt = self._create_verification_prompt()
        self.verification_chain = self.prompt | self.structured_llm

        # Store KB reference for looking up credibility
        self.kb = kb

    @staticmethod
    def _build_confidence_scoring_section() -> str:
        """Build the confidence scoring section of the prompt from CONFIDENCE_INTERPRETATIONS."""
        lines = [
            "CONFIDENCE SCORING (1-10)",
            "Rate your confidence in the verdict according to how clearly and consistently the retrieved evidence justifies it",
        ]
        for verdict, ranges in CONFIDENCE_INTERPRETATIONS.items():
            lines.append(f"\nFor {verdict} verdict:")
            for (min_conf, max_conf), description in sorted(ranges.items(), reverse=True):
                lines.append(f"- {min_conf}-{max_conf}: {description}")
        return "\n".join(lines)

    def _create_verification_prompt(self) -> ChatPromptTemplate:
        """Create the claim verification prompt template with detailed guidelines."""

        confidence_section = self._build_confidence_scoring_section()

        system_message = f"""You are a scientific claim verification system. Your task is to evaluate whether a given claim is supported, refuted, or has insufficient evidence based on the provided scientific evidence.

VERIFICATION GUIDELINES:
- **SUPPORTS**: The evidence clearly supports the claim with specific facts, data, or findings
- **REFUTES**: The evidence directly contradicts the claim with contrary facts or data
- **INSUFFICIENT_EVIDENCE**: The evidence is irrelevant, too vague, conflicting, or does not provide enough reliable information to verify the claim

When determining your verdict, always base your reasoning on the relationship between the claim and the retrieved evidence, not on external world knowledge.

{confidence_section}

REASONING REQUIREMENTS:
- Cite specific evidence from the provided sources (like this "[Source X]", or "[Sources X, Y, Z]")
- Weigh evidence by credibility (higher credibility = more weight)
- Consider study types (RCTs and meta-analyses carry more weight than case reports)
- Explain precisely why the evidence supports, refutes, or is insufficient for the claim
- Distinguish between what the evidence shows and what the claim asserts
- If evidence is insufficient, specify what kind of evidence would be needed to reach a stronger conclusion

Evidence from scientific literature (with credibility scores):
{{evidence}}

Claim to verify: {{claim}}

Provide your verification analysis:"""

        return ChatPromptTemplate.from_messages([("system", system_message), ("human", "{claim}")])

    # ======================== CORE VERIFICATION ========================

    def verify_claim(
        self,
        claim: str,
        evidence: List[Proposition],
    ) -> VerificationResult:
        """Verify a claim against provided evidence propositions.

        Args:
            claim: The scientific claim to verify
            evidence: List of Proposition objects (evidence)

        Returns:
            VerificationResult domain object
        """
        # Format evidence from propositions (credibility looked up automatically)
        evidence_str = self._format_evidence(evidence)

        # Generate verification using LLM (with include_raw=True, returns dict with 'parsed' and 'raw')
        response = Config.retry_llm_call(
            lambda: self.verification_chain.invoke({"claim": claim, "evidence": evidence_str})
        )

        # Extract parsed verification and raw response
        verification = response.get("parsed") if isinstance(response, dict) else response
        raw_message = response.get("raw") if isinstance(response, dict) else None

        # Extract token usage if available from raw response
        token_usage = None
        if raw_message and hasattr(raw_message, 'usage_metadata') and raw_message.usage_metadata:
            token_usage = {
                "input_tokens": raw_message.usage_metadata.get("input_tokens", 0),
                "output_tokens": raw_message.usage_metadata.get("output_tokens", 0),
                "total_tokens": raw_message.usage_metadata.get("total_tokens", 0),
            }

        # Create VerificationResult domain object
        return VerificationResult(
            claim=claim,
            verdict=verification.verdict,
            confidence=float(verification.confidence),
            reasoning=verification.reasoning,
            evidence=evidence,
            token_usage=token_usage,
        )

    # ======================== FORMATTING UTILITIES ========================

    def _format_evidence(self, propositions: List[Proposition]) -> str:
        """Format evidence propositions into a structured string with credibility info.

        Args:
            propositions: Proposition domain objects to format

        Returns:
            Formatted evidence string for the LLM prompt
        """
        if not propositions:
            return "No relevant evidence found in the knowledge base."

        evidence_parts = []
        for i, prop in enumerate(propositions, 1):
            # Look up paper metadata from KB if available
            metadata_parts = []
            if self.kb:
                paper = self.kb.get_paper(prop.paper_id)
                if paper:
                    # Build comprehensive metadata on a single line
                    metadata_parts.append(f"Title: {paper.title}")

                    if paper.authors:
                        authors_str = ", ".join(paper.authors[:3])
                        if len(paper.authors) > 3:
                            authors_str += f" et al. ({len(paper.authors)} authors)"
                        metadata_parts.append(f"Authors: {authors_str}")

                    metadata_parts.append(f"Year: {paper.year or 'Unknown'}")
                    metadata_parts.append(f"Citations: {paper.citations or 0}")

                    if paper.doi:
                        metadata_parts.append(f"DOI: {paper.doi}")

                    # Credibility scores
                    if paper.credibility:
                        cred = paper.credibility
                        metadata_parts.append(f"Rating: {cred.rating:.1f}/5 ★")
                        metadata_parts.append(f"Study Type: {cred.study_type}")
                        metadata_parts.append(f"Evidence Type: {cred.evidence_type}")

                        # Methodology metadata (if available)
                        if cred.sample_size:
                            metadata_parts.append(f"Sample Size: {cred.sample_size}")
                        if cred.study_duration:
                            metadata_parts.append(f"Duration: {cred.study_duration}")
                        if cred.randomized is not None:
                            metadata_parts.append(f"Randomized: {'Yes' if cred.randomized else 'No'}")
                        if cred.blinding:
                            metadata_parts.append(f"Blinding: {cred.blinding}")
                        if cred.control_type:
                            metadata_parts.append(f"Control: {cred.control_type}")
                        if cred.population_type:
                            metadata_parts.append(f"Population: {cred.population_type}")

            if metadata_parts:
                metadata_line = " | ".join(metadata_parts)
                evidence_parts.append(f"[Source {i}] {metadata_line}\nProposition: {prop.text}")
            else:
                evidence_parts.append(f"[Source {i} - {prop.source}]\nProposition: {prop.text}")

        return "\n\n".join(evidence_parts)

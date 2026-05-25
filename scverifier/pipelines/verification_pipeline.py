"""Stateless verification pipeline using modular components."""

from typing import List

from scverifier.core.extraction.proposition_extractor import PropositionExtractor
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.knowledge.literature_search import LiteratureSearch
from scverifier.core.verification.claim_verifier import ClaimVerifier
from scverifier.core.verification.paper_scorer import PaperScorer
from scverifier.data.models import Proposition, VerificationResult
from scverifier.config.settings import Config


class VerificationPipeline:
    """Stateless pipeline for scientific claim verification.

    This pipeline orchestrates the verification process without maintaining state.
    It has two main verification methods:
    1. verify_claim_with_search() - Searches for papers online, processes them, verifies
    2. verify_claim_from_kb() - Uses only existing knowledge base data to verify

    All persistent data is stored in the KnowledgeBase (single source of truth).
    """

    def __init__(self, kb: KnowledgeBase):
        """Initialize pipeline with a knowledge base.

        Args:
            kb: KnowledgeBase instance (single source of truth for all data)
        """
        # Core components (stateless services)
        self.kb = kb
        self.literature_search = LiteratureSearch()
        self.extractor = PropositionExtractor()
        self.paper_scorer = PaperScorer()
        self.claim_verifier = ClaimVerifier(kb=kb)  # Pass KB for credibility lookup

    # ======================== VERIFICATION WITH SEARCH ========================

    def verify_claim_with_search(
        self, claim: str, max_papers: int = 30, use_full_text: bool = False, quality_claims: bool = True, max_props_per_paper: int = 5, max_propositions: int = Config.PROPOSITION_RETRIEVAL_K
    ) -> VerificationResult:
        """Verify a claim by searching for papers online, processing them, and verifying.

        This method:
        1. Searches for relevant papers (supporting, refuting, neutral)
        2. Extracts propositions from new papers
        3. Scores paper credibility
        4. Stores everything in KB
        5. Verifies claim against all evidence

        Args:
            claim: Scientific claim to verify
            max_papers: Maximum number of papers to search for and process
            use_full_text: If True, extract from full_text. If False, extract from abstract only. Default False.
            quality_claims: Whether to use only quality propositions during verification. Default True.
            max_props_per_paper: Maximum propositions to use from each paper for diversity
            max_propositions: Maximum total propositions to retrieve per query

        Returns:
            VerificationResult with verdict, confidence, reasoning, and evidence
        """
        print("\n CLAIM VERIFICATION WITH SEARCH")
        print("=" * 60)
        print(f"Claim: {claim}\n")

        # Phase 1: Search for relevant papers
        print(f" Phase 1: Searching for {max_papers} relevant papers...")
        search_queries = self.literature_search.generate_search_queries(claim)

        print(f"   • Original query: {search_queries['original']}")
        print(f"   • Opposite query: {search_queries.get('opposite', 'N/A')}")
        print(f"   • Neutral query: {search_queries.get('neutral', 'N/A')}")

        papers_found = self.literature_search.search_papers(
            query=claim, search_queries=search_queries, max_papers=max_papers, verbose=True
        )
        print(f"   Retrieved {len(papers_found)} papers")

        # Phase 2: Process new papers (skip already processed ones)
        print("\n Phase 2: Processing new papers...")
        new_papers = []
        existing_papers = []

        for paper in papers_found:
            # Skip papers without abstracts
            if not paper.abstract or not paper.abstract.strip():
                print(f"    Skipping paper '{paper.title[:50]}...' (no abstract)")
                continue

            existing = self.kb.get_paper(paper.id)
            if existing and existing.propositions:
                existing_papers.append(existing)
            else:
                new_papers.append(paper)

        print("    Papers to process:")
        print(f"      New papers: {len(new_papers)}")
        print(f"      Already processed: {len(existing_papers)}")

        # Extract propositions from new papers
        if new_papers:
            print(f"\n    Extracting from {len(new_papers)} new papers...")
            processed_papers = self.extractor.extract_from_papers(
                new_papers, show_steps=True, use_full_text=use_full_text
            )

            # Score papers
            print("\n    Scoring paper credibility...")
            for paper in processed_papers:
                self.paper_scorer.score_paper(paper)

            # Add to knowledge base
            print("\n    Adding new papers to knowledge base...")
            self.kb.add_papers(processed_papers, verbose=False)

            total_quality_props = sum(len(p.get_quality_propositions()) for p in processed_papers)
            total_props = sum(len(p.propositions) for p in processed_papers)
            print(
                f"   Added {len(processed_papers)} papers with {total_quality_props} quality propositions out of {total_props} total propositions"
            )

            # Save KB incrementally to avoid losing data if pipeline crashes
            print("\n    Saving knowledge base (incremental save)...")
            self.kb.save()
            print(f"   Knowledge base saved to {Config.DB_NAME}")
        else:
            print("\n   No new papers to process")

        # Phase 3: Verify claim using all available evidence in KB
        print("\n  Phase 3: Verifying claim with all available evidence...")
        return self._verify_with_kb_evidence(claim, search_queries, quality_claims=quality_claims, max_props_per_paper=max_props_per_paper, max_propositions=max_propositions)

    # ======================== VERIFICATION FROM KB ONLY ========================

    def verify_claim_from_kb(self, claim: str, quality_claims: bool = True, use_abstract_only: bool = False, max_props_per_paper: int = 5, max_propositions: int = 50) -> VerificationResult:
        """Verify a claim using only existing knowledge base data.

        This method:
        1. Generates search queries
        2. Retrieves relevant propositions from KB
        3. Verifies claim against retrieved evidence

        No online search, no new papers processed. Uses only existing KB data.

        Args:
            claim: Scientific claim to verify
            quality_claims: Whether to use only quality propositions
            use_abstract_only: Whether to use only propositions from abstract sections
            max_props_per_paper: Maximum propositions to use from each paper for diversity
            max_propositions: Maximum total propositions to retrieve per query

        Returns:
            VerificationResult with verdict, confidence, reasoning, and evidence
        """
        print("\n CLAIM VERIFICATION FROM KNOWLEDGE BASE")
        print("=" * 60)
        print(f"Claim: {claim}\n")

        # Check if KB has data
        stats = self.kb.get_statistics()
        if stats["papers"] == 0:
            print("  Knowledge base is empty!")
            print("   Use verify_claim_with_search() to search for papers first.")
            return VerificationResult(
                claim=claim,
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=1.0,
                reasoning="Knowledge base is empty. No papers available for verification.",
                evidence=[],
            )

        print(" Knowledge Base Stats:")
        print(f"   Papers: {stats['papers']}")
        print(f"   Quality propositions: {stats['propositions_quality']}")
        if use_abstract_only:
            print("   Filter: Abstract propositions only")

        # Generate search queries
        print("\n Generating search queries...")
        search_queries = self.literature_search.generate_search_queries(claim)
        print(f"   • Original: {search_queries['original']}")
        print(f"   • Opposite: {search_queries.get('opposite', 'N/A')}")
        print(f"   • Neutral: {search_queries.get('neutral', 'N/A')}")

        # Verify with KB evidence
        print("\n  Verifying claim with knowledge base evidence...")
        return self._verify_with_kb_evidence(claim, search_queries, quality_claims=quality_claims, use_abstract_only=use_abstract_only, max_props_per_paper=max_props_per_paper, max_propositions=max_propositions)

    # ======================== INTERNAL VERIFICATION LOGIC ========================

    def _verify_with_kb_evidence(
        self, claim: str, search_queries: dict, quality_claims: bool = True, use_abstract_only: bool = False, max_props_per_paper: int = 5, max_propositions: int = Config.PROPOSITION_RETRIEVAL_K
    ) -> VerificationResult:
        """Verify claim using evidence from knowledge base.

        Internal method used by both verification approaches.

        Args:
            claim: The scientific claim to verify
            search_queries: Dictionary with original, opposite, and neutral queries
            quality_claims: Whether to use only quality propositions
            use_abstract_only: Whether to use only propositions from abstract sections
            max_props_per_paper: Maximum propositions to use from each paper for diversity
            max_propositions: Maximum total propositions to retrieve per query

        Returns:
            VerificationResult
        """
        print("   • Retrieving relevant propositions...")

        # Query knowledge base with different query types
        all_retrieved = []
        seen_contents = set()

        # Retrieve propositions for EACH query type
        for query_type, query in search_queries.items():
            retrieved = self.kb.search_propositions(query, top_k=max_propositions)
            if quality_claims:
                retrieved = [p for p in retrieved if p.is_high_quality()]

            # Filter by abstract-only if requested
            if use_abstract_only:
                retrieved = self._filter_abstract_propositions(retrieved)

            # Deduplicate by content
            for prop in retrieved:
                if prop.text not in seen_contents:
                    seen_contents.add(prop.text)
                    all_retrieved.append(prop)

            print(f"   • Retrieved {len(retrieved)} propositions from {query_type} query")

        print(f"   • Total unique propositions: {len(all_retrieved)}")

        # # Alternative approach: distribute max_propositions across ALL queries
        # num_queries = len(search_queries)
        # per_query_limit = max(1, max_propositions // num_queries)
        # print(f"   • Distributing ~{per_query_limit} propositions per query (total cap {max_propositions})")

        # for query_type, query in search_queries.items():
        #     retrieved = self.kb.search_propositions(query, top_k=per_query_limit)
        #     if quality_claims:
        #         retrieved = [p for p in retrieved if p.is_high_quality()]

        #     # Filter by abstract-only if requested
        #     if use_abstract_only:
        #         retrieved = self._filter_abstract_propositions(retrieved)

        #     # Deduplicate by content
        #     for prop in retrieved:
        #         if prop.text not in seen_contents:
        #             seen_contents.add(prop.text)
        #             all_retrieved.append(prop)
        #             # Enforce global cap
        #             if len(all_retrieved) >= max_propositions:
        #                 break

        #     print(f"   • Retrieved {len(retrieved)} propositions from {query_type} query")

        # print(f"   • Total unique propositions (capped to {max_propositions}): {len(all_retrieved)}")

        # Diversify propositions (limit per paper for source diversity)
        retrieved_props = self._diversify_propositions(all_retrieved, max_per_paper=max_props_per_paper)
        print(f"   • Diversified to {len(retrieved_props)} propositions (max {max_props_per_paper} per paper)")

        # Re-rank by credibility
        retrieved_props = self._rerank_by_credibility(retrieved_props)
        print("   • Re-ranked by credibility")

        # Verify using the claim verifier (it looks up credibility automatically)
        verification = self.claim_verifier.verify_claim(claim, retrieved_props)

        return verification

    def _diversify_propositions(self, propositions: List[Proposition], max_per_paper: int) -> List[Proposition]:
        """Limit propositions from each paper to ensure source diversity.

        Args:
            propositions: List of Proposition objects
            max_per_paper: Maximum propositions to take from each paper

        Returns:
            Diversified list of propositions
        """
        paper_counts = {}
        diversified = []

        for prop in propositions:
            paper_id = prop.paper_id
            count = paper_counts.get(paper_id, 0)

            if count < max_per_paper:
                diversified.append(prop)
                paper_counts[paper_id] = count + 1

        return diversified

    def _rerank_by_credibility(self, propositions: List[Proposition]) -> List[Proposition]:
        """Re-rank propositions by paper credibility scores (rating).

        Args:
            propositions: List of Proposition objects

        Returns:
            Sorted list (highest credibility first)
        """

        def get_rating(prop: Proposition) -> float:
            paper = self.kb.get_paper(prop.paper_id)
            if paper and paper.credibility:
                return paper.credibility.rating
            return 0

        return sorted(propositions, key=get_rating, reverse=True)

    def _filter_abstract_propositions(self, propositions: List[Proposition]) -> List[Proposition]:
        """Filter propositions to only include those from abstract sections.

        Args:
            propositions: List of Proposition objects

        Returns:
            Filtered list containing only propositions from abstract sections
        """
        filtered = []
        for prop in propositions:
            # Get the chunk that this proposition came from
            chunk = self.kb.get_chunk(prop.chunk_id)
            if chunk and chunk.section.lower() == "abstract":
                filtered.append(prop)

        return filtered

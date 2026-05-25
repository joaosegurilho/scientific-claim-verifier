"""Literature search functionality for finding papers from external APIs."""

from typing import List, Dict, Any
from time import sleep
from langchain_google_genai import ChatGoogleGenerativeAI

from scverifier.api.api_core import CoreAPI
from scverifier.api.api_pmc import PMCAPI
from scverifier.api.api_pubmed import PubMedAPI
from scverifier.api.api_semantic_scholar import SemanticScholarAPI
from scverifier.api.api_openalex import OpenAlexAPI
from scverifier.api.api import API
from scverifier.data.models import Paper
from scverifier.config.settings import Config


class LiteratureSearch:
    """Handles searching for scientific papers across multiple academic databases.

    This component:
    - Generates search queries (original, opposite, neutral)
    - Searches multiple APIs (Semantic Scholar, PubMed, CORE)
    - Deduplicates results
    - Returns Paper domain objects directly
    """

    def __init__(self):
        """Initialize API clients."""
        self.semantic_scholar = SemanticScholarAPI()
        self.pubmed = PubMedAPI()
        self.core = CoreAPI()
        self.openalex = OpenAlexAPI()

    # ======================== QUERY GENERATION ========================

    def generate_search_queries(self, claim: str) -> Dict[str, str]:
        """Generate original, opposite, and neutral queries using LLM.

        This helps find diverse evidence by searching for:
        - Supporting evidence (original)
        - Contradicting evidence (opposite)
        - Related but neutral studies (neutral)

        Args:
            claim: Original scientific claim

        Returns:
            Dictionary with 'original', 'opposite', and 'neutral' queries
        """
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE, timeout=Config.LLM_TIMEOUT
        )

        prompt = f"""You are a scientific literature search expert. Given this scientific claim: "{claim}"

Generate two related claims:

1. OPPOSITE: A statement that contradicts the original claim. 
   Use natural phrasing (e.g., "has no effect", "is not linked to", "may worsen", "fails to", "increases").
   Keep it concise and scientifically plausible.

2. NEUTRAL: A statement that is related but does not agree or disagree. 
   Include core and related keywords that could appear in scientific papers on the same topic 
   (e.g., synonyms, broader or adjacent concepts).

Respond in this exact format:
OPPOSITE: <text>
NEUTRAL: <text>

Example:
Claim: "Green tea improves memory."
OPPOSITE: Green tea shows no significant effect on memory function.
NEUTRAL: Studies explore green tea, cognition, neuroprotection, brain health, and memory-related biomarkers.

Now generate for the given claim:"""

        response = Config.retry_llm_call(lambda: llm.invoke(prompt))
        text = response.content

        # Parse the response
        queries = {"original": claim}

        for line in text.strip().split("\n"):
            if line.startswith("OPPOSITE:"):
                queries["opposite"] = line.replace("OPPOSITE:", "").strip()
            elif line.startswith("NEUTRAL:"):
                queries["neutral"] = line.replace("NEUTRAL:", "").strip()

        return queries

    def generate_paper_titles(self, search_queries: Dict[str, str], temp: float = 0.0) -> List[str]:
        """Use LLM to generate specific paper titles based on search queries.

        Args:
            search_queries: Dictionary with original, opposite, and neutral queries
            temp: Temperature for generation (higher = more diverse)

        Returns:
            List of generated paper title queries
        """
        llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=temp, timeout=Config.LLM_TIMEOUT)

        prompt = f"""You are a scientific literature search expert. Given these research queries:

ORIGINAL: {search_queries.get('original', '')}
OPPOSITE: {search_queries.get('opposite', '')}  
NEUTRAL: {search_queries.get('neutral', '')}

Generate 10 SPECIFIC paper titles that would help find relevant research papers.
Return ONLY the titles, one per line, no numbering.

Example for "Green tea prevents cancer":
Green tea polyphenols and cancer prevention mechanisms
Meta-analysis of green tea consumption and cancer risk
No association between green tea intake and cancer incidence

Now generate titles for the queries above:"""

        response = Config.retry_llm_call(lambda: llm.invoke(prompt))
        titles = [line.strip() for line in response.content.strip().split("\n") if line.strip()]

        # Fallback if LLM fails
        if not titles:
            return [
                search_queries.get("original", ""),
                search_queries.get("opposite", ""),
                search_queries.get("neutral", ""),
            ]

        return titles

    # ======================== PAPER SEARCH ========================

    def search_papers(
        self, query: str, search_queries: Dict[str, str] = None, max_papers: int = 30, verbose: bool = True
    ) -> List[Paper]:
        """Search for papers across multiple academic APIs.

        - First round uses temperature=0 for reproducibility
        - Later rounds use temperature=0.7 for diversity
        - Stops early if max_papers found or no new papers for several rounds

        Args:
            query: Original search query
            search_queries: Optional pre-generated queries (original, opposite, neutral)
            max_papers: Maximum number of unique papers to retrieve
            verbose: Whether to print progress messages

        Returns:
            List of unique Paper domain objects from APIs
        """
        # Generate queries if not provided
        if search_queries is None:
            if verbose:
                print("   • Generating search queries...")
            search_queries = self.generate_search_queries(query)
            if verbose:
                print(f"     Original: {search_queries['original']}")
                print(f"     Opposite: {search_queries.get('opposite', 'N/A')}")
                print(f"     Neutral: {search_queries.get('neutral', 'N/A')}")

        all_papers: list[Paper] = []
        seen_ids = set()
        seen_titles = set()
        seen_dois = set()
        apis: List[tuple[str, API]] = [
            ("Semantic Scholar", self.semantic_scholar),
            ("PubMed", self.pubmed),
            ("CORE", self.core),
            ("OpenAlex", self.openalex),
        ]

        if verbose:
            print(f"   • Target: {max_papers} unique papers")

        max_rounds = 5  # Safety to prevent infinite loops
        max_api_calls = 10 * max_papers  # Hard limit on total API calls
        api_call_count = 0
        rounds_without_new = 0
        round_index = 0

        while len(all_papers) < max_papers and rounds_without_new < max_rounds and api_call_count < max_api_calls:
            round_index += 1
            temp = 0.0 if round_index == 1 else 0.7

            if verbose:
                print(f"\n    Search round {round_index} / {max_rounds}")

            paper_titles = self.generate_paper_titles(search_queries, temp=temp)

            if verbose:
                print(f"   • Generated {len(paper_titles)} paper title queries")
                for i, title_query in enumerate(paper_titles):
                    print(f"     {i}. {title_query}")

            new_found = 0
            stop_search = False

            for title_query in paper_titles:
                if len(all_papers) >= max_papers or stop_search:
                    break

                for api_name, api_client in apis:
                    if len(all_papers) >= max_papers or api_call_count >= max_api_calls:
                        stop_search = True
                        break

                    try:
                        api_call_count += 1
                        paper_dicts = api_client.search_papers(title_query, limit=1)
                        for paper_dict in paper_dicts:
                            paper_id = paper_dict.get("id")
                            title = paper_dict.get("title", "").lower().strip()
                            doi = paper_dict.get("doi", None)
                            doi_key = None
                            if doi and isinstance(doi, str):
                                doi_key = doi.lower().rstrip(".").strip()

                            is_duplicate = False
                            if doi_key and doi_key != "unknown":
                                if doi_key in seen_dois:
                                    is_duplicate = True
                            if paper_id in seen_ids or title in seen_titles:
                                is_duplicate = True

                            if not is_duplicate:
                                seen_ids.add(paper_id)
                                seen_titles.add(title)
                                if doi_key and doi_key != "unknown":
                                    seen_dois.add(doi_key)

                                # Convert to Paper domain object immediately
                                paper = self._dict_to_paper(paper_dict)

                                # Enrich with PMC full text if available
                                if paper.pmc_id:
                                    pmc_api = PMCAPI()
                                    full_text_data = pmc_api.get_full_text(paper.pmc_id)
                                    if full_text_data and full_text_data.get("has_full_text"):
                                        paper.full_text = full_text_data["full_text_sections"]

                                all_papers.append(paper)
                                new_found += 1

                                if verbose:
                                    short_title = title[:77] + "..." if len(title) > 80 else title[:80]
                                    print(f"     Added '{short_title}' from {api_name}")

                                if len(all_papers) >= max_papers:
                                    stop_search = True
                                    break

                    except Exception as e:
                        if verbose:
                            print(f"       {api_name} failed: {e}")

                    if stop_search:
                        break  # stop API loop

                    sleep(0.5)  # Rate limiting

                if stop_search:
                    break  # stop title loop

            if new_found == 0:
                rounds_without_new += 1
            else:
                rounds_without_new = 0  # reset if progress made
                if verbose and new_found > 3:
                    print(f"     Added {new_found} papers in this round")

            if stop_search:
                break  # stop the while loop early if max reached

        if verbose:
            print(f"\n    Found {len(all_papers)} unique papers total (API calls: {api_call_count}/{max_api_calls}).")
            if api_call_count >= max_api_calls:
                print(f"     Search stopped: API call limit reached ({max_api_calls} calls).")
            elif rounds_without_new >= max_rounds:
                print("     Search stopped early (no new papers found).")

        return all_papers[:max_papers]

    # ======================== PAPER CONVERSION (PRIVATE) ========================

    def _dict_to_paper(self, paper_dict: Dict[str, Any]) -> Paper:
        """Convert API paper dictionary to Paper object.

        Private method - only used internally during search.

        Args:
            paper_dict: Raw paper data from API

        Returns:
            Paper domain object
        """
        # Get full_text_sections from API if available, otherwise empty list
        full_text_sections = paper_dict.get("full_text_sections", [])

        # Fallback: if no full_text_sections but has full_text string, create single section
        if not full_text_sections and paper_dict.get("full_text"):
            full_text_sections = [("full_text", paper_dict["full_text"])]

        return Paper(
            id=paper_dict.get("id", paper_dict.get("title", "unknown")),
            doi=paper_dict.get("doi", "unknown"),
            title=paper_dict.get("title", "Unknown"),
            abstract=paper_dict.get("abstract", ""),
            authors=paper_dict.get("authors", []),
            year=paper_dict.get("year"),
            citations=paper_dict.get("citations", 0),
            url=paper_dict.get("url", ""),
            pdf_url=paper_dict.get("pdf_url", ""),
            source=paper_dict.get("source", "unknown"),
            has_pdf=paper_dict.get("has_pdf", False),
            pmc_id=paper_dict.get("pmc_id"),
            full_text=full_text_sections,
        )

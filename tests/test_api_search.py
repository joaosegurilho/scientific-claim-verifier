"""Interactive test interface for all APIs."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scverifier.api.api_core import CoreAPI
from scverifier.api.api_pmc import PMCAPI
from scverifier.api.api_pubmed import PubMedAPI
from scverifier.api.api_semantic_scholar import SemanticScholarAPI
from scverifier.api.api_openalex import OpenAlexAPI


class UnifiedAPISearch:
    """Unified interface for searching all APIs."""

    def __init__(self):
        self.core_api = CoreAPI()
        self.pmc_api = PMCAPI()
        self.pubmed_api = PubMedAPI()
        self.s2_api = SemanticScholarAPI()
        self.openalex_api = OpenAlexAPI()

    def search_core(self, query: str, limit: int = 5):
        """Search CORE."""

        print(f"\nSearching CORE for: '{query}'...")

        try:
            papers = self.core_api.search_papers(query, limit)
            self._print_results(papers, "CORE")
            return papers
        except Exception as e:
            print(f"Error: {e}")
            return []

    def search_pubmed(self, query: str, limit: int = 5):
        """Search PubMed."""

        print(f"\nSearching PubMed for: '{query}'...")

        try:
            papers = self.pubmed_api.search_papers(query, limit)
            self._print_results(papers, "PubMed")
            return papers
        except Exception as e:
            print(f"Error: {e}")
            return []

    def search_semantic_scholar(self, query: str, limit: int = 5):
        """Search Semantic Scholar."""

        print(f"\nSearching Semantic Scholar for: '{query}'...")

        try:
            papers = self.s2_api.search_papers(query, limit)
            self._print_results(papers, "Semantic Scholar")
            return papers
        except Exception as e:
            print(f"Error: {e}")
            return []

    def search_openalex(self, query: str, limit: int = 5):
        """Search OpenAlex."""

        print(f"\nSearching OpenAlex for: '{query}'...")

        try:
            papers = self.openalex_api.search_papers(query, limit)
            self._print_results(papers, "OpenAlex")
            return papers
        except Exception as e:
            print(f"Error: {e}")
            return []

    def search_all(self, query: str, limit_per_source: int = 5):
        """Search all APIs."""

        print(f"\nSearching ALL APIs for: '{query}'...")
        print("=" * 60)

        all_papers = []

        # CORE
        core_papers = self.search_core(query, limit_per_source)
        all_papers.extend(core_papers)

        # Semantic Scholar
        s2_papers = self.search_semantic_scholar(query, limit_per_source)
        all_papers.extend(s2_papers)

        # PubMed
        pubmed_papers = self.search_pubmed(query, limit_per_source)
        all_papers.extend(pubmed_papers)

        # OpenAlex
        openalex_papers = self.search_openalex(query, limit_per_source)
        all_papers.extend(openalex_papers)

        # Summary
        print(f"\nTotal papers found: {len(all_papers)}")
        print(f"   - Semantic Scholar: {len(s2_papers)}")
        print(f"   - PubMed: {len(pubmed_papers)}")
        print(f"   - CORE: {len(core_papers)}")
        print(f"   - OpenAlex: {len(openalex_papers)}")

        pdf_available = sum(1 for p in all_papers if p.get("has_pdf"))
        print(f"   - With PDF available: {pdf_available}/{len(all_papers)}")

        return all_papers

    def get_pmc_full_text(self, pmc_id: str):
        """Fetch and print PMC article data in a user-friendly, CLI-style way."""

        print(f"Command: pmc: {pmc_id}\n")
        print(f"Fetching full-text for PMC ID: {pmc_id}...")

        result = self.pmc_api.get_full_text(pmc_id)

        if not result:
            print(f"Failed to retrieve PMC article {pmc_id}.")
            return {
                "pmc_id": pmc_id,
                "title": None,
                "abstract": None,
                "sections": {},
                "full_text": "",
                "pdf_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/",
                "is_scanned": False,
                "has_full_text": False,
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
                "source": "pmc",
            }

        title = result.get("title", "No title")
        pdf_url = result.get("pdf_url")
        sections = result.get("sections", {})
        is_scanned = result.get("is_scanned", False)

        if is_scanned:
            print("This article is scanned (PDF only, no full text available in XML).")
        elif not sections:
            print("No body sections found in XML (possibly abstract-only).")
        else:
            print(f"Article contains {len(sections)} full-text section(s).")

        total_len = len(result.get("full_text", ""))
        print(f"   Title: {title}")
        print(f"   Sections: {len(sections) if sections else 'None'}")
        print(f"   Total length: {total_len} characters")
        print(f"   PDF URL: {pdf_url}\n")

        return result

    def _print_results(self, papers, source_name):
        """Print search results in a readable format."""

        if not papers:
            print(f"No papers found in {source_name}")
            return

        print(f"\nFound {len(papers)} papers in {source_name}:")
        print("=" * 60)

        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            authors = paper.get("authors", [])
            if authors:
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += "..."
                print(f"   Authors: {author_str}")

            if paper.get("year"):
                print(f"   Year: {paper['year']}", end="")
            if paper.get("citations") is not None:
                print(f" | Citations: {paper['citations']}", end="")
            if paper.get("doi") is not None:
                print(f" | DOI: {paper['doi']}", end="")
            print()

            # PDF availability
            if paper.get("has_pdf"):
                print(f"   PDF available: {paper.get('pdf_url', 'Yes')}")
            else:
                print("   No PDF available")

            # PMC ID for PubMed papers
            if paper.get("pmc_id"):
                print(f"   PMC ID: {paper['pmc_id']} (full-text available)")

            # Abstract preview
            if paper.get("abstract"):
                abstract = paper["abstract"][:150] + "..." if len(paper["abstract"]) > 150 else paper["abstract"]
                print(f"   Abstract: {abstract}")

            print(f"   URL: {paper.get('url', 'N/A')}")


def interactive_mode():
    """Interactive search interface."""
    searcher = UnifiedAPISearch()

    print("\n" + "=" * 60)
    print("UNIFIED API SEARCH INTERFACE")
    print("=" * 60)
    print("Commands:")
    print("  'core: <query>' - Search CORE")
    print("  's2: <query>' - Search Semantic Scholar")
    print("  'pubmed: <query>' - Search PubMed")
    print("  'openalex: <query>' - Search OpenAlex")
    print("  'all: <query>' - Search all APIs")
    print("  'pmc: <pmc_id>' - Get full-text from PMC")
    print("  'quit' - Exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYour command: ").strip()
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.startswith("core:"):
                query = user_input[5:].strip()
                searcher.search_core(query)
            elif user_input.startswith("s2:"):
                query = user_input[3:].strip()
                searcher.search_semantic_scholar(query)
            elif user_input.startswith("pubmed:"):
                query = user_input[7:].strip()
                searcher.search_pubmed(query)
            elif user_input.startswith("openalex:"):
                query = user_input[9:].strip()
                searcher.search_openalex(query)
            elif user_input.startswith("all:"):
                query = user_input[4:].strip()
                searcher.search_all(query)
            elif user_input.startswith("pmc:"):
                pmc_id = user_input[4:].strip()
                searcher.get_pmc_full_text(pmc_id)
            else:
                print("Invalid command. Use 'core:', 's2:', 'pubmed:', 'openalex:', 'all:', or 'pmc:'")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_quick_tests():
    """Run quick tests on all APIs."""
    searcher = UnifiedAPISearch()
    print("\nRunning Quick Tests...")
    print("=" * 60)

    # Test 3: CORE
    print("\nTest 1: CORE")
    _ = searcher.search_core("machine learning healthcare", limit=3)

    # Test 1: Semantic Scholar
    print("\nTest 2: Semantic Scholar")
    _ = searcher.search_semantic_scholar("transformer neural networks", limit=3)

    # Test 2: PubMed
    print("\nTest 3: PubMed")
    pubmed_papers = searcher.search_pubmed("COVID-19 vaccine efficacy", limit=3)

    # Test 3: OpenAlex
    print("\nTest 4: OpenAlex")
    _ = searcher.search_openalex("deep learning computer vision", limit=3)

    # Test 4: PMC Full-text (if we found a PMC ID)
    pmc_paper = next((p for p in pubmed_papers if p.get("pmc_id")), None)
    if pmc_paper:
        print("\nTest 5: PMC Full-text")
        searcher.get_pmc_full_text(pmc_paper["pmc_id"])

    # Test 5: Search all
    print("\nTest 6: Search All APIs")
    searcher.search_all("BERT language model", limit_per_source=2)
    print("\n" + "=" * 60)
    print("All tests complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_quick_tests()
    else:
        interactive_mode()

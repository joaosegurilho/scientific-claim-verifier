"""OpenAlex API interface for accessing open research papers."""

import requests
import time
from typing import List, Dict, Any
from scverifier.config.settings import Config
from scverifier.api.api import API


class OpenAlexAPI(API):
    """OpenAlex API interface with comprehensive paper metadata."""

    def __init__(self):
        Config.setup_environment()
        self.api_key = Config.OPENALEX_API_KEY  # Optional
        self.mailto = Config.OPENALEX_MAILTO  # Required for polite pool
        self.base_url = "https://api.openalex.org/works"
        self.last_request_time = 0

    def _convert_inverted_index_to_text(self, inverted_index: Dict[str, List[int]]) -> str:
        """Convert OpenAlex inverted index to plaintext.

        OpenAlex stores abstracts as inverted indexes where each word maps to
        a list of positions. This function reconstructs the original text.

        Args:
            inverted_index: Dictionary mapping words to position lists

        Returns:
            Reconstructed plaintext string
        """
        try:
            if not inverted_index or not isinstance(inverted_index, dict):
                return ""

            # Collect all (position, word) pairs
            all_positions = []
            for word, positions in inverted_index.items():
                if not isinstance(positions, list):
                    continue
                for pos in positions:
                    if not isinstance(pos, int):
                        continue
                    all_positions.append((pos, word))

            if not all_positions:
                return ""

            # Sort by position and join
            all_positions.sort(key=lambda x: x[0])
            return " ".join(word for _, word in all_positions)

        except Exception as e:
            print(f"[OpenAlex] Error converting inverted index: {e}")
            return ""

    def _make_request(self, params: Dict) -> Dict:
        """Make a request with rate limiting and polite pool access.

        Args:
            params: Query parameters for the request

        Returns:
            JSON response as dictionary
        """
        # Rate limiting: 10 requests per second = 0.1 second delay
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self.last_request_time = time.time()

        # Add mailto parameter (required for polite pool)
        params["mailto"] = self.mailto

        # Add API key if available (optional)
        if self.api_key:
            params["api_key"] = self.api_key

        # Make request
        response = requests.get(self.base_url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search OpenAlex for papers.

        Args:
            query: Search query string
            limit: Maximum number of results (OpenAlex max is 200 per page)

        Returns:
            List of standardized paper dictionaries
        """
        params = {
            "search": query,
            "per_page": min(limit, 200),  # OpenAlex max per page
        }

        try:
            data = self._make_request(params)
        except requests.exceptions.Timeout:
            print("[OpenAlex] Request timeout")
            return []
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("[OpenAlex] Rate limit exceeded")
            else:
                print(f"[OpenAlex] HTTP error {e.response.status_code}: {e}")
            return []
        except Exception as e:
            print(f"[OpenAlex] Request failed: {str(e)}")
            return []

        papers = []
        for item in data.get("results", []):
            try:
                # Extract OpenAlex ID (format: "https://openalex.org/W1234567")
                openalex_id = item.get("id", "")
                short_id = openalex_id.split("/")[-1] if openalex_id else ""

                # Extract DOI (format: "https://doi.org/10.1234/...")
                doi_url = item.get("doi", "")
                doi = doi_url.replace("https://doi.org/", "") if doi_url else None

                # Convert abstract from inverted index
                inverted_abstract = item.get("abstract_inverted_index")
                abstract = self._convert_inverted_index_to_text(inverted_abstract) if inverted_abstract else ""

                # Extract authors from authorships array
                authors = []
                for authorship in item.get("authorships", []):
                    author = authorship.get("author", {})
                    name = author.get("display_name")
                    if name:
                        authors.append(name)

                # Extract PDF URL from open access info
                pdf_url = None
                best_oa = item.get("best_oa_location", {})
                open_access = item.get("open_access", {})

                if best_oa and best_oa.get("pdf_url"):
                    pdf_url = best_oa["pdf_url"]
                elif open_access.get("oa_url"):
                    pdf_url = open_access["oa_url"]

                # Use display_name as fallback for title
                title = item.get("title") or item.get("display_name", "")

                papers.append(
                    {
                        "id": short_id or openalex_id,
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": item.get("publication_year"),
                        "citations": item.get("cited_by_count", 0),
                        "url": openalex_id,
                        "pdf_url": pdf_url,
                        "has_pdf": pdf_url is not None,
                        "source": "openalex",
                    }
                )
            except Exception as e:
                print(f"[OpenAlex] Error parsing paper: {e}")
                continue

        if not papers:
            print(f"[OpenAlex] No results found for query: {query}")
        else:
            print(f"[OpenAlex] Found {len(papers)} results for query: {query}")

        return papers

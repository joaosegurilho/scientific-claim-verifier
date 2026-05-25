"""Simplified Semantic Scholar API interface with DOI support."""

import requests
import time
from typing import List, Dict, Any
from scverifier.config.settings import Config
from scverifier.api.api import API


class SemanticScholarAPI(API):
    """Semantic Scholar API interface with DOI and PDF support."""

    def __init__(self):
        Config.setup_environment()
        self.api_key = Config.SEMANTIC_SCHOLAR_API_KEY
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        self.last_request_time = 0

    def _make_request(self, endpoint: str, params: Dict | None = None) -> Dict:
        """Make a request with rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self.last_request_time = time.time()

        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params or {})
        response.raise_for_status()
        return response.json()

    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers and include DOI and PDF info.
        """
        params = {
            "query": query,
            "limit": min(limit, 100),
            # Add externalIds to get DOI
            "fields": "paperId,title,abstract,authors,year,citationCount,url,venue,openAccessPdf,externalIds",
        }

        try:
            data = self._make_request("paper/search", params)
        except Exception as e:
            print(f"[Semantic Scholar] Request failed: {str(e)}")
            return []
        
        papers = []

        for item in data.get("data", []):
            authors = [a.get("name", "Unknown") for a in item.get("authors", [])]
            pdf_info = item.get("openAccessPdf")
            pdf_url = pdf_info.get("url") if pdf_info else None
            doi = item.get("externalIds", {}).get("DOI", None)

            papers.append(
                {
                    "id": item.get("paperId", ""),
                    "doi": doi,
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", ""),
                    "authors": authors,
                    "year": item.get("year"),
                    "citations": item.get("citationCount", 0),
                    "url": item.get("url", ""),
                    "venue": item.get("venue", ""),
                    "pdf_url": pdf_url,
                    "has_pdf": pdf_url is not None,
                    "source": "semantic_scholar",
                }
            )

        if not papers:
            print(f"[Semantic Scholar] No results found for query: {query}")
        else:
            print(f"[Semantic Scholar] Found {len(papers)} results for query: {query}")
        
        return papers

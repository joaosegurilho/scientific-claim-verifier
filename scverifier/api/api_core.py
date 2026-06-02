"""CORE API interface for open access research papers."""

import logging
import requests
import time
from typing import List, Dict, Any
from scverifier.config.settings import Config
from scverifier.api.api import API

logger = logging.getLogger(__name__)


class CoreAPI(API):
    """CORE API interface for accessing open access research papers."""

    def __init__(self):
        Config.setup_environment()
        self.api_key = Config.CORE_API_KEY
        self.base_url = "https://api.core.ac.uk/v3"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.last_request_time = 0

    def _make_request(self, endpoint: str, params: Dict = None, json_data: Dict = None) -> Dict:
        """Make a request with rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.5:  # 2 requests per second to be safe
            time.sleep(0.5 - elapsed)
        self.last_request_time = time.time()

        url = f"{self.base_url}/{endpoint}"

        try:
            # Add timeout to prevent hanging
            if json_data:
                response = requests.post(url, headers=self.headers, json=json_data, timeout=20)
            else:
                response = requests.get(url, headers=self.headers, params=params or {}, timeout=20)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("CORE API timeout for: %s", endpoint)
            return {}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 522:
                logger.warning("CORE API temporarily unavailable (522)")
            else:
                logger.warning("CORE API HTTP error %d: %s", e.response.status_code, e)
            return {}
        except requests.exceptions.ConnectionError:
            logger.warning("CORE API connection error")
            return {}
        except Exception as e:
            logger.error("CORE API unexpected error: %s", e)
            return {}

    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search CORE for papers.

        Args:
            query: Search terms
            limit: Number of results (max 100)

        Returns:
            List of papers with abstracts and PDFs
        """
        # CORE API uses POST for search
        json_data = {
            "q": query,
            "limit": min(limit, 100),
        }

        try:
            data = self._make_request("search/works", json_data=json_data)

            if not data or not data.get("results"):
                logger.info("No results found for query: %s", query)
                return []

            papers = []

            for item in data.get("results", []):
                # Extract authors
                authors = []
                if item.get("authors"):
                    authors = [author.get("name", "Unknown") for author in item["authors"]]

                # Get PDF URL
                pdf_url = item.get("downloadUrl")

                # Get abstract
                abstract = item.get("abstract", "")

                paper = {
                    "id": str(item.get("id", "")),
                    "doi": str(item.get("doi", "")),
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "authors": authors,
                    "year": item.get("yearPublished"),
                    "citations": None,  # CORE doesn't provide citation counts in search
                    "url": item.get("links", [{}])[0].get("url", "") if item.get("links") else "",
                    "pdf_url": pdf_url,
                    "has_pdf": pdf_url is not None,
                    "source": "core",
                }
                papers.append(paper)

            if papers:
                logger.info("Found %d results for query: %s", len(papers), query)

            return papers

        except Exception as e:
            logger.error("API error: %s", e)
            return []

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific CORE paper."""
        try:
            data = self._make_request(f"works/{paper_id}")

            # Extract authors
            authors = []
            if data.get("authors"):
                authors = [author.get("name", "Unknown") for author in data["authors"]]

            # Get PDF URL
            pdf_url = data.get("downloadUrl")

            return {
                "id": str(data.get("id", "")),
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "authors": authors,
                "year": data.get("yearPublished"),
                "citations": None,
                "url": data.get("links", [{}])[0].get("url", "") if data.get("links") else "",
                "pdf_url": pdf_url,
                "has_pdf": pdf_url is not None,
                "source": "core",
            }

        except Exception as e:
            logger.error("Error fetching CORE paper details: %s", e)
            return {}

"""PubMed API interface for biomedical literature."""

import json
import logging
import requests
import time
from typing import List, Dict, Any
from xml.etree import ElementTree as ET
from scverifier.api.api import API

logger = logging.getLogger(__name__)


class PubMedAPI(API):
    """PubMed API interface."""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.last_request_time = 0

    def _make_request(self, endpoint: str, params: Dict, max_retries: int = 3, retry_delay: int = 2) -> str:
        """Make a request with rate limiting and retries.

        Args:
            endpoint: API endpoint
            params: Query parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
        """
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                elapsed = time.time() - self.last_request_time
                if elapsed < 0.34:  # ~3 requests per second
                    time.sleep(0.34 - elapsed)
                self.last_request_time = time.time()

                url = f"{self.base_url}/{endpoint}"
                response = requests.get(url, params=params)
                response.raise_for_status()

                content = response.text
                content_type = response.headers.get("content-type", "").lower()

                # Check for maintenance page or HTML error pages
                if '<meta name="maintenance"' in content or (
                    content.startswith("<!DOCTYPE html>") or content.startswith("<html>")
                ):
                    if attempt < max_retries - 1:
                        logger.warning("PubMed appears to be in maintenance. Retrying in %d seconds...", retry_delay)
                        time.sleep(retry_delay)
                        continue
                    raise requests.RequestException("PubMed is currently under maintenance")

                # Verify response format matches requested format
                if "json" in params.get("retmode", "") and "json" not in content_type:
                    if attempt < max_retries - 1:
                        logger.warning("Expected JSON but got %s. Retrying in %d seconds...", content_type, retry_delay)
                        time.sleep(retry_delay)
                        continue
                    raise requests.RequestException(f"Expected JSON response but got {content_type}")

                return content

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning("Request failed: %s. Retrying in %d seconds...", str(e), retry_delay)
                    time.sleep(retry_delay)
                else:
                    raise

        # This should never be reached, but add it to satisfy type checker
        raise requests.RequestException("All retry attempts failed")

    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed for papers.

        Args:
            query: Search terms
            limit: Number of results (max 100)

        Returns:
            List of papers with abstracts
        """
        # Step 1: Search for paper IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(limit, 100),
            "retmode": "json",
        }

        try:
            search_result = self._make_request("esearch.fcgi", search_params)

            # Try to detect HTML error pages
            if search_result.strip().startswith("<!DOCTYPE html>") or search_result.strip().startswith("<html>"):
                logger.warning(
                    "PubMed returned an HTML page instead of JSON. The service might be experiencing issues."
                )
                return []

            search_data = json.loads(search_result)
            pmids = search_data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                logger.info("No results found for query: %s", query)
            else:
                logger.info("Found %d results for query: %s", len(pmids), query)

        except requests.RequestException as e:
            logger.warning("Request failed: %s", str(e))
            return []
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON response: %s", str(e))
            logger.debug("Response content: %s...", search_result[:200])
            return []

        if not pmids:
            return []

        # Step 2: Fetch paper details
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }

        fetch_result = self._make_request("efetch.fcgi", fetch_params)
        return self._parse_pubmed_xml(fetch_result)

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response."""
        root = ET.fromstring(xml_text)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            try:
                # Extract basic info
                pmid_elem = article.find(".//PMID")
                if pmid_elem is None or pmid_elem.text is None:
                    continue
                pmid = pmid_elem.text

                title_elem = article.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()) if title_elem is not None else "No title"

                # Extract authors
                authors = []
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None and lastname.text is not None:
                        name = lastname.text
                        if forename is not None and forename.text is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)

                # Extract abstract
                abstract_texts = []
                for abstract_elem in article.findall(".//AbstractText"):
                    abstract_texts.append("".join(abstract_elem.itertext()))
                abstract_text = " ".join(abstract_texts) if abstract_texts else ""

                # Extract year
                year = None
                pub_date = article.find(".//PubDate/Year")
                if pub_date is not None and pub_date.text is not None:
                    try:
                        year = int(pub_date.text)
                    except ValueError:
                        pass

                # Check for PMC ID (indicates full-text availability)
                pmc_id = None
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "pmc":
                        pmc_id = article_id.text
                        break

                doi = None
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break

                paper = {
                    "id": pmid,
                    "doi": doi,
                    "pmc_id": pmc_id,
                    "title": title,
                    "abstract": abstract_text,
                    "authors": authors,
                    "year": year,
                    "citations": None,  # PubMed doesn't provide citation counts
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "pdf_url": None,  # Will be handled by PMC API if pmc_id exists
                    "has_pdf": pmc_id is not None,
                    "source": "pubmed",
                }
                papers.append(paper)

            except Exception as e:
                logger.warning("Error parsing article: %s", e)
                continue

        return papers

    def get_paper_details(self, pmid: str) -> Dict[str, Any]:
        """Get detailed information for a specific PubMed paper."""
        papers = self.search_papers(f"{pmid}[PMID]", limit=1)
        return papers[0] if papers else {}

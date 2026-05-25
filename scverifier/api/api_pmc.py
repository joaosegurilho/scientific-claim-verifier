"""PubMed Central API interface for full-text articles."""

import requests
import time
from typing import Dict, Any
from xml.etree import ElementTree as ET


class PMCAPI:
    """PubMed Central API for full-text article access."""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.last_request_time = 0

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Make a request with rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.34:
            time.sleep(0.34 - elapsed)
        self.last_request_time = time.time()

        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.text

    def get_full_text(self, pmc_id: str) -> Dict[str, Any] | None:
        """
        Get full-text content from PMC.

        Args:
            pmc_id: PMC ID (with or without 'PMC' prefix)

        Returns:
            Dictionary with full-text sections or None if not available.
        """
        if not pmc_id.startswith("PMC"):
            pmc_id = f"PMC{pmc_id}"

        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml",
        }

        try:
            xml_text = self._make_request("efetch.fcgi", params)
            return self._parse_pmc_xml(xml_text, pmc_id)
        except Exception:
            return None

    def _parse_pmc_xml(self, xml_text: str, pmc_id: str) -> Dict[str, Any] | None:
        """Parse PMC XML to extract full-text sections."""
        root = ET.fromstring(xml_text)
        article = root.find(".//article")
        if article is None:
            return None

        # Title
        title_elem = article.find(".//article-title")
        title = "".join(title_elem.itertext()) if title_elem is not None else "No title"

        # Abstract
        abstract_elem = article.find(".//abstract")
        abstract = "".join(abstract_elem.itertext()) if abstract_elem is not None else ""

        # Check if scanned PDF only
        is_scanned = any(
            m.find("meta-name") is not None
            and m.find("meta-name").text == "pmc-prop-is-scanned-article"
            and m.find("meta-value") is not None
            and m.find("meta-value").text == "yes"
            for m in root.findall(".//custom-meta")
        )

        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"

        if is_scanned:
            # For scanned PDFs, only abstract available
            full_text_sections = []
            if abstract:
                full_text_sections.append(("abstract", abstract))

            return {
                "pmc_id": pmc_id,
                "title": title,
                "abstract": abstract,
                "sections": {},
                "full_text": abstract or "",
                "full_text_sections": full_text_sections,
                "pdf_url": pdf_url,
                "is_scanned": True,
                "has_full_text": False,
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
                "source": "pmc",
            }

        # Extract body sections
        sections: Dict[str, str] = {}
        body = article.find(".//body")
        if body is not None:
            for sec in body.findall(".//sec"):
                title_elem = sec.find("title")
                section_title = "".join(title_elem.itertext()) if title_elem is not None else "Untitled"

                paragraphs = []
                for p in sec.findall(".//p"):
                    para_text = "".join(p.itertext()).strip()
                    if para_text:
                        paragraphs.append(para_text)

                if paragraphs:
                    sections[section_title] = "\n\n".join(paragraphs)

        full_text = self._combine_sections(abstract, sections)
        has_full_text = len(sections) > 0 and not is_scanned

        # Convert sections to list of tuples for Paper.full_text
        full_text_sections = []
        if abstract:
            full_text_sections.append(("abstract", abstract))
        for section_title, content in sections.items():
            full_text_sections.append((section_title.lower().replace(" ", "_"), content))

        return {
            "pmc_id": pmc_id,
            "title": title,
            "abstract": abstract,
            "sections": sections,
            "full_text": full_text,
            "full_text_sections": full_text_sections,
            "pdf_url": pdf_url,
            "is_scanned": is_scanned,
            "has_full_text": has_full_text,
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
            "source": "pmc",
        }

    def _combine_sections(self, abstract: str, sections: Dict[str, str]) -> str:
        """Combine abstract and sections into a single text string."""
        parts = []
        if abstract:
            parts.append(f"ABSTRACT\n{abstract}")
        for section_title, content in sections.items():
            parts.append(f"{section_title.upper()}\n{content}")
        return "\n\n".join(parts)

    def check_availability(self, pmc_id: str) -> bool:
        """Check if the given PMC ID has full-text available (not scanned PDF-only)."""
        result = self.get_full_text(pmc_id)
        return bool(result and result.get("has_full_text"))

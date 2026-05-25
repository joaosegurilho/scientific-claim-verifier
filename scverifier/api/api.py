from typing import List, Dict, Any


class API:
    """Base class for API clients."""

    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers using the API.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            A list of dictionaries, where each dictionary represents a paper and contains:
                - id (str): Unique identifier for the paper.
                - title (str): Title of the paper.
                - abstract (str): Abstract of the paper.
                - authors (List[str]): List of author names.
                - year (int): Year of publication.
                - citations (int): Number of citations.
                - url (str): URL to the paper.
                - pdf_url (str): URL to the PDF, if available.
                - has_pdf (bool): Whether a PDF is available.
                - source (str): Source of the paper (e.g., 'core', 'pubmed', 'semantic_scholar').
        """
        raise NotImplementedError("Subclasses must implement this method.")

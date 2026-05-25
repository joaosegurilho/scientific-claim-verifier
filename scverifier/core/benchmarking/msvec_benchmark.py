"""MSVEC benchmark implementation.

Loads and manages the MSVEC (Multi-Domain Scientific Claim Verification Evaluation Corpus)
dataset for claim verification benchmarking.

MSVEC contains scientific claims from multiple domains (Biology, Medicine, Space, Science,
Geology, Political, COVID-related) with verified truthfulness labels from fact-checking
websites (Snopes.com, Politifact.com).

Dataset source: https://github.com/lamps-lab/msvec
Paper: "MSVEC: A Multidomain Testing Dataset for Scientific Claim Verification" (MobiQuitous 2023)
"""

import json
from pathlib import Path
from typing import Optional, List

from scverifier.core.benchmarking.base import Benchmark, BenchmarkItem, VerificationMethod


class MSVEC(Benchmark):
    """MSVEC benchmark dataset.

    MSVEC (Multi-Domain Scientific Claim Verification Evaluation Corpus) contains
    scientific claims from multiple domains with verified truthfulness labels.

    The dataset includes 56 scientific claims.

    Labels:
    - true: Claim is verified as true
    - false: Claim is verified as false
    - mixed: Claim has mixed evidence (mapped to INSUFFICIENT_EVIDENCE)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        verification_method: VerificationMethod = VerificationMethod.AGENTLESS
    ):
        """Initialize MSVEC benchmark.

        Args:
            data_dir: Path to MSVEC data directory. If None, uses default location.
            verification_method: Method to use for verification
        """
        super().__init__(name="MSVEC")

        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default location relative to project root
            self.data_dir = Path("data/msvec_data")

        self.verification_method = verification_method

    def load(self, max_items: Optional[int] = None) -> List[BenchmarkItem]:
        """Load MSVEC claims.

        The dataset can be in different formats depending on the version:
        1. claims.jsonl - One claim per line in JSONL format
        2. claims.json - Array of claims in JSON format
        3. Individual claim files in a claims/ subdirectory

        Args:
            max_items: Maximum number of items to load (None for all)

        Returns:
            List of BenchmarkItem objects

        Raises:
            FileNotFoundError: If data directory or claims file not found
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MSVEC data directory not found at {self.data_dir}. "
                f"Please download from https://github.com/lamps-lab/msvec"
            )

        # Try different file formats
        claims_jsonl = self.data_dir / "claims.jsonl"
        claims_json = self.data_dir / "claims.json"
        claims_csv = self.data_dir / "claims.csv"

        if claims_jsonl.exists():
            self._load_jsonl(claims_jsonl, max_items)
        elif claims_json.exists():
            self._load_json(claims_json, max_items)
        elif claims_csv.exists():
            self._load_csv(claims_csv, max_items)
        else:
            # Try to find any jsonl or json file
            json_files = list(self.data_dir.glob("*.json")) + list(self.data_dir.glob("*.jsonl"))
            if json_files:
                file_path = json_files[0]
                if file_path.suffix == ".jsonl":
                    self._load_jsonl(file_path, max_items)
                else:
                    self._load_json(file_path, max_items)
            else:
                raise FileNotFoundError(
                    f"No claims file found in {self.data_dir}. "
                    f"Expected claims.jsonl, claims.json, or claims.csv"
                )

        return self.items

    def _load_jsonl(self, file_path: Path, max_items: Optional[int] = None):
        """Load claims from JSONL file."""
        self.items = []

        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_items and len(self.items) >= max_items:
                    break

                data = json.loads(line)
                item = self._parse_claim(data, idx)
                if item:
                    self.items.append(item)

    def _load_json(self, file_path: Path, max_items: Optional[int] = None):
        """Load claims from JSON file."""
        self.items = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array and dict with 'claims' key
        if isinstance(data, list):
            claims_list = data
        elif isinstance(data, dict) and "claims" in data:
            claims_list = data["claims"]
        else:
            claims_list = [data]

        for idx, claim_data in enumerate(claims_list):
            if max_items and len(self.items) >= max_items:
                break

            item = self._parse_claim(claim_data, idx)
            if item:
                self.items.append(item)

    def _load_csv(self, file_path: Path, max_items: Optional[int] = None):
        """Load claims from CSV file."""
        import csv

        self.items = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if max_items and len(self.items) >= max_items:
                    break

                item = self._parse_claim(row, idx)
                if item:
                    self.items.append(item)

    def _parse_claim(self, data: dict, idx: int) -> Optional[BenchmarkItem]:
        """Parse a single claim from data dictionary.

        The MSVEC dataset uses SciFact-style format:
        - evidence dict with doc_id -> list of {sentences, label} where label is SUPPORT/CONTRADICT
        - Empty evidence dict means INSUFFICIENT_EVIDENCE

        Args:
            data: Dictionary containing claim data
            idx: Index for generating claim ID if not present

        Returns:
            BenchmarkItem or None if claim cannot be parsed
        """
        # Extract claim text (try different field names)
        claim_text = (
            data.get("claim") or
            data.get("claim_text") or
            data.get("text") or
            data.get("sentence")
        )

        if not claim_text:
            return None

        # Extract claim ID
        claim_id = str(data.get("id", data.get("claim_id", f"msvec_{idx}")))

        # Determine label from evidence (SciFact-style format)
        evidence_dict = data.get("evidence", {})
        expected_result = self._get_label_from_evidence(evidence_dict)

        # If no evidence-based label, try explicit label field
        if expected_result == "INSUFFICIENT_EVIDENCE" and not evidence_dict:
            label = str(data.get("label", data.get("verdict", ""))).lower().strip()
            if label:
                expected_result = self._map_label(label)

        # Extract metadata
        metadata = {
            "evidence": evidence_dict,
            "doc_ids": data.get("doc_ids", []),
            "domain": data.get("domain", data.get("category", "")),
            "source": data.get("source", ""),
        }

        return BenchmarkItem(
            claim_id=claim_id,
            claim=claim_text,
            expected_result=expected_result,
            verification_method=self.verification_method,
            metadata=metadata,
        )

    def _get_label_from_evidence(self, evidence_dict: dict) -> str:
        """Determine label from SciFact-style evidence dictionary.

        Args:
            evidence_dict: Dict mapping doc_id to list of evidence items

        Returns:
            SUPPORTS, REFUTES, or INSUFFICIENT_EVIDENCE
        """
        if not evidence_dict:
            return "INSUFFICIENT_EVIDENCE"

        label_counts = {"SUPPORT": 0, "CONTRADICT": 0}

        for doc_evidences in evidence_dict.values():
            if isinstance(doc_evidences, list):
                for ev in doc_evidences:
                    label = ev.get("label", "").strip().upper()
                    if label in label_counts:
                        label_counts[label] += 1

        # Determine majority
        if label_counts["SUPPORT"] > label_counts["CONTRADICT"]:
            return "SUPPORTS"
        elif label_counts["CONTRADICT"] > label_counts["SUPPORT"]:
            return "REFUTES"
        elif label_counts["SUPPORT"] == label_counts["CONTRADICT"] and label_counts["SUPPORT"] > 0:
            return "INSUFFICIENT_EVIDENCE"
        else:
            return "INSUFFICIENT_EVIDENCE"

    def _map_label(self, label: str) -> str:
        """Map MSVEC labels to standard verification labels.

        MSVEC uses: true, false, mixed, mostly true, mostly false, etc.

        Args:
            label: Original label from dataset

        Returns:
            Standardized label (SUPPORTS, REFUTES, or INSUFFICIENT_EVIDENCE)
        """
        label = label.lower().strip()

        # True variants -> SUPPORTS
        if label in ["true", "mostly true", "correct", "verified", "supported"]:
            return "SUPPORTS"

        # False variants -> REFUTES
        if label in ["false", "mostly false", "incorrect", "wrong", "refuted", "pants on fire"]:
            return "REFUTES"

        # Mixed/unclear -> INSUFFICIENT_EVIDENCE
        if label in ["mixed", "half true", "unproven", "unclear", "inconclusive"]:
            return "INSUFFICIENT_EVIDENCE"

        # Default for unknown labels
        return "INSUFFICIENT_EVIDENCE"

    def get_statistics(self) -> dict:
        """Get statistics about the MSVEC benchmark.

        Returns:
            Dictionary with statistics including domain distribution
        """
        base_stats = super().get_statistics()

        if not self.items:
            return base_stats

        # Domain distribution
        domain_counts = {}
        source_counts = {}

        for item in self.items:
            domain = item.metadata.get("domain", "unknown")
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            source = item.metadata.get("source", "unknown")
            if source:
                source_counts[source] = source_counts.get(source, 0) + 1

        base_stats["domain_distribution"] = domain_counts
        base_stats["source_distribution"] = source_counts

        return base_stats

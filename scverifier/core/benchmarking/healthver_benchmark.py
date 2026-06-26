"""HealthVer benchmark implementation.

Loads and manages the HealthVer dataset for claim verification benchmarking.
HealthVer contains 14,330 evidence-claim pairs of health-related claims verified
against scientific research articles.
Auto-downloads from the official S3 bucket when data is missing.

Dataset source: https://aclanthology.org/2021.findings-emnlp.297/
Original data: https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz
"""

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm

from scverifier.core.benchmarking.base import Benchmark, BenchmarkItem, VerificationMethod

HEALTHVER_DATA_DIR = Path("data/healthver_data")
HEALTHVER_URL = "https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz"
_REQUIRED_FILES = ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl", "corpus.jsonl"]

logger = logging.getLogger(__name__)


class HealthVer(Benchmark):
    """HealthVer benchmark dataset.

    HealthVer contains 14,330 evidence-claim pairs for fact-checking health-related claims
    against scientific articles. The dataset was created by extracting claims from
    search engine results about COVID-19 and verifying them against the CORD-19 corpus.
    """

    def __init__(self, verification_method: VerificationMethod = VerificationMethod.AGENTLESS):
        """Initialize HealthVer benchmark.

        Args:
            verification_method: Method to use for verification
        """
        super().__init__(name="HealthVer")
        self.data_dir = HEALTHVER_DATA_DIR
        self.verification_method = verification_method

    def _ensure_data_downloaded(self) -> None:
        if HEALTHVER_DATA_DIR.exists():
            missing = [f for f in _REQUIRED_FILES if not (HEALTHVER_DATA_DIR / f).exists()]
            if not missing:
                return

        HEALTHVER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        print("HealthVer data not found. Downloading...")
        response = requests.get(HEALTHVER_URL, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                buf.write(chunk)
                pbar.update(len(chunk))

        buf.seek(0)
        print("Extracting...")
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(member.name.endswith(f) for f in _REQUIRED_FILES):
                    member.name = Path(member.name).name
                    tar.extract(member, path=HEALTHVER_DATA_DIR)

        print(f"HealthVer data downloaded and extracted to {HEALTHVER_DATA_DIR}")

    def _load_corpus(self) -> dict:
        corpus = {}
        corpus_path = self.data_dir / "corpus.jsonl"
        if not corpus_path.exists():
            return corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["doc_id"]] = doc
        return corpus

    def _flatten(self, xss):
        return [x for xs in xss for x in xs]

    def load(self, max_items: Optional[int] = None, split: str = "test") -> List[BenchmarkItem]:
        """Load HealthVer claims.

        Args:
            max_items: Maximum number of items to load (None for all)
            split: Dataset split to load ('train', 'validation', or 'test')

        Returns:
            List of BenchmarkItem objects
        """
        self._ensure_data_downloaded()

        split_map = {
            "train": "claims_train.jsonl",
            "validation": "claims_dev.jsonl",
            "test": "claims_test.jsonl",
        }
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'")

        claims_file = self.data_dir / split_map[split]
        if not claims_file.exists():
            raise FileNotFoundError(f"HealthVer claims file not found at {claims_file}")

        corpus = self._load_corpus()

        # Load claims
        claims = []
        with open(claims_file, "r", encoding="utf-8") as f:
            for line in f:
                claims.append(json.loads(line))

        label_map = {
            "SUPPORT": "SUPPORTS",
            "CONTRADICT": "REFUTES",
            "NEI": "INSUFFICIENT_EVIDENCE",
        }

        self.items = []
        idx = 0
        for claim in claims:
            if max_items and len(self.items) >= max_items:
                break

            evidence = {int(k): v for k, v in claim.get("evidence", {}).items()}

            for cited_doc_id in claim.get("doc_ids", []):
                cited_doc = corpus.get(cited_doc_id, {})

                if cited_doc_id in evidence:
                    this_evidence = evidence[cited_doc_id]
                    verdict = this_evidence[0]["label"]
                    evidence_sents = self._flatten([entry["sentences"] for entry in this_evidence])
                else:
                    verdict = "NEI"
                    evidence_sents = []

                expected_result = label_map.get(verdict, "INSUFFICIENT_EVIDENCE")
                claim_id = f"healthver_{split}_{idx}"

                item = BenchmarkItem(
                    claim_id=claim_id,
                    claim=claim["claim"],
                    expected_result=expected_result,
                    verification_method=self.verification_method,
                    metadata={
                        "abstract_id": cited_doc_id,
                        "title": cited_doc.get("title", ""),
                        "abstract": cited_doc.get("abstract", []),
                        "evidence_sentences": evidence_sents,
                        "verdict_original": verdict,
                        "split": split,
                    },
                )
                self.items.append(item)
                idx += 1

        return self.items

    def get_statistics(self) -> dict:
        """Get statistics about the HealthVer benchmark.

        Returns:
            Dictionary with statistics including split distribution
        """
        base_stats = super().get_statistics()

        if not self.items:
            return base_stats

        # Additional HealthVer-specific statistics
        split_counts = {}
        for item in self.items:
            split = item.metadata.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1

        base_stats["split_distribution"] = split_counts

        return base_stats

"""SciFact benchmark implementation.

Loads and manages the SciFact dataset for claim verification benchmarking.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, List

from scverifier.core.benchmarking.base import Benchmark, BenchmarkItem, VerificationMethod


class SciFact(Benchmark):
    """SciFact benchmark dataset.

    SciFact is a dataset of scientific claims paired with evidence from research papers.
    Automatically combines all SciFact claim files (dev, test, train) if no specific file is provided.
    """

    def __init__(self, claims_file: Optional[str] = None, split: Optional[str] = None, verification_method: VerificationMethod = VerificationMethod.AGENTLESS):
        """Initialize SciFact benchmark.

        Args:
            claims_file: Path to a specific SciFact claims JSONL file. Takes precedence over split.
            split: Which split to use: 'train', 'dev', 'test', or None for all (default: None)
            verification_method: Method to use for verification
        """
        super().__init__(name="SciFact")

        # If claims_file is provided, use it directly
        if claims_file:
            self.claims_file = Path(claims_file)
        # If split is specified, use the corresponding file
        elif split:
            if split not in ['train', 'dev', 'test']:
                raise ValueError(f"Invalid split '{split}'. Must be 'train', 'dev', 'test', or None")
            self.claims_file = Path(f"data/scifact_data/claims_{split}.jsonl")
        # Otherwise, combine all files
        else:
            self.claims_file = None

        self.split = split
        self.verification_method = verification_method
        self._temp_combined_file = None

    def _combine_all_claims_files(self) -> Path:
        """Combine all SciFact claim files (dev, test, train) into one temporary file.
        
        Returns:
            Path to the temporary combined file
        """
        scifact_dir = Path("data/scifact_data")
        if not scifact_dir.exists():
            raise FileNotFoundError(f"SciFact data directory not found at {scifact_dir}")
        
        claim_files = [
            scifact_dir / "claims_train.jsonl",
            scifact_dir / "claims_dev.jsonl",
            scifact_dir / "claims_test.jsonl",
        ]
        
        # Check which files exist
        existing_files = [f for f in claim_files if f.exists()]
        if not existing_files:
            raise FileNotFoundError(
                f"No SciFact claim files found in {scifact_dir}. "
                f"Expected: claims_train.jsonl, claims_dev.jsonl, claims_test.jsonl"
            )
        
        # Create temporary combined file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        print(f"Combining SciFact claim files:")
        total_claims = 0
        for claim_file in existing_files:
            count = 0
            with open(claim_file, "r", encoding="utf-8") as f:
                for line in f:
                    temp_file.write(line)
                    count += 1
                    total_claims += 1
            print(f"  - {claim_file.name}: {count} claims")
        
        temp_file.flush()
        temp_file.close()
        
        print(f"Total claims combined: {total_claims}\n")
        self._temp_combined_file = temp_file.name
        return Path(temp_file.name)

    def load(self, max_items: Optional[int] = None) -> List[BenchmarkItem]:
        """Load SciFact claims.

        Args:
            max_items: Maximum number of items to load (None for all)

        Returns:
            List of BenchmarkItem objects
        """
        # If no specific file provided, combine all files
        if self.claims_file is None:
            self.claims_file = self._combine_all_claims_files()
        
        if not self.claims_file.exists():
            raise FileNotFoundError(f"SciFact claims file not found at {self.claims_file}")

        # Load all claims first
        all_items = []
        with open(self.claims_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # Determine expected result from evidence
                expected_result = self._get_label_from_evidence(data.get("evidence", {}))

                item = BenchmarkItem(
                    claim_id=str(data["id"]),
                    claim=data["claim"],
                    expected_result=expected_result,
                    verification_method=self.verification_method,
                    metadata={
                        "evidence": data.get("evidence", {}),
                        "cited_doc_ids": data.get("cited_doc_ids", []),
                    }
                )
                all_items.append(item)

        # Sort by claim_id (converting to int for proper numeric sorting)
        all_items.sort(key=lambda x: int(x.claim_id))
        
        # Apply max_items limit after sorting
        if max_items:
            self.items = all_items[:max_items]
        else:
            self.items = all_items

        return self.items

    def _get_label_from_evidence(self, evidence_dict) -> str:
        """Determine the majority label from evidence.

        Args:
            evidence_dict: Dictionary mapping doc_id to evidence list (SciFact-Orig format)
                          or evidence dict (SciFact-Open format)

        Returns:
            Label string (SUPPORTS, REFUTES, or INSUFFICIENT_EVIDENCE)
        """
        label_counts = {"SUPPORT": 0, "CONTRADICT": 0}

        if isinstance(evidence_dict, dict):
            for doc_evidences in evidence_dict.values():
                # Handle both SciFact-Orig (list) and SciFact-Open (dict) formats
                if isinstance(doc_evidences, list):
                    # SciFact-Orig: list of evidence sets
                    for ev in doc_evidences:
                        label = ev.get("label", "").strip().upper()
                        if label in label_counts:
                            label_counts[label] += 1
                elif isinstance(doc_evidences, dict):
                    # SciFact-Open: single evidence dict
                    label = doc_evidences.get("label", "").strip().upper()
                    if label in label_counts:
                        label_counts[label] += 1

        # Determine majority
        if label_counts["SUPPORT"] > label_counts["CONTRADICT"]:
            return "SUPPORTS"
        elif label_counts["CONTRADICT"] > label_counts["SUPPORT"]:
            return "REFUTES"
        elif label_counts["SUPPORT"] == label_counts["CONTRADICT"] and label_counts["SUPPORT"] > 0:
            # Tie with at least one label: treat as insufficient evidence
            return "INSUFFICIENT_EVIDENCE"
        else:
            # No evidence labels
            return "INSUFFICIENT_EVIDENCE"
    
    def __del__(self):
        """Clean up temporary combined file if it was created."""
        if self._temp_combined_file and Path(self._temp_combined_file).exists():
            try:
                Path(self._temp_combined_file).unlink()
            except Exception:
                pass

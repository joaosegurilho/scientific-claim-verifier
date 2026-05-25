"""HealthVer benchmark implementation.

Loads and manages the HealthVer dataset for claim verification benchmarking.
HealthVer contains 14,330 evidence-claim pairs of health-related claims verified
against scientific research articles.

Dataset source: https://aclanthology.org/2021.findings-emnlp.297/
HuggingFace: dwadden/healthver_entailment
"""

from typing import Optional, List

from scverifier.core.benchmarking.base import Benchmark, BenchmarkItem, VerificationMethod


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
        self.verification_method = verification_method

    def load(self, max_items: Optional[int] = None, split: str = "validation") -> List[BenchmarkItem]:
        """Load HealthVer claims.

        Args:
            max_items: Maximum number of items to load (None for all)
            split: Dataset split to load ('train', 'validation', or 'test')

        Returns:
            List of BenchmarkItem objects

        Raises:
            ImportError: If datasets library is not installed
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to load HealthVer. "
                "Install it with: pip install datasets"
            )

        # Load the dataset
        dataset = load_dataset("dwadden/healthver_entailment")

        # Get the specified split
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(
                f"Split '{split}' not found. Available splits: {available_splits}"
            )

        data_split = dataset[split]

        self.items = []
        for idx, example in enumerate(data_split):
            if max_items and len(self.items) >= max_items:
                break

            # Map the label to our format
            # HealthVer uses: 'support', 'refute', 'neutral'
            label_map = {
                "support": "SUPPORTS",
                "refute": "REFUTES",
                "neutral": "INSUFFICIENT_EVIDENCE",
            }

            label = example.get("label", "neutral")
            expected_result = label_map.get(label.lower(), "INSUFFICIENT_EVIDENCE")

            # Create unique claim ID
            claim_id = f"healthver_{split}_{idx}"

            item = BenchmarkItem(
                claim_id=claim_id,
                claim=example["claim"],
                expected_result=expected_result,
                verification_method=self.verification_method,
                metadata={
                    "evidence": example.get("evidence", ""),
                    "label_original": label,
                    "split": split,
                }
            )
            self.items.append(item)

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

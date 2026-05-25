"""CoverBench benchmark implementation.

Loads and manages the CoverBench dataset for claim verification benchmarking.
CoverBench is a challenging benchmark for complex claim verification with rich grounding contexts.
"""

from typing import Optional, List

from scverifier.core.benchmarking.base import Benchmark, BenchmarkItem, VerificationMethod


class CoverBench(Benchmark):
    """CoverBench benchmark dataset.

    CoverBench contains 733 examples of complex claims with rich grounding contexts.
    The dataset focuses on complex reasoning including long-context understanding,
    multi-step reasoning, and quantitative analysis.
    """

    def __init__(self, verification_method: VerificationMethod = VerificationMethod.AGENTLESS):
        """Initialize CoverBench benchmark.

        Args:
            verification_method: Method to use for verification
        """
        super().__init__(name="CoverBench")
        self.verification_method = verification_method

    def load(self, max_items: Optional[int] = None) -> List[BenchmarkItem]:
        """Load CoverBench claims.

        Args:
            max_items: Maximum number of items to load (None for all)

        Returns:
            List of BenchmarkItem objects

        Raises:
            ImportError: If datasets library is not installed
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to load CoverBench. "
                "Install it with: pip install datasets"
            )

        # Load the dataset
        dataset = load_dataset("google/coverbench")['eval']

        self.items = []
        for idx, example in enumerate(dataset):
            if max_items and len(self.items) >= max_items:
                break

            # Map the binary label to our format
            # CoverBench has binary labels: True (entailment) / False (no entailment)
            label = example.get("label", False)
            expected_result = "SUPPORTS" if label else "REFUTES"

            item = BenchmarkItem(
                claim_id=f"coverbench_{idx}",
                claim=example["claim"],
                expected_result=expected_result,
                verification_method=self.verification_method,
                metadata={
                    "context": example.get("context", ""),
                    "domain": example.get("domain", ""),
                    "complexity_sources": example.get("complexity_sources", []),
                    "dataset_source": example.get("dataset_source", ""),
                }
            )
            self.items.append(item)

        return self.items

    def get_statistics(self) -> dict:
        """Get statistics about the CoverBench benchmark.

        Returns:
            Dictionary with statistics including domain and complexity distribution
        """
        base_stats = super().get_statistics()

        if not self.items:
            return base_stats

        # Additional CoverBench-specific statistics
        domain_counts = {}
        source_counts = {}
        for item in self.items:
            domain = item.metadata.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            source = item.metadata.get("dataset_source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        base_stats["domain_distribution"] = domain_counts
        base_stats["source_distribution"] = source_counts

        return base_stats

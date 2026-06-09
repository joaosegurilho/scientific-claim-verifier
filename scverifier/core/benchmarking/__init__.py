"""Benchmarking module for scientific claim verification.

This module provides a common interface for running benchmarks on different
datasets (SciFact, CoverBench, etc.) with various verification methods.

Example usage:
    from scverifier.core.benchmarking import SciFact, CoverBench, VerificationMethod

    # Load SciFact benchmark
    scifact = SciFact(claims_file="./data/scifact_data/claims_dev.jsonl")
    scifact.load(max_items=10)

    # Load CoverBench benchmark
    coverbench = CoverBench()
    coverbench.load(max_items=10)

    # Run benchmark using the runner script:
    # python -m scverifier.core.benchmarking.run_benchmark scifact --max-items 10
"""

from typing import Optional

from scverifier.core.benchmarking.base import (
    Benchmark,
    BenchmarkItem,
    VerificationMethod,
    EvaluationMetrics,
    BenchmarkEvaluator,
)
from scverifier.core.benchmarking.scifact_benchmark import SciFact
from scverifier.core.benchmarking.coverbench_benchmark import CoverBench
from scverifier.core.benchmarking.healthver_benchmark import HealthVer
from scverifier.core.benchmarking.msvec_benchmark import MSVEC

BENCHMARK_MAP = {
    "scifact": SciFact,
    "coverbench": CoverBench,
    "healthver": HealthVer,
    "msvec": MSVEC,
}

METHOD_MAP = {
    "agent": VerificationMethod.AGENT,
    "agentless": VerificationMethod.AGENTLESS,
    "agent_with_search": VerificationMethod.AGENT_WITH_SEARCH,
    "agentless_with_search": VerificationMethod.AGENTLESS_WITH_SEARCH,
}


def get_benchmark(
    dataset: str,
    method: str,
    split: Optional[str] = None,
) -> Benchmark:
    benchmark_cls = BENCHMARK_MAP[dataset]
    verification_method = METHOD_MAP[method]

    if dataset == "scifact":
        benchmark = benchmark_cls(split=split, verification_method=verification_method)
    elif dataset == "healthver":
        benchmark = benchmark_cls(verification_method=verification_method)
    else:
        benchmark = benchmark_cls(verification_method=verification_method)

    benchmark.load(max_items=None)
    return benchmark


__all__ = [
    "Benchmark",
    "BenchmarkItem",
    "VerificationMethod",
    "EvaluationMetrics",
    "BenchmarkEvaluator",
    "SciFact",
    "CoverBench",
    "HealthVer",
    "MSVEC",
    "BENCHMARK_MAP",
    "METHOD_MAP",
    "get_benchmark",
]

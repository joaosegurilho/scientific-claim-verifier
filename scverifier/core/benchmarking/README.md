# Benchmarking Module

This module provides a unified interface for running benchmarks on different scientific claim verification datasets.

## Supported Datasets

### SciFact
- **Size**: ~300 claims (dev set)
- **Source**: Scientific papers from PubMed
- **Labels**: SUPPORTS, REFUTES, INSUFFICIENT_EVIDENCE
- **Format**: JSONL file with claims and evidence annotations

### CoverBench
- **Size**: 733 claims
- **Source**: Multiple datasets (FinQA, TabFact, PubMedQA, etc.)
- **Labels**: SUPPORTS, REFUTES (binary: entailment/no entailment)
- **Format**: HuggingFace dataset (google/coverbench)
- **Note**: Requires `datasets` library: `pip install datasets`

### HealthVer
- **Size**: 14,330 evidence-claim pairs
- **Source**: Health-related claims from COVID-19 search results, verified against CORD-19 corpus
- **Labels**: SUPPORTS, REFUTES, INSUFFICIENT_EVIDENCE (mapped from support/refute/neutral)
- **Format**: HuggingFace dataset (dwadden/healthver_entailment)
- **Splits**: train, validation, test
- **Note**: Requires `datasets` library: `pip install datasets`
- **Reference**: https://aclanthology.org/2021.findings-emnlp.297/

## Quick Start

### Running SciFact Benchmark

```bash
# Run on first 10 claims without online search
python -m scverifier.core.benchmarking.run_benchmark scifact \
    --max-items 10 \
    --method agentless

# Run with specific split
python -m scverifier.core.benchmarking.run_benchmark scifact \
    --scifact-split dev \
    --max-items 10 \
    --method agentless

# Run with online paper search
python -m scverifier.core.benchmarking.run_benchmark scifact \
    --max-items 10 \
    --method agentless_with_search \
    --max-papers 5
```

### Running CoverBench Benchmark

```bash
# Run on first 10 claims
python -m scverifier.core.benchmarking.run_benchmark coverbench \
    --max-items 10 \
    --method agentless

# Run all claims with search
python -m scverifier.core.benchmarking.run_benchmark coverbench \
    --method agent_with_search \
    --max-papers 10
```

### Running HealthVer Benchmark

```bash
# Run on first 100 validation claims
python -m scverifier.core.benchmarking.run_benchmark healthver \
    --max-items 100 \
    --split validation \
    --method agentless

# Run on test split with search
python -m scverifier.core.benchmarking.run_benchmark healthver \
    --split test \
    --max-items 50 \
    --method agentless_with_search \
    --max-papers 5
```

## Command-Line Options

```
positional arguments:
  dataset               Benchmark dataset to run (scifact, coverbench, healthver, msvec)

optional arguments:
  --method METHOD       Verification method:
                        - agentless (default)
                        - agent
                        - agentless_with_search
                        - agent_with_search
  --max-items N         Maximum number of items to evaluate (default: all)
  --max-papers N        Max papers to retrieve per claim (default: 10, for search methods)
  --results-dir PATH    Directory to save results (default: ./benchmark_results)
  --scifact-split SPLIT SciFact split: train, dev, or test (default: combined)
  --split SPLIT         Dataset split (for HealthVer: train, validation, test; default: validation)
  --data-dir PATH       Data directory (for MSVEC)
  --resume PATH         Resume incomplete run from directory
```

## Output Structure

Each benchmark run creates a timestamped directory with:

```
benchmark_results/
└── scifact_agentless_20260128_153045/
    ├── logs/
    │   ├── claim_0.json         # Structured result with metadata
    │   ├── claim_1.json
    │   └── ...
    ├── figures/
    │   ├── confidence_distribution.png
    │   └── label_distribution.png
    └── summary.json             # Overall statistics

# Knowledge base is saved to the default location (data/output/)
# This is the same KB used by the webapp and other scripts
```

### Result Files

Each `claim_X.json` contains:
- Claim text and ID
- Expected result (ground truth)
- Predicted result (verdict)
- Confidence score
- Reasoning and justification
- Evidence propositions used with citations
- Timestamp and processing metadata

### Summary File

The `summary.json` contains:
- Overall accuracy
- Per-label accuracy
- Label distributions
- Confidence statistics
- Individual results for each claim

## Programmatic Usage

```python
from scverifier.core.benchmarking import SciFact, CoverBench, HealthVer, VerificationMethod
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.pipelines.verification_pipeline import VerificationPipeline

# Load benchmark (example with HealthVer)
benchmark = HealthVer(verification_method=VerificationMethod.AGENTLESS)
benchmark.load(max_items=10, split="validation")

# Setup pipeline with default KB
kb = KnowledgeBase()
kb.load()  # Load from default location (data/output)
pipeline = VerificationPipeline(kb=kb)

# Verify each claim
for item in benchmark.items:
    result = pipeline.verify_claim_from_kb(item.claim)
    item.result = result

    print(f"Claim: {item.claim}")
    print(f"Expected: {item.expected_result}")
    print(f"Predicted: {result.verdict}")
    print(f"Correct: {result.verdict == item.expected_result}")

# Save KB back to default location
kb.save()  # Saves to data/output by default
```

## Architecture

### Base Classes

- **Benchmark**: Abstract base class for all benchmarks
- **BenchmarkItem**: Represents a single claim with expected result and verification result
- **VerificationMethod**: Enum for different verification approaches

### Benchmark Implementations

- **SciFact**: Loads claims from JSONL file
- **CoverBench**: Loads claims from HuggingFace dataset
- **HealthVer**: Loads claims from HuggingFace dataset (with train/validation/test splits)

### Runner

- **BenchmarkRunner**: Orchestrates verification, logging, and saving results

## Adding New Benchmarks

To add a new benchmark:

1. Create a new file: `scverifier/core/benchmarking/mybenchmark_benchmark.py`

2. Implement the `Benchmark` interface:

```python
from scverifier.core.benchmarking.base import Benchmark, BenchmarkItem, VerificationMethod

class MyBenchmark(Benchmark):
    def __init__(self, verification_method: VerificationMethod = VerificationMethod.AGENTLESS):
        super().__init__(name="MyBenchmark")
        self.verification_method = verification_method

    def load(self, max_items=None):
        # Load your dataset and create BenchmarkItem objects
        # Map labels to: SUPPORTS, REFUTES, INSUFFICIENT_EVIDENCE
        self.items = [...]
        return self.items
```

3. Add to `__init__.py`:

```python
from scverifier.core.benchmarking.mybenchmark_benchmark import MyBenchmark
__all__ = [..., "MyBenchmark"]
```

4. Update `run_benchmark.py` to support the new dataset

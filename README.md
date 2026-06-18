# Scientific Claim Verifier

**Work in progress -> Fork** of Exemocaro/Scientific-Claim-Verifier. 

A framework for automated verification of scientific claims using LLMs and proposition-based retrieval.

## Installation

```bash
# pip
pip install -e .

# uv
uv sync
uv sync --group dev          # with dev dependencies
```

## Usage

The package exposes a `scverifier` CLI (entry point defined in `pyproject.toml` → `scverifier.cli:main`).

```bash
# Verify a claim
scverifier verify "Vitamin D prevents COVID-19" --max-papers 10
scverifier verify "claim" --kb-only
scverifier verify "claim" --kb-only --use-all-propositions

# Extract propositions from documents (all sections by default)
scverifier extract                    # demo paper (data/demo_paper.pdf)
scverifier extract data/demo_paper.pdf
scverifier extract folder/

# Run web app
scverifier webapp --port 8080

# Bulk verify claims from a CSV/JSONL file
scverifier bulk data/claims.csv --method agentless_with_search
scverifier bulk data/claims.jsonl --output results.jsonl --max-items 50

# Run benchmarks from YAML config
scverifier benchmark --config config/experiments/baseline.yaml

# Dry-run — validate config and print plan without running
scverifier benchmark --config config/experiments/baseline.yaml --dry-run
```

Config files define model, knowledge base, features, and benchmark datasets. See `config/experiments/baseline.yaml` for a minimal example (used for performance testing).

### Environment Setup

Create a `.env` file in the project root. The provider is auto-detected: set `GEMINI_API_KEY` for Gemini or `AZURE_API_KEY` (+ endpoint/deployment) for Azure OpenAI.

```bash
# --- LLM Provider (pick one) ---
GEMINI_API_KEY=your_key_here              # → Gemini
# AZURE_API_KEY=your_key_here             # → Azure OpenAI
# AZURE_ENDPOINT=https://your-endpoint
# AZURE_DEPLOYMENT_NAMES=...

# --- Paper search ---
SEMANTIC_SCHOLAR_API_KEY=your_key_here
CORE_API_KEY=your_key_here

# --- OpenAlex (polite pool) ---
OPENALEX_MAILTO=your@email.com
OPENALEX_API_KEY=your_key_here            # optional
```

Hugging Face authentication is required for the CoverBench dataset.
Set your token in `.env`:

```bash
HF_TOKEN=hf_your_token_here
```

Or run `huggingface-cli login` instead.

### Batch Processing

Batch proposition extraction uses Google's Batch API (50% cost discount). **Currently only works with Gemini and is not yet available through the `scverifier` CLI.** Run from `scripts/batch_processing/`:

```bash
python create_batch_extraction_jobs.py --benchmark coverbench --papers-per-claim 10
python submit_batch.py
python monitor_batch_jobs.py
python process_batch_results.py
```

See `scripts/batch_processing/README.md` for full workflow.

## Tests

```bash
uv run pytest tests/ -v
```

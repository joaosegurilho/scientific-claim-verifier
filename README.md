# Scientific Claim Verifier

Full code for the Master Thesis **"Development of a Language Model-Based Framework for Credibility Assessment and Verification of Scientific Claims"** by **Mateus Pereira**.
A framework for automated verification of scientific claims using Large Language Models and proposition-based retrieval. Made in Python. Uses langchain and fastapi.

## Quick Start

1. **Navigate to the project directory:**

   ```bash
   cd scientific-claim-verifier
   ```

2. **Install dependencies:**

    ```bash
    pip install -e .
    # or
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Create a `.env` file in the `scientific-claim-verifier` directory with the following variables:

    ```bash
    # LLM Provider: "gemini" or "azure"
    LLM_PROVIDER=gemini

    # Gemini API Key
    GEMINI_API_KEY=your_gemini_api_key_here

    # API keys for paper search
    SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
    CORE_API_KEY=your_core_api_key_here

    # OpenAlex API (required for polite pool access)
    OPENALEX_MAILTO=your_email@example.com
    OPENALEX_API_KEY=your_openalex_key_here  # Optional
    ```

    **Note on OpenAlex:** The `OPENALEX_MAILTO` parameter is required for polite pool access (better rate limits and service). The API key is optional for basic usage.

4. **Login to Hugging Face (Required for Benchmarks):**

    To access benchmark datasets like CoverBench, HealthVer, and SciFact, you need to authenticate with Hugging Face:

    ```bash
    # Install huggingface-cli if not already installed
    pip install huggingface_hub

    # Login using one of these methods:
    huggingface-cli login
    # Or set token as environment variable:
    # export HF_TOKEN=your_token_here
    ```

    Follow the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) for detailed authentication instructions.

5. **Run the pipeline:**

    Check examples below!

### Command Line Usage

```bash
# Extract propositions from documents (all sections by default)
python scripts/run_extraction_pipeline.py  # Process demo paper (data/demo_paper.pdf)
python scripts/run_extraction_pipeline.py data/demo_paper.pdf  # Process specific file
python scripts/run_extraction_pipeline.py folder/  # Process all files in folder

# Verify claims
python scripts/run_verification_pipeline.py --help
python scripts/run_verification_pipeline.py "Vitamin D prevents COVID-19" --max-papers 10
python scripts/run_verification_pipeline.py "claim" --kb-only  # Use only existing KB
python scripts/run_verification_pipeline.py "claim" --skip-extraction-eval  # Skip quality evaluation during extraction (faster)
python scripts/run_verification_pipeline.py "claim" --kb-only --use-all-propositions  # Use all propositions, not just quality ones

# Query knowledge base
python scripts/query_knowledge_base.py  # Interactive mode
python scripts/query_knowledge_base.py "search query"
python scripts/query_knowledge_base.py --stats  # Show statistics
python scripts/query_knowledge_base.py --list-papers  # List all papers

# Test autonomous agent
python scripts/test_agents.py
```

### Batch Processing

The project supports batch processing of proposition extraction using Google's Batch API (50% cost discount). This is useful for processing large benchmark datasets.

```bash
cd scripts/batch_processing

# Step 1: Create batch job for a benchmark
python create_batch_extraction_jobs.py --benchmark coverbench --papers-per-claim 10
python create_batch_extraction_jobs.py --benchmark healthver --papers-per-claim 10
python create_batch_extraction_jobs.py --benchmark scifact --papers-per-claim 10

# Chunked processing - Process claims in smaller batches (resumable)
python create_batch_extraction_jobs.py --benchmark coverbench --offset 0 --limit 100
python create_batch_extraction_jobs.py --benchmark coverbench --offset 100 --limit 100
python create_batch_extraction_jobs.py --benchmark coverbench --offset 200 --limit 100

# Test with single claim first
python create_batch_extraction_jobs.py --benchmark coverbench --test-single-claim

# Step 2: Monitor batch job (optional)
python monitor_batch_jobs.py  # Monitor latest job
python monitor_batch_jobs.py --job-id batches/abc123  # Monitor specific job
python monitor_batch_jobs.py --list-all  # List all jobs

# Step 3: Process results and add to knowledge base
python process_batch_results.py  # Process latest results
python process_batch_results.py --results-file path/to/results.jsonl
```

Features:
- **Chunked Processing**: Use `--offset` and `--limit` flags to process claims in manageable batches
- **Resumable**: Process large datasets incrementally (e.g., CoverBench's 733 claims in 100-claim batches)
- **Timing**: Automatically tracks and displays processing time for each batch
- **Metadata Tracking**: Saves offset, limit, claim range, and elapsed time to metadata files

See [scripts/batch_processing/README.md](scientific-claim-verifier/scripts/batch_processing/README.md) for detailed documentation.

### Benchmarking

The framework supports several scientific claim verification benchmarks:

```bash
cd scientific-claim-verifier

# Run benchmarks
python -m scverifier.core.benchmarking.run_benchmark coverbench --max-items 10
python -m scverifier.core.benchmarking.run_benchmark healthver --max-items 10
python -m scverifier.core.benchmarking.run_benchmark scifact --max-items 10
```

Available benchmarks:
- **CoverBench**: 733 complex claims with rich grounding contexts (Google)
- **HealthVer**: Health-related claim verification dataset
- **SciFact**: Scientific fact verification dataset

Note: Requires Hugging Face authentication (see Quick Start step 4).

### Web Application

The project includes a FastAPI web interface for browsing and verifying claims:

```bash
# Run the web application
cd scientific-claim-verifier
uvicorn scverifier.webapp.main:app --reload
```

Features:
- Browse papers in the knowledge base
- View paper details, chunks, and propositions
- Search propositions and chunks
- Verify scientific claims with visual timeline
- Upload and process documents with/without claim extraction
- Autonomous agent-based verification


### Run tests
```bash
# All at once (pytest discovers test_*.py files)
uv run pytest tests/ -v
```

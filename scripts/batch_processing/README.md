# Batch Proposition Extraction Scripts

These scripts enable batch processing of proposition extraction from papers using Google's Batch API, which offers a 50% cost discount compared to synchronous API calls.

## Overview

The batch processing workflow consists of **4 independent scripts**:

1. **create_batch_extraction_jobs.py** - Search papers, chunk them, create input + metadata files (does NOT submit)
2. **submit_batch.py** - Submit a single batch file to Google Batch API
3. **monitor_batch_jobs.py** - Monitor job status and download results when complete
4. **process_batch_results.py** - Parse results and add propositions to knowledge base

## Design Philosophy

Each batch file is **completely independent** with its own metadata:
- Files always use `_0`, `_1`, `_2` suffix (even single files)
- Each input file has its own complete metadata file
- No coordination between files - each is tracked separately
- Paper chunks always stay together in the same file when splitting
- Submit and process files independently, no need to wait for all parts

## Requirements

- Google Gemini API key configured in `.env`
- Semantic Scholar API key for paper search
- Python dependencies installed (see main requirements.txt)

## Complete Workflow

### Step 1: Create Batch Files (Does NOT Submit)

Create input and metadata files for batch processing:

```bash
cd scripts/batch_processing
python create_batch_extraction_jobs.py --benchmark scifact --papers-per-claim 10 --limit 50
```

**Parameters**:
- `--benchmark`: Which benchmark to use (`scifact`, `coverbench`, `healthver`)
- `--papers-per-claim`: How many papers to retrieve per claim (default: 10)
- `--limit`: Process only first N claims (useful for testing)
- `--test-single-claim`: Process just one claim for quick testing

**What it does**:
- Loads claims from the specified benchmark
- Searches for papers using Semantic Scholar
- Scores papers locally using PaperScorer
- Chunks papers (abstract + full_text) using DocumentProcessor
- Calculates token usage and estimated cost
- **Splits into multiple files if > BATCH_FILE_SPLIT_LIMIT lines** (keeps paper chunks together)
- Creates input JSONL + metadata JSON pairs
- Status: `CREATED` (ready for submission)

**Output files**:
```
data/batch_jobs/input/batch_20251211_143022_0.jsonl
data/batch_jobs/metadata/batch_20251211_143022_0.json
```

If split (> BATCH_FILE_SPLIT_LIMIT lines):
```
data/batch_jobs/input/batch_20251211_143022_0.jsonl
data/batch_jobs/input/batch_20251211_143022_1.jsonl
data/batch_jobs/input/batch_20251211_143022_2.jsonl
data/batch_jobs/metadata/batch_20251211_143022_0.json
data/batch_jobs/metadata/batch_20251211_143022_1.json
data/batch_jobs/metadata/batch_20251211_143022_2.json
```

### Step 2: Submit Batch File

Submit a single batch file to Google:

```bash
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_0.jsonl
```

**What it does**:
- Loads corresponding metadata file (same name, `.json` extension)
- Validates metadata has all required fields (papers, chunks, chunk_keys, claims)
- Uploads to Google File API
- Creates batch job
- Updates metadata with job ID and submission time
- Status: `SUBMITTED`

**If you have multiple parts, submit each one**:
```bash
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_0.jsonl
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_1.jsonl
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_2.jsonl
```

### Step 3: Monitor Job Status

**List all jobs**:
```bash
python monitor_batch_jobs.py --list-all
```

Shows all batch files with their statuses independently.

**Monitor latest submitted job**:
```bash
python monitor_batch_jobs.py
```

**Monitor specific job by ID**:
```bash
python monitor_batch_jobs.py --job-id batches/abc123xyz
```

**Monitor from metadata file**:
```bash
python monitor_batch_jobs.py --metadata-file data/batch_jobs/metadata/batch_20251211_143022_0.json
```

**What it does**:
- Polls job status every 30 seconds (configurable with `--poll-interval`)
- Downloads results when job completes (typically 1-24 hours)
- Saves to `data/batch_jobs/results/batch_20251211_143022_0_results.jsonl`
- Updates metadata with completion time
- Status: `COMPLETED`

### Step 4: Process Results

Process a single results file:

```bash
python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_0_results.jsonl
```

**Or process latest results**:
```bash
python process_batch_results.py
```

**If you have multiple parts, process each one**:
```bash
python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_0_results.jsonl
python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_1_results.jsonl
python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_2_results.jsonl
```

**What it does**:
- Loads corresponding metadata file
- Reconstructs Paper and Chunk objects from metadata
- Parses propositions from batch results
- Creates Proposition objects with proper IDs
- Adds papers to the main knowledge base (with deduplication check)
- Shows summary of what was added

## Example: SciFact Test Dataset in Batches

**Goal**: Process SciFact test dataset with 50 claims per batch, 10 papers per claim

### Batch 1: First 50 Claims

```bash
# Step 1: Create batch files
python create_batch_extraction_jobs.py --benchmark scifact --papers-per-claim 10 --limit 50

# Output shows:
# Created: batch_20251211_143022_0.jsonl (500 papers, 2,450 chunks)
# Status: CREATED

# Step 2: Submit to Google
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_0.jsonl

# Output shows:
# Batch job ID: batches/abc123xyz
# Status: SUBMITTED

# Step 3: Monitor (wait for completion, ~1-24 hours)
python monitor_batch_jobs.py

# Or check status later:
python monitor_batch_jobs.py --list-all

# Step 4: Process results when complete
python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_0_results.jsonl

# Output shows:
# Successfully parsed: 2,450 chunks
# Total propositions extracted: 12,250
# Added to knowledge base: 500 papers
```

### Batch 2: Next 50 Claims (Claims 51-100)

```bash
# Create next batch (use different timestamp)
python create_batch_extraction_jobs.py --benchmark scifact --papers-per-claim 10 --offset 50 --limit 50

# Submit
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_150000_0.jsonl

# Monitor
python monitor_batch_jobs.py

# Process when complete
python process_batch_results.py
```

### Batch 3: Remaining Claims

```bash
# Process all remaining claims
python create_batch_extraction_jobs.py --benchmark scifact --papers-per-claim 10 --offset 100

# Submit and process as above
```

**Note**: If a batch gets split (>4,000 lines), you'll see multiple files:
- `batch_20251211_143022_0.jsonl` (first 4,000 lines)
- `batch_20251211_143022_1.jsonl` (remaining lines)

Submit and process each file independently.

## Directory Structure

**Example with split files**:

```text
data/batch_jobs/
├── input/
│   ├── batch_20251211_143022_0.jsonl  # First part
│   └── batch_20251211_143022_1.jsonl  # Second part (if split)
├── metadata/
│   ├── batch_20251211_143022_0.json   # Metadata for part 0
│   └── batch_20251211_143022_1.json   # Metadata for part 1
└── results/
    ├── batch_20251211_143022_0_results.jsonl
    └── batch_20251211_143022_1_results.jsonl
```

Each metadata file contains:
- **papers**: Dict of all Paper objects in this file
- **chunks**: Dict of all Chunk objects in this file
- **chunk_keys**: List of all chunk keys in this file
- **claims**: Dict of claim data
- **token_stats**: Token usage statistics
- **batch_job_id**: Google job ID (after submission)
- **status**: CREATED → SUBMITTED → COMPLETED
```

## Cost Savings

Batch API pricing is 50% of the normal API rate:

**Example (SciFact test, 50 claims, 10 papers each):**
- ~500 papers, ~2,500 chunks
- ~750K tokens total
- Normal cost: ~$0.056
- Batch cost: ~$0.028 (50% off)

**Full SciFact (300 claims):**
- ~3,000 papers, ~15,000 chunks
- ~4.5M tokens
- Normal cost: ~$0.34
- Batch cost: ~$0.17 (50% off)

## Supported Benchmarks

- `scifact` - SciFact dataset (~300 claims)
- `coverbench` - CoverBench dataset (733 claims)
- `healthver` - HealthVer dataset

## Quick Reference

### Common Commands

```bash
# Create batch files (does NOT submit)
python create_batch_extraction_jobs.py --benchmark scifact --papers-per-claim 10 --limit 50

# Submit single file
python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_0.jsonl

# List all jobs with status
python monitor_batch_jobs.py --list-all

# Monitor latest job
python monitor_batch_jobs.py

# Process results
python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_0_results.jsonl
```

### File Naming Pattern

All files use `_0`, `_1`, `_2` suffix:
- Input: `batch_{timestamp}_0.jsonl`
- Metadata: `batch_{timestamp}_0.json`
- Results: `batch_{timestamp}_0_results.jsonl`

### Key Features

✅ Independent files - no coordination needed  
✅ Each file has complete metadata  
✅ Paper chunks stay together when splitting  
✅ Submit and process files in any order  
✅ Deduplication prevents adding papers twice  
✅ Always shows `_0`, `_1`, `_2` suffix for clarity  

## Troubleshooting

**"Metadata file not found"**
- Make sure you've run `create_batch_extraction_jobs.py` first
- Check `data/batch_jobs/metadata/` for available files

**"Results file not found"**
- Make sure the batch job has completed (use `monitor_batch_jobs.py --list-all`)
- Results are downloaded to `data/batch_jobs/results/`

**"Already submitted" warning**
- The file was already submitted to Google
- Check job status with `monitor_batch_jobs.py --list-all`
- If you really want to resubmit, type 'yes' when prompted

**Job taking too long**
- Google's Batch API can take up to 24 hours
- Check status: `python monitor_batch_jobs.py --list-all`
- Wait patiently - it will complete eventually

**"Metadata incomplete" error**
- The metadata file is missing required fields
- Re-run `create_batch_extraction_jobs.py` to regenerate

## Notes

- Batch jobs typically complete in 1-24 hours
- Processing happens asynchronously on Google's servers
- You can close the terminal after submitting
- Monitor from a different machine if needed
- All data is saved to the main knowledge base (Config.DB_NAME)
- Proposition tracking includes paper_id, chunk_id, section, and page number
- Each file is independent - no need to wait for all parts before processing

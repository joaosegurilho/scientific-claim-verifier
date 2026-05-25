#!/usr/bin/env python3
"""Submit a batch input file to Google's Batch API.

This script:
1. Takes a single batch JSONL file as input
2. Loads its corresponding metadata file  
3. Validates metadata completeness
4. Uploads to Google File API
5. Creates batch job
6. Updates metadata with job ID and submission time

Usage:
    python submit_batch.py --batch-file data/batch_jobs/input/batch_20251211_143022_0.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from scverifier.config.settings import Config


def validate_metadata(metadata: dict, metadata_file: Path) -> None:
    """Validate that metadata contains all required fields.
    
    Args:
        metadata: Metadata dictionary to validate
        metadata_file: Path to metadata file (for error messages)
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ["papers", "chunks", "chunk_keys", "claims"]
    missing_fields = [field for field in required_fields if field not in metadata]
    
    if missing_fields:
        raise ValueError(
            f"Metadata file {metadata_file} is incomplete. Missing fields: {', '.join(missing_fields)}\n"
            f"This metadata file was probably created incorrectly. Please regenerate it using create_batch_extraction_jobs.py"
        )
    
    # Check that fields are not empty
    if not metadata["papers"]:
        raise ValueError(f"Metadata file {metadata_file} has no papers. Cannot submit empty batch.")
    
    if not metadata["chunks"]:
        raise ValueError(f"Metadata file {metadata_file} has no chunks. Cannot submit empty batch.")
    
    if not metadata["chunk_keys"]:
        raise ValueError(f"Metadata file {metadata_file} has no chunk_keys. Cannot submit empty batch.")
    
    print(f"✓ Metadata validation passed:")
    print(f"  - {len(metadata['papers'])} papers")
    print(f"  - {len(metadata['chunks'])} chunks")
    print(f"  - {len(metadata['chunk_keys'])} chunk_keys")
    print(f"  - {len(metadata['claims'])} claims")


def submit_batch(batch_file_path: str) -> str:
    """Submit a single batch file to Google's Batch API.
    
    Args:
        batch_file_path: Path to the batch JSONL file
        
    Returns:
        batch_job_id: The ID of the created batch job
        
    Raises:
        FileNotFoundError: If batch file or metadata file doesn't exist
        ValueError: If metadata is incomplete
    """
    input_file = Path(batch_file_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file_path}")
    
    # Find corresponding metadata file (same basename, .json extension)
    metadata_file = input_file.parent.parent / "metadata" / input_file.name.replace(".jsonl", ".json")
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Expected metadata at: {metadata_file}\n"
            f"Please ensure the metadata file exists before submitting."
        )
    
    # Load and validate metadata
    print(f"\nLoading metadata from: {metadata_file.name}")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    validate_metadata(metadata, metadata_file)
    
    # Check if already submitted
    if metadata.get("batch_job_id"):
        print(f"\nWarning: This batch was already submitted!")
        print(f"  Previous job ID: {metadata['batch_job_id']}")
        print(f"  Submitted at: {metadata.get('submitted_at', 'unknown')}")
        
        response = input("\nDo you want to submit again? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Submission cancelled.")
            return None
    
    # Extract timestamp and part number from filename
    stem = input_file.stem  # e.g., "batch_20251211_143022_0"
    parts = stem.split("_")
    timestamp = "_".join(parts[1:3])  # "20251211_143022"
    part_num = int(parts[3])  # 0
    
    file_size_mb = input_file.stat().st_size / (1024 * 1024)
    
    print(f"\nSubmitting batch file: {input_file.name}")
    print(f"  Size: {file_size_mb:.1f} MB")
    # print(f"  Part: {part_num} of {metadata.get('total_parts', '?')}")
    
    # Setup Google API client
    Config.setup_environment()
    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    
    # Upload to Google File API with retry logic
    print("\nUploading to Google File API...")
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            uploaded_file = client.files.upload(
                file=str(input_file),
                config={
                    "display_name": f"batch_{timestamp}_{part_num}",
                    "mime_type": "application/jsonl"
                }
            )
            print(f"✓ Upload successful: {uploaded_file.name}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"✗ Upload failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"✗ Upload failed after {max_retries} attempts")
                raise
    
    # Create batch job with retry logic
    print("\nCreating batch job...")
    print(f"  Using model: {Config.BATCH_LLM_MODEL}")
    print(f"  Using uploaded file: {uploaded_file.name}")
    
    for attempt in range(max_retries):
        try:
            batch_job = client.batches.create(
                model=Config.BATCH_LLM_MODEL,
                src=uploaded_file.name,
                config={
                    "display_name": f"proposition_extraction_{timestamp}_{part_num}"
                }
            )
            print(f"✓ Batch job created: {batch_job.name}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"✗ Batch creation failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"✗ Batch creation failed after {max_retries} attempts")
                raise
    
    # Update metadata
    metadata["batch_job_id"] = batch_job.name
    metadata["submitted_at"] = datetime.now().isoformat()
    metadata["status"] = "SUBMITTED"
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata updated: {metadata_file.name}")
    
    print(f"\n{'='*60}")
    print("SUBMISSION SUCCESSFUL")
    print(f"{'='*60}")
    print(f"Batch job ID: {batch_job.name}")
    print(f"Estimated completion: 12-24 hours")
    print(f"\nTo monitor this job:")
    print(f"  python monitor_batch_jobs.py --job-id {batch_job.name}")
    print(f"\nOr monitor all jobs:")
    print(f"  python monitor_batch_jobs.py --list-all")
    
    return batch_job.name


def main():
    parser = argparse.ArgumentParser(description="Submit a batch file to Google's Batch API")
    parser.add_argument(
        "--batch-file",
        required=True,
        help="Path to the batch JSONL file to submit (e.g., data/batch_jobs/input/batch_20251211_143022_0.jsonl)"
    )
    
    args = parser.parse_args()
    
    try:
        batch_job_id = submit_batch(args.batch_file)
        if batch_job_id:
            return 0
        else:
            return 1
    except Exception as e:
        print(f"\n✗ Error submitting batch job: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

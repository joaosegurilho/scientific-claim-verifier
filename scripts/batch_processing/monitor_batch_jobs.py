#!/usr/bin/env python3
"""Monitor batch extraction jobs and download results when complete.

This script:
1. Lists all batch jobs with their statuses
2. Monitors a specific job by polling status every N seconds
3. Downloads results when job completes
4. Updates metadata with completion info

Each batch file is independent with its own metadata file.

Usage:
    python monitor_batch_jobs.py                              # Monitor latest job
    python monitor_batch_jobs.py --job-id batches/abc123      # Monitor specific job
    python monitor_batch_jobs.py --list-all                   # List all jobs
    python monitor_batch_jobs.py --metadata-file batch_...json  # Monitor from metadata file
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


def get_metadata_files():
    """Get all batch job metadata files sorted by timestamp (newest first)."""
    metadata_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "metadata"
    metadata_files = sorted(metadata_dir.glob("*batch_*.json"), reverse=True)
    return metadata_files


def load_metadata(metadata_file: Path) -> dict:
    """Load metadata from file."""
    with open(metadata_file, "r") as f:
        return json.load(f)


def list_all_jobs():
    """List all batch jobs with their statuses.
    
    Each file is shown independently with its own status.
    """
    print("\n=== All Batch Jobs ===\n")

    Config.setup_environment()
    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    metadata_files = get_metadata_files()

    if not metadata_files:
        print("No batch jobs found.")
        return

    for metadata_file in metadata_files:
        metadata = load_metadata(metadata_file)
        
        filename = metadata_file.name
        timestamp = metadata.get("timestamp", "unknown")
        num_papers = len(metadata.get("papers", {}))
        num_chunks = len(metadata.get("chunks", {}))
        num_claims = metadata.get("num_claims", 0)
        batch_job_id = metadata.get("batch_job_id")

        print(f"File: {filename}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Papers: {num_papers}, Chunks: {num_chunks}, Claims: {num_claims}")

        if not batch_job_id:
            print(f"  Status: NOT_SUBMITTED")
        else:
            try:
                batch_job = client.batches.get(name=batch_job_id)
                status = batch_job.state.name
                print(f"  Status: {status}")
                print(f"  Job ID: {batch_job_id}")
            except Exception as e:
                print(f"  Status: ERROR - {str(e)[:80]}")
                print(f"  Job ID: {batch_job_id}")
        
        print()


def find_metadata_by_job_id(job_id: str) -> Path:
    """Find metadata file containing the given job ID."""
    metadata_files = get_metadata_files()
    
    for metadata_file in metadata_files:
        metadata = load_metadata(metadata_file)
        if metadata.get("batch_job_id") == job_id:
            return metadata_file
    
    return None


def find_latest_job():
    """Find the latest submitted batch job."""
    metadata_files = get_metadata_files()

    if not metadata_files:
        print("No batch jobs found.")
        return None, None

    # Find first metadata file with a batch_job_id
    for metadata_file in metadata_files:
        metadata = load_metadata(metadata_file)
        if metadata.get("batch_job_id"):
            return metadata.get("batch_job_id"), metadata_file
    
    print("No submitted batch jobs found.")
    return None, None


def monitor_job(job_id: str, metadata_file: Path = None, poll_interval: int = 30):
    """Monitor a specific batch job and download results when complete.
    
    Args:
        job_id: Google batch job ID to monitor
        metadata_file: Path to metadata file for this job
        poll_interval: Seconds between status checks
    """
    Config.setup_environment()
    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    # Find metadata file if not provided
    if not metadata_file:
        metadata_file = find_metadata_by_job_id(job_id)
        if not metadata_file:
            print(f"Warning: Could not find metadata file for job {job_id}")
            metadata = None
            results_base_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_0"
        else:
            metadata = load_metadata(metadata_file)
            # Use metadata filename stem as base for results filename
            # e.g., "corpus_batch_20251214_193006_0.json" -> "corpus_batch_20251214_193006_0"
            results_base_name = metadata_file.stem
    else:
        metadata = load_metadata(metadata_file)
        # Use metadata filename stem as base for results filename
        # e.g., "corpus_batch_20251214_193006_0.json" -> "corpus_batch_20251214_193006_0"
        results_base_name = metadata_file.stem

    print(f"\nMonitoring batch job: {job_id}")
    if metadata_file:
        print(f"Metadata file: {metadata_file.name}")
    print(f"Poll interval: {poll_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            try:
                batch_job = client.batches.get(name=job_id)
                status = batch_job.state.name

                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")

                if status == "JOB_STATE_SUCCEEDED":
                    print("\n✓ Job completed successfully!")

                    # Download results
                    if batch_job.dest and batch_job.dest.file_name:
                        result_file_name = batch_job.dest.file_name
                        print(f"Downloading results from: {result_file_name}")

                        file_content = client.files.download(file=result_file_name)

                        # Save results with same naming pattern as input
                        results_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "results"
                        results_dir.mkdir(parents=True, exist_ok=True)

                        # Use the metadata filename as base for results filename
                        # e.g., "corpus_batch_20251214_193006_0.json" -> "corpus_batch_20251214_193006_0_results.jsonl"
                        results_file = results_dir / f"{results_base_name}_results.jsonl"

                        with open(results_file, "wb") as f:
                            f.write(file_content)

                        file_size_mb = len(file_content) / (1024 * 1024)
                        print(f"✓ Results saved: {results_file.name} ({file_size_mb:.1f} MB)")

                        # Update metadata
                        if metadata and metadata_file:
                            metadata["status"] = "COMPLETED"
                            metadata["completed_at"] = datetime.now().isoformat()
                            metadata["results_file"] = str(results_file)

                            with open(metadata_file, "w") as f:
                                json.dump(metadata, f, indent=2)

                            print(f"✓ Metadata updated: {metadata_file.name}")

                        # Count results
                        num_results = file_content.decode("utf-8").count("\n")
                        print("\nSummary:")
                        print(f"  - Results: {num_results} lines")

                        print(f"\nNext step:")
                        print(f"  python process_batch_results.py --results-file {results_file}")

                    else:
                        print("Warning: No results file found in batch job response")

                    break

                elif status in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                    print(f"\n✗ Job ended with status: {status}")

                    if hasattr(batch_job, "error") and batch_job.error:
                        print(f"Error: {batch_job.error}")

                    # Update metadata
                    if metadata and metadata_file:
                        metadata["status"] = status
                        metadata["failed_at"] = datetime.now().isoformat()
                        if hasattr(batch_job, "error"):
                            metadata["error"] = str(batch_job.error)

                        with open(metadata_file, "w") as f:
                            json.dump(metadata, f, indent=2)

                        print(f"✓ Metadata updated: {metadata_file.name}")

                    break

                else:
                    # Job still running
                    time.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\n\nMonitoring stopped by user.")
                break
            except Exception as e:
                print(f"\n✗ Error checking job status: {e}")
                print(f"Retrying in {poll_interval} seconds...")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Monitor batch extraction jobs")
    parser.add_argument("--job-id", help="Specific batch job ID to monitor")
    parser.add_argument("--metadata-file", help="Path to metadata file to monitor")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status checks (default: 30)")
    parser.add_argument("--list-all", action="store_true", help="List all batch jobs and their statuses")

    args = parser.parse_args()

    if args.list_all:
        list_all_jobs()
        return 0

    # Determine which job to monitor
    if args.metadata_file:
        # Monitor from metadata file
        metadata_file = Path(args.metadata_file)
        if not metadata_file.exists():
            print(f"Error: Metadata file not found: {metadata_file}")
            return 1
        
        metadata = load_metadata(metadata_file)
        job_id = metadata.get("batch_job_id")
        
        if not job_id:
            print(f"Error: Metadata file has no batch_job_id. Has it been submitted?")
            return 1
        
        print(f"Monitoring job from metadata file: {metadata_file.name}")
        monitor_job(job_id, metadata_file, args.poll_interval)
        
    elif args.job_id:
        # Monitor specific job ID
        job_id = args.job_id
        monitor_job(job_id, None, args.poll_interval)
        
    else:
        # Find and monitor latest job
        job_id, metadata_file = find_latest_job()
        if not job_id:
            return 1
        print(f"Monitoring latest job: {job_id}")
        monitor_job(job_id, metadata_file, args.poll_interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())

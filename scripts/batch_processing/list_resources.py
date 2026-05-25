#!/usr/bin/env python3
"""List, delete files and cancel batch jobs on Google's Gemini API.

Usage:
    python list_resources.py                      # List both files and batches
    python list_resources.py --files              # List only files
    python list_resources.py --batches            # List only batches
    python list_resources.py --delete-all-files   # Delete all uploaded files
    python list_resources.py --cancel-all-batches # Cancel all pending/running batches
    python list_resources.py --delete-file FILE   # Delete a specific file
    python list_resources.py --cancel-batch BATCH # Cancel a specific batch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from scverifier.config.settings import Config


def get_client() -> genai.Client:
    """Create Gemini client using Config API key."""
    return genai.Client(api_key=Config.GEMINI_API_KEY)


def list_files(client: genai.Client) -> None:
    """List all uploaded files."""
    print("=" * 60)
    print("UPLOADED FILES")
    print("=" * 60)

    files = list(client.files.list())

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} file(s):\n")

    for f in files:
        print(f"  Name: {f.name}")
        print(f"  Display Name: {getattr(f, 'display_name', 'N/A')}")
        print(f"  Size: {getattr(f, 'size_bytes', 'N/A')} bytes")
        print(f"  State: {getattr(f, 'state', 'N/A')}")
        print(f"  Created: {getattr(f, 'create_time', 'N/A')}")
        print(f"  URI: {getattr(f, 'uri', 'N/A')}")
        print("-" * 40)


def list_batches(client: genai.Client) -> None:
    """List all batch jobs."""
    print("=" * 60)
    print("BATCH JOBS")
    print("=" * 60)

    batches = list(client.batches.list())

    if not batches:
        print("No batch jobs found.")
        return

    print(f"Found {len(batches)} batch job(s):\n")

    for b in batches:
        print(f"  Name: {b.name}")
        print(f"  State: {getattr(b, 'state', 'N/A')}")
        print(f"  Model: {getattr(b, 'model', 'N/A')}")
        print(f"  Created: {getattr(b, 'create_time', 'N/A')}")

        # Show source file if available
        src = getattr(b, 'src', None)
        if src:
            print(f"  Source: {src}")

        # Show destination if available
        dest = getattr(b, 'dest', None)
        if dest:
            print(f"  Destination: {dest}")

        print("-" * 40)


def delete_file(client: genai.Client, file_name: str) -> bool:
    """Delete a specific file."""
    try:
        client.files.delete(name=file_name)
        print(f"Deleted: {file_name}")
        return True
    except Exception as e:
        print(f"Failed to delete {file_name}: {e}")
        return False


def delete_all_files(client: genai.Client) -> None:
    """Delete all uploaded files."""
    files = list(client.files.list())

    if not files:
        print("No files to delete.")
        return

    print(f"Deleting {len(files)} file(s)...")
    deleted = 0
    for f in files:
        if delete_file(client, f.name):
            deleted += 1

    print(f"\nDeleted {deleted}/{len(files)} files.")


def cancel_batch(client: genai.Client, batch_name: str) -> bool:
    """Cancel a specific batch job."""
    try:
        client.batches.cancel(name=batch_name)
        print(f"Cancelled: {batch_name}")
        return True
    except Exception as e:
        print(f"Failed to cancel {batch_name}: {e}")
        return False


def cancel_all_batches(client: genai.Client) -> None:
    """Cancel all pending/running batch jobs."""
    batches = list(client.batches.list())

    # Filter to only pending/running
    active_states = ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]
    active_batches = [
        b for b in batches
        if str(getattr(b, 'state', '')).split('.')[-1] in active_states
    ]

    if not active_batches:
        print("No active batch jobs to cancel.")
        return

    print(f"Cancelling {len(active_batches)} active batch job(s)...")
    cancelled = 0
    for b in active_batches:
        if cancel_batch(client, b.name):
            cancelled += 1

    print(f"\nCancelled {cancelled}/{len(active_batches)} batch jobs.")


def main():
    parser = argparse.ArgumentParser(description="List Gemini API resources")
    parser.add_argument("--files", action="store_true", help="List only files")
    parser.add_argument("--batches", action="store_true", help="List only batches")
    parser.add_argument("--delete-all-files", action="store_true", help="Delete all uploaded files")
    parser.add_argument("--cancel-all-batches", action="store_true", help="Cancel all pending/running batches")
    parser.add_argument("--delete-file", type=str, help="Delete a specific file by name")
    parser.add_argument("--cancel-batch", type=str, help="Cancel a specific batch by name")
    args = parser.parse_args()

    client = get_client()

    # Handle delete/cancel operations
    if args.delete_all_files:
        delete_all_files(client)
        return

    if args.cancel_all_batches:
        cancel_all_batches(client)
        return

    if args.delete_file:
        delete_file(client, args.delete_file)
        return

    if args.cancel_batch:
        cancel_batch(client, args.cancel_batch)
        return

    # If neither specified, show both
    show_files = args.files or (not args.files and not args.batches)
    show_batches = args.batches or (not args.files and not args.batches)

    if show_files:
        list_files(client)
        print()

    if show_batches:
        list_batches(client)


if __name__ == "__main__":
    main()

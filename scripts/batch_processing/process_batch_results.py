#!/usr/bin/env python3
"""Process batch extraction results and add to knowledge base.

This script:
1. Loads a single batch results JSONL file
2. Loads its corresponding metadata file
3. Parses propositions from each result
4. Reconstructs Paper and Chunk objects from metadata
5. Creates Proposition objects with proper tracking
6. Adds papers to the main knowledge base

Each results file is processed independently with its own metadata.

Usage:
    python process_batch_results.py --results-file data/batch_jobs/results/batch_20251211_143022_0_results.jsonl
    python process_batch_results.py  # Process latest results file
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scverifier.data.models import Paper, Chunk, Proposition, CredibilityScores
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.utils.id_generator import get_next_prop_id


def find_latest_results():
    """Find the latest completed results file.

    Returns:
        tuple: (results_file, metadata_file)
    """
    results_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "results"

    # Find all result files
    all_results = sorted(results_dir.glob("batch_*_results.jsonl"), reverse=True)

    if not all_results:
        print("No results files found.")
        return None, None

    # Return first (most recent) file
    latest_file = all_results[0]
    
    # Find corresponding metadata
    # batch_20251211_143022_0_results.jsonl -> batch_20251211_143022_0.json
    stem = latest_file.stem.replace("_results", "")  # batch_20251211_143022_0
    metadata_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "metadata"
    metadata_file = metadata_dir / f"{stem}.json"

    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return None, None

    print(f"\nLatest results file: {latest_file.name}")
    return latest_file, metadata_file


def load_metadata(metadata_file: Path) -> dict:
    """Load metadata from file."""
    with open(metadata_file, "r") as f:
        return json.load(f)


def parse_results(results_file: Path, metadata: dict):
    """Parse batch results and create proposition objects.

    Args:
        results_file: Path to single results file
        metadata: Metadata dictionary with papers, chunks, etc.

    Returns:
        tuple: (papers_dict, chunks_dict, paper_propositions)
    """
    print(f"\nParsing results from: {results_file.name}")

    # Reconstruct papers from metadata
    papers_dict = {}
    for paper_id, paper_data in metadata["papers"].items():
        paper_obj = Paper(
            id=paper_data["id"],
            doi=paper_data.get("doi"),
            title=paper_data["title"],
            abstract=paper_data["abstract"],
            authors=paper_data["authors"],
            year=paper_data["year"],
            citations=paper_data["citations"],
            url=paper_data.get("url", ""),
            source=paper_data.get("source", "unknown"),
            full_text=paper_data.get("full_text", []),
        )

        # Reconstruct credibility if available
        if paper_data.get("credibility"):
            cred_data = paper_data["credibility"]
            paper_obj.credibility = CredibilityScores(
                rating=cred_data.get("rating"),
                study_type=cred_data.get("study_type"),
                evidence_type=cred_data.get("evidence_type"),
                sample_size=cred_data.get("sample_size"),
                study_duration=cred_data.get("study_duration"),
                control_type=cred_data.get("control_type"),
                blinding=cred_data.get("blinding"),
                randomized=cred_data.get("randomized"),
                population_type=cred_data.get("population_type"),
                citation_bonus=cred_data.get("citation_bonus", 0),
                recency_penalty=cred_data.get("recency_penalty", 0),
                open_access_bonus=cred_data.get("open_access_bonus", 0),
                fulltext_content_bonus=cred_data.get("fulltext_content_bonus", 0),
            )

        papers_dict[paper_id] = paper_obj

    # Reconstruct chunks from metadata
    chunks_dict = {}
    for chunk_id, chunk_data in metadata["chunks"].items():
        chunk_obj = Chunk(
            text=chunk_data["text"],
            chunk_id=chunk_data["chunk_id"],
            source=chunk_data["source"],
            paper_id=chunk_data["paper_id"],
            page=chunk_data.get("page"),
            section=chunk_data.get("section"),
        )
        chunks_dict[chunk_id] = chunk_obj

    # Parse results
    paper_propositions = defaultdict(list)
    successful_count = 0
    failed_count = 0
    total_propositions = 0

    with open(results_file, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} results...")

            print("\n" + "=" * 80)
            try:
                result = json.loads(line)
                key = result.get("key")

                print(f'Processing result line {line_num}, key "{key}"...\n')

                if not key:
                    failed_count += 1
                    continue

                # Parse key: claim_id|paper_id|chunk_id
                try:
                    claim_id, paper_id, chunk_id = key.split("|")
                except ValueError:
                    print(f"  Warning: Invalid key format: {key}")
                    failed_count += 1
                    continue

                # Get chunk and paper
                chunk: Chunk | None = chunks_dict.get(chunk_id)
                paper: Paper | None = papers_dict.get(paper_id)

                if not chunk or not paper:
                    print(f"  Warning: Missing chunk or paper for key: {key}")
                    failed_count += 1
                    continue

                # Log the extraction
                print(f"Paper: {paper_id} | {paper.title}")
                print(f"Chunk ID: {chunk_id}\n")

                # Check if result has response or error
                if "response" in result and result["response"]:
                    # Extract propositions from response
                    try:
                        response = result["response"]

                        # Navigate the response structure
                        if "candidates" in response and len(response["candidates"]) > 0:
                            candidate = response["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                if len(parts) > 0 and "text" in parts[0]:
                                    text_response = parts[0]["text"].strip()

                                    # Parse Python list format (e.g., ['prop1', 'prop2'])
                                    import re
                                    import ast

                                    propositions_list = []
                                    try:
                                        # Remove markdown code fences if present
                                        if text_response.startswith("```"):
                                            # Extract content between code fences
                                            fence_match = re.search(r"```python\n(.*)\n```", text_response, re.DOTALL)
                                            if fence_match:
                                                text_response = fence_match.group(1).strip()
                                            else:
                                                # Try without python label
                                                fence_match = re.search(r"```\n?(.*)\n?```", text_response, re.DOTALL)
                                                if fence_match:
                                                    text_response = fence_match.group(1).strip()

                                        # Find the list pattern in the response
                                        match = re.search(r"\[.*\]", text_response, re.DOTALL)
                                        if match:
                                            raw = match.group(0).strip()

                                            # Try multiple parsing strategies
                                            # 1. Regex extraction of items
                                            try:
                                                items = re.findall(r"'((?:\\'|[^'])*)'", raw)
                                                propositions_list = [item.replace("\\'", "'") for item in items]
                                            except Exception as e:
                                                print(f"REGEX PARSING FAILED for key: {key} with error: {e}")

                                            # 2. JSON parsing
                                            if not propositions_list:
                                                try:
                                                    propositions_list = json.loads(raw)
                                                    if not isinstance(propositions_list, list):
                                                        propositions_list = []
                                                except Exception as e:
                                                    print(f"JSON PARSING FAILED for key: {key} with error: {e}")

                                            # 3. Convert Python-style to JSON
                                            if not propositions_list:
                                                normalized = raw.replace("'", '"')
                                                try:
                                                    propositions_list = json.loads(normalized)
                                                    if not isinstance(propositions_list, list):
                                                        propositions_list = []
                                                except Exception as e:
                                                    print(f"JSON NORMALIZED PARSING FAILED for key: {key} with error: {e}")

                                            # 4. ast.literal_eval as last resort
                                            if not propositions_list:
                                                try:
                                                    propositions_list = ast.literal_eval(raw)
                                                    if not isinstance(propositions_list, list):
                                                        propositions_list = []
                                                except Exception as e:
                                                    print(f"AST PARSING FAILED for key: {key} with error: {e}")

                                            if not propositions_list:
                                                print(f"PARSING FAILED for key: {key}")
                                                propositions_list = []
                                            
                                            # Debug output
                                            print(f"\nFull text response:\n{repr(text_response)}")
                                        else:
                                            # No list found in response
                                            print(f"  Warning: No list found in response for key {key}")
                                            propositions_list = []
                                    except Exception as e:
                                        print(f"  Warning: Unexpected error parsing propositions for key {key}: {e}")
                                        propositions_list = []

                                    if not propositions_list:
                                        print("No propositions extracted.")
                                    else:
                                        print(f"Extracted {len(propositions_list)} propositions:")
                                        print("-" * 80)

                                        for i, prop_text in enumerate(propositions_list, 1):
                                            if not prop_text or not prop_text.strip():
                                                continue

                                            cleaned_text = prop_text.strip()
                                            print(f"{i}. {cleaned_text}")

                                            prop = Proposition(
                                                text=cleaned_text,
                                                chunk_id=chunk_id,
                                                source=paper.title,
                                                paper_id=paper_id,
                                                prop_id=get_next_prop_id(),
                                                page=chunk.page,
                                            )
                                            paper_propositions[paper_id].append(prop)
                                            total_propositions += 1

                                    successful_count += 1
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1

                    except (KeyError, IndexError) as e:
                        print(f"  Warning: Failed to parse response for key {key}: {e}")
                        failed_count += 1

                elif "error" in result:
                    # Log error
                    failed_count += 1
                else:
                    failed_count += 1

            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line {line_num}: {e}")
                failed_count += 1

    print("\n" + "=" * 80)
    print("PARSING SUMMARY")
    print("=" * 80)
    print(f"[OK] Successfully parsed: {successful_count} chunks")
    print(f"[FAIL] Failed to parse: {failed_count} chunks")
    print(f"[TOTAL] Total propositions extracted: {total_propositions}")
    print()
    print("Propositions per paper and chunk:")

    # Group propositions by paper and chunk
    paper_chunk_props = defaultdict(lambda: defaultdict(list))
    for paper_id, props in paper_propositions.items():
        for prop in props:
            paper_chunk_props[paper_id][prop.chunk_id].append(prop)

    for paper_id, chunks in paper_chunk_props.items():
        paper = papers_dict.get(paper_id)
        paper_title = paper.title if paper else paper_id
        print(f"\n  Paper: {paper_title[:70]}")
        for chunk_id, props in sorted(chunks.items()):
            print(f"    - {chunk_id}: {len(props)} propositions")
    print("=" * 80)

    return papers_dict, chunks_dict, paper_propositions


def add_to_knowledge_base(kb: KnowledgeBase, papers_dict: dict, chunks_dict: dict, paper_propositions: dict):
    """Add papers with propositions to the main knowledge base.
    
    Includes deduplication check to prevent adding papers already in KB.
    """
    print("\n" + "=" * 80)
    print("KNOWLEDGE BASE UPDATE")
    print("=" * 80)

    # Print KB stats BEFORE
    print("\nKnowledge Base BEFORE:")
    print(f"  - Total papers: {len(kb.papers)}")
    print(f"  - Total chunks: {sum(len(p.chunks) for p in kb.papers.values())}")
    print(f"  - Total propositions: {sum(len(p.propositions) for p in kb.papers.values())}")

    # Attach propositions and chunks to papers
    papers_to_add = []
    for paper_id, paper in papers_dict.items():
        paper.propositions = paper_propositions.get(paper_id, [])
        paper.chunks = [chunk for chunk in chunks_dict.values() if chunk.paper_id == paper_id]

        # Only add papers that have propositions
        if paper.propositions:
            papers_to_add.append(paper)

    if not papers_to_add:
        print("\nWarning: No papers with propositions to add to KB")
        return

    # Filter out papers already in KB (safety check against duplicates)
    new_papers = [p for p in papers_to_add if p.id not in kb.papers]

    if not new_papers:
        print("\nAll papers already in KB. Skipping to avoid duplicates.")
        return

    if len(new_papers) < len(papers_to_add):
        print(f"\nFiltered out {len(papers_to_add) - len(new_papers)} papers already in KB")

    # Calculate what will be added
    new_chunks_count = sum(len(p.chunks) for p in new_papers)
    new_propositions_count = sum(len(p.propositions) for p in new_papers)

    print(f"\nAdding {len(new_papers)} papers to knowledge base...")
    kb.add_papers(new_papers, verbose=True)
    kb.save()

    # Print KB stats AFTER
    print("\nKnowledge Base AFTER:")
    print(f"  - Total papers: {len(kb.papers)}")
    print(f"  - Total chunks: {sum(len(p.chunks) for p in kb.papers.values())}")
    print(f"  - Total propositions: {sum(len(p.propositions) for p in kb.papers.values())}")

    # Print what was added
    print("\n" + "-" * 80)
    print("ADDED TO KNOWLEDGE BASE:")
    print(f"  - Papers: {len(new_papers)}")
    print(f"  - Chunks: {new_chunks_count}")
    print(f"  - Propositions: {new_propositions_count}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Process batch extraction results")
    parser.add_argument(
        "--results-file",
        help="Path to results JSONL file (e.g., data/batch_jobs/results/batch_20251211_143022_0_results.jsonl)"
    )

    args = parser.parse_args()

    # Find results file and metadata
    if args.results_file:
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"Error: Results file not found: {results_file}")
            return 1

        # Find corresponding metadata
        # batch_20251211_143022_0_results.jsonl -> batch_20251211_143022_0.json
        stem = results_file.stem.replace("_results", "")
        metadata_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "metadata"
        metadata_file = metadata_dir / f"{stem}.json"

        if not metadata_file.exists():
            print(f"Error: Metadata file not found: {metadata_file}")
            return 1

        print(f"Processing results file: {results_file.name}")
        print(f"Using metadata file: {metadata_file.name}")
    else:
        # Find latest
        results_file, metadata_file = find_latest_results()
        if not results_file:
            return 1

    # Load metadata
    metadata = load_metadata(metadata_file)

    # Load KB FIRST to initialize ID counters
    print("\nLoading knowledge base to initialize ID counters...")
    kb = KnowledgeBase()
    kb.load()

    # Parse results
    papers_dict, chunks_dict, paper_propositions = parse_results(results_file, metadata)

    # Add to knowledge base (with deduplication check)
    add_to_knowledge_base(kb, papers_dict, chunks_dict, paper_propositions)

    print("\n✓ Processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

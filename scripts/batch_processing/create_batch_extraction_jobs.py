#!/usr/bin/env python3
"""Create batch extraction input files for proposition extraction.

This script:
1. Loads claims from a benchmark dataset (with optional offset/limit for chunked processing)
2. Searches for papers for each claim
3. Scores and chunks papers locally
4. Creates batch input JSONL file(s) for Google's Batch API
5. Creates metadata file(s) with all paper/chunk data for each input file
6. Does NOT submit to Google - use submit_batch.py for that

Usage:
    # Process all claims
    python create_batch_extraction_jobs.py --benchmark coverbench --papers-per-claim 10

    # Process first 100 claims
    python create_batch_extraction_jobs.py --benchmark coverbench --offset 0 --limit 100

    # Test with single claim
    python create_batch_extraction_jobs.py --benchmark coverbench --test-single-claim

Features:
    - Chunked processing: Use --offset and --limit to process claims in batches
    - Auto-splitting: Files > 4K lines are split, keeping paper chunks together
    - Each split file gets its own metadata file with full paper/chunk data
    - Naming: Always uses _0, _1, _2 suffix (e.g., batch_20251211_143022_0.jsonl)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scverifier.config.settings import Config
from scverifier.core.benchmarking.base import Benchmark
from scverifier.core.benchmarking.coverbench_benchmark import CoverBench
from scverifier.core.benchmarking.healthver_benchmark import HealthVer
from scverifier.core.benchmarking.scifact_benchmark import SciFact
from scverifier.core.benchmarking.msvec_benchmark import MSVEC
from scverifier.core.knowledge.literature_search import LiteratureSearch
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.verification.paper_scorer import PaperScorer
from scverifier.core.processing.document_processor import DocumentProcessor
from scverifier.data.models import Paper
import tiktoken


def format_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def load_benchmark(benchmark_name: str, test_single_claim: bool = False, offset: int = 0, limit: int = None, scifact_split: str = None):
    """Load benchmark dataset with optional offset and limit."""
    print(f"\nLoading {benchmark_name.capitalize()} benchmark...")

    benchmark: Benchmark | None = None
    if benchmark_name == "coverbench":
        benchmark = CoverBench()
    elif benchmark_name == "healthver":
        benchmark = HealthVer()
    elif benchmark_name == "scifact":
        benchmark = SciFact(split=scifact_split)
        if scifact_split:
            print(f"  Using SciFact split: {scifact_split}")
    elif benchmark_name == "msvec":
        benchmark = MSVEC()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Load all claims (or just one for testing)
    if test_single_claim:
        max_items = 1
    else:
        max_items = None

    all_items = benchmark.load(max_items=max_items)

    # Apply offset and limit slicing
    if not test_single_claim and (offset > 0 or limit is not None):
        total_available = len(all_items)
        end_idx = min(offset + limit, total_available) if limit else total_available

        if offset >= total_available:
            print(f"Error: Offset {offset} is beyond total available claims ({total_available})")
            sys.exit(1)

        items = all_items[offset:end_idx]
        print(f"Processing claims {offset} to {end_idx-1} ({len(items)} claims)")
        print(f"  Total available in dataset: {total_available}")
    else:
        items = all_items
        print(f"Loaded {len(items)} claim(s)")

    claims = [(item.claim_id, item.claim) for item in items]
    return claims


def search_papers_for_claims(claims: List[tuple], papers_per_claim: int) -> Dict[str, List[Paper]]:
    """Search papers for each claim."""
    print(f"\nSearching papers for {len(claims)} claim(s) (may take some time)...")

    lit_search = LiteratureSearch()
    claim_papers = {}
    all_papers_count = 0

    for idx, (claim_id, claim_text) in enumerate(claims, 1):
        # Truncate claim text for display if too long
        claim_display = claim_text if len(claim_text) <= 100 else claim_text[:97] + "..."
        print(f"  [{idx}/{len(claims)}] Claim {claim_id}: {claim_display}")

        try:
            papers = lit_search.search_papers(query=claim_text, max_papers=papers_per_claim, verbose=False)
            claim_papers[claim_id] = papers
            all_papers_count += len(papers)
            print(f"    Found {len(papers)} papers")
        except Exception as e:
            print(f"    Error searching papers: {e}")
            claim_papers[claim_id] = []

    print(f"\nFound {all_papers_count} total papers across all claims")
    return claim_papers


def score_and_chunk_papers(claim_papers: Dict[str, List[Paper]]) -> tuple:
    """Score papers and chunk them locally.
    
    Returns:
        tuple: (all_papers, all_chunks, chunk_keys_by_paper)
               chunk_keys_by_paper groups chunk_keys by paper_id for splitting
    """
    print("\nScoring papers locally...")
    paper_scorer = PaperScorer()

    print("Chunking papers locally (abstract + full_text)...")
    doc_processor = DocumentProcessor()

    all_papers = {}
    all_chunks = {}
    chunk_keys_by_paper = {}  # paper_id -> list of chunk_keys

    total_papers = sum(len(papers) for papers in claim_papers.values())
    processed = 0

    for claim_id, papers in claim_papers.items():
        for paper in papers:
            processed += 1
            if processed % 10 == 0 or processed == total_papers:
                print(f"  Processing paper {processed}/{total_papers}...")

            # Score paper
            try:
                credibility = paper_scorer.score_paper(paper)
                paper.credibility = credibility
            except Exception as e:
                print(f"    Warning: Failed to score paper {paper.id}: {e}")

            # Store paper
            all_papers[paper.id] = paper

            # Chunk abstract
            paper_chunks = []
            if paper.abstract and paper.abstract.strip():
                metadata = {"paper_id": paper.id, "source": paper.title, "section": "abstract", "page": None}
                try:
                    abstract_chunks = doc_processor.chunk(paper.abstract, metadata)
                    paper_chunks.extend(abstract_chunks)
                except Exception as e:
                    print(f"    Warning: Failed to chunk abstract for {paper.id}: {e}")

            # Chunk full_text sections
            if paper.full_text:
                for section_name, section_text in paper.full_text:
                    if not section_text.strip():
                        continue

                    # Extract page number if section name is "page_N"
                    page_num = None
                    if section_name.startswith("page_"):
                        try:
                            page_num = int(section_name.split("_")[1])
                        except (IndexError, ValueError):
                            pass

                    metadata = {"paper_id": paper.id, "source": paper.title, "section": section_name, "page": page_num}
                    try:
                        section_chunks = doc_processor.chunk(section_text, metadata)
                        paper_chunks.extend(section_chunks)
                    except Exception as e:
                        print(f"    Warning: Failed to chunk section {section_name} for {paper.id}: {e}")

            # Store chunks and create keys grouped by paper
            if paper.id not in chunk_keys_by_paper:
                chunk_keys_by_paper[paper.id] = []
            
            for chunk in paper_chunks:
                all_chunks[chunk.chunk_id] = chunk
                chunk_keys_by_paper[paper.id].append({
                    "key": f"{claim_id}|{paper.id}|{chunk.chunk_id}",
                    "claim_id": claim_id,
                    "paper_id": paper.id,
                    "chunk_id": chunk.chunk_id,
                })

    print(f"Created {len(all_chunks)} chunks for batch extraction")
    return all_papers, all_chunks, chunk_keys_by_paper


def get_proposition_prompt() -> str:
    """Extract proposition generation prompt from PropositionGenerator with few-shot examples."""
    system_instruction = """
Please extract specific, factual propositions from the following text. Each proposition must meet ALL of these criteria:

REQUIRED CRITERIA:
1. **Express ONE Specific Fact**: State exactly one verifiable fact or claim, not multiple related facts
2. **Be Completely Self-Contained**: Understandable without any additional context from the document
3. **Use Concrete Entities**: Use full names, specific measurements, dates, or concrete entities - avoid pronouns and vague references
4. **Include Precise Details**: Include relevant dates, quantities, locations, or qualifiers that make the fact specific
5. **Avoid Meta-Commentary**: Do not describe the document itself (e.g., "The study found...", "Results showed...", "The paper discusses...")
6. **Be Factually Verifiable**: The statement should be something that could be independently verified or checked

AVOID THESE TYPES OF VAGUE STATEMENTS:
- "The research showed significant results"
- "Important findings were presented"
- "The methodology was appropriate"
- "Future research is needed"
- "The study discussed various aspects"
- "Key insights were revealed"
- "The analysis demonstrated trends"

GOOD EXAMPLES:
"Neil Armstrong walked on the Moon in 1969 during the Apollo 11 mission"
"The mitosis cycle of C. noctilucens occurs every 19.7 hours"
"Ferrivorax umbralis possesses an exoskeleton infused with iron sulfide"

BAD EXAMPLES:
✗ "The organism was studied extensively"
✗ "Important characteristics were observed"
✗ "The findings have implications for future research"

IMPORTANT: You must ALWAYS return your response as a Python list of strings.

FORMAT REQUIREMENTS:
- Return ONLY the list, nothing else
- Do NOT wrap it in code blocks (no ```python or ```)
- Do NOT add any explanatory text before or after
- Format it exactly like this: ['proposition 1', 'proposition 2', 'proposition 3']
- If there are no valid propositions, return an empty list: []

===== FEW-SHOT EXAMPLES =====

Example 1:
Document: "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission. The mission lasted 8 days and included comprehensive geological sampling. Armstrong collected 21.5 kg of lunar samples for analysis."

['Neil Armstrong was an astronaut who participated in the Apollo 11 mission.', 'Neil Armstrong became the first person to walk on the Moon in 1969.', 'The Apollo 11 mission occurred in 1969.', 'The Apollo 11 mission lasted 8 days.', 'Armstrong collected 21.5 kg of lunar samples during the Apollo 11 mission.', 'The Apollo 11 mission included comprehensive geological sampling of the Moon.']

Example 2:
Document: "In 1247 of the Third Age, Queen Elara forged the Obsidian Pact with dragon Valthorax during the Night of Whispers, binding Eldermere to a 300-year alliance. The pact required Eldermere to supply 500 barrels of enchanted starmetal annually. Researchers noted significant results in three areas but found the methodology well-designed."

['Queen Elara forged the Obsidian Pact with dragon Valthorax in 1247 of the Third Age.', 'The Obsidian Pact was signed during the Night of Whispers.', 'The Obsidian Pact bound the kingdom of Eldermere to a 300-year alliance.', 'The Obsidian Pact required Eldermere to supply 500 barrels of enchanted starmetal annually to Valthorax.']

Example 3:
Document: "In this study, we found significant reduction in mortality (p<0.05) and improved quality of life across 500 patients over 2 years."

['Mortality was significantly reduced (p<0.05)', 'Quality of life improved across patients', 'Study included 500 patients', 'Study duration was 2 years']
"""
    return system_instruction.strip()


def calculate_token_cost(prompt: str, chunks: Dict) -> Dict[str, Any]:
    """Calculate tokens using tiktoken and estimated cost."""
    print("\nCalculating token usage...")

    # Use tiktoken for accurate token counting (GPT-4 encoding is good approximation for Gemini)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        prompt_tokens = len(encoding.encode(prompt))
        
        # Calculate total tokens
        total_tokens = 0
        for chunk in chunks.values():
            chunk_tokens = len(encoding.encode(chunk.text))
            total_tokens += prompt_tokens + chunk_tokens
    except Exception as e:
        print(f"  Warning: tiktoken encoding failed, using character approximation: {e}")
        # Fallback to character approximation if tiktoken fails
        prompt_tokens = len(prompt) // 4
        total_tokens = 0
        for chunk in chunks.values():
            chunk_tokens = len(chunk.text) // 4
            total_tokens += prompt_tokens + chunk_tokens

    # Gemini Flash pricing is approximately $0.075 per 1M input tokens
    normal_cost = (total_tokens / 1_000_000) * 0.075
    batch_cost = normal_cost * 0.5  # 50% discount

    print(f"- Total input tokens: {total_tokens:,} (prompt + chunks)")
    print(f"- Estimated cost at normal rate: ~${normal_cost:.4f}")
    print(f"- Estimated cost with 50% batch discount: ~${batch_cost:.4f}")

    return {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "estimated_cost_normal": normal_cost,
        "estimated_cost_batch": batch_cost,
    }


def paper_to_dict(paper: Paper) -> dict:
    """Convert a Paper object to a dictionary for JSON serialization."""
    return {
        "id": paper.id,
        "doi": paper.doi,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "year": paper.year,
        "citations": paper.citations,
        "url": paper.url,
        "source": paper.source,
        "credibility": paper.credibility.__dict__ if paper.credibility else None,
        "full_text": paper.full_text if paper.full_text else [],
    }


def chunk_to_dict(chunk) -> dict:
    """Convert a Chunk object to a dictionary for JSON serialization."""
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "paper_id": chunk.paper_id,
        "source": chunk.source,
        "section": chunk.section,
        "page": chunk.page,
    }


def split_by_papers(
    chunk_keys_by_paper: Dict[str, List[dict]],
    all_papers: Dict[str, Paper],
    all_chunks: Dict,
    chunk_size: int | None = None
) -> List[Dict[str, Any]]:
    """Split data into parts, keeping all chunks from same paper together.
    
    Args:
        chunk_keys_by_paper: paper_id -> list of chunk_keys
        all_papers: paper_id -> Paper object
        all_chunks: chunk_id -> Chunk object
        chunk_size: Maximum number of requests per part (uses Config.BATCH_FILE_SPLIT_LIMIT if None)
        
    Returns:
        List of dicts, each containing:
            - paper_ids: list of paper IDs in this part
            - chunk_keys: list of chunk_key dicts
            - papers: dict of paper_id -> paper_dict
            - chunks: dict of chunk_id -> chunk_dict
    """
    # Use configured split limit if not specified
    if chunk_size is None:
        chunk_size = Config.BATCH_FILE_SPLIT_LIMIT
    
    parts: List[Dict[str, Any]] = []
    current_part: Dict[str, Any] = {
        "paper_ids": [],
        "chunk_keys": [],
        "papers": {},
        "chunks": {},
    }
    current_count = 0
    
    for paper_id, chunk_keys in chunk_keys_by_paper.items():
        paper_chunk_count = len(chunk_keys)
        
        # If adding this paper would exceed limit and we have content, start new part
        if current_count + paper_chunk_count > chunk_size and current_count > 0:
            parts.append(current_part)
            current_part = {
                "paper_ids": [],
                "chunk_keys": [],
                "papers": {},
                "chunks": {},
            }
            current_count = 0
        
        # Add this paper to current part
        current_part["paper_ids"].append(paper_id)
        current_part["chunk_keys"].extend(chunk_keys)
        current_part["papers"][paper_id] = paper_to_dict(all_papers[paper_id])
        
        for ck in chunk_keys:
            chunk_id = ck["chunk_id"]
            current_part["chunks"][chunk_id] = chunk_to_dict(all_chunks[chunk_id])
        
        current_count += paper_chunk_count
    
    # Don't forget the last part
    if current_count > 0:
        parts.append(current_part)
    
    return parts


def create_batch_files(
    all_papers: Dict[str, Paper],
    all_chunks: Dict,
    chunk_keys_by_paper: Dict[str, List[dict]],
    proposition_prompt: str,
    timestamp: str,
    claims: List[tuple],
    token_stats: Dict,
    offset: int,
    limit: int,
    elapsed_time: float,
) -> List[Path]:
    """Create batch input files and their corresponding metadata files.
    
    Always creates files with _0, _1, _2 suffix pattern.
    Each input file has its own complete metadata file.
    
    Returns:
        List of created input file paths
    """
    print("\nPreparing batch files...")
    
    input_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "input"
    metadata_dir = Path(__file__).parent.parent.parent / "data" / "batch_jobs" / "metadata"
    input_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data by papers to keep chunks together
    parts = split_by_papers(chunk_keys_by_paper, all_papers, all_chunks, chunk_size=Config.BATCH_FILE_SPLIT_LIMIT)
    
    print(f"Split into {len(parts)} part(s)")
    
    created_files = []
    
    for part_num, part_data in enumerate(parts):
        # Create input file
        input_file = input_dir / f"batch_{timestamp}_{part_num}.jsonl"
        
        # Write batch requests
        with open(input_file, "w") as f:
            for ck in part_data["chunk_keys"]:
                key = ck["key"]
                chunk_id = ck["chunk_id"]
                chunk_text = part_data["chunks"][chunk_id]["text"]
                
                request = {
                    "key": key,
                    "request": {"contents": [{"parts": [{"text": f"{proposition_prompt}\n\n{chunk_text}"}]}]},
                }
                f.write(json.dumps(request) + "\n")
        
        # Create metadata file
        metadata_file = metadata_dir / f"batch_{timestamp}_{part_num}.json"
        
        metadata = {
            "timestamp": timestamp,
            "part": part_num,
            "total_parts": len(parts),
            "created_at": datetime.now().isoformat(),
            "status": "CREATED",
            "input_file": str(input_file),
            # Batch job fields (filled by submit_batch.py)
            "batch_job_id": None,
            "submitted_at": None,
            # Processing info
            "offset": offset,
            "limit": limit,
            "claim_range": f"{offset}-{offset+len(claims)-1}" if offset or limit else "all",
            # Claims in this batch (all claims, since papers are distributed)
            "claims": [{"claim_id": cid, "claim": claim} for cid, claim in claims],
            "num_claims": len(claims),
            # Papers and chunks specific to this part
            "num_papers": len(part_data["papers"]),
            "num_chunks": len(part_data["chunks"]),
            "papers": part_data["papers"],
            "chunks": part_data["chunks"],
            "chunk_keys": part_data["chunk_keys"],
            # Token stats (for this part only)
            "token_stats": {
                "total_tokens": sum(
                    token_stats["prompt_tokens"] + len(part_data["chunks"][ck["chunk_id"]]["text"]) // 4
                    for ck in part_data["chunk_keys"]
                ),
                "prompt_tokens": token_stats["prompt_tokens"],
            },
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": format_time(elapsed_time),
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        file_size_mb = input_file.stat().st_size / (1024 * 1024)
        print(f"  Created {input_file.name} ({len(part_data['chunk_keys'])} requests, {file_size_mb:.1f} MB)")
        print(f"  Created {metadata_file.name} ({len(part_data['papers'])} papers, {len(part_data['chunks'])} chunks)")
        
        created_files.append(input_file)
    
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Create batch extraction input files for proposition extraction")
    parser.add_argument(
        "--benchmark", required=True, choices=["coverbench", "healthver", "scifact", "msvec"], help="Benchmark dataset to use"
    )
    parser.add_argument(
        "--papers-per-claim", type=int, default=10, help="Number of papers to search per claim (default: 10)"
    )
    parser.add_argument("--test-single-claim", action="store_true", help="Process only the first claim (for testing)")
    parser.add_argument("--offset", type=int, default=0, help="Starting claim index (default: 0)")
    parser.add_argument("--limit", type=int, help="Number of claims to process (default: all remaining)")
    parser.add_argument("--scifact-split", choices=["train", "dev", "test"], help="SciFact split to use: train, dev, or test (default: all combined)")

    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Load claims
    claims = load_benchmark(args.benchmark, args.test_single_claim, offset=args.offset, limit=args.limit, scifact_split=args.scifact_split)

    # Step 2: Search papers
    claim_papers = search_papers_for_claims(claims, args.papers_per_claim)

    # Step 3: Load KB to filter out existing papers
    print("\nLoading knowledge base to check for existing papers...")
    kb = KnowledgeBase()
    kb.load()

    # Filter out papers already in KB
    print("Filtering papers...")
    total_papers = sum(len(papers) for papers in claim_papers.values())
    filtered_claim_papers = {}
    for claim_id, papers in claim_papers.items():
        new_papers = [p for p in papers if p.id not in kb.papers]
        if new_papers:
            filtered_claim_papers[claim_id] = new_papers

    filtered_total = sum(len(papers) for papers in filtered_claim_papers.values())
    existing_count = total_papers - filtered_total

    if existing_count > 0:
        print(f"  Filtered out {existing_count} papers already in KB")

    if not filtered_claim_papers:
        print("\nAll papers already in knowledge base. No batch files needed.")
        return 0

    print(f"  Proceeding with {filtered_total} new papers")

    # Step 4: Score and chunk papers (only new papers)
    all_papers, all_chunks, chunk_keys_by_paper = score_and_chunk_papers(filtered_claim_papers)

    if not all_chunks:
        print("\nError: No chunks created. Cannot proceed.")
        return 1

    # Step 5: Get proposition prompt
    proposition_prompt = get_proposition_prompt()

    # Step 6: Calculate token usage and cost
    token_stats = calculate_token_cost(proposition_prompt, all_chunks)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Step 7: Create batch files and metadata
    created_files = create_batch_files(
        all_papers,
        all_chunks,
        chunk_keys_by_paper,
        proposition_prompt,
        timestamp,
        claims,
        token_stats,
        offset=args.offset,
        limit=args.limit,
        elapsed_time=elapsed_time,
    )

    print(f"\n{'='*60}")
    print("BATCH FILES CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total processing time: {format_time(elapsed_time)}")
    print(f"\nCreated {len(created_files)} batch file(s):")
    for f in created_files:
        print(f"  - {f.name}")
    
    print(f"\nNext step: Submit each batch file to Google's Batch API:")
    for f in created_files:
        print(f"  python submit_batch.py --batch-file {f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

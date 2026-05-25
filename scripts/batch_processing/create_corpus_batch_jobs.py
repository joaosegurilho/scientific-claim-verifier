#!/usr/bin/env python3
"""Create batch extraction input files from corpus papers referenced by SciFact dev set claims.

This script:
1. Loads claims from SciFact dev set ONLY
2. Collects all unique doc_ids/cited_doc_ids referenced by claims
3. Loads those papers from the corpus.jsonl file
4. Chunks papers locally
5. Creates batch input JSONL file(s) for Google's Batch API
6. Creates metadata file(s) with all paper/chunk data for each input file
7. Does NOT submit to Google - use submit_batch.py for that

Usage:
    # Process all corpus papers referenced by SciFact dev set claims
    python create_corpus_batch_jobs.py

    # Process with custom corpus path
    python create_corpus_batch_jobs.py --corpus data/msvec_data/corpus.jsonl

    # Test with limited papers
    python create_corpus_batch_jobs.py --limit 10

Features:
    - Loads papers from local corpus.jsonl (no API calls)
    - Auto-splitting: Files > 2K lines are split, keeping paper chunks together
    - Each split file gets its own metadata file with full paper/chunk data
    - Naming: Always uses _0, _1, _2 suffix (e.g., corpus_batch_20251214_143022_0.jsonl)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scverifier.config.settings import Config
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
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


def load_corpus(corpus_path: Path, candidates_path: Path = None) -> Dict[int, dict]:
    """Load corpus.jsonl as {doc_id: entry} dictionary, with optional fallback to candidates.

    Args:
        corpus_path: Path to corpus.jsonl file
        candidates_path: Optional path to corpus_candidates.jsonl for fallback

    Returns:
        Dictionary mapping doc_id (int) to corpus entry dict
    """
    print(f"\nLoading corpus from {corpus_path}...")
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            corpus[entry['doc_id']] = entry
    print(f"  Loaded {len(corpus)} papers from main corpus")

    # Load candidates as fallback if provided
    if candidates_path and candidates_path.exists():
        print(f"  Loading additional papers from {candidates_path}...")
        candidates_count = 0
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry['doc_id']
                # Only add if not already in main corpus
                if doc_id not in corpus:
                    corpus[doc_id] = entry
                    candidates_count += 1
        print(f"  Loaded {candidates_count} additional papers from candidates")
        print(f"  Total corpus size: {len(corpus)} papers")

    return corpus


def get_all_claim_doc_ids(data_dir: Path) -> Set[int]:
    """Get all doc_ids referenced by SciFact dev set claims.

    Args:
        data_dir: Path to data directory containing scifact_data

    Returns:
        Set of unique doc_ids (integers)
    """
    print(f"\nCollecting doc_ids from SciFact dev set claims in {data_dir}...")
    doc_ids = set()
    scifact_count = 0

    # SciFact claims (DEV SET ONLY)
    scifact_dir = data_dir / "scifact_data"
    claims_file = scifact_dir / "claims_dev.jsonl"
    if claims_file.exists():
        with open(claims_file, 'r', encoding='utf-8') as f:
            for line in f:
                claim = json.loads(line)
                # Get cited_doc_ids
                doc_ids.update(claim.get("cited_doc_ids", []))
                # Also get doc_ids from evidence keys
                for doc_id in claim.get("evidence", {}).keys():
                    doc_ids.add(int(doc_id))
                scifact_count += 1
        print(f"  SciFact claims_dev.jsonl: {scifact_count} claims")
    else:
        print(f"  Error: SciFact dev file not found at {claims_file}")
        return set()

    print(f"\n  Total claims: {scifact_count}")
    print(f"  Unique doc_ids: {len(doc_ids)}")

    return doc_ids


def corpus_entry_to_paper(entry: dict) -> Paper:
    """Convert corpus entry to Paper object.

    Args:
        entry: Dictionary from corpus.jsonl with doc_id, title, abstract

    Returns:
        Paper object with abstract as joined string
    """
    # Join abstract sentences into single string
    abstract = " ".join(entry["abstract"]) if isinstance(entry["abstract"], list) else entry["abstract"]

    return Paper(
        id=str(entry["doc_id"]),
        doi=None,
        title=entry["title"],
        abstract=abstract,
        source="corpus",
    )


def chunk_papers(papers: List[Paper]) -> tuple:
    """Chunk papers using DocumentProcessor.

    Args:
        papers: List of Paper objects

    Returns:
        tuple: (all_papers dict, all_chunks dict, chunk_keys_by_paper dict)
    """
    print("\nChunking papers...")
    doc_processor = DocumentProcessor()

    all_papers = {}
    all_chunks = {}
    chunk_keys_by_paper = {}  # paper_id -> list of chunk_keys

    for idx, paper in enumerate(papers, 1):
        if idx % 100 == 0 or idx == len(papers):
            print(f"  Processing paper {idx}/{len(papers)}...")

        # Store paper
        all_papers[paper.id] = paper

        # Chunk abstract (corpus papers only have abstracts)
        paper_chunks = []
        if paper.abstract and paper.abstract.strip():
            metadata = {"paper_id": paper.id, "source": paper.title, "section": "abstract", "page": None}
            try:
                abstract_chunks = doc_processor.chunk(paper.abstract, metadata)
                paper_chunks.extend(abstract_chunks)
            except Exception as e:
                print(f"    Warning: Failed to chunk abstract for {paper.id}: {e}")

        # Store chunks and create keys grouped by paper
        if paper.id not in chunk_keys_by_paper:
            chunk_keys_by_paper[paper.id] = []

        for chunk in paper_chunks:
            all_chunks[chunk.chunk_id] = chunk
            chunk_keys_by_paper[paper.id].append({
                "key": f"corpus|{paper.id}|{chunk.chunk_id}",
                "paper_id": paper.id,
                "chunk_id": chunk.chunk_id,
            })

    print(f"  Created {len(all_chunks)} chunks from {len(all_papers)} papers")
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
x "The organism was studied extensively"
x "Important characteristics were observed"
x "The findings have implications for future research"

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

    print(f"  Total input tokens: {total_tokens:,} (prompt + chunks)")
    print(f"  Estimated cost at normal rate: ~${normal_cost:.4f}")
    print(f"  Estimated cost with 50% batch discount: ~${batch_cost:.4f}")

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
    token_stats: Dict,
    elapsed_time: float,
    offset: int,
    limit: int,
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

    print(f"  Split into {len(parts)} part(s)")

    created_files = []

    for part_num, part_data in enumerate(parts):
        # Create input file with corpus_ prefix to distinguish from online search batches
        input_file = input_dir / f"corpus_batch_{timestamp}_{part_num}.jsonl"

        # Write batch requests
        with open(input_file, "w", encoding="utf-8") as f:
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
        metadata_file = metadata_dir / f"corpus_batch_{timestamp}_{part_num}.json"

        metadata = {
            "timestamp": timestamp,
            "part": part_num,
            "total_parts": len(parts),
            "created_at": datetime.now().isoformat(),
            "status": "CREATED",
            "input_file": str(input_file),
            "source": "corpus",  # Mark as corpus-based
            # Batch job fields (filled by submit_batch.py)
            "batch_job_id": None,
            "submitted_at": None,
            # Processing info
            "offset": offset,
            "limit": limit,
            "paper_range": f"{offset}-{offset+len(all_papers)-1}" if offset or limit else "all",
            # Claims (empty for corpus-based processing, but required by submit_batch.py)
            "claims": [],
            "num_claims": 0,
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

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        file_size_mb = input_file.stat().st_size / (1024 * 1024)
        print(f"  Created {input_file.name} ({len(part_data['chunk_keys'])} requests, {file_size_mb:.2f} MB)")
        print(f"  Created {metadata_file.name} ({len(part_data['papers'])} papers, {len(part_data['chunks'])} chunks)")

        created_files.append(input_file)

    return created_files


def main():
    parser = argparse.ArgumentParser(description="Create batch extraction jobs from corpus papers")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/msvec_data/corpus.jsonl",
        help="Path to corpus.jsonl file (default: data/msvec_data/corpus.jsonl)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing scifact_data and msvec_data (default: data)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting paper index (default: 0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Number of papers to process (default: all remaining after offset)"
    )
    parser.add_argument(
        "--skip-kb-filter",
        action="store_true",
        help="Skip filtering out papers already in knowledge base"
    )

    args = parser.parse_args()

    # Start timing
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CREATE CORPUS BATCH JOBS")
    print("=" * 70)

    # Step 1: Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        return 1

    # Also check for corpus_candidates.jsonl as fallback
    candidates_path = Path(args.data_dir) / "corpus_candidates.jsonl"

    corpus = load_corpus(corpus_path, candidates_path)

    # Step 2: Get all doc_ids from claims
    data_dir = Path(args.data_dir)
    doc_ids = get_all_claim_doc_ids(data_dir)

    # Step 3: Convert to Paper objects
    print("\nConverting corpus entries to Paper objects...")
    all_papers = []
    missing = []
    for doc_id in sorted(doc_ids):
        if doc_id in corpus:
            all_papers.append(corpus_entry_to_paper(corpus[doc_id]))
        else:
            missing.append(doc_id)

    if missing:
        print(f"  Warning: {len(missing)} doc_ids not found in corpus: {missing[:10]}...")

    print(f"  Created {len(all_papers)} Paper objects")

    # Step 4: Apply offset and limit slicing
    total_available = len(all_papers)

    if args.offset >= total_available:
        print(f"\nError: Offset {args.offset} is beyond total available papers ({total_available})")
        return 1

    end_idx = min(args.offset + args.limit, total_available) if args.limit else total_available
    papers = all_papers[args.offset:end_idx]

    if args.offset > 0 or args.limit:
        print(f"  Processing papers {args.offset} to {end_idx-1} ({len(papers)} papers)")
        print(f"  Total available in corpus: {total_available}")
    else:
        print(f"  Processing all {len(papers)} papers")

    # Step 5: Load KB and filter existing papers (optional)
    if not args.skip_kb_filter:
        print("\nLoading knowledge base to check for existing papers...")
        kb = KnowledgeBase()
        try:
            kb.load()

            original_count = len(papers)
            papers = [p for p in papers if p.id not in kb.papers]
            filtered_count = original_count - len(papers)

            if filtered_count > 0:
                print(f"  Filtered out {filtered_count} papers already in KB")

            if not papers:
                print("\nAll papers already in knowledge base. No batch files needed.")
                return 0

            print(f"  Proceeding with {len(papers)} new papers")
        except Exception as e:
            print(f"  Warning: Could not load KB ({e}), processing all papers")

    # Step 6: Chunk papers
    all_papers, all_chunks, chunk_keys_by_paper = chunk_papers(papers)

    if not all_chunks:
        print("\nError: No chunks created. Cannot proceed.")
        return 1

    # Step 7: Get proposition prompt
    proposition_prompt = get_proposition_prompt()

    # Step 8: Calculate token usage and cost
    token_stats = calculate_token_cost(proposition_prompt, all_chunks)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Step 9: Create batch files and metadata
    created_files = create_batch_files(
        all_papers,
        all_chunks,
        chunk_keys_by_paper,
        proposition_prompt,
        timestamp,
        token_stats,
        elapsed_time,
        offset=args.offset,
        limit=args.limit,
    )

    print(f"\n{'='*70}")
    print("BATCH FILES CREATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Total processing time: {format_time(elapsed_time)}")
    print(f"Papers processed: {len(all_papers)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"\nCreated {len(created_files)} batch file(s):")
    for f in created_files:
        print(f"  - {f.name}")

    print(f"\nNext step: Submit each batch file to Google's Batch API:")
    for f in created_files:
        print(f"  python submit_batch.py --batch-file {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Simple FastAPI web application for Scientific Claim Verification System.

This provides a web interface to:
- Browse papers in the knowledge base
- View paper details (chunks, propositions)
- Search propositions
- View all chunks
- Verify scientific claims
"""

import json
import os
import tempfile
import traceback
import uvicorn
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse

from scverifier.api.api_pmc import PMCAPI
from scverifier.core.agents.autonomous_agent import AutonomousClaimAgent
from scverifier.core.extraction.proposition_evaluator import PropositionEvaluator
from scverifier.webapp.verify_agent_simple import stream_agent_verification
from scverifier.core.extraction.proposition_extractor import PropositionExtractor
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.knowledge.literature_search import LiteratureSearch
from scverifier.core.retrieval.response_generator import ResponseGenerator
from scverifier.core.verification.confidence_interpreter import get_confidence_interpretation, get_confidence_level
from scverifier.core.verification.paper_scorer import PaperScorer
from scverifier.core.visualization.timeline_generator import TimelineGenerator
from scverifier.data.file_loader import FileLoader
from scverifier.data.local_paper_processor import LocalPaperProcessor
from scverifier.pipelines.verification_pipeline import VerificationPipeline

# Initialize FastAPI app
app = FastAPI(title="Scientific Claim Verification System")

# Setup templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Initialize knowledge base (loaded once at startup)
kb = KnowledgeBase()
response_generator = ResponseGenerator()
lit_search = LiteratureSearch()

try:
    kb.load()
    print(f" Knowledge base loaded with {len(kb.papers)} papers")
except FileNotFoundError:
    print(" No knowledge base found. Please run extraction pipeline first.")
except Exception as e:
    print(f" Error loading knowledge base: {e}")


# ======================== JINJA2 CUSTOM FILTERS ========================


def get_chunk_section(chunk_id: str) -> str:
    """Get the section from a chunk by its ID."""
    for paper in kb.papers.values():
        for chunk in paper.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.section
    return ""


def format_source_badge(section: str = "", page: Optional[int] = None) -> str:
    """Format a source badge showing section or page.

    Priority: section > page
    """
    if section:
        # Format section name (replace underscores, title case)
        formatted = section.replace("_", " ").title()
        return f'<span class="section-badge">{formatted}</span>'
    elif page:
        return f'<span class="page-badge">Page {page}</span>'
    return ""


# Register custom filters
templates.env.filters["get_chunk_section"] = get_chunk_section
templates.env.filters["format_source_badge"] = format_source_badge

# Register custom global functions (for use as function calls in templates)
templates.env.globals["format_source_badge"] = format_source_badge
templates.env.globals["get_chunk_section"] = get_chunk_section


# ======================== PAGINATION HELPER ========================


def paginate_items(items: list, page: int, per_page: int = 100) -> dict:
    """Paginate a list of items and return pagination metadata.

    Args:
        items: List of items to paginate
        page: Current page number (1-indexed)
        per_page: Items per page (default: 100)

    Returns:
        dict with:
            - items: Sliced list for current page
            - current_page: Current page number
            - total_pages: Total number of pages
            - total_items: Total number of items
            - has_prev: Whether there's a previous page
            - has_next: Whether there's a next page
            - prev_page: Previous page number (or None)
            - next_page: Next page number (or None)
    """
    # Ensure page is at least 1
    page = max(1, page)

    total_items = len(items)
    total_pages = max(1, (total_items + per_page - 1) // per_page)  # Ceiling division

    # Clamp page to valid range
    page = min(page, total_pages)

    # Calculate slice indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    # Calculate item range for display (1-indexed)
    item_start = start_idx + 1 if total_items > 0 else 0
    item_end = min(end_idx, total_items)

    return {
        "items": items[start_idx:end_idx],
        "current_page": page,
        "total_pages": total_pages,
        "total_items": total_items,
        "item_start": item_start,
        "item_end": item_end,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages else None,
    }


# ======================== HOME PAGE ========================


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - redirect to papers."""
    return RedirectResponse(url="/papers", status_code=302)


# ======================== PAPERS ========================


@app.get("/papers", response_class=HTMLResponse)
async def papers_page(
    request: Request,
    full_text: str = "all",
    source: str = "all",
    year_from: str = "",
    credibility_min: str = "",
    page: int = 1,
):
    """Display all papers in the knowledge base."""
    papers_list = list(kb.papers.values())

    # Apply full text filter
    if full_text == "available":
        papers_list = [p for p in papers_list if p.full_text and len(p.full_text) > 0]
    elif full_text == "abstract_only":
        papers_list = [p for p in papers_list if not p.full_text or len(p.full_text) == 0]

    # Apply source filter
    if source != "all":
        papers_list = [p for p in papers_list if p.source == source]

    # Apply year filter (from year onwards)
    if year_from and year_from.strip():
        try:
            year_val = int(year_from)
            papers_list = [p for p in papers_list if p.year and p.year >= year_val]
        except ValueError:
            pass  # Invalid year input, ignore filter

    # Apply credibility threshold filter
    if credibility_min and credibility_min.strip():
        try:
            cred_val = float(credibility_min)
            papers_list = [p for p in papers_list if p.credibility and p.credibility.rating >= cred_val]
        except ValueError:
            pass  # Invalid credibility input, ignore filter

    # Sort by citations (descending) then year (descending)
    papers_list.sort(key=lambda p: (-(p.citations or 0), -(p.year or 0)))

    # Apply pagination
    pagination = paginate_items(papers_list, page, per_page=100)

    return templates.TemplateResponse(
        "papers.html",
        {
            "request": request,
            "papers": pagination["items"],
            "stats": kb.get_statistics(),
            "full_text_filter": full_text,
            "source_filter": source,
            "year_from_filter": year_from,
            "credibility_min_filter": credibility_min,
            "pagination": pagination,
        },
    )


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
async def paper_detail(request: Request, paper_id: str):
    """Display detailed information about a specific paper."""
    paper = kb.get_paper(paper_id)

    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper with ID '{paper_id}' not found.")

    # Get statistics
    stats = paper.get_statistics()

    # Get quality propositions
    quality_props = paper.get_quality_propositions()

    # Check if propositions were extracted from full text
    has_fulltext_propositions = paper.extracted_from in ["full_text", "both"]

    return templates.TemplateResponse(
        "paper_detail.html",
        {
            "request": request,
            "paper": paper,
            "stats": stats,
            "quality_props": quality_props,
            "chunks": paper.chunks,
            "all_propositions": paper.propositions,
            "has_fulltext_propositions": has_fulltext_propositions,
        },
    )


@app.get("/api/papers/{paper_id}/similar")
async def get_similar_papers(paper_id: str, top_k: int = 5):
    """Get papers similar to the given paper based on proposition similarity."""
    similar_papers = kb.find_similar_papers(paper_id, top_k)

    # Convert to JSON-serializable format
    results = []
    for paper, score in similar_papers:
        results.append(
            {
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "citations": paper.citations,
                "similarity_score": float(score),  # Convert numpy float32 to Python float
                "url": f"/papers/{paper.id}",
            }
        )

    return JSONResponse(content={"similar_papers": results})


@app.get("/api/propositions/{proposition_id}/similar")
async def get_similar_propositions(proposition_id: str, top_k: int = 5):
    """Get propositions similar to the given proposition based on semantic similarity."""
    similar_propositions = kb.find_similar_propositions(proposition_id, top_k)

    # Convert to JSON-serializable format
    results = []
    for prop, score in similar_propositions:
        # Get the source paper for context
        paper = kb.get_paper(prop.paper_id)
        # Get section from chunk
        section = ""
        if paper:
            for chunk in paper.chunks:
                if chunk.chunk_id == prop.chunk_id:
                    section = chunk.section
                    break
        results.append(
            {
                "id": prop.prop_id,
                "text": prop.text,
                "is_quality": prop.is_high_quality(),
                "paper_id": prop.paper_id,
                "paper_title": paper.title if paper else "Unknown",
                "chunk_id": prop.chunk_id,
                "page": prop.page,
                "section": section,
                "similarity_score": float(score),  # Convert numpy float32 to Python float
                "url": f"/propositions/{prop.prop_id}",
            }
        )

    return JSONResponse(content={"similar_propositions": results})


@app.get("/api/chunks/{chunk_id}/similar")
async def get_similar_chunks(chunk_id: str, top_k: int = 5):
    """Get chunks similar to the given chunk based on semantic similarity."""
    similar_chunks = kb.find_similar_chunks(chunk_id, top_k)

    # Convert to JSON-serializable format
    results = []
    for chunk, score in similar_chunks:
        # Get the source paper for context
        paper = kb.get_paper(chunk.paper_id)
        
        results.append(
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "paper_id": chunk.paper_id,
                "paper_title": paper.title if paper else "Unknown",
                "section": chunk.section,
                "page": chunk.page,
                "similarity_score": float(score),  # Convert numpy float32 to Python float
                "url": f"/chunks/{chunk.paper_id}/{chunk.chunk_id}",
            }
        )

    return JSONResponse(content={"similar_chunks": results})


# ======================== PROPOSITIONS ========================


@app.get("/propositions", response_class=HTMLResponse)
async def propositions_page(request: Request, filter: str = "quality", page: int = 1):
    """Display all propositions in the knowledge base."""
    # Collect propositions based on filter
    all_props = []
    for paper in kb.papers.values():
        if filter == "all":
            all_props.extend(paper.propositions)
        else:
            all_props.extend(paper.get_quality_propositions())

    # Apply pagination
    pagination = paginate_items(all_props, page, per_page=100)

    return templates.TemplateResponse(
        "propositions.html",
        {
            "request": request,
            "propositions": pagination["items"],
            "total": pagination["total_items"],
            "filter_mode": filter,
            "pagination": pagination,
        },
    )


@app.post("/propositions/search", response_class=HTMLResponse)
async def search_propositions(request: Request, query: str = Form(...)):
    """Search for similar propositions."""
    if not query.strip():
        return templates.TemplateResponse(
            "propositions.html",
            {
                "request": request,
                "propositions": [],
                "total": 0,
                "query": query,
                "error": "Please enter a search query.",
            },
        )

    # Search using knowledge base
    results = kb.search_propositions(query)

    return templates.TemplateResponse(
        "propositions.html", {"request": request, "propositions": results, "total": len(results), "query": query}
    )


@app.get("/propositions/{prop_id}", response_class=HTMLResponse)
async def proposition_detail(request: Request, prop_id: str):
    """Display detailed information about a proposition."""
    # Search through all papers to find the proposition with this unique ID
    proposition = None
    paper = None

    for p in kb.papers.values():
        prop = p.get_proposition(prop_id)
        if prop:
            proposition = prop
            paper = p
            break

    if not proposition or not paper:
        raise HTTPException(status_code=404, detail=f"Proposition with ID '{prop_id}' not found.")

    # Find the chunk
    chunk = None
    for c in paper.chunks:
        if c.chunk_id == proposition.chunk_id:
            chunk = c
            break

    if not chunk:
        raise HTTPException(status_code=404, detail=f"Chunk with ID '{proposition.chunk_id}' not found.")

    return templates.TemplateResponse(
        "proposition_detail.html",
        {
            "request": request,
            "paper": paper,
            "chunk": chunk,
            "proposition": proposition,
        },
    )


@app.get("/chunks/{paper_id}/{chunk_id}", response_class=HTMLResponse)
async def chunk_detail_with_paper(request: Request, paper_id: str, chunk_id: str):
    """Display detailed information about a chunk and all its propositions."""
    # Find the paper
    paper = kb.papers.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper with ID '{paper_id}' not found.")

    # Find the chunk in this paper
    chunk = None
    for c in paper.chunks:
        if c.chunk_id == chunk_id:
            chunk = c
            break

    if not chunk:
        raise HTTPException(status_code=404, detail=f"Chunk with ID '{chunk_id}' not found in paper '{paper_id}'.")

    # Find all propositions from this chunk
    chunk_propositions = [p for p in paper.propositions if p.chunk_id == chunk_id]

    return templates.TemplateResponse(
        "chunk_detail.html", {"request": request, "paper": paper, "chunk": chunk, "propositions": chunk_propositions}
    )


@app.get("/chunks/{chunk_id}", response_class=HTMLResponse)
async def chunk_detail(request: Request, chunk_id: str):
    """Display detailed information about a chunk and all its propositions (legacy single-parameter route)."""
    # Search through all papers to find the chunk with this unique ID
    chunk = None
    paper = None

    for p in kb.papers.values():
        for c in p.chunks:
            if c.chunk_id == chunk_id:
                chunk = c
                paper = p
                break
        if chunk:
            break

    if not chunk or not paper:
        raise HTTPException(status_code=404, detail=f"Chunk with ID '{chunk_id}' not found.")

    # Find all propositions from this chunk
    chunk_propositions = [p for p in paper.propositions if p.chunk_id == chunk_id]

    return templates.TemplateResponse(
        "chunk_detail.html", {"request": request, "paper": paper, "chunk": chunk, "propositions": chunk_propositions}
    )


# ======================== CHUNKS ========================


@app.get("/chunks", response_class=HTMLResponse)
async def chunks_page(request: Request, page: int = 1):
    """Display all chunks in the knowledge base."""
    # Collect all chunks from all papers
    all_chunks = []
    for paper in kb.papers.values():
        all_chunks.extend(paper.chunks)

    # Apply pagination
    pagination = paginate_items(all_chunks, page, per_page=100)

    return templates.TemplateResponse(
        "chunks.html",
        {
            "request": request,
            "chunks": pagination["items"],
            "total": pagination["total_items"],
            "pagination": pagination,
        },
    )


@app.post("/chunks/search", response_class=HTMLResponse)
async def search_chunks(request: Request, query: str = Form(...)):
    """Search for similar chunks."""
    if not query.strip():
        return templates.TemplateResponse(
            "chunks.html",
            {"request": request, "chunks": [], "total": 0, "query": query, "error": "Please enter a search query."},
        )

    # Search using knowledge base
    results = kb.search_chunks(query)

    return templates.TemplateResponse(
        "chunks.html", {"request": request, "chunks": results, "total": len(results), "query": query}
    )


# ======================== CLAIM VERIFICATION ========================


@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    """Display claim verification page."""
    stats = kb.get_statistics()
    return templates.TemplateResponse("verify.html", {"request": request, "kb_stats": stats})


@app.post("/verify", response_class=HTMLResponse)
async def verify_claim(
    request: Request,
    claim: str = Form(...),
    verification_mode: str = Form("agentless"),  # agentless, agent
    search_mode: str = Form("kb_quality"),
    max_papers: int = Form(30),
    max_props_per_paper: int = Form(5),
    max_propositions: int = Form(50),
    use_full_text: bool = Form(False),
    skip_evaluation: bool = Form(False),
    use_abstract_only: bool = Form(False),
    allow_online_search: bool = Form(False),
    agent_quality_only: bool = Form(False),
):
    """Verify a scientific claim."""
    if not claim.strip():
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": "Please enter a claim to verify.", "kb_stats": kb.get_statistics()},
        )

    # If agent mode is selected, render streaming page
    if verification_mode == "agent":
        return templates.TemplateResponse(
            "verify.html",
            {
                "request": request,
                "claim": claim,
                "verification_mode": verification_mode,
                "agent_mode": True,
                "allow_online_search": allow_online_search,
                "agent_quality_only": agent_quality_only,
                "kb_stats": kb.get_statistics(),
            },
        )

    logs = []

    # If the proposition vector store is missing, rebuild it automatically
    if (
        not hasattr(kb.retrieval_system, "proposition_vectorstore")
        or kb.retrieval_system.proposition_vectorstore is None
    ):
        logs.append("Proposition vector store not initialized. Rebuilding vector stores from current papers...")
        kb.load()  # either loads or rebuilds from scratch
        kb.save()
    try:
        # Initialize verification pipeline with skip_evaluation
        logs.append("Initializing verification pipeline...")
        pipeline = VerificationPipeline(kb=kb)
        pipeline.extractor.skip_evaluation = skip_evaluation

        # Run verification based on mode
        if search_mode == "with_search":
            logs.append(f" Searching for up to {max_papers} papers online...")
            logs.append(" This may take a few minutes...")
            result = pipeline.verify_claim_with_search(
                claim,
                max_papers=max_papers,
                use_full_text=use_full_text,
                max_props_per_paper=max_props_per_paper,
                max_propositions=max_propositions,
            )
            logs.append(" Verification with search complete!")
        elif search_mode == "kb_quality":
            logs.append(" Verifying using existing knowledge base and quality claims...")
            if use_abstract_only:
                logs.append("   Filter: Using abstract propositions only")
            result = pipeline.verify_claim_from_kb(
                claim,
                quality_claims=True,
                use_abstract_only=use_abstract_only,
                max_props_per_paper=max_props_per_paper,
                max_propositions=max_propositions,
            )
            logs.append(" Verification from KB complete!")
        else:
            logs.append(" Verifying using existing knowledge base and all claims...")
            if use_abstract_only:
                logs.append("   Filter: Using abstract propositions only")
            result = pipeline.verify_claim_from_kb(
                claim,
                quality_claims=False,
                use_abstract_only=use_abstract_only,
                max_props_per_paper=max_props_per_paper,
                max_propositions=max_propositions,
            )
            logs.append(" Verification from KB complete!")

        # Get paper details for evidence
        evidence_papers: List[Dict[str, Any]] = []
        for prop in result.evidence:
            paper = kb.get_paper(prop.paper_id)
            if paper and paper.id not in [p["id"] for p in evidence_papers]:
                evidence_papers.append(
                    {
                        "id": paper.id,
                        "title": paper.title,
                        "doi": paper.doi,
                        "year": paper.year,
                        "citations": paper.citations,
                        "url": paper.url,
                        "credibility": paper.credibility,
                    }
                )

        # Get confidence interpretation
        confidence_interpretation = get_confidence_interpretation(result.verdict, int(round(result.confidence)))
        confidence_level = get_confidence_level(int(round(result.confidence)))

        return templates.TemplateResponse(
            "verify.html",
            {
                "request": request,
                "claim": claim,
                "search_mode": search_mode,
                "max_papers": max_papers,
                "max_props_per_paper": max_props_per_paper,
                "max_propositions": max_propositions,
                "use_full_text": use_full_text,
                "skip_evaluation": skip_evaluation,
                "use_abstract_only": use_abstract_only,
                "result": result,
                "evidence_papers": evidence_papers,
                "confidence_interpretation": confidence_interpretation,
                "confidence_level": confidence_level,
                "logs": logs,
                "kb_stats": kb.get_statistics(),
            },
        )

    except Exception as e:
        error_details = traceback.format_exc()
        logs.append(f" Error: {str(e)}")

        return templates.TemplateResponse(
            "verify.html",
            {
                "request": request,
                "claim": claim,
                "search_mode": search_mode,
                "error": f"Error during verification: {str(e)}",
                "error_details": error_details,
                "logs": logs,
                "kb_stats": kb.get_statistics(),
                "skip_evaluation": skip_evaluation,
            },
        )


@app.get("/api/verify-agent-stream")
async def verify_agent_stream(claim: str, allow_online_search: bool = False, agent_quality_only: bool = False):
    """Stream agent verification progress using Server-Sent Events.

    Args:
        claim: The scientific claim to verify
        allow_online_search: Whether to allow searching online databases
        agent_quality_only: Whether to search only quality propositions
    """
    print("\n[API] /api/verify-agent-stream called", flush=True)
    print(f"[API] Claim: {claim[:80]}...", flush=True)
    print(f"[API] Online: {allow_online_search}, Quality: {agent_quality_only}", flush=True)

    try:
        # Initialize agent
        print("[API] Creating agent...", flush=True)
        agent = AutonomousClaimAgent(
            kb=kb, lit_search=lit_search, allow_online_search=allow_online_search, quality_only=agent_quality_only
        )
        print("[API] Agent created successfully", flush=True)

        # Use the simple streaming function
        print("[API] Starting stream...", flush=True)
        return StreamingResponse(
            stream_agent_verification(agent, claim, kb),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[API ERROR] Failed to create agent or stream: {error_msg}", flush=True)
        print(f"[API ERROR] Traceback:\n{error_trace}", flush=True)

        async def error_generator():
            error_data = {
                "type": "error",
                "error": f"Failed to initialize verification: {error_msg}",
                "details": error_trace,
            }
            yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )


# ======================== Q&A (QUESTION & ANSWER) ========================


@app.get("/qa", response_class=HTMLResponse)
async def qa_page(request: Request):
    """Display Q&A page for asking questions to the knowledge base."""
    stats = kb.get_statistics()
    return templates.TemplateResponse("qa.html", {"request": request, "kb_stats": stats})


@app.post("/qa", response_class=HTMLResponse)
async def ask_question(
    request: Request,
    question: str = Form(...),
    retrieval_mode: str = Form("propositions"),
    max_results: int = Form(5),
):
    """Process a question and return an AI-generated answer."""
    if not question.strip():
        return templates.TemplateResponse(
            "qa.html",
            {
                "request": request,
                "error": "Please enter a question.",
                "kb_stats": kb.get_statistics(),
            },
        )

    logs = []

    try:
        if retrieval_mode == "quality_propositions":
            logs.append(f" Searching for relevant quality propositions (top {max_results})...")
            propositions = kb.search_propositions(question, top_k=max_results)
            propositions = [p for p in propositions if p.is_high_quality()]
            logs.append(f" Found {len(propositions)} quality propositions")
        else:
            logs.append(f" Searching for all relevant propositions (top {max_results})...")
            propositions = kb.search_propositions(question, top_k=max_results)
            logs.append(f" Found {len(propositions)} propositions")

        if not propositions:
            return templates.TemplateResponse(
                "qa.html",
                {
                    "request": request,
                    "question": question,
                    "retrieval_mode": retrieval_mode,
                    "max_results": max_results,
                    "error": "No relevant information found in the knowledge base for this question.",
                    "logs": logs,
                    "kb_stats": kb.get_statistics(),
                },
            )

        # Generate response
        logs.append(" Generating AI response...")
        response_data = response_generator.generate_response(question, propositions)
        logs.append(" Response generated successfully!")

        return templates.TemplateResponse(
            "qa.html",
            {
                "request": request,
                "question": question,
                "retrieval_mode": retrieval_mode,
                "max_results": max_results,
                "answer": response_data["answer"],
                "sources": response_data["sources"],
                "logs": logs,
                "kb_stats": kb.get_statistics(),
            },
        )

    except Exception as e:
        error_details = traceback.format_exc()
        logs.append(f" Error: {str(e)}")

        return templates.TemplateResponse(
            "qa.html",
            {
                "request": request,
                "question": question,
                "retrieval_mode": retrieval_mode,
                "error": f"Error processing question: {str(e)}",
                "error_details": error_details,
                "logs": logs,
                "kb_stats": kb.get_statistics(),
            },
        )


# ======================== VISUALIZATION ========================


@app.get("/visualize", response_class=HTMLResponse)
async def visualize_page(request: Request):
    """Display proposition timeline visualization page."""
    stats = kb.get_statistics()
    return templates.TemplateResponse("visualize.html", {"request": request, "kb_stats": stats})


@app.post("/visualize", response_class=HTMLResponse)
async def generate_visualization(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(500),
    quality_only: bool = Form(False),
):
    """Generate proposition timeline visualization."""
    if not query.strip():
        return templates.TemplateResponse(
            "visualize.html",
            {
                "request": request,
                "error": "Please enter a query.",
                "kb_stats": kb.get_statistics(),
            },
        )

    try:
        # Get propositions with similarity scores
        # Fetch more if filtering for quality to ensure we get enough results
        fetch_k = top_k * 2 if quality_only else top_k
        propositions_with_scores = kb.retrieval_system.query_propositions_with_scores(query, top_k=fetch_k)

        # Filter for quality propositions if requested
        if quality_only:
            propositions_with_scores = [(p, s) for p, s in propositions_with_scores if p.is_high_quality()][:top_k]

        if not propositions_with_scores:
            return templates.TemplateResponse(
                "visualize.html",
                {
                    "request": request,
                    "query": query,
                    "top_k": top_k,
                    "quality_only": quality_only,
                    "error": "No propositions found for this query.",
                    "kb_stats": kb.get_statistics(),
                },
            )

        # Generate visualization
        timeline_gen = TimelineGenerator()

        # Get total propositions by year for background bars
        total_props_by_year = timeline_gen.get_all_propositions_by_year(kb.papers)

        viz_data = timeline_gen.generate_visualization_data(
            query, propositions_with_scores, kb.papers, total_props_by_year
        )

        # Extract unique source papers
        unique_paper_ids = set()
        for prop, _ in propositions_with_scores:
            unique_paper_ids.add(prop.paper_id)

        source_papers = [kb.papers[paper_id] for paper_id in unique_paper_ids if paper_id in kb.papers]

        return templates.TemplateResponse(
            "visualize.html",
            {
                "request": request,
                "query": query,
                "top_k": top_k,
                "quality_only": quality_only,
                "timeline_html": viz_data["timeline_html"],
                "cumulative_html": viz_data["cumulative_html"],
                "heatmap_html": viz_data["heatmap_html"],
                "year_data": viz_data["year_data"],
                "summary": viz_data["summary"],
                "propositions": propositions_with_scores,
                "source_papers": source_papers,
                "papers": kb.papers,
                "kb_stats": kb.get_statistics(),
            },
        )

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)

        return templates.TemplateResponse(
            "visualize.html",
            {
                "request": request,
                "query": query,
                "top_k": top_k,
                "error": f"Error generating visualization: {str(e)}",
                "kb_stats": kb.get_statistics(),
            },
        )


# ======================== API ENDPOINTS (Optional) ========================


@app.get("/api/stats")
async def api_stats():
    """Get knowledge base statistics as JSON."""
    return kb.get_statistics()


@app.get("/api/papers")
async def api_papers():
    """Get all papers as JSON."""
    return [paper.to_dict() for paper in kb.papers.values()]


@app.get("/api/papers/{paper_id}")
async def api_paper_detail(paper_id: str):
    """Get paper details as JSON."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    return {
        "paper": paper.to_dict(),
        "stats": paper.get_statistics(),
        "propositions_count": len(paper.propositions),
        "quality_propositions_count": len(paper.get_quality_propositions()),
    }

@app.post("/api/papers/{paper_id}/extract-fulltext")
async def api_extract_fulltext(paper_id: str):
    """Extract propositions from full text of a paper."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    if not paper.full_text:
        raise HTTPException(status_code=400, detail="Paper has no full text available")

    if paper.extracted_from in ["full_text", "both"]:
        raise HTTPException(status_code=400, detail="Propositions already extracted from full text")

    try:
        # Create extractor and scorer
        extractor = PropositionExtractor()
        extractor.skip_evaluation = False
        scorer = PaperScorer()

        # Store existing propositions (from abstract)
        existing_prop_count = len(paper.propositions)

        # Extract from full text only (not abstract, since we already have those)
        # We'll clear existing and re-extract from full text
        paper.chunks = []
        paper.propositions = []
        extractor.extract_from_paper(paper, show_steps=True, use_full_text=True)

        # Score the paper
        paper.credibility = scorer.score_paper(paper)

        # Update in knowledge base
        kb.papers[paper_id] = paper

        # Save knowledge base with reextracted paper
        kb.add_paper(paper, verbose=False)
        kb.save()

        return {
            "success": True,
            "message": f"Extracted propositions from full text. Previous: {existing_prop_count}, Now: {len(paper.propositions)}",
            "propositions_count": len(paper.propositions),
            "quality_propositions_count": len(paper.get_quality_propositions()),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during extraction: {str(e)}")


@app.post("/api/papers/{paper_id}/regenerate-fulltext")
async def api_regenerate_fulltext(paper_id: str):
    """Regenerate all chunks and propositions from full text (without evaluation)."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    if not paper.full_text or paper.full_text == "":
        raise HTTPException(status_code=400, detail="Paper has no full text available")

    try:
        # Create extractor
        extractor = PropositionExtractor()
        extractor.skip_evaluation = True

        # Clear existing chunks and propositions
        paper.chunks = []
        paper.propositions = []

        print("Test")

        # Delete paper
        kb.delete_paper(paper_id, verbose=True)
        kb.save()

        print("Test")
        # Re-extract from full text without evaluation
        extractor.extract_from_paper(paper, show_steps=True, use_full_text=True)

        # Save knowledge base with reextracted paper
        kb.add_paper(paper, verbose=False)
        kb.save()

        return {
            "success": True,
            "message": "Successfully regenerated from full text",
            "chunks_count": len(paper.chunks),
            "propositions_count": len(paper.propositions),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during regeneration: {str(e)}")


@app.post("/api/papers/{paper_id}/regenerate-abstract")
async def api_regenerate_abstract(paper_id: str):
    """Regenerate all chunks and propositions from abstract only (without evaluation)."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    if not paper.abstract:
        raise HTTPException(status_code=400, detail="Paper has no abstract available")

    try:
        # Create extractor
        extractor = PropositionExtractor()
        extractor.skip_evaluation = True

        # Clear existing chunks and propositions
        paper.chunks = []
        paper.propositions = []

        # Delete paper
        kb.delete_paper(paper_id, verbose=True)
        kb.save()

        # Re-extract from abstract without evaluation
        extractor.extract_from_paper(paper, show_steps=True, use_full_text=False)

        # Add diagnostic logging to understand proposition count
        print(f"DEBUG: Abstract length: {len(paper.abstract)} chars")
        print(f"DEBUG: Total chunks generated: {len(paper.chunks)}")
        print(f"DEBUG: Total propositions generated: {len(paper.propositions)}")

        # Check quality filtering
        quality_props = paper.get_quality_propositions()
        print(f"DEBUG: Quality propositions: {len(quality_props)}")

        # Check evaluation status
        if paper.propositions:
            has_eval = any(prop.evaluation is not None for prop in paper.propositions)
            print(f"DEBUG: Propositions have evaluation: {has_eval}")

            # Print first few propositions
            for i, prop in enumerate(paper.propositions[:5]):
                eval_str = str(prop.evaluation.to_dict()) if prop.evaluation else "None"
                print(f"DEBUG Prop {i}: {prop.text[:80]}...")
                print(f"       Evaluation: {eval_str}")

        # Save knowledge base with reextracted paper
        kb.add_paper(paper, verbose=False)
        kb.save()

        return {
            "success": True,
            "message": "Successfully regenerated from abstract",
            "chunks_count": len(paper.chunks),
            "propositions_count": len(paper.propositions),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during regeneration: {str(e)}")


@app.post("/api/papers/{paper_id}/reevaluate")
async def api_reevaluate_propositions(paper_id: str):
    """Re-evaluate all existing propositions without re-extraction."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    if not paper.propositions:
        raise HTTPException(status_code=400, detail="Paper has no propositions to evaluate")

    try:
        # Create evaluator
        evaluator = PropositionEvaluator()

        # Re-evaluate all propositions
        print(f"\nRe-evaluating {len(paper.propositions)} propositions...")
        for prop in paper.propositions:
            chunk = kb.get_chunk(prop.chunk_id)
            if not chunk:
                print(f"  Warning: Chunk '{prop.chunk_id}' not found for proposition '{prop.prop_id}'")
                print(f"  Skipping evaluation for proposition: {prop.text[:50]}...")
                continue
            evaluation = evaluator.evaluate_proposition(prop.text, chunk.text)
            prop.evaluation = evaluation
            print(f"  Evaluated: {prop.text[:50]}... (Avg: {evaluation.average_score():.1f}/10)")

        # Update vectorstore metadata with new evaluation scores
        # This is necessary because propositions are loaded from the vectorstore on startup
        print("\nUpdating vectorstore metadata with evaluation scores...")
        updated_count = 0
        if kb.retrieval_system.proposition_vectorstore:
            for prop in paper.propositions:
                # Find the document in the vectorstore by prop_id
                for doc_id, doc in kb.retrieval_system.proposition_vectorstore.docstore._dict.items():
                    if doc.metadata.get("prop_id") == prop.prop_id:
                        # Update metadata with evaluation scores
                        if prop.evaluation:
                            doc.metadata.update(prop.evaluation.to_dict())
                            doc.metadata["passed_quality"] = prop.evaluation.passes_quality_check()
                            updated_count += 1
                        break
        print(f"  Updated {updated_count}/{len(paper.propositions)} propositions in vectorstore")

        # Save knowledge base (paper is already updated in kb.papers since we got it by reference)
        # NOTE: Do NOT call kb.add_paper() here - that would duplicate propositions in vectorstore!
        kb.save()

        quality_props = paper.get_quality_propositions()

        return {
            "success": True,
            "message": f"Successfully re-evaluated {len(paper.propositions)} propositions",
            "propositions_count": len(paper.propositions),
            "quality_propositions_count": len(quality_props),
            "credibility_score": paper.credibility.rating if paper.credibility else None,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during re-evaluation: {str(e)}")


@app.post("/api/papers/{paper_id}/score")
async def api_score_paper(paper_id: str):
    """Score a paper's credibility and extract metadata."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    try:
        # Score the paper
        scorer = PaperScorer()
        scores = scorer.score_paper(paper)

        # Update in knowledge base
        kb.papers[paper_id] = paper
        kb.save()

        return {
            "success": True,
            "message": "Paper scored successfully",
            "credibility": scores.to_dict(),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error scoring paper: {str(e)}")


@app.delete("/api/papers/{paper_id}")
async def api_delete_paper(paper_id: str):
    """Delete a paper and its associated chunks/propositions."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    try:
        # Delete paper and rebuild vector stores
        kb.delete_paper(paper_id, verbose=True)

        # Save knowledge base
        kb.save()

        return {
            "success": True,
            "message": f"Paper '{paper_id}' deleted successfully",
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during deletion: {str(e)}")


@app.post("/api/papers/{paper_id}/fetch-pmc-fulltext")
async def api_fetch_pmc_fulltext(paper_id: str):
    """Fetch full text from PMC if pmc_id exists and full text is not available."""
    paper = kb.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    # Check if paper has pmc_id
    if not paper.pmc_id:
        return {"success": False, "message": "Paper does not have a PMC ID"}

    # Check if paper already has full text
    if paper.full_text and len(paper.full_text) > 0:
        return {"success": False, "message": "Paper already has full text"}

    try:
        # Initialize PMC API
        pmc_api = PMCAPI()

        print(f"\nFetching full text from PMC for paper '{paper.title}'...")
        print(f"PMC ID: {paper.pmc_id}")

        # Fetch full text from PMC
        result = pmc_api.get_full_text(paper.pmc_id)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to fetch data from PMC")

        if not result.get("has_full_text", False):
            message = "Full text not available on PMC"
            if result.get("is_scanned", False):
                message = "Paper is available as scanned PDF only (no extractable text)"

            return {"success": False, "message": message, "pdf_url": result.get("pdf_url", "")}

        # Update paper with full text
        paper.full_text = result.get("full_text_sections", [])
        if result.get("pdf_url"):
            paper.pdf_url = result.get("pdf_url", "")

        print(f"Successfully fetched {len(paper.full_text)} sections from PMC")

        # Save to knowledge base
        kb.add_paper(paper, verbose=False)
        kb.save()

        return {
            "success": True,
            "message": "Successfully fetched full text from PMC",
            "sections_count": len(paper.full_text),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching full text from PMC: {str(e)}")


@app.post("/api/papers/upload")
async def api_upload_paper(file: UploadFile = File(...), extraction_method: str = Form("pymupdf")):
    """Upload and process a single paper file.

    Args:
        file: PDF, TXT, or MD file to upload
        extraction_method: PDF extraction method (pymupdf, marker, pdfplumber, pypdf)
    """
    try:
        # Validate file extension
        allowed_extensions = [".pdf", ".txt", ".md"]
        file_ext = os.path.splitext(str(file.filename or ""))[1].lower()
        if file_ext not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            )

        # Validate extraction method
        allowed_methods = ["pymupdf", "marker", "pdfplumber", "pypdf"]
        if extraction_method not in allowed_methods:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Invalid extraction method. Allowed: {', '.join(allowed_methods)}",
                },
            )

        # Store original filename for paper ID
        original_filename = file.filename

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Extract metadata and create Paper object with specified extraction method
            # For PDFs, let LocalPaperProcessor handle extraction to preserve page info
            local_paper_processor = LocalPaperProcessor(extraction_method=extraction_method)

            # Check if it's a PDF - if so, don't pre-load text to preserve pages
            if file_ext == ".pdf":
                paper = local_paper_processor.extract_from_file(tmp_file_path, None, original_filename)
            else:
                # For text files, we need to load the content first
                file_loader = FileLoader()
                documents = file_loader.load_file(tmp_file_path)
                full_text_content = "\n\n".join([doc.page_content for doc in documents])
                paper = local_paper_processor.extract_from_file(tmp_file_path, full_text_content, original_filename)

            # Check if paper already exists
            if kb.has_paper(paper.id):
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": f"Paper '{paper.title}' already exists in the knowledge base"},
                )

            # Extract propositions
            extractor = PropositionExtractor()
            extractor.skip_evaluation = True
            extractor.extract_from_paper(paper, show_steps=True, use_full_text=True)

            # Score paper credibility (extract metadata from first page/abstract)
            scorer = PaperScorer()
            scorer.score_paper(paper)
            if paper.credibility:
                print(f"   Scored paper: {paper.credibility.rating:.1f}/5 ({paper.credibility.study_type})")

            # Add to knowledge base
            kb.add_paper(paper, verbose=True)
            kb.save()

            return {
                "success": True,
                "message": f"Successfully uploaded and processed '{file.filename}'",
                "paper_id": paper.id,
                "paper_title": paper.title,
                "chunks_count": len(paper.chunks),
                "propositions_count": len(paper.propositions),
            }

        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": f"Error processing file: {str(e)}"})


@app.post("/api/papers/upload-multiple")
async def api_upload_multiple_papers(files: List[UploadFile] = File(...), extraction_method: str = Form("pymupdf")):
    """Upload and process multiple paper files.

    Args:
        files: List of PDF, TXT, or MD files to upload
        extraction_method: PDF extraction method (pymupdf, marker, pdfplumber, pypdf)
    """
    results: List[Dict] = []

    # Validate extraction method
    allowed_methods = ["pymupdf", "marker", "pdfplumber", "pypdf"]
    if extraction_method not in allowed_methods:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Invalid extraction method. Allowed: {', '.join(allowed_methods)}"},
        )

    for file in files:
        try:
            # Validate file extension
            allowed_extensions = [".pdf", ".txt", ".md"]
            file_ext = os.path.splitext(file.filename or "")[1].lower()
            if file_ext not in allowed_extensions:
                results.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
                    }
                )
                continue

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Extract metadata and create Paper object with specified extraction method
                # For PDFs, let LocalPaperProcessor handle extraction to preserve page info
                local_paper_processor = LocalPaperProcessor(extraction_method=extraction_method)

                # Check if it's a PDF - if so, don't pre-load text to preserve pages
                if file_ext == ".pdf":
                    paper = local_paper_processor.extract_from_file(tmp_file_path, None, file.filename)
                else:
                    # For text files, we need to load the content first
                    file_loader = FileLoader()
                    documents = file_loader.load_file(tmp_file_path)
                    full_text_content = "\n\n".join([doc.page_content for doc in documents])
                    paper = local_paper_processor.extract_from_file(tmp_file_path, full_text_content, file.filename)

                # Check if paper already exists
                if kb.has_paper(paper.id):
                    results.append(
                        {"filename": file.filename, "success": False, "error": "Paper already exists in knowledge base"}
                    )
                    continue

                # Extract propositions
                extractor = PropositionExtractor()
                extractor.skip_evaluation = True
                extractor.extract_from_paper(paper, show_steps=True, use_full_text=True)

                # Score paper credibility
                scorer = PaperScorer()
                scorer.score_paper(paper)

                # Add to knowledge base
                kb.add_paper(paper, verbose=True)

                results.append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "paper_id": paper.id,
                        "paper_title": paper.title,
                        "chunks_count": len(paper.chunks),
                        "propositions_count": len(paper.propositions),
                    }
                )

            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        except Exception as e:
            traceback.print_exc()
            results.append({"filename": file.filename, "success": False, "error": str(e)})

    # Save knowledge base once after all uploads
    try:
        kb.save()
    except Exception as e:
        print(f"Error saving knowledge base: {e}")

    success_count = sum(1 for r in results if r["success"])

    return {"total": len(files), "successful": success_count, "failed": len(files) - success_count, "results": results}


@app.post("/api/papers/upload-multiple-no-extraction")
async def api_upload_multiple_papers_no_extraction(
    files: List[UploadFile] = File(...), extraction_method: str = Form("pymupdf")
):
    """Upload multiple paper files without extracting propositions/claims.

    Args:
        files: List of PDF, TXT, or MD files to upload
        extraction_method: PDF extraction method (pymupdf, marker, pdfplumber, pypdf)
    """
    results: List[Dict] = []

    # Validate extraction method
    allowed_methods = ["pymupdf", "marker", "pdfplumber", "pypdf"]
    if extraction_method not in allowed_methods:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Invalid extraction method. Allowed: {', '.join(allowed_methods)}"},
        )

    for file in files:
        try:
            # Validate file extension
            allowed_extensions = [".pdf", ".txt", ".md"]
            file_ext = os.path.splitext(file.filename or "")[1].lower()
            if file_ext not in allowed_extensions:
                results.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
                    }
                )
                continue

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Extract metadata and create Paper object with specified extraction method
                # For PDFs, let LocalPaperProcessor handle extraction to preserve page info
                local_paper_processor = LocalPaperProcessor(extraction_method=extraction_method)

                # Check if it's a PDF - if so, don't pre-load text to preserve pages
                if file_ext == ".pdf":
                    paper = local_paper_processor.extract_from_file(tmp_file_path, None, file.filename)
                else:
                    # For text files, we need to load the content first
                    file_loader = FileLoader()
                    documents = file_loader.load_file(tmp_file_path)
                    full_text_content = "\n\n".join([doc.page_content for doc in documents])
                    paper = local_paper_processor.extract_from_file(tmp_file_path, full_text_content, file.filename)

                # Check if paper already exists
                if kb.has_paper(paper.id):
                    results.append(
                        {"filename": file.filename, "success": False, "error": "Paper already exists in knowledge base"}
                    )
                    continue

                # Score paper (even without proposition extraction)
                scorer = PaperScorer()
                scorer.score_paper(paper)

                # Skip proposition extraction - just add paper to knowledge base
                kb.add_paper(paper, verbose=True)

                results.append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "paper_id": paper.id,
                        "paper_title": paper.title,
                        "chunks_count": 0,
                        "propositions_count": 0,
                    }
                )

            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        except Exception as e:
            traceback.print_exc()
            results.append({"filename": file.filename, "success": False, "error": str(e)})

    # Save knowledge base once after all uploads
    try:
        kb.save()
    except Exception as e:
        print(f"Error saving knowledge base: {e}")

    success_count = sum(1 for r in results if r["success"])

    return {"total": len(files), "successful": success_count, "failed": len(files) - success_count, "results": results}


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" Starting Scientific Claim Verification Web Application")
    print("=" * 70)
    print(f"\n Knowledge Base: {len(kb.papers)} papers loaded")
    print(" Server: http://localhost:8000")
    print(" Documentation: http://localhost:8000/docs")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

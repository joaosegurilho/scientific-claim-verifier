"""Simplified agent verification endpoint - easier to debug and understand."""

import asyncio
import json
import traceback
from typing import AsyncGenerator, Dict, Any


async def stream_agent_verification(
    agent,
    claim: str,
    kb,
) -> AsyncGenerator[str, None]:
    """
    Simple, straightforward agent streaming.

    This is a simplified version that's easy to debug.
    Returns SSE-formatted strings ready to yield.
    """
    print("\n" + "="*80, flush=True)
    print(f"[STREAM START] Claim: {claim[:80]}...", flush=True)
    print("="*80 + "\n", flush=True)

    # Step 1: Initialize tracking variables
    final_agent_message = None
    event_count = 0
    tool_call_count = 0

    try:
        # Step 2: Get the agent stream
        print("[STEP 1] Creating agent stream...", flush=True)
        stream = agent.verify_claim_stream(claim, debug=False)
        print("[STEP 1] Agent stream created successfully", flush=True)

        # Step 3: Process each event from the stream
        print("[STEP 2] Starting to process events...", flush=True)
        async for event in stream:
            event_count += 1
            event_type = event.get('type', 'unknown')

            # Log the event
            if event_type == 'tool_call':
                tool_call_count += 1
                tool_name = event.get('tool', 'unknown')
                print(f"[EVENT {event_count}] Tool call #{tool_call_count}: {tool_name}", flush=True)

            elif event_type == 'tool_result':
                result_preview = str(event.get('content', ''))[:80]
                print(f"[EVENT {event_count}] Tool result: {result_preview}...", flush=True)

            elif event_type == 'agent_message':
                final_agent_message = event.get('content')
                msg_preview = final_agent_message[:100] if final_agent_message else 'None'
                print(f"[EVENT {event_count}] Agent message: {msg_preview}...", flush=True)

            else:
                print(f"[EVENT {event_count}] Type: {event_type}", flush=True)

            # Send event to client
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.01)

        print(f"\n[STEP 3] Stream completed. Total events: {event_count}", flush=True)

        # Step 4: Check if we got a final message
        if not final_agent_message:
            print("[ERROR] No final agent message received!", flush=True)
            error_data = {
                'type': 'error',
                'error': 'Agent did not provide a final response',
                'details': f'Processed {event_count} events, {tool_call_count} tool calls'
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        # Step 5: Parse the final message
        print("[STEP 4] Parsing agent response...", flush=True)
        verdict, confidence, reasoning, evidence_ids = agent._parse_agent_response(final_agent_message)
        print(f"[STEP 4] Parsed: verdict={verdict}, confidence={confidence}", flush=True)

        # Step 6: Build the completion response
        print("[STEP 5] Building completion response...", flush=True)

        # Get tracked IDs
        seen_proposition_ids = set(getattr(agent, '_last_seen_proposition_ids', []))
        seen_paper_ids = set(getattr(agent, '_last_seen_paper_ids', []))
        seen_chunk_ids = set(getattr(agent, '_last_seen_chunk_ids', []))

        print(f"[STEP 5] Tracked: {len(seen_proposition_ids)} props, {len(seen_paper_ids)} papers, {len(seen_chunk_ids)} chunks", flush=True)

        # Collect evidence
        evidence = _collect_evidence(kb, seen_proposition_ids, evidence_ids)
        evidence_chunks = _collect_chunks(kb, seen_chunk_ids)
        evidence_papers = _collect_papers(kb, seen_paper_ids, evidence_ids)

        print(f"[STEP 5] Collected: {len(evidence)} evidence props, {len(evidence_papers)} papers", flush=True)

        # Format for JSON
        evidence_list = _format_evidence(evidence, kb)
        chunks_list = _format_chunks(evidence_chunks, kb)

        # Get confidence interpretation
        from scverifier.core.verification.confidence_interpreter import (
            get_confidence_interpretation,
            get_confidence_level
        )
        confidence_interpretation = get_confidence_interpretation(verdict, int(round(confidence)))
        confidence_level = get_confidence_level(int(round(confidence)))

        # Build final result
        result_data = {
            'type': 'complete',
            'result': {
                'verdict': verdict,
                'confidence': confidence,
                'reasoning': reasoning,
                'evidence': evidence_list,
                'evidence_ids': evidence_ids
            },
            'evidence_papers': evidence_papers,
            'evidence_chunks': chunks_list,
            'confidence_interpretation': confidence_interpretation,
            'confidence_level': confidence_level
        }

        print(f"[SUCCESS] Verification complete: {verdict} ({confidence:.1f}/10)", flush=True)
        print("="*80 + "\n", flush=True)

        yield f"data: {json.dumps(result_data)}\n\n"

    except Exception as e:
        print(f"\n[EXCEPTION] Error during streaming: {str(e)}", flush=True)
        print(f"[EXCEPTION] Traceback:\n{traceback.format_exc()}", flush=True)

        error_data = {
            'type': 'error',
            'error': str(e),
            'details': traceback.format_exc()
        }
        yield f"data: {json.dumps(error_data)}\n\n"


def _collect_evidence(kb, seen_proposition_ids, evidence_ids):
    """Collect evidence propositions."""
    evidence = []
    collected_prop_ids = set()

    # First, get directly seen propositions
    for prop_id in seen_proposition_ids:
        if prop_id not in collected_prop_ids:
            prop = kb.get_proposition(prop_id)
            if prop:
                evidence.append(prop)
                collected_prop_ids.add(prop_id)

    # Then add quality propositions from cited papers
    for paper_id in evidence_ids:
        paper = kb.get_paper(paper_id)
        if paper:
            quality_props = paper.get_quality_propositions()
            for prop in quality_props[:5]:
                if prop.prop_id not in collected_prop_ids:
                    evidence.append(prop)
                    collected_prop_ids.add(prop.prop_id)

    return evidence


def _collect_chunks(kb, seen_chunk_ids):
    """Collect evidence chunks."""
    chunks = []
    for chunk_id in seen_chunk_ids:
        chunk = kb.get_chunk(chunk_id)
        if chunk:
            chunks.append(chunk)
    return chunks


def _collect_papers(kb, seen_paper_ids, evidence_ids):
    """Collect paper details."""
    papers = []
    seen_ids = set()

    all_paper_ids = seen_paper_ids.union(set(evidence_ids))
    for paper_id in all_paper_ids:
        if paper_id in seen_ids:
            continue

        paper = kb.get_paper(paper_id)
        if paper:
            papers.append({
                "id": paper.id,
                "title": paper.title,
                "doi": paper.doi,
                "year": paper.year,
                "citations": paper.citations,
                "url": paper.url,
                "credibility": paper.credibility.to_dict() if paper.credibility else None,
            })
            seen_ids.add(paper_id)

    return papers


def _format_evidence(evidence, kb):
    """Format evidence propositions for JSON."""
    evidence_list = []
    for p in evidence:
        paper = kb.get_paper(p.paper_id)
        section = ""
        if paper:
            for chunk in paper.chunks:
                if chunk.chunk_id == p.chunk_id:
                    section = chunk.section
                    break

        evidence_list.append({
            'prop_id': p.prop_id,
            'text': p.text,
            'source': paper.title if paper else 'Unknown',
            'paper_id': p.paper_id,
            'chunk_id': p.chunk_id,
            'page': p.page,
            'section': section,
            'evaluation': p.evaluation.to_dict() if p.evaluation else None
        })

    return evidence_list


def _format_chunks(chunks, kb):
    """Format chunks for JSON."""
    chunks_list = []
    for chunk in chunks:
        paper = kb.get_paper(chunk.paper_id)
        chunks_list.append({
            'chunk_id': chunk.chunk_id,
            'text': chunk.text,
            'source': paper.title if paper else 'Unknown',
            'paper_id': chunk.paper_id,
            'section': chunk.section,
            'page': chunk.page
        })

    return chunks_list

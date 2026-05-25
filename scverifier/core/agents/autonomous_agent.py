"""Autonomous agent for claim verification using pattern.

This agent autonomously decides which tools to use and how to gather evidence
to verify a scientific claim. It uses LangGraph's pattern to:
1. Think about what information it needs
2. Use tools to gather evidence
3. Reason about the evidence
4. Make a final verification decision
"""

import re
from typing import Optional, AsyncGenerator, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage

from scverifier.config.settings import Config
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.knowledge.literature_search import LiteratureSearch
from scverifier.core.agents import tools
from scverifier.data.models import VerificationResult, Proposition
from scverifier.core.verification.confidence_interpreter import (
    get_confidence_interpretation,
    CONFIDENCE_INTERPRETATIONS,
)


class AutonomousClaimAgent:
    """Autonomous agent that decides how to verify claims using available tools.

    This agent uses an autonomous pattern:
    - Reasons about what information is needed
    - Acts by calling appropriate tools
    - Continues until it has enough evidence to make a decision
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        lit_search: Optional[LiteratureSearch] = None,
        allow_online_search: bool = True,
        quality_only: bool = False,
    ):
        """Initialize the autonomous agent.

        Args:
            kb: Knowledge base to search
            lit_search: Literature search for online papers (optional)
            allow_online_search: Whether to allow searching online databases
            quality_only: Whether to search only quality propositions
        """
        Config.setup_environment()

        self.kb = kb
        self.lit_search = lit_search or LiteratureSearch()
        self.allow_online_search = allow_online_search
        self.quality_only = quality_only

        # Initialize tools
        tools.set_knowledge_base(kb)
        tools.set_literature_search(self.lit_search)
        tools.set_quality_filter(quality_only)

        # Select tools based on configuration
        if allow_online_search:
            self.tools = tools.get_all_tools()
        else:
            self.tools = tools.get_kb_only_tools()

        # Initialize LLM with higher max_output_tokens to prevent truncation
        self.llm = ChatGoogleGenerativeAI(
            model=Config.AGENT_MODEL,
            temperature=Config.AGENT_TEMPERATURE,
            timeout=Config.LLM_TIMEOUT,
            max_output_tokens=Config.AGENT_MAX_OUTPUT_TOKENS,  # Ensure agent has enough tokens for final response
        )

        # System prompt for the agent (create before agent initialization)
        self.system_prompt = self._create_system_prompt()

        # Create autonomous agent with memory
        self.memory = MemorySaver()
        self.agent: Any = create_agent(
            model=self.llm, tools=self.tools, checkpointer=self.memory, system_prompt=self.system_prompt
        )

        # Set default recursion limit for agent execution
        self.recursion_limit = Config.RECURSION_LIMIT  # number of reasoning steps

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        tools_available = "search_similar_propositions, search_similar_chunks, search_propositions_in_paper, find_similar_papers, get_paper_details, get_kb_statistics, get_proposition_source_chunk"
        if self.allow_online_search:
            tools_available += ", search_online_papers (slow, use only when KB is insufficient)"

        # Format confidence scale from CONFIDENCE_INTERPRETATIONS
        confidence_scale = self._format_confidence_scale()

        return f"""You are a scientific claim verification agent. Your task is to verify whether a scientific claim is supported, refuted, or has insufficient evidence.

AVAILABLE TOOLS:
{tools_available}

YOUR APPROACH:
You have autonomy in how you investigate claims. Adapt your strategy based on what you find. However, keep these principles in mind:

- Start by understanding what's available (get_kb_statistics gives you an overview)
- Search broadly first, then dive deeper into promising papers
- Actively look for BOTH supporting AND contradicting evidence - don't stop at the first match
- When you find relevant papers, explore them thoroughly and check related research
- Consider the quality and credibility of sources (study type, sample size, methodology)
- Use different search angles and query formulations to find diverse perspectives
- Only reach a verdict after gathering substantial evidence from multiple sources (aim for 5-10)

CRITICAL: Base your verdict solely on the evidence you find, not on external knowledge you might have. If the evidence contradicts the claim, say so - even if you "know" otherwise.

CONFIDENCE SCALE (1-10):
Use this scale to assign your confidence score based on the verdict and evidence quality:

{confidence_scale}

OUTPUT FORMAT:
After gathering sufficient evidence to confidently verify the claim, provide your final answer in this format.
Start by writing "#RESULT:":

#RESULT:
VERDICT: [SUPPORTS/REFUTES/INSUFFICIENT_EVIDENCE]
CONFIDENCE: [1-10]
REASONING: [Detailed reasoning citing specific evidence with paper IDs and credibility ratings]
EVIDENCE_IDS: [Comma-separated list of paper IDs ONLY - e.g., "12345678, 87654321, 41300993" - NO explanatory text here]

IMPORTANT: The EVIDENCE_IDS field must contain ONLY paper IDs separated by commas. Any explanations about evidence quality or limitations should go in the REASONING section above, not in EVIDENCE_IDS.

Now verify the following claim:
"""

    def _format_confidence_scale(self) -> str:
        """Format the confidence scale for the system prompt.

        Returns:
            Formatted string describing the confidence scale for each verdict type
        """
        scale_text = []

        for verdict in ["SUPPORTS", "REFUTES", "INSUFFICIENT_EVIDENCE"]:
            scale_text.append(f"\n{verdict}:")
            interpretations = CONFIDENCE_INTERPRETATIONS[verdict]

            # Sort by confidence range (descending)
            sorted_ranges = sorted(interpretations.items(), key=lambda x: x[0][1], reverse=True)

            for (min_conf, max_conf), description in sorted_ranges:
                scale_text.append(f"  {min_conf}-{max_conf}: {description}")

        return "\n".join(scale_text)

    async def verify_claim_stream(
        self, claim: str, thread_id: str = "default", debug: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Verify a claim with streaming output for real-time updates.

        Args:
            claim: The scientific claim to verify
            thread_id: Thread ID for conversation memory
            debug: Whether to output debug messages

        Yields:
            Dict with event type and data for streaming to UI
        """
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

        # Track all entities seen during search
        seen_proposition_ids = set()
        seen_paper_ids = set()
        seen_chunk_ids = set()
        final_message_received = False

        # Create initial message (system prompt is now in agent configuration)
        messages = [{"role": "user", "content": claim}]

        try:
            # Stream agent execution
            async for event in self.agent.astream({"messages": messages}, config=config, stream_mode="values"):
                # Extract the last message
                if "messages" in event:
                    last_message = event["messages"][-1]
                    message_type = type(last_message).__name__

                    if debug:
                        print(f"[DEBUG] Stream event - message type: {message_type}", flush=True)

                    # Check for tool calls first
                    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
                    has_content = hasattr(last_message, "content") and last_message.content

                    if debug:
                        print(f"[DEBUG] has_tool_calls: {has_tool_calls}, has_content: {has_content}", flush=True)

                    # Determine event type
                    if has_tool_calls:
                        # Agent is calling a tool
                        for tool_call in last_message.tool_calls:
                            yield {
                                "type": "tool_call",
                                "tool": tool_call["name"],
                                "args": tool_call["args"],
                                "timestamp": None,
                            }
                    elif isinstance(last_message, ToolMessage) and last_message.content:
                        # Tool result - extract IDs from the result
                        content_str = str(last_message.content)

                        # Extract proposition IDs, paper IDs, and chunk IDs from tool results
                        prop_ids, paper_ids, chunk_ids = self._extract_ids_from_tool_result(content_str)
                        seen_proposition_ids.update(prop_ids)
                        seen_paper_ids.update(paper_ids)
                        seen_chunk_ids.update(chunk_ids)

                        yield {"type": "tool_result", "content": content_str, "timestamp": None}
                    elif isinstance(last_message, AIMessage) and last_message.content:
                        # Agent is thinking/responding
                        content = last_message.content

                        if debug:
                            print(f"[DEBUG] AIMessage content type: {type(content)}", flush=True)

                        # Handle both string and list-of-dicts content formats
                        if isinstance(content, str):
                            text_content = content
                        elif isinstance(content, list) and len(content) > 0:
                            # Content is a list of dicts like [{'type': 'text', 'text': '...'}]
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    text_parts.append(item["text"])
                            text_content = "\n".join(text_parts) if text_parts else ""
                        else:
                            text_content = str(content)

                        # Remove #RESULT: tag if present
                        if text_content.startswith("#RESULT:"):
                            text_content = text_content[8:].strip()

                        if text_content:
                            # Check if this looks like a final verdict message
                            if "VERDICT:" in text_content and "REASONING:" in text_content:
                                final_message_received = True

                            if debug:
                                print(f"[DEBUG] Yielding agent message, preview: {text_content[:100]}...", flush=True)
                            yield {"type": "agent_message", "content": text_content, "timestamp": None}
                    else:
                        if debug:
                            print(
                                f"[DEBUG] Message not yielded - type: {message_type}, isinstance AIMessage: {isinstance(last_message, AIMessage)}",
                                flush=True,
                            )

            # Store tracked IDs for later retrieval
            self._last_seen_proposition_ids = list(seen_proposition_ids)
            self._last_seen_paper_ids = list(seen_paper_ids)
            self._last_seen_chunk_ids = list(seen_chunk_ids)

            # If no final message was received, try to force one
            if not final_message_received:
                if debug:
                    print("[DEBUG] No final message received, attempting to force verdict...", flush=True)

                force_decision_message = """Based on all the evidence you've gathered, provide your final verdict NOW in the required format:

#RESULT:
VERDICT: [SUPPORTS/REFUTES/INSUFFICIENT_EVIDENCE]
CONFIDENCE: [1-10]
REASONING: [Your reasoning based on the evidence you've gathered]
EVIDENCE_IDS: [Comma-separated list of paper IDs you examined]"""

                messages.append({"role": "user", "content": force_decision_message})

                try:
                    force_config = {"configurable": {"thread_id": thread_id + "_force"}, "recursion_limit": 5}

                    async for event in self.agent.astream(
                        {"messages": messages}, config=force_config, stream_mode="values"
                    ):
                        if "messages" in event:
                            last_message = event["messages"][-1]

                            if isinstance(last_message, AIMessage) and last_message.content:
                                content = last_message.content
                                if isinstance(content, str):
                                    text_content = content
                                elif isinstance(content, list) and len(content) > 0:
                                    text_parts = [
                                        item["text"] for item in content if isinstance(item, dict) and "text" in item
                                    ]
                                    text_content = "\n".join(text_parts) if text_parts else ""
                                else:
                                    text_content = str(content)

                                if text_content.startswith("#RESULT:"):
                                    text_content = text_content[8:].strip()

                                if text_content and ("VERDICT:" in text_content or "REASONING:" in text_content):
                                    if debug:
                                        print("[DEBUG] Forced verdict received", flush=True)
                                    yield {"type": "agent_message", "content": text_content, "timestamp": None}
                                    break

                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Failed to force verdict: {e}", flush=True)

        except Exception as e:
            # Check if this is a recursion limit error
            error_msg = str(e)
            if "recursion limit" in error_msg.lower():
                # Force the agent to provide a final answer based on what it gathered
                # Create a prompt that asks the agent to make a decision NOW
                force_decision_message = """You have reached the maximum number of reasoning steps. Based on all the evidence you've gathered so far, you MUST provide your final verdict NOW.

Review what you've learned and provide your answer in the required format:

#RESULT:
VERDICT: [SUPPORTS/REFUTES/INSUFFICIENT_EVIDENCE]
CONFIDENCE: [1-10]
REASONING: [Your reasoning based on the evidence you've gathered]
EVIDENCE_IDS: [Comma-separated list of paper IDs you examined]

You MUST respond with a verdict. Do not ask for more information or more time."""

                # Add the force decision message to the conversation
                messages.append({"role": "user", "content": force_decision_message})

                # Try to get a final response with reduced recursion limit
                try:
                    force_config = {
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 5,  # Very short limit, just enough for final answer
                    }

                    async for event in self.agent.astream(
                        {"messages": messages}, config=force_config, stream_mode="values"
                    ):
                        if "messages" in event:
                            last_message = event["messages"][-1]

                            # Only capture the final AI response
                            if isinstance(last_message, AIMessage) and last_message.content:
                                content = last_message.content
                                if isinstance(content, str):
                                    text_content = content
                                elif isinstance(content, list) and len(content) > 0:
                                    text_parts = [
                                        item["text"] for item in content if isinstance(item, dict) and "text" in item
                                    ]
                                    text_content = "\n".join(text_parts) if text_parts else ""
                                else:
                                    text_content = str(content)

                                if text_content.startswith("#RESULT:"):
                                    text_content = text_content[8:].strip()

                                if text_content and ("VERDICT:" in text_content or "REASONING:" in text_content):
                                    yield {"type": "agent_message", "content": text_content, "timestamp": None}
                                    break

                except Exception:
                    # If forcing a decision also fails, provide a basic fallback
                    yield {
                        "type": "agent_message",
                        "content": f"VERDICT: INSUFFICIENT_EVIDENCE\nCONFIDENCE: 3\nREASONING: The agent could not complete the verification within the step limit and was unable to formulate a final verdict.\n\nEVIDENCE_IDS: {', '.join(list(seen_paper_ids)[:10])}",
                        "timestamp": None,
                    }

                # Store tracked IDs
                self._last_seen_proposition_ids = list(seen_proposition_ids)
                self._last_seen_paper_ids = list(seen_paper_ids)
                self._last_seen_chunk_ids = list(seen_chunk_ids)
            else:
                # For other errors, propagate them
                yield {"type": "error", "error": str(e), "timestamp": None}

    def verify_claim(self, claim: str, thread_id: str = "default") -> VerificationResult:
        """Verify a claim (synchronous version).

        Args:
            claim: The scientific claim to verify
            thread_id: Thread ID for conversation memory

        Returns:
            VerificationResult with verdict, confidence, and reasoning
        """
        from langgraph.errors import GraphRecursionError

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

        # Create initial message (system prompt is now in agent configuration)
        messages = [{"role": "user", "content": claim}]

        # Track IDs from tool messages
        seen_proposition_ids: set[str] = set()
        seen_paper_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()

        # Track token usage
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            # Run agent
            print(f"  Starting agent verification...")
            result = self.agent.invoke({"messages": messages}, config=config)

            # Extract final response
            final_response = result["messages"][-1].content
            print(f"  Agent completed. Parsing response...")

            # Extract IDs and token usage from tool messages in conversation history
            for msg in result["messages"]:
                # Collect token usage from all messages
                if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                    total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
                    total_output_tokens += msg.usage_metadata.get("output_tokens", 0)

                # Extract IDs from tool messages
                if isinstance(msg, ToolMessage) and msg.content:
                    prop_ids, paper_ids, chunk_ids = self._extract_ids_from_tool_result(str(msg.content))
                    seen_proposition_ids.update(prop_ids)
                    seen_paper_ids.update(paper_ids)
                    seen_chunk_ids.update(chunk_ids)

        except GraphRecursionError:
            # Agent hit recursion limit - force a final verdict
            print(f"  Agent hit recursion limit ({self.recursion_limit} steps). Forcing verdict...")

            force_decision_message = """You have reached the maximum number of reasoning steps. Based on all the evidence you've gathered so far, you MUST provide your final verdict NOW.

Review what you've learned and provide your answer in the required format:

#RESULT:
VERDICT: [SUPPORTS/REFUTES/INSUFFICIENT_EVIDENCE]
CONFIDENCE: [1-10]
REASONING: [Your reasoning based on the evidence you've gathered]
EVIDENCE_IDS: [Comma-separated list of paper IDs you examined]

You MUST respond with a verdict. Do not ask for more information or more time."""

            messages.append({"role": "user", "content": force_decision_message})

            try:
                force_config = {
                    "configurable": {"thread_id": thread_id + "_force"},
                    "recursion_limit": 5,  # Very short limit, just enough for final answer
                }
                result = self.agent.invoke({"messages": messages}, config=force_config)
                final_response = result["messages"][-1].content
                print(f"  Forced verdict received.")

                # Extract IDs from forced conversation
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage) and msg.content:
                        prop_ids, paper_ids, chunk_ids = self._extract_ids_from_tool_result(str(msg.content))
                        seen_proposition_ids.update(prop_ids)
                        seen_paper_ids.update(paper_ids)
                        seen_chunk_ids.update(chunk_ids)

            except Exception as e:
                # If forcing also fails, return a basic fallback
                print(f"  Failed to force verdict: {e}")
                return VerificationResult(
                    claim=claim,
                    verdict="INSUFFICIENT_EVIDENCE",
                    confidence=3.0,
                    reasoning=f"The agent could not complete the verification within the step limit ({self.recursion_limit} steps) and was unable to formulate a final verdict.",
                    evidence=[],
                )

        # Parse the response
        verdict, confidence, reasoning, evidence_ids = self._parse_agent_response(final_response)

        # Collect evidence propositions from all seen propositions
        evidence = self._collect_evidence_from_seen(list(seen_proposition_ids), list(seen_paper_ids), evidence_ids)

        # Create token usage dict
        token_usage = None
        if total_input_tokens > 0 or total_output_tokens > 0:
            token_usage = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            }

        return VerificationResult(
            claim=claim, verdict=verdict, confidence=confidence, reasoning=reasoning, evidence=evidence,
            token_usage=token_usage
        )

    def _normalize_content(self, content) -> str:
        """Normalize message content to a string.

        Gemini models can return content as either a string or a list of dicts.
        This method handles both formats.

        Args:
            content: The message content (string or list)

        Returns:
            Normalized string content
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Content is a list of dicts like [{'type': 'text', 'text': '...'}]
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(text_parts) if text_parts else ""
        else:
            return str(content) if content else ""

    def _parse_agent_response(self, response) -> tuple[str, float, str, list[str]]:
        """Parse the agent's final response.

        Args:
            response: The agent's response text (string or list)

        Returns:
            Tuple of (verdict, confidence, reasoning, evidence_ids)
        """
        # Normalize content to string (handles both string and list formats)
        response_text = self._normalize_content(response)

        # Default values
        verdict = "INSUFFICIENT_EVIDENCE"
        confidence = 5.0
        reasoning = response_text
        evidence_ids = []
        evidence_reasoning = ""

        # Parse structured output
        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("VERDICT:"):
                verdict_text = line.replace("VERDICT:", "").strip()
                if "SUPPORTS" in verdict_text.upper():
                    verdict = "SUPPORTS"
                elif "REFUTES" in verdict_text.upper():
                    verdict = "REFUTES"
                else:
                    verdict = "INSUFFICIENT_EVIDENCE"

            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    # Extract first number found
                    match = re.search(r"\d+", conf_text)
                    if match:
                        confidence = float(match.group())
                except Exception:
                    confidence = 5.0

            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
                # Also collect any following lines until we hit EVIDENCE_IDS
                idx = lines.index(line)
                for i in range(idx + 1, len(lines)):
                    if lines[i].startswith("EVIDENCE_IDS:"):
                        break
                    reasoning += "\n" + lines[i]

            elif line.startswith("EVIDENCE_IDS:"):
                ids_text = line.replace("EVIDENCE_IDS:", "").strip()

                # Store the full text for potential use as evidence reasoning
                evidence_reasoning = ids_text

                # Extract valid paper IDs from the text
                # Paper IDs are typically numeric (e.g., "41300993") or alphanumeric with specific patterns
                # Look for patterns like "Paper ID: 12345" or standalone IDs
                potential_ids = []

                # First, try to find "Paper ID: XXXXXX" patterns
                id_matches = re.findall(r"Paper ID[:\s]+(\w+)", ids_text, re.IGNORECASE)
                potential_ids.extend(id_matches)

                # Also try to find standalone numeric IDs (7+ digits, which is typical for paper IDs)
                numeric_ids = re.findall(r"\b(\d{7,})\b", ids_text)
                potential_ids.extend(numeric_ids)

                # Also split by commas and check if each part looks like a paper ID
                # (all digits or short alphanumeric)
                comma_parts = [part.strip() for part in ids_text.split(",")]
                for part in comma_parts:
                    # Remove brackets if present
                    part = part.strip("[]() ")
                    # Check if it looks like a simple paper ID (all digits or simple alphanumeric)
                    if part and (
                        part.isdigit() or (len(part) < 20 and part.replace("-", "").replace("_", "").isalnum())
                    ):
                        potential_ids.append(part)

                # Deduplicate while preserving order
                seen = set()
                for pid in potential_ids:
                    if pid not in seen and pid:
                        evidence_ids.append(pid)
                        seen.add(pid)

        # If we have evidence reasoning text that's more than just IDs,
        # append it to the main reasoning
        if evidence_reasoning and len(evidence_reasoning) > 50:
            # Clean up the evidence reasoning text
            cleaned = evidence_reasoning.strip("[] ")

            # Check if the evidence_reasoning is just a list of IDs (short and mostly numeric)
            # vs. a long explanation (which we want to preserve)
            is_just_ids = all(
                part.strip().replace(",", "").replace(" ", "").replace("-", "").replace("_", "").isalnum()
                and len(part.strip()) < 30
                for part in cleaned.split(",")
            )

            # Only append if it contains substantial explanatory text and isn't already in reasoning
            if not is_just_ids and cleaned not in reasoning:
                reasoning += f"\n\n**Evidence Note:** {cleaned}"

        return verdict, confidence, reasoning, evidence_ids

    def _extract_ids_from_tool_result(self, tool_result: str) -> tuple[set[str], set[str], set[str]]:
        """Extract proposition IDs, paper IDs, and chunk IDs from a tool result.

        Args:
            tool_result: The string content from a tool result

        Returns:
            Tuple of (proposition_ids, paper_ids, chunk_ids) as sets
        """
        prop_ids: set[str] = set()
        paper_ids: set[str] = set()
        chunk_ids: set[str] = set()

        # Extract Paper IDs (format: "Paper ID: XXXXX")
        paper_id_matches = re.findall(r"Paper ID[:\s]+(\S+)", tool_result, re.IGNORECASE)
        paper_ids.update(paper_id_matches)

        # Extract Proposition IDs (format: "Proposition ID: XXXXX")
        prop_id_matches = re.findall(r"Proposition ID[:\s]+(\S+)", tool_result, re.IGNORECASE)
        prop_ids.update(prop_id_matches)

        # Extract Chunk IDs (format: "Chunk ID: XXXXX")
        # This will get the explicit "Chunk ID: chunk_abc123" lines
        chunk_id_matches = re.findall(r"Chunk ID[:\s]+(chunk_[a-z0-9]+)", tool_result, re.IGNORECASE)
        chunk_ids.update(chunk_id_matches)

        return prop_ids, paper_ids, chunk_ids

    def _collect_evidence_from_seen(
        self, seen_proposition_ids: list[str], seen_paper_ids: list[str], evidence_ids: list[str]
    ) -> list[Proposition]:
        """Collect evidence propositions from all propositions the agent examined.

        Args:
            seen_proposition_ids: List of all proposition IDs seen during search
            seen_paper_ids: List of all paper IDs seen during search
            evidence_ids: List of paper IDs explicitly cited in agent's response

        Returns:
            List of propositions the agent examined
        """
        evidence: list[Proposition] = []
        collected_prop_ids: set[str] = set()

        # First, collect propositions directly referenced by ID
        for prop_id in seen_proposition_ids:
            if prop_id in collected_prop_ids:
                continue
            prop = self.kb.get_proposition(prop_id)
            if prop:
                evidence.append(prop)
                collected_prop_ids.add(prop_id)

        # Then, if we have paper IDs from evidence_ids (agent's final citations),
        # add quality propositions from those papers
        for paper_id in evidence_ids:
            paper = self.kb.get_paper(paper_id)
            if paper:
                quality_props = paper.get_quality_propositions()
                for prop in quality_props[:5]:  # Top 5 from cited papers
                    if prop.prop_id not in collected_prop_ids:
                        evidence.append(prop)
                        collected_prop_ids.add(prop.prop_id)

        return evidence

    def _collect_evidence(self, paper_ids: list[str]) -> list[Proposition]:
        """Collect evidence propositions from specified papers.

        Args:
            paper_ids: List of paper IDs

        Returns:
            List of propositions from those papers
        """
        evidence = []

        for paper_id in paper_ids:
            paper = self.kb.get_paper(paper_id)
            if paper:
                # Get quality propositions from this paper
                quality_props = paper.get_quality_propositions()
                # Limit to top 3 per paper
                evidence.extend(quality_props[:3])

        # If no specific evidence found, return empty list
        return evidence if evidence else []

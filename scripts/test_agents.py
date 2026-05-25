"""Test script for autonomous claim verification agent.

This script tests the autonomous agent to verify it works correctly.
"""

import asyncio
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.core.agents.autonomous_agent import AutonomousClaimAgent

# Debug toggle - set to True to see all debug messages
DEBUG = False


async def test_autonomous_agent():
    """Test the autonomous agent."""
    print("\n" + "=" * 70)
    print(" Testing Autonomous Agent")
    print("=" * 70)

    # Load knowledge base
    print("\n Loading knowledge base...")
    kb = KnowledgeBase()
    try:
        kb.load()
        print(f"   Loaded {len(kb.papers)} papers")
    except Exception as e:
        print(f"   Error loading KB: {e}")
        return

    # Initialize agent
    print("\n Initializing autonomous agent...")
    agent = AutonomousClaimAgent(
        kb=kb,
        allow_online_search=False  # Disable online search for faster testing
    )

    # Test claim
    claim = "Vitamin D prevents COVID-19"
    print(f"\n Testing claim: '{claim}'")
    print("\n Agent reasoning and tool use (streaming):")

    # Run verification with streaming
    print("\n Running verification...")
    event_count = 0
    async for event in agent.verify_claim_stream(claim, debug=DEBUG):
        event_count += 1
        event_type = event.get("type")

        if DEBUG:
            print(f"\n   [DEBUG] Event #{event_count}, type: {event_type}")

        if event_type == "tool_call":
            tool_name = event.get("tool")
            print(f"\n   [Tool Call] {tool_name}")
            args = event.get("args", {})
            for key, value in args.items():
                print(f"      {key}: {value}")

        elif event_type == "tool_result":
            content = event.get("content", "")
            # Show only first line unless DEBUG is on
            if DEBUG:
                print(f"\n   [Tool Result]\n{content}")
            else:
                first_line = content.split("\n")[0]
                print(f"   [Tool Result] {first_line}...")

        elif event_type == "agent_message":
            content = event.get("content", "")
            print(f"\n   [Agent Final Answer]\n{content}")

        elif event_type == "error":
            error = event.get("error")
            print(f"\n   [Error] {error}")
        else:
            if DEBUG:
                print(f"\n   [Unknown Event] {event}")

    if DEBUG:
        print(f"\n   [DEBUG] Total events received: {event_count}")

    print("\n" + "=" * 70)
    print(" Streaming Complete!")
    print("=" * 70)


async def main():
    """Run the test."""
    print("\n" + "=" * 70)
    print(" AUTONOMOUS AGENT TEST SUITE")
    print("=" * 70)

    await test_autonomous_agent()

    print("\n" + "=" * 70)
    print(" Test Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

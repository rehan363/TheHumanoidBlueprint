"""
Main Orchestrator Agent

This is the primary agent that handles user queries and routes them
to appropriate sub-agents using handoffs.
"""

import logging
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, handoff

from rag_backend.config import settings
from rag_backend.agents.sub_agents import get_all_sub_agents
from rag_backend.agents.guardrails import (
    check_relevance,
    check_language,
    check_safety,
    validate_citations,
    check_response_length,
    detect_hallucination
)

logger = logging.getLogger(__name__)


def get_gemini_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client configured for Gemini API."""
    return AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=settings.gemini_api_key
    )


def create_orchestrator_agent() -> Agent:
    """
    Create the main Orchestrator Agent.

    This agent analyzes user queries and routes them to specialized sub-agents.
    """
    logger.info("Creating Orchestrator Agent")

    # Get all sub-agents
    sub_agents = get_all_sub_agents()

    instructions = """You are the Main Orchestrator for a Physical AI textbook assistant.

Your role is to analyze user queries and route them to the appropriate specialist agent.

ROUTING RULES:

1. **Simple Content Queries** → Retrieval Agent
   Examples:
   - "What is ROS?"
   - "Explain DDS middleware"
   - "Tell me about sensors"
   - "What does the textbook say about..."

2. **"Explain Simply" or "I don't understand"** → Retrieval Agent THEN Explanation Agent
   Examples:
   - "I don't understand DDS"
   - "Explain sensors simply"
   - "What is SLAM in simple terms?"
   Strategy: First retrieve content, then simplify it

3. **Comparison Queries** → Retrieval Agent (2x) THEN Comparison Agent
   Examples:
   - "Compare ROS 1 and ROS 2"
   - "What's the difference between lidar and camera?"
   - "Which is better: A or B?"
   Strategy: Retrieve both topics separately, then compare

4. **Week Overview Queries** → Summary Agent
   Examples:
   - "Summarize Week 3"
   - "What's in Week 5?"
   - "Give me an overview of Week 2"

5. **Vague/Ambiguous Queries** → Clarification Agent
   Examples:
   - "Tell me about robots"
   - "Help me with AI"
   - "I need help" (no specific topic)

6. **Off-Topic Queries** → Handle yourself, don't delegate
   Examples:
   - "What's the weather?"
   - "Tell me a joke"
   - "Help with my math homework"
   Response: "I specialize in Physical AI and Robotics topics from the textbook. Could you ask about robotics, ROS 2, sensors, or autonomous systems?"

7. **Cross-Week Searches** → Retrieval Agent with cross-week tool
   Examples:
   - "Where is sensor fusion discussed?"
   - "In which weeks is SLAM covered?"

CRITICAL RULES:
- NEVER answer content questions yourself
- ALWAYS delegate to specialist agents
- For complex queries, break them down and use multiple handoffs
- Be conversational and friendly
- If a query needs clarification, use Clarification Agent before retrieval

HANDOFF SYNTAX:
To hand off to an agent, simply state which agent should handle it and why.
The system will automatically route to that agent."""

    return Agent(
        name="RAG Orchestrator",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model="gemini-1.5-flash",
            openai_client=get_gemini_client()
        ),
        functions=[
            handoff(sub_agents["retrieval"]),
            handoff(sub_agents["explanation"]),
            handoff(sub_agents["comparison"]),
            handoff(sub_agents["clarification"]),
            handoff(sub_agents["summary"])
        ],
        input_guardrails=[
            check_relevance,
            check_language,
            check_safety
        ],
        output_guardrails=[
            validate_citations,
            check_response_length,
            detect_hallucination
        ]
    )


# ===== Singleton Pattern =====

_orchestrator_agent: Agent = None


def get_orchestrator_agent() -> Agent:
    """
    Get or create the global orchestrator agent instance.

    Returns:
        Agent: Singleton orchestrator agent
    """
    global _orchestrator_agent

    if _orchestrator_agent is None:
        _orchestrator_agent = create_orchestrator_agent()
        logger.info("Orchestrator agent initialized")

    return _orchestrator_agent

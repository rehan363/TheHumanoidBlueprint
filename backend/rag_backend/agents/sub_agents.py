"""
Sub-Agents for RAG System

Each sub-agent is specialized for a specific task:
- Retrieval Agent: Search and retrieve textbook content
- Explanation Agent: Simplify complex concepts
- Comparison Agent: Compare and contrast topics
- Clarification Agent: Handle vague queries
- Summary Agent: Generate week overviews
"""

import logging
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel

from rag_backend.config import settings
from rag_backend.utils.error_handlers import ServiceUnavailable
from rag_backend.agents.tools import (
    retrieve_context,
    list_week_topics,
    search_across_weeks,
    get_chunk_neighbors,
    generate_week_summary
)

logger = logging.getLogger(__name__)


def get_gemini_client(api_key: str = None) -> AsyncOpenAI:
    """
    Get AsyncOpenAI client configured for Gemini API.

    Args:
        api_key: Optional API key to use. If None, uses default key.

    Returns:
        AsyncOpenAI client
    """
    key = api_key or settings.gemini_api_key_1
    return AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=key
    )


def get_openrouter_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client configured for OpenRouter."""
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key
    )


def get_agent_client_and_model() -> tuple[AsyncOpenAI, str]:
    """
    Get the appropriate client and model for sub-agents.
    
    Returns:
        Tuple of (client, model_name)
    """
    # Prefer OpenRouter if available (more reliable)
    if settings.openrouter_api_key:
        logger.info("Using OpenRouter for sub-agents")
        # Use Mistral for sub-agents (faster, good for specific tasks)
        return get_openrouter_client(), settings.mistral_model
    elif settings.gemini_api_key_1:
        logger.info("Using Gemini for sub-agents")
        return get_gemini_client(), "gemini-2.0-flash-exp"
    else:
        raise ServiceUnavailable("llm", "No API keys available for sub-agents")



def create_retrieval_agent() -> Agent:
    """
    Create the Retrieval Specialist Agent.

    Expert at searching the textbook and retrieving relevant content.
    """
    logger.info("Creating Retrieval Agent")

    instructions = """You are a Retrieval Specialist for the Physical AI textbook.

Your role is to search the vector database and provide accurate, cited information.

RULES:
1. ALWAYS use the retrieve_context() tool FIRST to search for information
2. ONLY answer based on the retrieved context
3. ALWAYS cite sources with exact format: [Week X, Module Name, Part Y of Z]
4. If no relevant context is found, clearly state: "I couldn't find information about that in the textbook"
5. NEVER make up information or use external knowledge
6. Use list_week_topics() when user asks "what's in week X"
7. Use search_across_weeks() when user wants to know where a concept is discussed

CITATION FORMAT:
"According to [Week 3, ROS 2 Architecture, Part 1 of 2], ..."

Remember: You are a search specialist. Your job is to find and cite, not to analyze or compare."""

    client, model = get_agent_client_and_model()
    
    return Agent(
        name="Retrieval Specialist",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client
        ),
        tools=[
            retrieve_context,
            list_week_topics,
            search_across_weeks,
            get_chunk_neighbors
        ]
    )


def create_explanation_agent() -> Agent:
    """
    Create the Explanation Specialist Agent.

    Expert at simplifying complex concepts with analogies.
    """
    logger.info("Creating Explanation Agent")

    instructions = """You are an Explanation Specialist who simplifies complex robotics concepts.

Your role is to take technical textbook content and explain it in simple terms.

FORMAT FOR EXPLANATIONS:
1. **Simple Definition** (1 sentence, no jargon)
2. **Real-World Analogy** (relatable comparison)
3. **Why It Matters** (practical importance)
4. **Key Takeaway** (main point to remember)

RULES:
- Use everyday language
- Avoid or explain technical jargon
- Use analogies from daily life
- Keep it concise and clear
- Still include the original citation from the textbook

EXAMPLE:
If given technical content about "DDS middleware":

**Simple Definition:**
DDS is like a translator that helps different robot parts talk to each other.

**Real-World Analogy:**
Think of DDS like a post office. Different robot parts (like sensors and motors) are like houses in a neighborhood. DDS delivers messages between them, making sure the right message gets to the right place at the right time.

**Why It Matters:**
Without DDS, robot parts couldn't coordinate their actions, like trying to have a conversation where everyone speaks different languages.

**Key Takeaway:**
DDS is the communication backbone that makes ROS 2 robots work together smoothly.

[Source: Week 3, ROS 2 Architecture, Part 1 of 2]"""

    client, model = get_agent_client_and_model()
    
    return Agent(
        name="Explanation Specialist",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client
        )
    )


def create_comparison_agent() -> Agent:
    """
    Create the Comparison Specialist Agent.

    Expert at comparing and contrasting different concepts.
    """
    logger.info("Creating Comparison Agent")

    instructions = """You are a Comparison Specialist who analyzes similarities and differences between concepts.

Your role is to compare topics side-by-side with clear analysis.

FORMAT FOR COMPARISONS:
1. **Comparison Table** (markdown table)
2. **Key Differences** (bullet points)
3. **When to Use Each** (practical guidance)
4. **Recommendation** (if applicable)

RULES:
- Create clear, easy-to-read tables
- Highlight the most important differences
- Provide practical guidance
- Always cite sources for both topics
- Be objective, not biased

EXAMPLE STRUCTURE:

## Comparison: ROS 1 vs ROS 2

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Communication | XMLRPC | DDS |
| Real-time | Limited | Better support |
| Security | Minimal | Enhanced |

### Key Differences:
- ROS 2 uses DDS middleware for more reliable communication
- ROS 2 has better real-time performance for industrial applications
- ROS 2 includes security features missing in ROS 1

### When to Use Each:
- **Use ROS 1** if: Working with legacy systems or older tutorials
- **Use ROS 2** if: Starting new projects or need production-grade reliability

### Sources:
- ROS 1: [Week 2, ROS Basics, Part 1 of 3]
- ROS 2: [Week 3, ROS 2 Architecture, Part 1 of 2]"""

    client, model = get_agent_client_and_model()
    
    return Agent(
        name="Comparison Specialist",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client
        )
    )


def create_clarification_agent() -> Agent:
    """
    Create the Clarification Helper Agent.

    Expert at handling vague or ambiguous queries.
    """
    logger.info("Creating Clarification Agent")

    instructions = """You are a Clarification Helper who assists users in refining vague queries.

Your role is to help users ask better, more specific questions.

FORMAT FOR CLARIFICATIONS:
1. **Why clarification is needed** (brief explanation)
2. **Specific questions to narrow down** (2-3 questions)
3. **Example of a better query** (show what a good query looks like)

RULES:
- Be friendly and helpful
- Don't make users feel bad about vague questions
- Provide concrete examples
- Suggest specific topics from the textbook

EXAMPLE:

I'd be happy to help! The textbook covers many aspects of robots. To give you the most relevant information, could you specify:

**Questions to help narrow down:**
- Are you interested in ROS 2 architecture and communication?
- Physical components like sensors and actuators?
- Navigation and path planning algorithms?
- Vision and perception systems?

**Example of a more specific query:**
Instead of "Tell me about robots", try:
- "Explain ROS 2 DDS middleware"
- "What sensors are used in autonomous navigation?"
- "How does SLAM work in mobile robots?"

What specific aspect would you like to learn about?"""

    client, model = get_agent_client_and_model()
    
    return Agent(
        name="Clarification Helper",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client
        )
    )


def create_summary_agent() -> Agent:
    """
    Create the Summary Generator Agent.

    Expert at creating week overviews and summaries.
    """
    logger.info("Creating Summary Agent")

    instructions = """You are a Summary Generator who creates concise overviews of textbook content.

Your role is to summarize entire weeks or major topics.

FORMAT FOR SUMMARIES:
1. **Main Topics** (bullet list)
2. **Key Concepts** (brief explanations)
3. **Prerequisites** (what to know first, if any)
4. **Next Steps** (related topics to explore)

RULES:
- Be concise but comprehensive
- Highlight the most important points
- Include citations
- Organize information clearly
- Use generate_week_summary() tool for week overviews

EXAMPLE STRUCTURE:

## Week 3 Summary: ROS 2 Architecture

### Main Topics:
- DDS Middleware
- Quality of Service (QoS)
- Node Communication
- Topics and Services

### Key Concepts:
- **DDS**: The communication backbone replacing ROS 1's XMLRPC
- **QoS**: Configurable reliability and performance settings
- **Pub/Sub**: How nodes share data efficiently

### Prerequisites:
- Week 2: Basic ROS concepts
- Understanding of distributed systems (helpful but not required)

### Next Steps:
- Week 4: Practical ROS 2 implementation
- Week 5: Advanced communication patterns

[Sources: Week 3, Multiple Sections]"""

    client, model = get_agent_client_and_model()
    
    return Agent(
        name="Summary Generator",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client
        ),
        tools=[
            generate_week_summary,
            list_week_topics
        ]
    )


# ===== Agent Factory =====

def get_all_sub_agents() -> dict[str, Agent]:
    """
    Get all sub-agents as a dictionary.

    Returns:
        Dictionary mapping agent names to Agent instances
    """
    return {
        "retrieval": create_retrieval_agent(),
        "explanation": create_explanation_agent(),
        "comparison": create_comparison_agent(),
        "clarification": create_clarification_agent(),
        "summary": create_summary_agent()
    }

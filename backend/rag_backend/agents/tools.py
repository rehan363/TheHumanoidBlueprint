"""
Tools/Skills for RAG Agent System

These tools are used by agents to interact with the vector database
and perform specialized tasks.
"""

import logging
from typing import Optional, List, Dict, Any
from agents import function_tool

from rag_backend.services.embedding_service import get_embedding_service
from rag_backend.services.vector_store import get_vector_store
from rag_backend.config import settings

logger = logging.getLogger(__name__)


@function_tool
async def retrieve_context(
    query: str,
    top_k: int = 5,
    week_filter: Optional[int] = None
) -> str:
    """
    Search the vector database for relevant textbook content.

    This is the primary tool for finding information in the Physical AI textbook.
    It converts the query to a vector embedding and searches for similar content.

    Args:
        query: The search query (e.g., "What is DDS middleware?")
        top_k: Number of relevant chunks to retrieve (default: 5)
        week_filter: Optional week number (1-13) to search only in that week

    Returns:
        Formatted string with retrieved content and citations
    """
    try:
        logger.info(f"retrieve_context called: query='{query[:50]}...', top_k={top_k}, week={week_filter}")

        # Get services
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        # Generate query embedding
        query_embedding = await embedding_service.generate_query_embedding(query)

        # Search vector database
        results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=settings.similarity_threshold,
            week_filter=week_filter
        )

        if not results:
            return "No relevant content found in the textbook for this query."

        # Format results with citations
        formatted_chunks = []
        for idx, chunk in enumerate(results, 1):
            citation = (
                f"[Week {chunk['week']}, {chunk['module']}, "
                f"Part {chunk['chunk_index']+1} of {chunk['total_chunks']}]"
            )

            formatted_chunks.append(
                f"**Source {idx}** {citation}\n"
                f"Relevance: {chunk['score']:.1%}\n\n"
                f"{chunk['content']}\n\n"
                f"---"
            )

        context = "\n\n".join(formatted_chunks)
        logger.info(f"Retrieved {len(results)} chunks for query")

        return context

    except Exception as e:
        logger.error(f"Error in retrieve_context: {e}")
        return f"Error retrieving context: {str(e)}"


@function_tool
async def list_week_topics(week: int) -> str:
    """
    List all topics covered in a specific week of the textbook.

    Args:
        week: Week number (1-13)

    Returns:
        Formatted list of topics/modules in that week
    """
    try:
        logger.info(f"list_week_topics called: week={week}")

        if week < 1 or week > 13:
            return f"Invalid week number: {week}. Please use a week between 1 and 13."

        vector_store = get_vector_store()

        # Get all chunks for this week (using scroll to get all results)
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="week", match=MatchValue(value=week))]
            ),
            limit=100,
            with_payload=True
        )

        if not results:
            return f"No content found for Week {week}."

        # Extract unique modules
        modules = {}
        for point in results:
            module = point.payload.get("module", "Unknown")
            if module not in modules:
                modules[module] = {
                    "chunks": 0,
                    "file_path": point.payload.get("file_path", "")
                }
            modules[module]["chunks"] += 1

        # Format output
        output = f"## Week {week} Topics:\n\n"
        for idx, (module, info) in enumerate(sorted(modules.items()), 1):
            output += f"{idx}. **{module}** ({info['chunks']} sections)\n"

        logger.info(f"Found {len(modules)} topics in Week {week}")
        return output

    except Exception as e:
        logger.error(f"Error in list_week_topics: {e}")
        return f"Error listing topics: {str(e)}"


@function_tool
async def search_across_weeks(
    query: str,
    weeks: Optional[List[int]] = None
) -> str:
    """
    Search for a concept across multiple weeks to track where it's discussed.

    Args:
        query: The concept to search for (e.g., "sensor fusion")
        weeks: List of week numbers to search (default: all weeks 1-13)

    Returns:
        Summary of where the concept appears across weeks
    """
    try:
        logger.info(f"search_across_weeks called: query='{query}', weeks={weeks}")

        if weeks is None:
            weeks = list(range(1, 14))  # All weeks

        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        # Generate query embedding once
        query_embedding = await embedding_service.generate_query_embedding(query)

        # Search each week
        results_by_week = {}
        for week in weeks:
            week_results = await vector_store.search(
                query_embedding=query_embedding,
                top_k=2,  # Top 2 per week
                score_threshold=0.7,
                week_filter=week
            )

            if week_results:
                results_by_week[week] = week_results

        if not results_by_week:
            return f"The concept '{query}' was not found in the specified weeks."

        # Format output
        output = f"## Where '{query}' is discussed:\n\n"

        for week in sorted(results_by_week.keys()):
            chunks = results_by_week[week]
            output += f"### Week {week}:\n"

            for chunk in chunks:
                output += (
                    f"- **{chunk['module']}** "
                    f"(Relevance: {chunk['score']:.1%})\n"
                )

            output += "\n"

        logger.info(f"Found concept in {len(results_by_week)} weeks")
        return output

    except Exception as e:
        logger.error(f"Error in search_across_weeks: {e}")
        return f"Error searching across weeks: {str(e)}"


@function_tool
async def get_chunk_neighbors(
    chunk_id: str,
    before: int = 1,
    after: int = 1
) -> str:
    """
    Get surrounding chunks for more context.

    Useful when a user wants more detail about a specific section.

    Args:
        chunk_id: The ID of the target chunk
        before: Number of chunks before (default: 1)
        after: Number of chunks after (default: 1)

    Returns:
        Formatted text with surrounding context
    """
    try:
        logger.info(f"get_chunk_neighbors called: chunk_id={chunk_id}")

        vector_store = get_vector_store()

        # Get the target chunk
        target_chunk = await vector_store.get_chunk_by_id(chunk_id)

        if not target_chunk:
            return f"Chunk with ID '{chunk_id}' not found."

        # For now, return a message explaining this feature needs the chunk index
        # In a full implementation, we'd query neighboring chunks by index

        return (
            f"**Context for {chunk_id}:**\n\n"
            f"{target_chunk['content']}\n\n"
            f"_Note: Neighboring chunks feature requires index-based lookup._"
        )

    except Exception as e:
        logger.error(f"Error in get_chunk_neighbors: {e}")
        return f"Error getting neighbors: {str(e)}"


@function_tool
async def generate_week_summary(week: int) -> str:
    """
    Generate a high-level summary of an entire week's content.

    Args:
        week: Week number (1-13)

    Returns:
        Summary of key topics and concepts
    """
    try:
        logger.info(f"generate_week_summary called: week={week}")

        # First get all topics
        topics_list = await list_week_topics(week)

        # Then get a sample of content from each major topic
        vector_store = get_vector_store()
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="week", match=MatchValue(value=week))]
            ),
            limit=10,  # Sample 10 chunks
            with_payload=True
        )

        if not results:
            return f"No content available for Week {week}."

        # Build summary
        summary = f"# Week {week} Summary\n\n"
        summary += topics_list + "\n\n"

        summary += "## Key Content Samples:\n\n"
        for idx, point in enumerate(results[:5], 1):  # Top 5 samples
            module = point.payload.get("module", "Unknown")
            content_preview = point.payload.get("content", "")[:200]
            summary += f"{idx}. **{module}**: {content_preview}...\n\n"

        return summary

    except Exception as e:
        logger.error(f"Error in generate_week_summary: {e}")
        return f"Error generating summary: {str(e)}"

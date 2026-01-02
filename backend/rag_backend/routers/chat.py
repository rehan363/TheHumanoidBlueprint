"""
Chat query endpoint router for RAG system using OpenAI Agents SDK.
"""

from fastapi import APIRouter, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
import time
from datetime import datetime
from agents import Runner
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

from rag_backend.models.chat import ChatQueryRequest, ChatQueryResponse, Citation
from rag_backend.agents import get_orchestrator_agent, get_guardrail_response
from rag_backend.utils.rate_limiter import limiter
from rag_backend.utils.error_handlers import (
    InvalidRequest,
    RateLimitExceeded,
    ServiceUnavailable
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG chatbot",
    description="Submit a question to the RAG system and receive an AI-generated answer with citations"
)
@limiter.limit("3/minute")
async def query_chatbot(request: Request, query_request: ChatQueryRequest):
    """
    Process a chat query through the OpenAI Agents SDK system.

    Args:
        request: FastAPI request object (for rate limiting)
        query_request: Chat query request with query text and type

    Returns:
        ChatQueryResponse: AI-generated answer with citations

    Raises:
        400: Invalid request (validation failed)
        429: Rate limit exceeded
        500: Internal server error
        503: Service unavailable (backend offline)
    """
    start_time = time.time()

    try:
        # Validate query length
        if len(query_request.query) < 1 or len(query_request.query) > 500:
            raise InvalidRequest(
                "Query must be between 1 and 500 characters",
                {"query_length": len(query_request.query)}
            )

        # Sanitize query
        query_request.query = query_request.query.strip()

        if not query_request.query:
            raise InvalidRequest("Query cannot be empty after sanitization")

        # Log query
        logger.info(
            f"Processing {query_request.query_type} query from {request.client.host}: "
            f"{query_request.query[:50]}..."
        )

        # Get orchestrator agent
        orchestrator = get_orchestrator_agent()

        # Create runner
        runner = Runner()

        # Run agent with user query
        result = await runner.run(
            starting_agent=orchestrator,
            input=query_request.query
        )

        # Extract answer from result
        answer = result.final_output

        # Extract citations from answer (look for citation pattern)
        import re
        citations = []
        citation_pattern = r'\[Week (\d+), ([^\]]+), Part (\d+) of (\d+)\]'

        for match in re.finditer(citation_pattern, answer):
            week_num = int(match.group(1))
            module = match.group(2)
            chunk_idx = int(match.group(3)) - 1  # Convert to 0-based
            total_chunks = int(match.group(4))

            citation = Citation(
                source=f"Week {week_num}, {module}",
                chunk_index=chunk_idx,
                total_chunks=total_chunks,
                relevance_score=1.0,  # Agent-provided context assumed relevant
                content_preview=None
            )
            citations.append(citation)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Build response
        response = ChatQueryResponse(
            answer=answer,
            citations=citations,
            query_type=query_request.query_type,
            session_id=query_request.session_id,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )

        logger.info(
            f"Query processed successfully in {processing_time_ms}ms "
            f"with {len(citations)} citations"
        )

        return response

    except InputGuardrailTripwireTriggered as e:
        # Input guardrail blocked the query
        logger.warning(f"Input guardrail triggered: {e}")

        guardrail_name = str(e).split()[0] if str(e) else "unknown"
        error_message = get_guardrail_response(guardrail_name, e.guardrail_output)

        # Return error as response (not raising exception, per UX design)
        processing_time_ms = int((time.time() - start_time) * 1000)

        return ChatQueryResponse(
            answer=error_message,
            citations=[],
            query_type=query_request.query_type,
            session_id=query_request.session_id,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )

    except OutputGuardrailTripwireTriggered as e:
        # Output guardrail detected issue with response
        logger.warning(f"Output guardrail triggered: {e}")

        # Still return the response but with a warning appended
        guardrail_name = str(e).split()[0] if str(e) else "unknown"
        warning_message = get_guardrail_response(guardrail_name, e.guardrail_output)

        answer = result.final_output + warning_message

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ChatQueryResponse(
            answer=answer,
            citations=[],
            query_type=query_request.query_type,
            session_id=query_request.session_id,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )

    except InvalidRequest as e:
        logger.warning(f"Invalid request: {e.message}")
        raise

    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {request.client.host}")
        raise

    except ServiceUnavailable as e:
        logger.error(f"Service unavailable: {e.message}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error processing query: {e}")
        raise ServiceUnavailable("rag_system", "An unexpected error occurred while processing your query")

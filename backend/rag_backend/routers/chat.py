"""
Chat query endpoint router for RAG system.
"""

from fastapi import APIRouter, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
from rag_backend.models.chat import ChatQueryRequest, ChatQueryResponse, ErrorResponse
from rag_backend.services.rag_pipeline import get_rag_pipeline
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
    Process a chat query through the RAG pipeline.

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
    try:
        # Validate query length
        if len(query_request.query) < 1 or len(query_request.query) > 500:
            raise InvalidRequest(
                "Query must be between 1 and 500 characters",
                {"query_length": len(query_request.query)}
            )

        # Sanitize query (basic sanitization - strip whitespace)
        query_request.query = query_request.query.strip()

        if not query_request.query:
            raise InvalidRequest("Query cannot be empty after sanitization")

        # Log query
        logger.info(
            f"Processing {query_request.query_type} query from {request.client.host}: "
            f"{query_request.query[:50]}..."
        )

        # Get RAG pipeline
        rag_pipeline = get_rag_pipeline()

        # Process query through RAG pipeline
        response = await rag_pipeline.process_query(query_request)

        logger.info(
            f"Query processed successfully in {response.processing_time_ms}ms "
            f"with {len(response.citations)} citations"
        )

        return response

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

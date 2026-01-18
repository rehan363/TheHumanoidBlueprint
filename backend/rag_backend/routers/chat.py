import logging
import time
from datetime import datetime
from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, status, Request, Response, HTTPException

from rag_backend.models.chat import ChatQueryRequest, ChatQueryResponse, Citation
from rag_backend.models.session import QuerySession, SessionMessage
from rag_backend.services.rag_pipeline import get_rag_pipeline
from rag_backend.utils.rate_limiter import limiter
from rag_backend.utils.error_handlers import (
    InvalidRequest,
    RateLimitExceeded,
    ServiceUnavailable
)
from src.services.database_service import db_service # Import db_service
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post(
    "/query",
    response_model=ChatQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG chatbot",
    description="Submit a question to the RAG system and receive an AI-generated answer with citations"
)
@limiter.limit("3/minute")
async def query_chatbot(request: Request, response: Response, query_request: ChatQueryRequest):
    """
    Process a chat query through RAG pipeline.
    Persists user and assistant messages to the database.

    Args:
        request: FastAPI request object (for rate limiting)
        response: Response object (for rate limit headers)
        query_request: Chat query request with query text, type, and optional session_id

    Returns:
        ChatQueryResponse: AI-generated answer with citations and session_id

    Raises:
        400: Invalid request (validation failed)
        429: Rate limit exceeded
        500: Internal server error
        503: Service unavailable (backend offline)
    """
    start_time = time.time()
    session_id: Optional[UUID] = None

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

        # Log query start
        logger.info(
            f"Query Start: type={query_request.query_type}, "
            f"preview='{query_request.query[:50]}...', "
            f"session_id={query_request.session_id or 'new'}"
        )

        # Handle session: get existing or create new
        if query_request.session_id:
            session_id = query_request.session_id
        else:
            new_session = await db_service.create_session()
            session_id = new_session.id
            logger.info(f"Created new session: {session_id}")

        # Persist user message
        db_start = time.perf_counter()
        await db_service.save_message(
            session_id=session_id,
            role='user',
            content=query_request.query,
            query_type=query_request.query_type
        )
        logger.debug(f"Saved user message to DB in {time.perf_counter() - db_start:.4f}s")

        # Get RAG pipeline
        rag_pipeline = get_rag_pipeline()

        # Process query through RAG pipeline
        logger.info(f"Processing query through RAG pipeline for session {session_id}")
        rag_start = time.perf_counter()
        result = await rag_pipeline.process_query(query_request)
        logger.info(f"RAG pipeline completed in {time.perf_counter() - rag_start:.4f}s")

        # Persist assistant message
        db_start = time.perf_counter()
        await db_service.save_message(
            session_id=session_id,
            role='assistant',
            content=result.response,
            citations=result.citations,
            query_type=query_request.query_type
        )
        logger.debug(f"Saved assistant message to DB in {time.perf_counter() - db_start:.4f}s")

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Query processed successfully in {total_time:.2f}ms "
            f"with {len(result.citations)} citations for session {session_id}"
        )

        # Add session_id to the response
        chat_response = ChatQueryResponse(
            answer=result.response,
            citations=result.citations,
            query_type=query_request.query_type,
            session_id=session_id,
            processing_time_ms=int(total_time),
            timestamp=datetime.utcnow(),
        )
        return chat_response

    except InvalidRequest as e:
        logger.warning(f"Invalid request stage=validation: {e.message}, details={e.details}")
        raise

    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for host={request.client.host}")
        raise

    except ServiceUnavailable as e:
        logger.error(f"Service unavailable service={e.service_name}: {e.message}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error processing query for session {session_id}: {str(e)}")
        raise ServiceUnavailable("rag_system", "An unexpected error occurred while processing your query")

@router.get(
    "/history/{session_id}",
    response_model=List[SessionMessage],
    status_code=status.HTTP_200_OK,
    summary="Retrieve chat history for a session",
    description="Fetches all messages for a given session ID"
)
async def get_chat_history(session_id: UUID):
    """
    Retrieve chat history for a given session.

    Args:
        session_id: The UUID of the session to retrieve history for.

    Returns:
        List[SessionMessage]: A list of chat messages in chronological order.

    Raises:
        404: If the session ID does not exist.
        500: Internal server error.
    """
    try:
        messages = await db_service.get_session_messages(session_id)
        if not messages:
            # Check if session exists at all
            session_exists = await db_service.fetchrow("SELECT id FROM query_sessions WHERE id = $1", session_id)
            if not session_exists:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving chat history for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving chat history.")

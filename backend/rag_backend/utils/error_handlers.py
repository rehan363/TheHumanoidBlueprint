"""
Custom exception handlers and error definitions for the RAG backend.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from rag_backend.models.chat import ErrorResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Custom Exception Classes

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please try again later.", retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class ServiceUnavailable(Exception):
    """Exception raised when a required service is unavailable."""

    def __init__(self, service_name: str, message: str = None):
        self.service_name = service_name
        self.message = message or f"{service_name} service is temporarily unavailable"
        super().__init__(self.message)


class InvalidRequest(Exception):
    """Exception raised for invalid request parameters."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class EmbeddingGenerationError(Exception):
    """Exception raised when embedding generation fails."""

    def __init__(self, message: str = "Failed to generate embeddings"):
        self.message = message
        super().__init__(self.message)


class VectorSearchError(Exception):
    """Exception raised when vector search fails."""

    def __init__(self, message: str = "Vector search failed"):
        self.message = message
        super().__init__(self.message)


class LLMGenerationError(Exception):
    """Exception raised when LLM generation fails."""

    def __init__(self, message: str = "Failed to generate response from LLM"):
        self.message = message
        super().__init__(self.message)


from rag_backend.utils.logging_utils import get_request_id

# Exception Handlers

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded exceptions."""
    logger.warning(f"Rate limit exceeded for {request.client.host}: {exc.message}")

    error_response = ErrorResponse(
        error="rate_limit_exceeded",
        message=exc.message,
        details={"retry_after": exc.retry_after},
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=error_response.model_dump(),
        headers={"Retry-After": str(exc.retry_after)}
    )


async def service_unavailable_handler(request: Request, exc: ServiceUnavailable) -> JSONResponse:
    """Handle service unavailable exceptions."""
    logger.error(f"Service unavailable: {exc.service_name} - {exc.message}")

    error_response = ErrorResponse(
        error="service_unavailable",
        message="AI assistant temporarily offline. Browse content normally.",
        details={"service": exc.service_name},
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_response.model_dump()
    )


async def invalid_request_handler(request: Request, exc: InvalidRequest) -> JSONResponse:
    """Handle invalid request exceptions."""
    logger.warning(f"Invalid request from {request.client.host}: {exc.message}")

    error_response = ErrorResponse(
        error="invalid_request",
        message=exc.message,
        details=exc.details,
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump()
    )


async def embedding_generation_error_handler(request: Request, exc: EmbeddingGenerationError) -> JSONResponse:
    """Handle embedding generation errors."""
    logger.error(f"Embedding generation error: {exc.message}")

    error_response = ErrorResponse(
        error="embedding_error",
        message="Failed to process your query. Please try again.",
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


async def vector_search_error_handler(request: Request, exc: VectorSearchError) -> JSONResponse:
    """Handle vector search errors."""
    logger.error(f"Vector search error: {exc.message}")

    error_response = ErrorResponse(
        error="search_error",
        message="Failed to search textbook content. Please try again.",
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


async def llm_generation_error_handler(request: Request, exc: LLMGenerationError) -> JSONResponse:
    """Handle LLM generation errors."""
    logger.error(f"LLM generation error: {exc.message}")

    error_response = ErrorResponse(
        error="generation_error",
        message="Failed to generate response. Please try again.",
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general uncaught exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")

    error_response = ErrorResponse(
        error="internal_error",
        message="An unexpected error occurred. Please try again.",
        request_id=get_request_id(),
        timestamp=datetime.utcnow()
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


# Register all exception handlers with the FastAPI app
def register_exception_handlers(app):
    """Register all custom exception handlers with the FastAPI application."""
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_exception_handler(ServiceUnavailable, service_unavailable_handler)
    app.add_exception_handler(InvalidRequest, invalid_request_handler)
    app.add_exception_handler(EmbeddingGenerationError, embedding_generation_error_handler)
    app.add_exception_handler(VectorSearchError, vector_search_error_handler)
    app.add_exception_handler(LLMGenerationError, llm_generation_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)

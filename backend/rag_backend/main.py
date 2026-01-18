"""
FastAPI application entry point for RAG backend.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import logging
import sys
import time

from rag_backend.config import settings
from rag_backend.routers import health_router, chat_router
from rag_backend.utils.rate_limiter import limiter
from rag_backend.utils.error_handlers import register_exception_handlers
from rag_backend.utils.logging_utils import set_request_id, RequestIdFilter, get_request_id
import uuid

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add RequestIdFilter to the root logger
for handler in logging.root.handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG backend for Physical AI Textbook with Gemini and Qdrant",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter state
app.state.limiter = limiter

# Register rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Register custom exception handlers
register_exception_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests and outgoing responses, including processing time
    and response body for error statuses.
    Assumes a request ID is generated/set for each request.
    """
    # Generate or get request ID from headers
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)

    start_time = time.perf_counter()

    # Log incoming request details
    logger.info(
        f"Incoming Request: {request.client.host}:{request.client.port} "
        f"{request.method} {request.url.path}"
    )

    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        formatted_process_time = f"{process_time:.4f}s"

        # Log response status and processing time
        log_message = (
            f"Outgoing Response: {request.client.host}:{request.client.port} "
            f"{request.method} {request.url.path} - "
            f"Status {response.status_code} - Took {formatted_process_time}"
        )

        if response.status_code >= 400:
            # For error responses, try to read and log the response body
            # We need to buffer the response body to read it and then re-send it
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            try:
                # Attempt to decode as JSON for better readability
                log_message += f" - Body: {response_body.decode('utf-8')}"
            except UnicodeDecodeError:
                log_message += f" - Body: (non-text content)"

            if response.status_code >= 500:
                logger.error(log_message)
            else:
                logger.warning(log_message)
            
            # Re-generate the response since the body iterator was consumed
            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        else:
            logger.info(log_message)

        # Add Request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        process_time = time.perf_counter() - start_time
        formatted_process_time = f"{process_time:.4f}s"
        logger.exception(
            f"Request Failed: {request.client.host}:{request.client.port} "
            f"{request.method} {request.url.path} - Took {formatted_process_time} - Error: {e}"
        )
        # Re-raise the exception, it will be caught by the general_exception_handler
        raise


# Include routers
app.include_router(health_router)
app.include_router(chat_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Physical AI Textbook RAG Backend",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("="*60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("="*60)
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info(f"Rate limit: {settings.rate_limit_per_minute} requests/minute")
    logger.info("="*60)

    try:
        # Initialize RAG pipeline (triggers singleton creation)
        from rag_backend.services.rag_pipeline import get_rag_pipeline

        rag_pipeline = get_rag_pipeline()
        health_status = await rag_pipeline.health_check()

        logger.info("Service health check:")
        for service, status in health_status.items():
            status_emoji = "✅" if status else "❌"
            logger.info(f"  {status_emoji} {service}: {'healthy' if status else 'unavailable'}")

        if not all(health_status.values()):
            logger.warning("⚠️  Some services are unavailable! RAG functionality may be limited.")

        logger.info("="*60)
        logger.info("✅ Backend started successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.exception(f"❌ Failed to initialize services: {e}")
        logger.warning("⚠️  Backend started but some services may not be available.")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("="*60)
    logger.info("Shutting down backend...")
    logger.info("="*60)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag_backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower()
    )

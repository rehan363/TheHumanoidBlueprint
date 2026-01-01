"""
FastAPI application entry point for RAG backend.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import logging
import sys

from rag_backend.config import settings
from rag_backend.routers import health_router, chat_router
from rag_backend.utils.rate_limiter import limiter
from rag_backend.utils.error_handlers import register_exception_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG backend for Physical AI Textbook with Gemini and Qdrant",
    docs_url="/api/docs" if settings.debug_mode else None,
    redoc_url="/api/redoc" if settings.debug_mode else None,
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
    """Log all incoming requests and responses."""
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"Request failed: {e}")
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
        "docs": "/api/docs" if settings.debug_mode else "disabled"
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
        # Import and initialize services (triggers singleton creation)
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

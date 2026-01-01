"""
Health check endpoint router.
"""

from fastapi import APIRouter, status
from rag_backend.models.health import HealthCheckResponse
from rag_backend.services.rag_pipeline import get_rag_pipeline
from rag_backend.config import settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Check the health status of the RAG backend and its services"
)
async def health_check():
    """
    Health check endpoint to verify backend and service status.

    Returns:
        HealthCheckResponse: Health status with service details
    """
    try:
        # Get RAG pipeline
        rag_pipeline = get_rag_pipeline()

        # Check individual services
        service_health = await rag_pipeline.health_check()

        # Determine overall status
        all_healthy = all(service_health.values())
        overall_status = "healthy" if all_healthy else "degraded"

        # Build services status dict
        services = {
            "qdrant": "ok" if service_health.get("qdrant") else "unavailable",
            "gemini": "ok" if service_health.get("gemini") else "unavailable",
        }

        response = HealthCheckResponse(
            status=overall_status,
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            services=services,
            details={
                "app_name": settings.app_name,
                "debug_mode": settings.debug_mode
            }
        )

        logger.info(f"Health check: {overall_status}")
        return response

    except Exception as e:
        logger.exception(f"Health check failed: {e}")

        # Return unhealthy status even if check fails
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            services={},
            details={"error": str(e)}
        )

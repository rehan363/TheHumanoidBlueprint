"""
API routers for RAG backend.
"""

from rag_backend.routers.health import router as health_router

__all__ = ["health_router"]

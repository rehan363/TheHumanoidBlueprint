"""
API routers for RAG backend.
"""

from rag_backend.routers.health import router as health_router
from rag_backend.routers.chat import router as chat_router

__all__ = ["health_router", "chat_router"]

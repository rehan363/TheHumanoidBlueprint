"""
Embedding service using Google's Generative AI text-embedding-004 model.
"""

import google.generativeai as genai
from typing import List, Union
import logging
from rag_backend.config import settings
from rag_backend.utils.error_handlers import EmbeddingGenerationError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Google's embedding model."""

    def __init__(self):
        """Initialize the embedding service with Google Generative AI."""
        try:
            genai.configure(api_key=settings.embedding_api_key)
            self.model_name = settings.gemini_embedding_model
            logger.info(f"EmbeddingService initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            raise EmbeddingGenerationError(f"Embedding service initialization failed: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to generate embedding for

        Returns:
            List[float]: Vector embedding (768 dimensions)

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            if not text or not text.strip():
                raise EmbeddingGenerationError("Cannot generate embedding for empty text")

            result = genai.embed_content(
                model=self.model_name,
                content=text.strip(),
                task_type="retrieval_document"
            )

            embedding = result['embedding']

            if not embedding or len(embedding) != settings.qdrant_vector_size:
                raise EmbeddingGenerationError(
                    f"Invalid embedding dimension: expected {settings.qdrant_vector_size}, got {len(embedding)}"
                )

            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise EmbeddingGenerationError(f"Failed to generate embedding: {e}")

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query text (optimized for retrieval).

        Args:
            query: Query text to generate embedding for

        Returns:
            List[float]: Vector embedding (768 dimensions)

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            if not query or not query.strip():
                raise EmbeddingGenerationError("Cannot generate embedding for empty query")

            result = genai.embed_content(
                model=self.model_name,
                content=query.strip(),
                task_type="retrieval_query"
            )

            embedding = result['embedding']

            if not embedding or len(embedding) != settings.qdrant_vector_size:
                raise EmbeddingGenerationError(
                    f"Invalid embedding dimension: expected {settings.qdrant_vector_size}, got {len(embedding)}"
                )

            logger.debug(f"Generated query embedding for text of length {len(query)}")
            return embedding

        except Exception as e:
            logger.error(f"Query embedding generation error: {e}")
            raise EmbeddingGenerationError(f"Failed to generate query embedding: {e}")

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List[List[float]]: List of vector embeddings

        Raises:
            EmbeddingGenerationError: If batch embedding generation fails
        """
        try:
            if not texts:
                raise EmbeddingGenerationError("Cannot generate embeddings for empty list")

            embeddings = []
            for text in texts:
                if not text or not text.strip():
                    logger.warning("Skipping empty text in batch")
                    continue

                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)

            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation error: {e}")
            raise EmbeddingGenerationError(f"Failed to generate batch embeddings: {e}")


# Global embedding service instance
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.

    Returns:
        EmbeddingService: Singleton embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

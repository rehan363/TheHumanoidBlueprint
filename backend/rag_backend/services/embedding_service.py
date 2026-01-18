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
            logger.info(f"EmbeddingService initialized with model: {self.model_name}", extra={"model_name": self.model_name})
        except Exception as e:
            logger.error(
                f"Failed to initialize EmbeddingService: {e}",
                exc_info=True,
                extra={"error_type": "InitializationError", "exception_message": str(e)}
            )
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
        logger.info(f"Generating embedding for text length: {len(text)}")
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
                error_msg = f"Invalid embedding dimension: expected {settings.qdrant_vector_size}, got {len(embedding) if embedding else 'None'}"
                logger.error(
                    error_msg,
                    extra={
                        "error_type": "EmbeddingDimensionError",
                        "text_length": len(text),
                        "expected_dimension": settings.qdrant_vector_size,
                        "actual_dimension": len(embedding) if embedding else 0
                    }
                )
                raise EmbeddingGenerationError(error_msg)

            logger.debug(f"Generated embedding for text of length {len(text)}", extra={"text_length": len(text)})
            return embedding

        except EmbeddingGenerationError as e:
            logger.error(
                f"Embedding generation error: {e}",
                exc_info=True,
                extra={"error_type": "EmbeddingGenerationError", "text_length": len(text), "exception_message": str(e)}
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected embedding generation error: {e}",
                exc_info=True,
                extra={"error_type": "UnexpectedError", "text_length": len(text), "exception_message": str(e)}
            )

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
                error_msg = f"Invalid query embedding dimension: expected {settings.qdrant_vector_size}, got {len(embedding) if embedding else 'None'}"
                logger.error(
                    error_msg,
                    extra={
                        "error_type": "QueryEmbeddingDimensionError",
                        "query_length": len(query),
                        "expected_dimension": settings.qdrant_vector_size,
                        "actual_dimension": len(embedding) if embedding else 0
                    }
                )
                raise EmbeddingGenerationError(error_msg)

            logger.debug(f"Generated query embedding for text of length {len(query)}", extra={"query_length": len(query)})
            return embedding

        except EmbeddingGenerationError as e:
            logger.error(
                f"Query embedding generation error: {e}",
                exc_info=True,
                extra={"error_type": "QueryEmbeddingGenerationError", "query_length": len(query), "exception_message": str(e)}
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected query embedding generation error: {e}",
                exc_info=True,
                extra={"error_type": "UnexpectedError", "query_length": len(query), "exception_message": str(e)}
            )

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

            logger.info(f"Generated {len(embeddings)} embeddings in batch", extra={"num_generated_embeddings": len(embeddings), "num_input_texts": len(texts)})
            return embeddings

        except EmbeddingGenerationError as e:
            logger.error(
                f"Batch embedding generation error: {e}",
                exc_info=True,
                extra={"error_type": "BatchEmbeddingGenerationError", "num_texts": len(texts), "exception_message": str(e)}
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected batch embedding generation error: {e}",
                exc_info=True,
                extra={"error_type": "UnexpectedError", "num_texts": len(texts), "exception_message": str(e)}
            )


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

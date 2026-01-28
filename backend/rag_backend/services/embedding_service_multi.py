"""
Multi-provider embedding service supporting local sentence-transformers, Jina, and Gemini.

This service provides fallback capabilities between different embedding providers
to ensure embeddings can be generated even when one provider is unavailable.
"""

from typing import List, Optional, Literal
import logging
import asyncio
import httpx

from rag_backend.config import settings
from rag_backend.utils.error_handlers import EmbeddingGenerationError

logger = logging.getLogger(__name__)

EmbeddingProviderType = Literal["local", "jina", "gemini", "auto"]


class MultiProviderEmbeddingService:
    """Embedding service supporting multiple providers with automatic fallback."""

    def __init__(self):
        """Initialize embedding service with all available providers."""
        self.provider = settings.embedding_provider
        self.vector_size = settings.qdrant_vector_size  # 768
        
        # Track available providers
        self.local_available = False
        self.jina_available = bool(settings.jina_api_key)
        self.gemini_available = bool(settings.embedding_api_key)
        
        # Initialize local sentence-transformers model
        self.local_model = None
        self.local_model_name = settings.huggingface_embedding_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.local_model_name}")
            self.local_model = SentenceTransformer(self.local_model_name)
            self.local_available = True
            logger.info(f"Local embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load local embedding model: {e}")
        
        # Jina configuration
        self.jina_api_url = "https://api.jina.ai/v1/embeddings"
        self.jina_model = "jina-embeddings-v3"
        
        # Log provider status
        logger.info(f"Embedding Provider Strategy: {self.provider}")
        logger.info(f"Available providers: Local={self.local_available}, Jina={self.jina_available}, Gemini={self.gemini_available}")
        
        if not any([self.local_available, self.jina_available, self.gemini_available]):
            raise EmbeddingGenerationError("No embedding providers available")

    async def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for text using available providers with fallback.

        Args:
            text: Input text to generate embedding for
            task_type: Type of embedding task ("retrieval_document" or "retrieval_query")

        Returns:
            List[float]: Vector embedding (768 dimensions)

        Raises:
            EmbeddingGenerationError: If all providers fail
        """
        if not text or not text.strip():
            raise EmbeddingGenerationError("Cannot generate embedding for empty text")

        text = text.strip()
        errors = []

        # Determine provider order based on strategy - prioritize Gemini for consistency with indexed docs
        if self.provider == "auto":
            providers = ["gemini", "local", "jina"]  # Gemini first for consistency with existing vectors
        elif self.provider == "local":
            providers = ["local"]
        elif self.provider == "jina":
            providers = ["jina", "local"]
        elif self.provider == "gemini":
            providers = ["gemini", "local"]
        else:
            providers = ["gemini", "local"]

        for provider in providers:
            try:
                if provider == "local" and self.local_available:
                    embedding = await self._generate_local(text)
                    logger.debug(f"Generated embedding using local model ({len(embedding)} dims)")
                    return embedding
                elif provider == "jina" and self.jina_available:
                    embedding = await self._generate_jina(text, task_type)
                    logger.debug(f"Generated embedding using Jina ({len(embedding)} dims)")
                    return embedding
                elif provider == "gemini" and self.gemini_available:
                    embedding = await self._generate_gemini(text, task_type)
                    logger.debug(f"Generated embedding using Gemini ({len(embedding)} dims)")
                    return embedding
            except Exception as e:
                error_msg = f"{provider}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Embedding provider {provider} failed: {e}")
                continue

        # All providers failed
        raise EmbeddingGenerationError(f"All embedding providers failed: {'; '.join(errors)}")

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query (optimized for retrieval).

        Args:
            query: Query text to generate embedding for

        Returns:
            List[float]: Vector embedding (768 dimensions)
        """
        return await self.generate_embedding(query, task_type="retrieval_query")

    async def _generate_local(self, text: str) -> List[float]:
        """Generate embedding using local sentence-transformers model."""
        logger.info("Generating embedding with local sentence-transformers")
        
        if self.local_model is None:
            raise EmbeddingGenerationError("Local embedding model not loaded")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.local_model.encode(text, convert_to_numpy=True).tolist()
        )
        
        # Validate dimension
        if len(embedding) != self.vector_size:
            raise EmbeddingGenerationError(
                f"Local embedding dimension mismatch: expected {self.vector_size}, got {len(embedding)}"
            )
        
        return embedding

    async def _generate_jina(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding using Jina AI API."""
        logger.info("Generating embedding with Jina AI")
        
        # Map task type to Jina's task parameter
        jina_task = "retrieval.passage" if task_type == "retrieval_document" else "retrieval.query"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.jina_api_url,
                headers={
                    "Authorization": f"Bearer {settings.jina_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.jina_model,
                    "task": jina_task,
                    "dimensions": self.vector_size,  # Request 768 dims
                    "input": [text]
                }
            )
            
            if response.status_code != 200:
                raise EmbeddingGenerationError(
                    f"Jina API error: {response.status_code} - {response.text}"
                )
            
            result = response.json()
            
            if "data" not in result or len(result["data"]) == 0:
                raise EmbeddingGenerationError("Jina API returned empty response")
            
            embedding = result["data"][0]["embedding"]
            
            if len(embedding) != self.vector_size:
                raise EmbeddingGenerationError(
                    f"Jina embedding dimension mismatch: expected {self.vector_size}, got {len(embedding)}"
                )
            
            return embedding

    async def _generate_gemini(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding using Google Gemini API."""
        logger.info("Generating embedding with Gemini")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.embedding_api_key)
            
            result = genai.embed_content(
                model=settings.gemini_embedding_model,
                content=text,
                task_type=task_type
            )
            
            embedding = result['embedding']
            
            if len(embedding) != self.vector_size:
                raise EmbeddingGenerationError(
                    f"Gemini embedding dimension mismatch: expected {self.vector_size}, got {len(embedding)}"
                )
            
            return embedding
            
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "leaked" in error_str.lower():
                logger.warning("Gemini API key is invalid or leaked")
                self.gemini_available = False
            raise EmbeddingGenerationError(f"Gemini embedding failed: {e}")

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List[List[float]]: List of vector embeddings
        """
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

    async def health_check(self) -> bool:
        """Check if at least one embedding provider is healthy."""
        test_text = "health check"
        
        try:
            embedding = await self.generate_embedding(test_text)
            if embedding and len(embedding) == self.vector_size:
                logger.info("Embedding service health check: PASS")
                return True
        except Exception as e:
            logger.warning(f"Embedding health check failed: {e}")
        
        return False


# Global embedding service instance
_embedding_service: Optional[MultiProviderEmbeddingService] = None


def get_embedding_service() -> MultiProviderEmbeddingService:
    """
    Get or create the global embedding service instance.

    Returns:
        MultiProviderEmbeddingService: Singleton embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = MultiProviderEmbeddingService()
    return _embedding_service

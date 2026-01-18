"""
LLM service using LiteLLM with Gemini API and response caching.
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
import time
import logging
import hashlib
import google.generativeai as genai
from rag_backend.config import settings
from rag_backend.utils.error_handlers import LLMGenerationError, ServiceUnavailable

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM generation with caching and rate limiting."""

    def __init__(self):
        """Initialize LLM service with Gemini."""
        try:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
            self.generation_config = {
                "temperature": settings.gemini_temperature,
                "max_output_tokens": settings.gemini_max_tokens,
            }
            self._query_cache = {}  # Simple in-memory cache
            self._cache_timestamps = {}  # Track cache entry times
            logger.info(f"LLMService initialized with model: {settings.gemini_model}", extra={"model_name": settings.gemini_model})
        except Exception as e:
            logger.error(
                f"Failed to initialize LLMService: {e}",
                exc_info=True,
                extra={"error_type": "InitializationError", "exception_message": str(e)}
            )
            raise ServiceUnavailable("gemini", f"Failed to initialize Gemini: {e}")

    def _get_cache_key(self, prompt: str, context: str) -> str:
        """Generate cache key for a query."""
        content = f"{prompt}|||{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid (within TTL)."""
        if cache_key not in self._cache_timestamps:
            return False

        timestamp = self._cache_timestamps[cache_key]
        age = time.time() - timestamp

        return age < settings.cache_ttl_seconds

    def _clean_expired_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > settings.cache_ttl_seconds
        ]

        for key in expired_keys:
            self._query_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate response using Gemini with retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks from vector search
            system_prompt: Optional system prompt override
            use_cache: Whether to use caching (default: True)

        Returns:
            str: Generated response

        Raises:
            LLMGenerationError: If generation fails
        """
        try:
            # Build context string from chunks
            context_str = self._build_context_string(context_chunks)

            # Check cache first
            cache_key = self._get_cache_key(query, context_str)
            if use_cache and cache_key in self._query_cache and self._is_cache_valid(cache_key):
                logger.info("Cache hit for query")
                return self._query_cache[cache_key]

            # Clean expired cache entries periodically
            if len(self._cache_timestamps) > settings.cache_max_entries:
                self._clean_expired_cache()

            # Build prompt
            prompt = self._build_prompt(query, context_str, system_prompt)

            # Generate response with retry logic
            response = await self._generate_with_retry(prompt)

            # Cache the response
            if use_cache:
                self._query_cache[cache_key] = response
                self._cache_timestamps[cache_key] = time.time()

                # Enforce max cache size (LRU-like behavior)
                if len(self._query_cache) > settings.cache_max_entries:
                    oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                    self._query_cache.pop(oldest_key, None)
                    self._cache_timestamps.pop(oldest_key, None)
                    logger.debug(f"Evicted oldest cache entry: {oldest_key[:8]}...")

            return response

        except Exception as e:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            logger.error(
                f"LLM generation failed for query {query_hash[:8]}: {e}",
                exc_info=True,
                extra={
                    "error_type": "LLMGenerationError",
                    "query_hash": query_hash,
                    "context_chunks_count": len(context_chunks),
                    "exception_message": str(e)
                }
            )
            raise LLMGenerationError(f"Failed to generate response: {e}")

    def _build_context_string(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        if not context_chunks:
            return "No relevant context found in the textbook."

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            week = chunk.get("week", "Unknown")
            module = chunk.get("module", "Unknown")
            content = chunk.get("content", "")
            chunk_index = chunk.get("chunk_index", 0)
            total_chunks = chunk.get("total_chunks", 1)

            context_parts.append(
                f"[Context {i} - Week {week}, {module}, Part {chunk_index+1} of {total_chunks}]\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Build the full prompt for the LLM."""
        default_system_prompt = """You are an AI assistant for a Physical AI textbook. Your role is to answer questions based ONLY on the provided context from the textbook.

IMPORTANT RULES:
1. ONLY use information from the provided context to answer questions
2. If the context doesn't contain enough information to answer the question, respond with: "I couldn't find information about that in the textbook."
3. Always cite your sources by referencing the Week and section mentioned in the context
4. Be concise but accurate
5. Do not make up information or use knowledge outside the provided context
6. If the question is ambiguous, ask for clarification or provide the most relevant information from the context

Remember: Your credibility depends on accurate, cited answers from the textbook content only."""

        system = system_prompt or default_system_prompt

        prompt = f"""{system}

CONTEXT FROM TEXTBOOK:
{context}

USER QUESTION:
{query}

YOUR ANSWER:"""

        return prompt

    async def _generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> str:
        """
        Generate response with exponential backoff retry.

        Args:
            prompt: Full prompt
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            str: Generated response

        Raises:
            LLMGenerationError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )

                if not response or not response.text:
                    raise LLMGenerationError("Empty response from Gemini")

                return response.text.strip()

            except Exception as e:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                logger.warning(
                    f"Generation attempt {attempt + 1} failed for prompt {prompt_hash[:8]}: {e}",
                    exc_info=True,
                    extra={
                        "error_type": "GenerationAttemptFailed",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "prompt_hash": prompt_hash,
                        "exception_message": str(e)
                    }
                )

                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...", extra={"next_attempt_delay": delay})
                    time.sleep(delay)
                else:
                    raise LLMGenerationError(f"All retry attempts failed: {e}")

    async def health_check(self) -> bool:
        """
        Check if LLM service is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            test_response = self.model.generate_content(
                "Respond with 'OK' if you're working.",
                generation_config={"max_output_tokens": 10}
            )
            return bool(test_response and test_response.text)
        except Exception as e:
            logger.error(
                f"LLM health check failed: {e}",
                exc_info=True,
                extra={"error_type": "HealthCheckFailed", "exception_message": str(e)}
            )
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._query_cache),
            "max_cache_size": settings.cache_max_entries,
            "cache_ttl_seconds": settings.cache_ttl_seconds
        }


# Global LLM service instance
_llm_service: LLMService = None


def get_llm_service() -> LLMService:
    """
    Get or create the global LLM service instance.

    Returns:
        LLMService: Singleton LLM service instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

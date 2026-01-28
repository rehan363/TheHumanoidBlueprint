"""
Multi-provider LLM service supporting Gemini and OpenRouter with automatic fallback.
"""

from typing import List, Dict, Any, Optional, Literal
import time
import logging
import hashlib
from openai import AsyncOpenAI
import google.generativeai as genai

from rag_backend.config import settings
from rag_backend.utils.error_handlers import LLMGenerationError, ServiceUnavailable

logger = logging.getLogger(__name__)

ProviderType = Literal["gemini", "openrouter_deepseek", "openrouter_mistral", "auto"]


class MultiProviderLLMService:
    """LLM service supporting multiple providers with automatic fallback."""

    def __init__(self):
        """Initialize LLM service with all available providers."""
        self.provider = settings.llm_provider
        self._query_cache = {}
        self._cache_timestamps = {}
        
        # Initialize Gemini if keys available
        self.gemini_available = False
        if settings.gemini_api_key_1 or settings.gemini_api_key_2:
            try:
                genai.configure(api_key=settings.gemini_api_key_1 or settings.gemini_api_key_2)
                self.gemini_model = genai.GenerativeModel(settings.gemini_model)
                self.gemini_config = {
                    "temperature": settings.gemini_temperature,
                    "max_output_tokens": settings.gemini_max_tokens,
                }
                self.gemini_available = True
                logger.info(f"Gemini initialized: {settings.gemini_model}")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
        
        # Initialize OpenRouter if key available
        self.openrouter_available = False
        if settings.openrouter_api_key:
            try:
                self.openrouter_client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key,
                )
                self.openrouter_available = True
                logger.info(f"OpenRouter initialized with DeepSeek: {settings.deepseek_model}, Mistral: {settings.mistral_model}")
            except Exception as e:
                logger.warning(f"OpenRouter initialization failed: {e}")
        
        # Log provider status
        logger.info(f"LLM Provider Strategy: {self.provider}")
        logger.info(f"Available providers: Gemini={self.gemini_available}, OpenRouter={self.openrouter_available}")
        
        if not self.gemini_available and not self.openrouter_available:
            raise ServiceUnavailable("llm", "No LLM providers available")

    def _get_cache_key(self, prompt: str, context: str) -> str:
        """Generate cache key for a query."""
        content = f"{prompt}|||{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        age = time.time() - self._cache_timestamps[cache_key]
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
        user_profile: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate response using available LLM providers with fallback.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_prompt: Optional system prompt override
            user_profile: Optional user profile data for personalization
            use_cache: Whether to use caching
            
        Returns:
            Generated response
        """
        try:
            context_str = self._build_context_string(context_chunks)
            
            # Check cache - include profile in cache key if present
            cache_key = self._get_cache_key(query, context_str + str(user_profile))
            if use_cache and cache_key in self._query_cache and self._is_cache_valid(cache_key):
                logger.info("Cache hit for query")
                return self._query_cache[cache_key]
            
            # Clean cache periodically
            if len(self._cache_timestamps) > settings.cache_max_entries:
                self._clean_expired_cache()
            
            # Build prompt
            prompt = self._build_prompt(query, context_str, system_prompt, user_profile)
            
            # Try providers based on strategy
            response = await self._generate_with_fallback(prompt)
            
            # Cache response
            if use_cache:
                self._query_cache[cache_key] = response
                self._cache_timestamps[cache_key] = time.time()
                
                # Enforce max cache size
                if len(self._query_cache) > settings.cache_max_entries:
                    oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                    self._query_cache.pop(oldest_key, None)
                    self._cache_timestamps.pop(oldest_key, None)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise LLMGenerationError(f"Failed to generate response: {e}")

    async def _generate_with_fallback(self, prompt: str) -> str:
        """Generate response with provider fallback logic."""
        
        if self.provider == "auto":
            # Try Gemini first, fallback to OpenRouter
            if self.gemini_available:
                try:
                    return await self._generate_gemini(prompt)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "quota" in error_str.lower():
                        logger.warning("Gemini quota exceeded, falling back to OpenRouter")
                        if self.openrouter_available:
                            return await self._generate_openrouter(prompt, "deepseek")
                    raise
            elif self.openrouter_available:
                return await self._generate_openrouter(prompt, "deepseek")
                
        elif self.provider == "gemini":
            if not self.gemini_available:
                raise ServiceUnavailable("gemini", "Gemini not available")
            return await self._generate_gemini(prompt)
            
        elif self.provider == "openrouter_deepseek":
            if not self.openrouter_available:
                raise ServiceUnavailable("openrouter", "OpenRouter not available")
            return await self._generate_openrouter(prompt, "deepseek")
            
        elif self.provider == "openrouter_mistral":
            if not self.openrouter_available:
                raise ServiceUnavailable("openrouter", "OpenRouter not available")
            return await self._generate_openrouter(prompt, "mistral")
        
        raise LLMGenerationError(f"Unknown provider: {self.provider}")

    async def _generate_gemini(self, prompt: str) -> str:
        """Generate using Gemini."""
        logger.info("Generating with Gemini")
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=self.gemini_config
        )
        if not response or not response.text:
            raise LLMGenerationError("Empty response from Gemini")
        return response.text.strip()

    async def _generate_openrouter(self, prompt: str, model: str = "deepseek") -> str:
        """Generate using OpenRouter."""
        model_name = settings.deepseek_model if model == "deepseek" else settings.mistral_model
        logger.info(f"Generating with OpenRouter ({model_name})")
        
        response = await self.openrouter_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
        )
        
        if not response.choices or not response.choices[0].message.content:
            raise LLMGenerationError("Empty response from OpenRouter")
        
        return response.choices[0].message.content.strip()

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
        system_prompt: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the full prompt for the LLM."""
        default_system_prompt = """You are an AI assistant for a Physical AI textbook. Your role is to answer questions based ONLY on the provided context from the textbook.

IMPORTANT RULES:
1. ONLY use information from the provided context to answer questions
2. If the context doesn't contain enough information to answer the question, respond with: "I couldn't find information about that in the textbook."
3. Always cite your sources by referencing the Week and section mentioned in the context
4. Be concise but accurate
5. Do not make up information or use knowledge outside the provided context
6. If the question is ambiguous, ask for clarification or provide the most relevant information from the context"""

        if user_profile:
            python_exp = user_profile.get("python_experience", 5)
            hardware_exp = user_profile.get("hardware_experience", 5)
            name = user_profile.get("name", "Student")
            
            # Personalization Logic
            personalization = f"\n\nUSER PROFILE:\n- Name: {name}\n- Python Level: {python_exp}/10\n- Hardware/Robotics Level: {hardware_exp}/10\n"
            
            if python_exp < 3:
                personalization += "- Keep coding concepts extremely simple and explain all Python syntax.\n"
            elif python_exp > 8:
                personalization += "- You can use advanced technical terminology and complex code snippets.\n"
                
            if hardware_exp < 3:
                personalization += "- Explain physical components (servos, sensors) in basic terms.\n"
            
            default_system_prompt += personalization

        system = system_prompt or default_system_prompt

        prompt = f"""{system}

CONTEXT FROM TEXTBOOK:
{context}

USER QUESTION:
{query}

YOUR ANSWER:"""

        return prompt

    async def health_check(self) -> bool:
        """Check if at least one LLM provider is healthy."""
        if self.gemini_available:
            try:
                test_response = self.gemini_model.generate_content(
                    "Respond with 'OK' if you're working.",
                    generation_config={"max_output_tokens": 10}
                )
                if test_response and test_response.text:
                    logger.info("Gemini health check: PASS")
                    return True
            except Exception as e:
                logger.warning(f"Gemini health check failed: {e}")
        
        if self.openrouter_available:
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=settings.deepseek_model,
                    messages=[{"role": "user", "content": "Respond with 'OK'"}],
                    max_tokens=10
                )
                if response.choices:
                    logger.info("OpenRouter health check: PASS")
                    return True
            except Exception as e:
                logger.warning(f"OpenRouter health check failed: {e}")
        
        return False


# Global LLM service instance
_llm_service: Optional[MultiProviderLLMService] = None


def get_llm_service() -> MultiProviderLLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = MultiProviderLLMService()
    return _llm_service

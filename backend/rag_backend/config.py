"""
Configuration management for RAG backend.

This module handles all environment variables and application settings
using Pydantic Settings for validation and type safety.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Gemini API Configuration
    gemini_api_key: str = ""  # Will default to gemini_api_key_1 if not set
    gemini_api_key_1: str = ""  # Primary key for orchestrator and some sub-agents
    gemini_api_key_2: str = ""  # Secondary key for embeddings and other sub-agents
    gemini_api_key_3: str = ""  # Tertiary key for remaining sub-agents
    new_gemini_api_key: str = ""  # New fresh API key (highest priority)
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_embedding_model: str = "models/text-embedding-004"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 1024

    # OpenRouter Configuration
    openrouter_api_key: str = ""
    deepseek_model: str = "tngtech/deepseek-r1t2-chimera:free"
    mistral_model: str = "mistralai/devstral-2512:free"
    
    # LLM Provider Strategy
    # Options: "gemini", "openrouter_deepseek", "openrouter_mistral", "auto"
    # "auto" will try Gemini first, fallback to OpenRouter if quota exceeded
    llm_provider: str = "auto"
    primary_llm_model: str = "gemini-2.0-flash-exp"  # Used when provider is "auto"

    # Embedding Provider Configuration
    # Options: "huggingface", "jina", "gemini", "auto"
    # "auto" will try HuggingFace first (free, no key required)
    embedding_provider: str = "auto"
    huggingface_api_key: str = ""  # Optional - works without key for public models
    huggingface_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # 768 dims
    jina_api_key: str = ""  # Optional - for Jina AI embeddings

    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "physical_ai_textbook"
    qdrant_vector_size: int = 768  # Google text-embedding-004 dimensions

    # Neon Postgres Configuration
    neon_database_url: str
    neon_pool_size: int = 10
    neon_max_overflow: int = 5

    # Rate Limiting Configuration
    rate_limit_per_minute: int = 3
    rate_limit_strategy: str = "fixed-window"

    # Caching Configuration
    cache_max_entries: int = 100
    cache_ttl_seconds: int = 300  # 5 minutes

    # RAG Pipeline Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.3
    chunk_size: int = 768
    chunk_overlap: int = 100

    # Admin Configuration
    admin_token: Optional[str] = None

    # Application Configuration
    app_name: str = "Physical AI Textbook RAG Backend"
    app_version: str = "1.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"

    # CORS Configuration
    cors_origins: list[str] = [
        "http://localhost:3000",
        "https://rehan363.github.io",
        "https://panaversity-hackathon-i.vercel.app"
    ]

    # Documentation Path
    docs_base_path: str = "../physical-ai-textbook/docs"

    # API Key Properties (computed)
    orchestrator_api_key: Optional[str] = None
    embedding_api_key: Optional[str] = None
    retrieval_agent_api_key: Optional[str] = None
    explanation_agent_api_key: Optional[str] = None
    comparison_agent_api_key: Optional[str] = None
    clarification_agent_api_key: Optional[str] = None
    summary_agent_api_key: Optional[str] = None

    def __init__(self, **values):
        """Initialize settings and allocate API keys."""
        super().__init__(**values)

        # Use gemini_api_key_1 as default if gemini_api_key is empty
        if not self.gemini_api_key and self.gemini_api_key_1:
            self.gemini_api_key = self.gemini_api_key_1

        # Use new_gemini_api_key as primary if available
        primary_key = self.new_gemini_api_key or self.gemini_api_key_1 or self.gemini_api_key

        # Allocate keys across services - prioritize new key for all
        self.orchestrator_api_key = primary_key
        self.embedding_api_key = primary_key  # Use new key for embeddings
        self.retrieval_agent_api_key = primary_key
        self.explanation_agent_api_key = primary_key
        self.comparison_agent_api_key = primary_key
        self.clarification_agent_api_key = primary_key
        self.summary_agent_api_key = primary_key

        logger.info("="*60)
        logger.info("API Keys Allocated:")
        if self.embedding_api_key:
            logger.info(f"  - Embeddings: KEY_2 (***{self.embedding_api_key[-10:]}***)")
        if self.orchestrator_api_key:
            logger.info(f"  - Orchestrator: KEY_1 (***{self.orchestrator_api_key[-10:]}***)")
        if self.retrieval_agent_api_key:
            logger.info(f"  - Retrieval Agent: KEY_1")
        if self.explanation_agent_api_key:
            logger.info(f"  - Explanation Agent: KEY_1")
        if self.comparison_agent_api_key:
            logger.info(f"  - Comparison Agent: KEY_1")
        if self.clarification_agent_api_key:
            logger.info(f"  - Clarification Agent: KEY_3 (***{self.clarification_agent_api_key[-10:]}***)")
        if self.summary_agent_api_key:
            logger.info(f"  - Summary Agent: KEY_3 (***{self.summary_agent_api_key[-10:]}***)")
        logger.info("="*60)


# Global settings instance
settings = Settings()

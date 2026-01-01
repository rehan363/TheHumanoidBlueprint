"""
RAG pipeline orchestrating retrieval and generation.
"""

from typing import List, Optional, Tuple
import time
import logging
from rag_backend.services.embedding_service import get_embedding_service
from rag_backend.services.vector_store import get_vector_store
from rag_backend.services.llm_service import get_llm_service
from rag_backend.models.chat import ChatQueryRequest, ChatQueryResponse, Citation
from rag_backend.config import settings
from rag_backend.utils.error_handlers import (
    EmbeddingGenerationError,
    VectorSearchError,
    LLMGenerationError,
    ServiceUnavailable
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering with context retrieval."""

    def __init__(self):
        """Initialize RAG pipeline with all required services."""
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.llm_service = get_llm_service()
        logger.info("RAG Pipeline initialized")

    async def process_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """
        Process a chat query through the RAG pipeline.

        Args:
            request: Chat query request

        Returns:
            ChatQueryResponse: Response with answer and citations

        Raises:
            Various exceptions for different failure modes
        """
        start_time = time.time()

        try:
            # Step 1: Generate query embedding
            logger.info(f"Processing {request.query_type} query: {request.query[:50]}...")
            query_embedding = await self.embedding_service.generate_query_embedding(request.query)

            # Step 2: Retrieve relevant chunks
            context_chunks = await self._retrieve_context(query_embedding, request)

            # Step 3: Handle case with no relevant content
            if not context_chunks:
                logger.warning("No relevant chunks found for query")
                processing_time_ms = int((time.time() - start_time) * 1000)
                return ChatQueryResponse(
                    answer="I couldn't find information about that in the textbook. Could you try rephrasing your question or ask about a different topic covered in the course?",
                    citations=[],
                    query_type=request.query_type,
                    session_id=request.session_id,
                    processing_time_ms=processing_time_ms
                )

            # Step 4: Enhance context for text_selection queries
            if request.query_type == "text_selection" and request.context:
                context_chunks = await self._enhance_with_selected_text(context_chunks, request.context)

            # Step 5: Generate response using LLM
            answer = await self.llm_service.generate_response(
                query=request.query,
                context_chunks=context_chunks
            )

            # Step 6: Extract citations
            citations = self._extract_citations(context_chunks)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"Query processed successfully in {processing_time_ms}ms with {len(citations)} citations")

            return ChatQueryResponse(
                answer=answer,
                citations=citations,
                query_type=request.query_type,
                session_id=request.session_id,
                processing_time_ms=processing_time_ms
            )

        except (EmbeddingGenerationError, VectorSearchError, LLMGenerationError) as e:
            logger.error(f"RAG pipeline error: {e}")
            raise

        except Exception as e:
            logger.exception(f"Unexpected error in RAG pipeline: {e}")
            raise ServiceUnavailable("rag_pipeline", str(e))

    async def _retrieve_context(
        self,
        query_embedding: List[float],
        request: ChatQueryRequest
    ) -> List[dict]:
        """
        Retrieve relevant context chunks from vector store.

        Args:
            query_embedding: Query vector embedding
            request: Original request (for potential filtering)

        Returns:
            List of context chunks with metadata
        """
        try:
            # Perform vector search
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=settings.top_k_results,
                score_threshold=settings.similarity_threshold
            )

            logger.info(f"Retrieved {len(results)} relevant chunks (threshold={settings.similarity_threshold})")
            return results

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            raise VectorSearchError(f"Failed to retrieve context: {e}")

    async def _enhance_with_selected_text(
        self,
        context_chunks: List[dict],
        selected_context
    ) -> List[dict]:
        """
        Enhance context with selected text for text_selection queries.

        Args:
            context_chunks: Retrieved context chunks
            selected_context: Selected text context

        Returns:
            Enhanced context chunks with selected text prepended
        """
        # Create a special "selected text" chunk with high relevance
        selected_chunk = {
            "id": "selected_text",
            "score": 1.0,  # Maximum relevance
            "content": selected_context.text,
            "week": None,
            "module": "Selected Text",
            "file_path": selected_context.file_path,
            "chunk_index": 0,
            "total_chunks": 1,
            "heading_path": ["User Selected Text"]
        }

        # Prepend selected text to context
        enhanced_chunks = [selected_chunk] + context_chunks

        logger.info("Enhanced context with selected text")
        return enhanced_chunks

    def _extract_citations(self, context_chunks: List[dict]) -> List[Citation]:
        """
        Extract citations from context chunks.

        Args:
            context_chunks: Context chunks used for generation

        Returns:
            List of Citation objects
        """
        citations = []

        for chunk in context_chunks:
            # Build source reference
            if chunk.get("module") == "Selected Text":
                source = f"Selected Text"
            else:
                week = chunk.get("week", "Unknown")
                module = chunk.get("module", "Unknown")
                source = f"Week {week}, {module}"

            # Get chunk position info
            chunk_index = chunk.get("chunk_index", 0)
            total_chunks = chunk.get("total_chunks", 1)

            # Add positional info if multiple chunks
            if total_chunks > 1:
                source += f", Part {chunk_index + 1} of {total_chunks}"

            # Create citation
            citation = Citation(
                source=source,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                relevance_score=round(chunk.get("score", 0.0), 3),
                file_path=chunk.get("file_path"),
                content_preview=chunk.get("content", "")[:200]  # First 200 chars
            )

            citations.append(citation)

        return citations

    async def health_check(self) -> dict:
        """
        Check health of all pipeline services.

        Returns:
            dict: Health status of each service
        """
        health_status = {
            "qdrant": await self.vector_store.health_check(),
            "gemini": await self.llm_service.health_check(),
        }

        logger.info(f"Pipeline health check: {health_status}")
        return health_status


# Global RAG pipeline instance
_rag_pipeline: RAGPipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the global RAG pipeline instance.

    Returns:
        RAGPipeline: Singleton RAG pipeline instance
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline

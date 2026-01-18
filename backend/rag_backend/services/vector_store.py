"""
Vector store service using Qdrant for similarity search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import logging
from rag_backend.config import settings
from rag_backend.utils.error_handlers import VectorSearchError, ServiceUnavailable
from rag_backend.models.chunk import TextChunk, ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStore:
    """Service for vector similarity search using Qdrant."""

    def __init__(self):
        """Initialize Qdrant client."""
        try:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=10.0
            )
            self.collection_name = settings.qdrant_collection_name
            logger.info(f"VectorStore initialized with collection: {self.collection_name}", extra={"collection_name": self.collection_name, "qdrant_url": settings.qdrant_url})
        except Exception as e:
            logger.error(
                f"Failed to initialize VectorStore: {e}",
                exc_info=True,
                extra={"error_type": "InitializationError", "qdrant_url": settings.qdrant_url, "exception_message": str(e)}
            )
            raise ServiceUnavailable("qdrant", f"Failed to connect to Qdrant: {e}")

    async def health_check(self) -> bool:
        """
        Check if Qdrant service is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            collections = self.client.get_collections()
            logger.info("Qdrant health check successful", extra={"status": "healthy", "collection_name": self.collection_name})
            return True
        except Exception as e:
            logger.error(
                f"Qdrant health check failed: {e}",
                exc_info=True,
                extra={"error_type": "HealthCheckFailed", "collection_name": self.collection_name, "exception_message": str(e)}
            )
            return False

    async def collection_exists(self) -> bool:
        """
        Check if the collection exists.

        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            logger.debug(f"Collection {self.collection_name} exists: {exists}", extra={"collection_name": self.collection_name, "exists": exists})
            return exists
        except Exception as e:
            logger.error(
                f"Failed to check collection existence for {self.collection_name}: {e}",
                exc_info=True,
                extra={"error_type": "CollectionExistenceCheckFailed", "collection_name": self.collection_name, "exception_message": str(e)}
            )
            return False

    async def create_collection(self) -> bool:
        """
        Create the Qdrant collection with proper configuration.

        Returns:
            bool: True if created successfully, False otherwise
        """
        try:
            if await self.collection_exists():
                logger.info(f"Collection {self.collection_name} already exists", extra={"collection_name": self.collection_name})
                return True

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance.COSINE
                )
            )

            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="week",
                field_schema="integer"
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="module",
                field_schema="keyword"
            )

            logger.info(f"Created collection: {self.collection_name}", extra={"collection_name": self.collection_name, "vector_size": settings.qdrant_vector_size, "distance": Distance.COSINE.value})
            return True

        except Exception as e:
            logger.error(
                f"Failed to create collection {self.collection_name}: {e}",
                exc_info=True,
                extra={"error_type": "CollectionCreationError", "collection_name": self.collection_name, "exception_message": str(e)}
            )
            raise VectorSearchError(f"Failed to create collection: {e}")

    async def upsert_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Upsert text chunks into the vector store.

        Args:
            chunks: List of TextChunk objects with embeddings

        Returns:
            int: Number of chunks upserted

        Raises:
            VectorSearchError: If upsert operation fails
        """
        try:
            if not chunks:
                logger.warning("No chunks to upsert")
                return 0

            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Skipping chunk {chunk.chunk_id} without embedding")
                    continue

                payload = {
                    "content": chunk.content,
                    "week": chunk.metadata.week,
                    "module": chunk.metadata.module,
                    "file_path": chunk.metadata.file_path,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks,
                    "heading_path": chunk.metadata.heading_path or [],
                    "token_count": chunk.token_count or 0
                }

                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload=payload
                )
                points.append(point)

            if not points:
                logger.warning("No valid points to upsert", extra={"num_input_chunks": len(chunks), "collection_name": self.collection_name})
                return 0

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Upserted {len(points)} chunks to Qdrant", extra={"num_upserted_points": len(points), "collection_name": self.collection_name})
            return len(points)

        except Exception as e:
            logger.error(
                f"Failed to upsert chunks to collection {self.collection_name}: {e}",
                exc_info=True,
                extra={"error_type": "UpsertChunksError", "collection_name": self.collection_name, "num_input_chunks": len(chunks), "exception_message": str(e)}
            )
            raise VectorSearchError(f"Failed to upsert chunks: {e}")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        score_threshold: float = None,
        week_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector store.

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return (default: from settings)
            score_threshold: Minimum similarity score (default: from settings)
            week_filter: Optional week number to filter results

        Returns:
            List[Dict]: List of search results with content and metadata

        Raises:
            VectorSearchError: If search operation fails
        """
        try:
            top_k = top_k or settings.top_k_results
            score_threshold = score_threshold or settings.similarity_threshold

            # Build filter if week is specified
            query_filter = None
            if week_filter is not None:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="week",
                            match=MatchValue(value=week_filter)
                        )
                    ]
                )

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            ).points

            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "content": scored_point.payload.get("content", ""),
                    "week": scored_point.payload.get("week"),
                    "module": scored_point.payload.get("module", ""),
                    "file_path": scored_point.payload.get("file_path", ""),
                    "chunk_index": scored_point.payload.get("chunk_index", 0),
                    "total_chunks": scored_point.payload.get("total_chunks", 1),
                    "heading_path": scored_point.payload.get("heading_path", []),
                }
                results.append(result)

            logger.info(
                f"Found {len(results)} results for query (top_k={top_k}, threshold={score_threshold})",
                extra={"collection_name": self.collection_name, "num_results": len(results), "top_k": top_k, "score_threshold": score_threshold, "week_filter": week_filter}
            )
            return results

        except Exception as e:
            logger.error(
                f"Vector search failed for collection {self.collection_name}: {e}",
                exc_info=True,
                extra={"error_type": "VectorSearchError", "collection_name": self.collection_name, "top_k": top_k, "score_threshold": score_threshold, "week_filter": week_filter, "exception_message": str(e)}
            )
            raise VectorSearchError(f"Search operation failed: {e}")

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            Dict: Chunk data or None if not found

        Raises:
            VectorSearchError: If retrieval fails
        """
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True
            )

            if not point:
                return None

            return {
                "id": point[0].id,
                "content": point[0].payload.get("content", ""),
                "metadata": point[0].payload
            }

        except Exception as e:
            logger.error(
                f"Failed to retrieve chunk {chunk_id} from collection {self.collection_name}: {e}",
                exc_info=True,
                extra={"error_type": "RetrieveChunkError", "chunk_id": chunk_id, "collection_name": self.collection_name, "exception_message": str(e)}
            )
            raise VectorSearchError(f"Failed to retrieve chunk: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict: Collection statistics (count, size, etc.)
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status.value
            }
        except Exception as e:
            logger.error(
                f"Failed to get collection stats for {self.collection_name}: {e}",
                exc_info=True,
                extra={"error_type": "CollectionStatsError", "collection_name": self.collection_name, "exception_message": str(e)}
            )
            return {}


# Global vector store instance
_vector_store: VectorStore = None


def get_vector_store() -> VectorStore:
    """
    Get or create the global vector store instance.

    Returns:
        VectorStore: Singleton vector store instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

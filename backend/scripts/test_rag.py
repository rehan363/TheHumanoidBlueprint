"""
Test script for RAG retrieval from Qdrant
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import rag_backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_backend.services.vector_store import get_vector_store
from rag_backend.services.embedding_service import get_embedding_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_retrieval():
    try:
        query = "What is ROS2?"
        logger.info(f"Testing retrieval for query: '{query}'")
        
        # 1. Get embedding
        embed_service = get_embedding_service()
        query_embedding = await embed_service.generate_query_embedding(query)
        
        # 2. Search in Qdrant
        vector_store = get_vector_store()
        results = await vector_store.search(query_embedding, top_k=3)
        
        if not results:
            logger.error("❌ No results found!")
            return False
            
        logger.info(f"✅ Found {len(results)} results:")
        for i, res in enumerate(results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Score: {res['score']:.4f}")
            logger.info(f"  Chapter: {res['chapter']}")
            logger.info(f"  Module: {res['module']}")
            logger.info(f"  Snippet: {res['content'][:100]}...")
            
        return True
    except Exception as e:
        logger.exception(f"❌ Retrieval test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_retrieval())

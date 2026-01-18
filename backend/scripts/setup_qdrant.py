#!/usr/bin/env python3
"""
Script to initialize Qdrant collection for the RAG system.

Usage:
    python scripts/setup_qdrant.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import rag_backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_backend.services.vector_store import get_vector_store
from rag_backend.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_collection():
    """Initialize Qdrant collection with proper configuration."""
    try:
        logger.info("Initializing Qdrant collection setup...")

        vector_store = get_vector_store()

        # Check if collection exists
        exists = await vector_store.collection_exists()
        if exists:
            logger.info(f"Collection '{settings.qdrant_collection_name}' already exists")

            # Get stats
            stats = await vector_store.get_collection_stats()
            logger.info(f"Collection stats: {stats}")

            # Delete existing collection forcefully for setup
            logger.info("Deleting existing collection to reset schema...")
            vector_store.client.delete_collection(settings.qdrant_collection_name)
            logger.info("Existing collection deleted")

        # Create new collection
        logger.info(f"Creating collection '{settings.qdrant_collection_name}'...")
        success = await vector_store.create_collection()

        if success:
            logger.info("✅ Qdrant collection created successfully!")
            logger.info(f"   Collection name: {settings.qdrant_collection_name}")
            logger.info(f"   Vector size: {settings.qdrant_vector_size}")
            logger.info(f"   Distance metric: COSINE")
            logger.info(f"   Indexed fields: chapter, module")

            # Verify collection
            stats = await vector_store.get_collection_stats()
            logger.info(f"\nCollection stats: {stats}")

            return True
        else:
            logger.error("❌ Failed to create collection")
            return False

    except Exception as e:
        logger.exception(f"Error during setup: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Qdrant Collection Setup")
    logger.info("="*60)
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.qdrant_collection_name}")
    logger.info("="*60)

    success = asyncio.run(setup_collection())

    if success:
        logger.info("\n✅ Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs/")
        logger.info("  2. Start backend: uvicorn rag_backend.main:app --reload")
        sys.exit(0)
    else:
        logger.error("\n❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI tool to index documentation into Qdrant vector database.

Usage:
    python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs/
    python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs/ --week 3
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_backend.services.embedding_service import get_embedding_service
from rag_backend.services.vector_store import get_vector_store
from rag_backend.utils.chunking import get_chunker
from rag_backend.models.chunk import TextChunk
from rag_backend.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Document indexer for the RAG system."""

    def __init__(self):
        """Initialize indexer with services."""
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.chunker = get_chunker()
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }

    def discover_markdown_files(self, docs_path: Path, week_filter: int = None) -> List[Dict[str, Any]]:
        """
        Discover markdown files in the docs directory.

        Args:
            docs_path: Base docs directory
            week_filter: Optional week number to process only specific week

        Returns:
            List of file info dicts with path, week, module
        """
        files = []

        # Pattern: docs/week-{N}/README.md or docs/week-{N}/{topic}.md
        for week_dir in sorted(docs_path.glob("week-*")):
            if not week_dir.is_dir():
                continue

            # Extract week number
            try:
                week_num = int(week_dir.name.split("-")[1])
            except (IndexError, ValueError):
                logger.warning(f"Invalid week directory name: {week_dir.name}")
                continue

            # Apply week filter
            if week_filter is not None and week_num != week_filter:
                continue

            # Find markdown files in this week
            for md_file in week_dir.glob("*.md"):
                module_name = md_file.stem.replace("-", " ").title()
                if module_name.lower() == "readme":
                    module_name = f"Week {week_num} Overview"

                files.append({
                    "path": md_file,
                    "week": week_num,
                    "module": module_name
                })

        logger.info(f"Discovered {len(files)} markdown files")
        return files

    async def index_file(self, file_info: Dict[str, Any]) -> int:
        """
        Index a single markdown file.

        Args:
            file_info: File information dict

        Returns:
            int: Number of chunks indexed
        """
        try:
            file_path = file_info["path"]
            week = file_info["week"]
            module = file_info["module"]

            logger.info(f"Processing: Week {week} - {module} ({file_path.name})")

            # Chunk the file
            chunks = self.chunker.chunk_file(file_path, week, module)
            self.stats["chunks_created"] += len(chunks)

            if not chunks:
                logger.warning(f"No chunks created for {file_path.name}")
                return 0

            # Generate embeddings for each chunk
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            for chunk in chunks:
                try:
                    embedding = await self.embedding_service.generate_embedding(chunk.content)
                    chunk.embedding = embedding
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
                    self.stats["errors"] += 1

            # Filter chunks with embeddings
            chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
            logger.info(f"Generated {len(chunks_with_embeddings)} embeddings successfully")

            # Upsert to vector store
            if chunks_with_embeddings:
                indexed_count = await self.vector_store.upsert_chunks(chunks_with_embeddings)
                self.stats["chunks_indexed"] += indexed_count
                logger.info(f"✅ Indexed {indexed_count} chunks from {file_path.name}")
                return indexed_count

            return 0

        except Exception as e:
            logger.exception(f"Error indexing file {file_info['path']}: {e}")
            self.stats["errors"] += 1
            return 0

    async def index_all(self, docs_path: Path, week_filter: int = None) -> bool:
        """
        Index all markdown files in the docs directory.

        Args:
            docs_path: Base docs directory
            week_filter: Optional week number to process only specific week

        Returns:
            bool: True if successful
        """
        try:
            self.stats["start_time"] = datetime.utcnow()

            # Discover files
            files = self.discover_markdown_files(docs_path, week_filter)

            if not files:
                logger.warning("No markdown files found to index")
                return False

            # Index each file
            for file_info in files:
                await self.index_file(file_info)
                self.stats["files_processed"] += 1

            self.stats["end_time"] = datetime.utcnow()

            # Save metadata
            await self.save_metadata()

            return True

        except Exception as e:
            logger.exception(f"Error during indexing: {e}")
            return False

    async def save_metadata(self):
        """Save indexing metadata to JSON file."""
        try:
            metadata_path = Path(__file__).parent.parent / "data" / "index_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Get collection stats
            collection_stats = await self.vector_store.get_collection_stats()

            metadata = {
                "last_indexed": self.stats["end_time"].isoformat() if self.stats["end_time"] else None,
                "files_processed": self.stats["files_processed"],
                "chunks_created": self.stats["chunks_created"],
                "chunks_indexed": self.stats["chunks_indexed"],
                "errors": self.stats["errors"],
                "collection_stats": collection_stats,
                "settings": {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "embedding_model": settings.gemini_embedding_model,
                    "vector_size": settings.qdrant_vector_size
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def print_summary(self):
        """Print indexing summary."""
        logger.info("=" * 60)
        logger.info("INDEXING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Chunks created: {self.stats['chunks_created']}")
        logger.info(f"Chunks indexed: {self.stats['chunks_indexed']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            logger.info(f"Duration: {duration:.2f} seconds")

        logger.info("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index textbook documentation into Qdrant")
    parser.add_argument(
        "--docs-path",
        type=str,
        required=True,
        help="Path to the docs directory (e.g., ../physical-ai-textbook/docs/)"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Index only a specific week (optional)"
    )

    args = parser.parse_args()

    docs_path = Path(args.docs_path).resolve()

    if not docs_path.exists() or not docs_path.is_dir():
        logger.error(f"Invalid docs path: {docs_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Document Indexing Tool")
    logger.info("=" * 60)
    logger.info(f"Docs path: {docs_path}")
    if args.week:
        logger.info(f"Week filter: {args.week}")
    logger.info("=" * 60)

    indexer = DocumentIndexer()

    success = await indexer.index_all(docs_path, week_filter=args.week)

    indexer.print_summary()

    if success and indexer.stats["chunks_indexed"] > 0:
        logger.info("\n✅ Indexing completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Start backend: uvicorn rag_backend.main:app --reload")
        logger.info("  2. Test query: curl -X POST http://localhost:8000/api/chat/query")
        sys.exit(0)
    else:
        logger.error("\n❌ Indexing failed or no documents indexed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

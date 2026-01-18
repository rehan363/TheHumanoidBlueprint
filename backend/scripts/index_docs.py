#!/usr/bin/env python3
"""
CLI tool to index documentation into Qdrant vector database.

Usage:
    python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs/
    python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs/ --chapter 1
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

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
    """Document indexer for RAG system."""

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

    def discover_markdown_files(self, docs_path: Path, chapter_filter: int = None, module_filter: str = None) -> List[Dict[str, Any]]:
        """
        Discover markdown files in the actual docs directory structure.

        Actual structure:
            docs/
            ├── module1-ros2/
            │   ├── chapter1-foundations.md
            │   ├── chapter2-landscape.md
            │   └── chapter3-ros2-intro.md
            ├── module2-gazebo/
            ├── module3-isaac/
            └── module4-vla/

        Args:
            docs_path: Base docs directory
            chapter_filter: Optional chapter number to process only specific chapter
            module_filter: Optional module name to process only specific module

        Returns:
            List of file info dicts with path, chapter, module
        """
        files = []

        # Pattern: docs/moduleX-{name}/chapter{N}-{topic}.md
        for module_dir in sorted(docs_path.glob("module*-*")):
            if not module_dir.is_dir():
                continue

            # Extract module name (e.g., "module1-ros2" -> "ROS 2")
            module_name_raw = module_dir.name
            # Clean up module name for display
            if "-" in module_name_raw:
                # Format: module1-ros2 -> "ROS 2"
                parts = module_name_raw.split("-", 1)
                module_display = parts[1].replace("-", " ").title() if len(parts) > 1 else module_name_raw
            else:
                module_display = module_name_raw

            # Apply module filter
            if module_filter and module_filter.lower() not in module_name_raw.lower():
                continue

            # Find all chapter files in this module
            for md_file in sorted(module_dir.glob("chapter*.md")):
                # Extract chapter number from filename (e.g., "chapter1-foundations.md" -> 1)
                match = re.match(r'chapter(\d+)', md_file.stem)
                if not match:
                    logger.warning(f"Skipping file with unexpected naming: {md_file.name}")
                    continue

                chapter_num = int(match.group(1))

                # Apply chapter filter
                if chapter_filter and chapter_num != chapter_filter:
                    continue

                # Get topic from filename (e.g., "chapter1-foundations.md" -> "foundations")
                topic = md_file.stem.split("-", 1)[1] if "-" in md_file.stem else ""

                files.append({
                    "path": md_file,
                    "chapter": chapter_num,
                    "module": module_display,
                    "module_raw": module_name_raw,
                    "topic": topic
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
            chapter = file_info["chapter"]
            module = file_info["module"]
            topic = file_info.get("topic", "")

            logger.info(f"Processing: Chapter {chapter} - {module} - {topic} ({file_path.name})")

            # Chunk file
            chunks = self.chunker.chunk_file(file_path, chapter, module)
            self.stats["chunks_created"] += len(chunks)

            if not chunks:
                logger.warning(f"No chunks created for {file_path.name}")
                return 0

            # Generate embeddings for each chunk
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:
                    logger.info(f"  Embedding chunk {i}/{len(chunks)}...")
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
                logger.info(f"Indexed {indexed_count} chunks from {file_path.name}")
                return indexed_count

            return 0

        except Exception as e:
            logger.exception(f"Error indexing file {file_info['path']}: {e}")
            self.stats["errors"] += 1
            return 0

    async def index_all(self, docs_path: Path, chapter_filter: int = None, module_filter: str = None) -> bool:
        """
        Index all markdown files in the docs directory.

        Args:
            docs_path: Base docs directory
            chapter_filter: Optional chapter number to process only specific chapter
            module_filter: Optional module name to process only specific module

        Returns:
            bool: True if successful
        """
        try:
            self.stats["start_time"] = datetime.utcnow()

            # Discover files
            files = self.discover_markdown_files(docs_path, chapter_filter, module_filter)

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
        help="Path to docs directory (e.g., ../physical-ai-textbook/docs/)"
    )
    parser.add_argument(
        "--chapter",
        type=int,
        help="Index only a specific chapter (e.g., 1, 2, 3)"
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Index only a specific module (e.g., 'ros2', 'gazebo', 'isaac')"
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
    if args.chapter:
        logger.info(f"Chapter filter: {args.chapter}")
    if args.module:
        logger.info(f"Module filter: {args.module}")
    logger.info("=" * 60)

    indexer = DocumentIndexer()

    success = await indexer.index_all(docs_path, chapter_filter=args.chapter, module_filter=args.module)

    indexer.print_summary()

    if success and indexer.stats["chunks_indexed"] > 0:
        logger.info("")
        logger.info("Indexing completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Start backend: uvicorn rag_backend.main:app --reload")
        logger.info("  2. Test query: curl -X POST http://localhost:8000/api/chat/query")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("Indexing failed or no documents indexed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

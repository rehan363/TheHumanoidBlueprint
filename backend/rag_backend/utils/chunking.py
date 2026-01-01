"""
Text chunking utilities for markdown documents.

Implements markdown-aware hierarchical splitting with configurable chunk size and overlap.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from rag_backend.config import settings
from rag_backend.models.chunk import TextChunk, ChunkMetadata
import hashlib

logger = logging.getLogger(__name__)


class MarkdownChunker:
    """Markdown-aware text chunker with hierarchical splitting."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 100
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters (default: from settings)
            chunk_overlap: Overlap between chunks in characters (default: from settings)
            min_chunk_size: Minimum chunk size to avoid very small chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Markdown heading pattern
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def extract_heading_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract heading hierarchy from markdown text.

        Args:
            text: Markdown text

        Returns:
            List of heading info dicts with level, title, start_pos
        """
        headings = []
        for match in self.heading_pattern.finditer(text):
            level = len(match.group(1))  # Number of # symbols
            title = match.group(2).strip()
            start_pos = match.start()
            headings.append({
                "level": level,
                "title": title,
                "start_pos": start_pos
            })
        return headings

    def build_heading_path(self, headings: List[Dict[str, Any]], current_pos: int) -> List[str]:
        """
        Build hierarchical heading path for a given position in the text.

        Args:
            headings: List of heading info
            current_pos: Current position in text

        Returns:
            List of heading titles forming the path
        """
        path = []
        current_level = 0

        for heading in headings:
            if heading["start_pos"] > current_pos:
                break

            # If this heading is at a higher or same level, reset path
            if heading["level"] <= current_level:
                path = path[:heading["level"]-1]

            path.append(heading["title"])
            current_level = heading["level"]

        return path

    def split_into_chunks(self, text: str, metadata_base: Dict[str, Any]) -> List[TextChunk]:
        """
        Split text into chunks with overlap and metadata.

        Args:
            text: Input text to chunk
            metadata_base: Base metadata (week, module, file_path)

        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning(f"Text too short to chunk: {len(text)} chars")
            return []

        # Extract heading hierarchy
        headings = self.extract_heading_hierarchy(text)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # Try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break (double newline)
                last_para = text.rfind('\n\n', start, end)
                if last_para > start + self.min_chunk_size:
                    end = last_para + 2

                # Otherwise, look for sentence break
                elif end < len(text):
                    last_period = max(
                        text.rfind('. ', start, end),
                        text.rfind('.\n', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if last_period > start + self.min_chunk_size:
                        end = last_period + 2

            # Extract chunk text
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                # Build heading path for this chunk
                heading_path = self.build_heading_path(headings, start)

                # Create chunk metadata
                metadata = ChunkMetadata(
                    week=metadata_base["week"],
                    module=metadata_base["module"],
                    file_path=metadata_base["file_path"],
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated after all chunks are created
                    heading_path=heading_path
                )

                # Generate chunk ID
                chunk_id = self._generate_chunk_id(
                    metadata_base["file_path"],
                    chunk_index
                )

                # Create TextChunk
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    embedding=None,  # Will be populated by embedding service
                    metadata=metadata,
                    token_count=self._estimate_tokens(chunk_text)
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Avoid infinite loop
            if start >= len(text):
                break

        # Update total_chunks in all chunk metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata.total_chunks = total_chunks

        logger.info(f"Split text into {total_chunks} chunks (avg size: {sum(len(c.content) for c in chunks) // max(total_chunks, 1)} chars)")
        return chunks

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """
        Generate unique chunk ID.

        Args:
            file_path: Source file path
            chunk_index: Index of chunk in file

        Returns:
            Unique chunk ID
        """
        # Create hash from file path
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"chunk_{path_hash}_{chunk_index}"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: 4 chars per token on average
        return len(text) // 4

    def chunk_file(self, file_path: Path, week: int, module: str) -> List[TextChunk]:
        """
        Chunk a markdown file.

        Args:
            file_path: Path to markdown file
            week: Week number
            module: Module name

        Returns:
            List of TextChunk objects

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If chunking fails
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Create metadata base
            metadata_base = {
                "week": week,
                "module": module,
                "file_path": str(file_path)
            }

            # Split into chunks
            chunks = self.split_into_chunks(text, metadata_base)

            logger.info(f"Chunked file {file_path.name}: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            raise


# Global chunker instance
_chunker: MarkdownChunker = None


def get_chunker() -> MarkdownChunker:
    """
    Get or create the global chunker instance.

    Returns:
        MarkdownChunker: Singleton chunker instance
    """
    global _chunker
    if _chunker is None:
        _chunker = MarkdownChunker()
    return _chunker

"""
Pydantic models for chat API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime


class SelectedTextContext(BaseModel):
    """Context for text selection queries."""

    text: str = Field(..., min_length=1, max_length=5000, description="Selected text content")
    file_path: str = Field(..., description="Path to the source file")
    selection_range: Optional[dict] = Field(None, description="Start and end positions of selection")


class ChatQueryRequest(BaseModel):
    """Request model for chat query endpoint."""

    query: str = Field(..., min_length=1, max_length=500, description="User query text")
    query_type: Literal["full_text", "text_selection"] = Field(
        default="full_text",
        description="Type of query: full_text or text_selection"
    )
    context: Optional[SelectedTextContext] = Field(
        None,
        description="Context for text selection queries (required when query_type is text_selection)"
    )
    session_id: Optional[str] = Field(None, description="Session identifier for conversation history")

    @validator("context")
    def validate_context(cls, v, values):
        """Validate that context is provided when query_type is text_selection."""
        if values.get("query_type") == "text_selection" and v is None:
            raise ValueError("context is required when query_type is text_selection")
        return v

    @validator("query")
    def sanitize_query(cls, v):
        """Sanitize query text."""
        return v.strip()


class Citation(BaseModel):
    """Citation information for a response."""

    source: str = Field(..., description="Source reference (e.g., 'Week 3, ROS 2 Architecture')")
    chunk_index: int = Field(..., description="Position of this chunk in the section")
    total_chunks: int = Field(..., description="Total number of chunks in the section")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    file_path: Optional[str] = Field(None, description="Path to source file")
    content_preview: Optional[str] = Field(None, max_length=200, description="Preview of cited content")


class ChatQueryResponse(BaseModel):
    """Response model for chat query endpoint."""

    answer: str = Field(..., description="Generated answer from RAG pipeline")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    query_type: Literal["full_text", "text_selection"] = Field(..., description="Type of query processed")
    session_id: Optional[str] = Field(None, description="Session identifier")
    processing_time_ms: Optional[int] = Field(None, description="Time taken to process query (milliseconds)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Unique identifier for the request")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

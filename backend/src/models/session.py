from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any

from pydantic import BaseModel, Field

# Re-defining Citation model for backend consistency with frontend types
class Citation(BaseModel):
    source: str
    chunk_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    file_path: Optional[str] = None # Added from frontend types
    content_preview: Optional[str] = None # Added from frontend types

class QuerySession(BaseModel):
    """
    Pydantic model for a chat session.
    Corresponds to the 'query_sessions' table.
    """
    id: UUID = Field(default_factory=uuid4)
    session_token: str = Field(..., max_length=255)
    user_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0)

class SessionMessage(BaseModel):
    """
    Pydantic model for a message within a chat session.
    Corresponds to the 'session_messages' table.
    """
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: Literal['user', 'assistant']
    content: str
    citations: Optional[List[Citation]] = None # Use List[Citation] for consistency
    query_type: Optional[Literal['full_text', 'text_selection']] = None # Optional for messages not directly from query
    created_at: datetime = Field(default_factory=datetime.utcnow)

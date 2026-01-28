import contextvars
import logging
from typing import Optional

# Context variable to store the request ID
request_id_ctx_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)

def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_ctx_var.get()

def set_request_id(request_id: str) -> None:
    """Set the current request ID in context."""
    request_id_ctx_var.set(request_id)

class RequestIdFilter(logging.Filter):
    """
    Logging filter that adds request_id to log records.
    """
    def filter(self, record):
        record.request_id = get_request_id() or "no-request-id"
        return True

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ExternalAPIRequest(BaseModel):
    """Model for external API request parameters."""
    query: str = Field(..., description="Search query parameter")
    timestamp: datetime = Field(default_factory=datetime.now)


class ExternalAPIResponse(BaseModel):
    """Model for processed external API response."""
    status: str
    data: List[dict]
    processed_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
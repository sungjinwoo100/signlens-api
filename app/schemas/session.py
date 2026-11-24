from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    session_name: Optional[str] = "Live Detection Session"


class SessionAppendText(BaseModel):
    text: str
    confidence: Optional[float] = None


class SessionResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    session_name: str
    is_active: bool
    total_predictions: int
    average_confidence: float
    spelled_text: str
    started_at: datetime
    ended_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionStatsResponse(BaseModel):
    session_id: int
    session_name: str
    is_active: bool
    total_predictions: int
    average_confidence: float
    spelled_text: str
    most_frequent_sign: Optional[str] = None
    duration_seconds: float
    started_at: datetime
    ended_at: Optional[datetime] = None

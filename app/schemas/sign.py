from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class SignBase(BaseModel):
    label: str
    category: str = "alphabet"  # alphabet, numbers, phrases
    difficulty: str = "Beginner"  # Beginner, Intermediate, Advanced
    description: str
    hand_formation: str
    mnemonic_tip: str
    icon_key: Optional[str] = None


class SignCreate(SignBase):
    pass


class SignUpdate(BaseModel):
    label: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    description: Optional[str] = None
    hand_formation: Optional[str] = None
    mnemonic_tip: Optional[str] = None
    icon_key: Optional[str] = None


class SignResponse(SignBase):
    id: int
    created_at: datetime
    # Per-user flags (populated if authenticated, null/None if logged-out)
    saved: Optional[bool] = None
    mastered: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True)

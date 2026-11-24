from datetime import datetime
from typing import Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ProgressUpdate(BaseModel):
    # Accepts either {"saved": true} / {"mastered": true} or the is_* spelling.
    model_config = ConfigDict(populate_by_name=True)

    is_saved: Optional[bool] = Field(
        None, validation_alias=AliasChoices("is_saved", "saved")
    )
    is_mastered: Optional[bool] = Field(
        None, validation_alias=AliasChoices("is_mastered", "mastered")
    )


class ProgressResponse(BaseModel):
    id: int
    user_id: int
    sign_id: int
    is_saved: bool
    is_mastered: bool
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProgressSummary(BaseModel):
    total_signs: int
    saved_count: int
    mastered_count: int
    quiz_high_score: int
    best_streak: int


class ResetResponse(BaseModel):
    message: str
    cleared_progress_count: int
    cleared_quiz_count: int

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class QuizOption(BaseModel):
    sign_id: int
    label: str


class QuizQuestion(BaseModel):
    id: str
    prompt: str
    target_sign_id: int
    target_label: str
    hand_formation: str
    mnemonic_tip: str
    options: List[QuizOption]
    correct_sign_id: int


class QuizGenerateResponse(BaseModel):
    category: str
    total_questions: int
    questions: List[QuizQuestion]


class QuizAttemptCreate(BaseModel):
    total_questions: int = Field(gt=0)
    correct_answers: int = Field(ge=0)
    score: int = Field(ge=0)
    streak: int = Field(ge=0)
    category: Optional[str] = "all"


class QuizAttemptResponse(BaseModel):
    id: int
    user_id: int
    total_questions: int
    correct_answers: int
    score: int
    streak: int
    category: str
    created_at: datetime
    accuracy_percentage: float

    model_config = ConfigDict(from_attributes=True)

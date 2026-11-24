from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictRequest(BaseModel):
    """Accepts EITHER a base64-encoded camera frame (JPEG/PNG) OR 63 raw hand landmark floats."""

    image_base64: Optional[str] = Field(
        None,
        description="Base64-encoded JPEG/PNG camera frame (optional data URI prefix supported)",
    )
    landmarks: Optional[List[float]] = Field(
        None,
        description="63 raw float values representing 21 3D hand landmarks (x, y, z)",
    )
    handedness: Optional[str] = Field(
        None,
        description="Optional handedness ('Left' or 'Right') when sending raw landmarks",
    )
    already_normalized: bool = Field(
        False,
        description="Set True only if landmarks were already processed by normalize_landmarks()",
    )
    session_id: Optional[int] = Field(
        None,
        description="Optional active detection session ID to log prediction into",
    )

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks_count(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) != 63:
            raise ValueError(
                f"Expected exactly 63 floats (21 landmarks x 3 axes), received {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_input_provided(self) -> "PredictRequest":
        if not self.image_base64 and self.landmarks is None:
            raise ValueError("Provide either 'image_base64' or 'landmarks' (63 floats)")
        return self


class TopPrediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    predicted_label: str
    confidence: float
    hand_detected: bool = True
    latency_ms: Optional[float] = None
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    hand_landmarks: Optional[List[List[float]]] = Field(
        None,
        description="21 normalized [x, y] image-space hand landmark coordinates for UI skeleton overlay",
    )
    session_id: Optional[int] = None
    spelled_text: Optional[str] = None

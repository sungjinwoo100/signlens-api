import time
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.session import DetectionSession
from app.schemas.predict import PredictRequest, PredictResponse, TopPrediction
from app.services.ml_service import ml_service

router = APIRouter(prefix="/predict", tags=["Real-time Hand Sign Prediction"])


@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict ASL letter from a camera frame (JPEG/base64) or 63 hand landmarks",
)
def predict_sign(
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    """
    Guest-accessible real-time prediction endpoint.
    Accepts EITHER:
      - `image_base64`: JPEG/PNG camera frame -> resizes, runs MediaPipe Hands +
        `normalize_landmarks()` + MLP classifier. Returns `hand_detected=false`
        and `label="nothing"` if no hand is visible.
      - `landmarks`: 63 raw landmark floats (21 x [x, y, z]) -> runs
        `normalize_landmarks()` + MLP classifier.

    Optionally logs the prediction to an active `session_id` when a hand is detected.
    """
    hand_landmarks = None
    if request.image_base64:
        label, confidence, hand_detected, latency_ms, top_list, hand_landmarks = (
            ml_service.predict_from_image_base64(request.image_base64)
        )
    else:
        t0 = time.perf_counter()
        assert request.landmarks is not None
        label, confidence, top_list = ml_service.predict(
            request.landmarks,
            handedness=request.handedness,
            already_normalized=request.already_normalized,
        )
        hand_detected = True
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    spelled_text_result: Optional[str] = None

    # Log to active detection session only when a hand sign was actually detected
    if request.session_id and hand_detected:
        session = (
            db.query(DetectionSession)
            .filter(DetectionSession.id == request.session_id)
            .first()
        )
        if session and session.is_active:
            new_total = session.total_predictions + 1
            new_avg = (
                (session.average_confidence * session.total_predictions) + confidence
            ) / new_total

            session.total_predictions = new_total
            session.average_confidence = round(new_avg, 4)
            session.spelled_text = (session.spelled_text or "") + label
            db.commit()
            spelled_text_result = session.spelled_text

    return PredictResponse(
        label=label,
        predicted_label=label,
        confidence=confidence,
        hand_detected=hand_detected,
        latency_ms=latency_ms,
        top_predictions=[
            TopPrediction(label=lbl, confidence=conf) for lbl, conf in top_list
        ],
        hand_landmarks=hand_landmarks,
        session_id=request.session_id,
        spelled_text=spelled_text_result,
    )

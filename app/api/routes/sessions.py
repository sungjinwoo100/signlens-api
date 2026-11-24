from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from app.api.deps import get_optional_user
from app.db.database import get_db
from app.models.session import DetectionSession
from app.models.user import User
from app.schemas.session import (
    SessionAppendText,
    SessionCreate,
    SessionResponse,
    SessionStatsResponse,
)

router = APIRouter(prefix="/sessions", tags=["Detection Sessions"])


def _most_frequent_sign(spelled_text: Optional[str]) -> Optional[str]:
    """Each prediction appends one letter, so the modal letter is the top sign."""
    letters = [c for c in (spelled_text or "") if not c.isspace()]
    if not letters:
        return None
    return Counter(letters).most_common(1)[0][0]


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new SignLens detection session",
)
def start_session(
    session_in: SessionCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Starts a new live ASL detection logging session.
    """
    user_id = current_user.id if current_user else None
    name = session_in.session_name or "Live Detection Session"

    session = DetectionSession(
        user_id=user_id,
        session_name=name,
        is_active=True,
        total_predictions=0,
        average_confidence=0.0,
        spelled_text="",
        started_at=datetime.now(timezone.utc),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get(
    "",
    response_model=List[SessionResponse],
    summary="List detection sessions",
)
def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Lists detection sessions. If user is authenticated, returns their sessions;
    otherwise returns recent sessions.
    """
    query = db.query(DetectionSession)
    if current_user:
        query = query.filter(DetectionSession.user_id == current_user.id)
    return query.order_by(DetectionSession.started_at.desc()).offset(skip).limit(limit).all()


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get detection session details",
)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
):
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection session with ID {session_id} not found.",
        )
    return session


@router.post(
    "/{session_id}/stop",
    response_model=SessionResponse,
    summary="Stop/finalize an active detection session",
)
def stop_session(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    Stops an active detection session and records its end timestamp.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection session with ID {session_id} not found.",
        )

    if session.is_active:
        session.is_active = False
        session.ended_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)

    return session


@router.get(
    "/{session_id}/stats",
    response_model=SessionStatsResponse,
    summary="Get aggregated statistics for a detection session",
)
def get_session_stats(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    Returns total predictions, running average confidence, spelled text, and duration.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection session with ID {session_id} not found.",
        )

    # Compute duration in seconds
    end_time = session.ended_at or datetime.now(timezone.utc)
    # Ensure started_at is timezone-aware for comparison if needed
    start_time = session.started_at
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    duration = max(0.0, (end_time - start_time).total_seconds())

    return SessionStatsResponse(
        session_id=session.id,
        session_name=session.session_name,
        is_active=session.is_active,
        total_predictions=session.total_predictions,
        average_confidence=session.average_confidence,
        spelled_text=session.spelled_text or "",
        most_frequent_sign=_most_frequent_sign(session.spelled_text),
        duration_seconds=round(duration, 2),
        started_at=session.started_at,
        ended_at=session.ended_at,
    )


@router.get(
    "/{session_id}/export",
    response_class=PlainTextResponse,
    summary="Export session text log and summary as plain text",
)
def export_session_log(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    Exports the spelled letters and session metrics formatted as plain text (.txt).
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection session with ID {session_id} not found.",
        )

    started_str = session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    ended_str = (
        session.ended_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if session.ended_at
        else "In Progress"
    )

    log_content = (
        f"=====================================================\n"
        f" SIGNLENS ASL DETECTION SESSION LOG\n"
        f"=====================================================\n"
        f"Session ID:         {session.id}\n"
        f"Session Name:       {session.session_name}\n"
        f"Status:             {'ACTIVE' if session.is_active else 'COMPLETED'}\n"
        f"Started At:         {started_str}\n"
        f"Ended At:           {ended_str}\n"
        f"Total Predictions:  {session.total_predictions}\n"
        f"Average Confidence: {session.average_confidence * 100:.1f}%\n"
        f"Most Frequent Sign: {_most_frequent_sign(session.spelled_text) or '-'}\n"
        f"-----------------------------------------------------\n"
        f"SPELLED TEXT OUTPUT:\n"
        f"-----------------------------------------------------\n"
        f"{session.spelled_text or '(No letters recorded)'}\n"
        f"=====================================================\n"
    )

    return PlainTextResponse(
        content=log_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="signlens_session_{session.id}.txt"'
        },
    )

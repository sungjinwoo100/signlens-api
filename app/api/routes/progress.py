from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.database import get_db
from app.models.progress import UserProgress
from app.models.quiz import QuizAttempt
from app.models.sign import Sign
from app.models.user import User
from app.schemas.progress import (
    ProgressResponse,
    ProgressSummary,
    ProgressUpdate,
    ResetResponse,
)

router = APIRouter(prefix="/progress", tags=["Learning Progress"])


@router.put(
    "/signs/{sign_id}",
    response_model=ProgressResponse,
    summary="Update or toggle saved/mastered state for a sign (upsert)",
)
def update_sign_progress(
    sign_id: int,
    progress_in: ProgressUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Sets or toggles the saved or mastered status of a sign for the authenticated user.
    Creates (upserts) the record if it does not exist yet.
    """
    sign = db.query(Sign).filter(Sign.id == sign_id).first()
    if not sign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sign with ID {sign_id} not found.",
        )

    record = (
        db.query(UserProgress)
        .filter(
            UserProgress.user_id == current_user.id,
            UserProgress.sign_id == sign_id,
        )
        .first()
    )

    if not record:
        record = UserProgress(
            user_id=current_user.id,
            sign_id=sign_id,
            is_saved=progress_in.is_saved if progress_in.is_saved is not None else False,
            is_mastered=progress_in.is_mastered if progress_in.is_mastered is not None else False,
        )
        db.add(record)
    else:
        if progress_in.is_saved is not None:
            record.is_saved = progress_in.is_saved
        if progress_in.is_mastered is not None:
            record.is_mastered = progress_in.is_mastered

    db.commit()
    db.refresh(record)
    return record


@router.get(
    "/summary",
    response_model=ProgressSummary,
    summary="Get user learning progress statistics",
)
def get_progress_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns an aggregated statistical summary of total signs, saved count,
    mastered count, quiz all-time high score, and best streak.
    """
    total_signs = db.query(func.count(Sign.id)).scalar() or 0

    saved_count = (
        db.query(func.count(UserProgress.id))
        .filter(
            UserProgress.user_id == current_user.id,
            UserProgress.is_saved.is_(True),
        )
        .scalar()
        or 0
    )

    mastered_count = (
        db.query(func.count(UserProgress.id))
        .filter(
            UserProgress.user_id == current_user.id,
            UserProgress.is_mastered.is_(True),
        )
        .scalar()
        or 0
    )

    quiz_high_score = (
        db.query(func.max(QuizAttempt.score))
        .filter(QuizAttempt.user_id == current_user.id)
        .scalar()
        or 0
    )

    best_streak = (
        db.query(func.max(QuizAttempt.streak))
        .filter(QuizAttempt.user_id == current_user.id)
        .scalar()
        or 0
    )

    return ProgressSummary(
        total_signs=total_signs,
        saved_count=saved_count,
        mastered_count=mastered_count,
        quiz_high_score=quiz_high_score,
        best_streak=best_streak,
    )


@router.delete(
    "/reset",
    response_model=ResetResponse,
    summary="Reset user learning progress and quiz history",
)
def reset_progress(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Wipes all saved/mastered records and quiz attempt history for the authenticated user.
    """
    progress_count = (
        db.query(UserProgress)
        .filter(UserProgress.user_id == current_user.id)
        .delete(synchronize_session=False)
    )

    quiz_count = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.user_id == current_user.id)
        .delete(synchronize_session=False)
    )

    db.commit()

    return ResetResponse(
        message="Learning progress and quiz history have been reset successfully.",
        cleared_progress_count=progress_count,
        cleared_quiz_count=quiz_count,
    )

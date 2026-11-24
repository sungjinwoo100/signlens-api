import random
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.database import get_db
from app.models.quiz import QuizAttempt
from app.models.sign import Sign
from app.models.user import User
from app.schemas.quiz import (
    QuizAttemptCreate,
    QuizAttemptResponse,
    QuizGenerateResponse,
    QuizOption,
    QuizQuestion,
)

router = APIRouter(prefix="/quiz", tags=["Quiz"])


@router.get(
    "/questions",
    response_model=QuizGenerateResponse,
    summary="Generate multiple-choice quiz questions on the fly",
)
def generate_quiz_questions(
    category: str = Query("all", description="Category of signs (alphabet, numbers, phrases, or all)"),
    count: int = Query(10, ge=1, le=50, description="Number of questions to generate"),
    db: Session = Depends(get_db),
):
    """
    Generates multiple-choice questions from the ASL dictionary.
    Each question includes 1 correct sign and 3 shuffled distractors.
    Returns HTTP 400 if the requested category contains fewer than 4 signs.
    """
    query = db.query(Sign)
    category_clean = category.strip().lower()
    if category_clean != "all":
        query = query.filter(Sign.category == category_clean)

    pool: List[Sign] = query.all()
    if len(pool) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot generate multiple-choice questions for category '{category}'. "
                f"Found only {len(pool)} signs, but at least 4 are required to provide options."
            ),
        )

    # Pick target signs (sample without replacement up to available count)
    target_count = min(count, len(pool))
    target_signs = random.sample(pool, target_count)

    questions: List[QuizQuestion] = []
    for target in target_signs:
        # Pick 3 distinct distractors from the pool
        distractors = [s for s in pool if s.id != target.id]
        chosen_distractors = random.sample(distractors, 3)

        # Assemble and shuffle 4 options
        raw_options = [target] + chosen_distractors
        random.shuffle(raw_options)

        options = [
            QuizOption(sign_id=opt.id, label=opt.label)
            for opt in raw_options
        ]

        questions.append(
            QuizQuestion(
                id=str(uuid.uuid4())[:8],
                prompt=f"Which ASL sign matches the hand posture: '{target.hand_formation}'?",
                target_sign_id=target.id,
                target_label=target.label,
                hand_formation=target.hand_formation,
                mnemonic_tip=target.mnemonic_tip,
                options=options,
                correct_sign_id=target.id,
            )
        )

    return QuizGenerateResponse(
        category=category_clean,
        total_questions=len(questions),
        questions=questions,
    )


@router.post(
    "/attempts",
    response_model=QuizAttemptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a completed quiz attempt",
)
def record_quiz_attempt(
    attempt_in: QuizAttemptCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Saves a completed quiz round score, streak, and question metrics for the authenticated user.
    """
    if attempt_in.correct_answers > attempt_in.total_questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Correct answers cannot exceed total questions.",
        )

    accuracy = (
        round((attempt_in.correct_answers / attempt_in.total_questions) * 100, 2)
        if attempt_in.total_questions > 0
        else 0.0
    )

    attempt = QuizAttempt(
        user_id=current_user.id,
        total_questions=attempt_in.total_questions,
        correct_answers=attempt_in.correct_answers,
        score=attempt_in.score,
        streak=attempt_in.streak,
        category=attempt_in.category or "all",
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    return QuizAttemptResponse(
        id=attempt.id,
        user_id=attempt.user_id,
        total_questions=attempt.total_questions,
        correct_answers=attempt.correct_answers,
        score=attempt.score,
        streak=attempt.streak,
        category=attempt.category,
        created_at=attempt.created_at,
        accuracy_percentage=accuracy,
    )


@router.get(
    "/attempts",
    response_model=List[QuizAttemptResponse],
    summary="List past quiz attempts for current user",
)
def list_quiz_attempts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the quiz attempt history for the authenticated user, ordered newest first.
    """
    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.user_id == current_user.id)
        .order_by(QuizAttempt.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    results = []
    for a in attempts:
        accuracy = (
            round((a.correct_answers / a.total_questions) * 100, 2)
            if a.total_questions > 0
            else 0.0
        )
        results.append(
            QuizAttemptResponse(
                id=a.id,
                user_id=a.user_id,
                total_questions=a.total_questions,
                correct_answers=a.correct_answers,
                score=a.score,
                streak=a.streak,
                category=a.category,
                created_at=a.created_at,
                accuracy_percentage=accuracy,
            )
        )
    return results

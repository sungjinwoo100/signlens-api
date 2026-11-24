from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, get_optional_user
from app.db.database import get_db
from app.models.progress import UserProgress
from app.models.sign import Sign
from app.models.user import User
from app.schemas.sign import SignCreate, SignResponse, SignUpdate

router = APIRouter(prefix="/signs", tags=["ASL Dictionary"])


def _attach_user_progress(
    signs: List[Sign],
    db: Session,
    user_id: Optional[int],
) -> List[SignResponse]:
    """Helper to populate 'saved' and 'mastered' flags for a list of signs."""
    if not user_id:
        return [
            SignResponse(
                id=s.id,
                label=s.label,
                category=s.category,
                difficulty=s.difficulty,
                description=s.description,
                hand_formation=s.hand_formation,
                mnemonic_tip=s.mnemonic_tip,
                icon_key=s.icon_key,
                created_at=s.created_at,
                saved=None,
                mastered=None,
            )
            for s in signs
        ]

    # Pre-fetch user progress records for these signs
    sign_ids = [s.id for s in signs]
    progress_map = {}
    if sign_ids:
        records = (
            db.query(UserProgress)
            .filter(
                UserProgress.user_id == user_id,
                UserProgress.sign_id.in_(sign_ids),
            )
            .all()
        )
        progress_map = {r.sign_id: r for r in records}

    results = []
    for s in signs:
        user_prog = progress_map.get(s.id)
        results.append(
            SignResponse(
                id=s.id,
                label=s.label,
                category=s.category,
                difficulty=s.difficulty,
                description=s.description,
                hand_formation=s.hand_formation,
                mnemonic_tip=s.mnemonic_tip,
                icon_key=s.icon_key,
                created_at=s.created_at,
                saved=user_prog.is_saved if user_prog else False,
                mastered=user_prog.is_mastered if user_prog else False,
            )
        )
    return results


@router.get(
    "",
    response_model=List[SignResponse],
    summary="List, search, and filter ASL signs",
)
def list_signs(
    q: Optional[str] = Query(None, description="Search query across label, description, and tips"),
    category: Optional[str] = Query(None, description="Category filter (alphabet, numbers, phrases)"),
    difficulty: Optional[str] = Query(None, description="Difficulty filter (Beginner, Intermediate, Advanced)"),
    saved_only: bool = Query(False, description="Filter for saved/bookmarked signs only (requires auth)"),
    mastered_only: bool = Query(False, description="Filter for mastered signs only (requires auth)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Retrieves signs from the ASL dictionary.
    Supports search query (?q=), category filter (?category=), and per-user
    filters (?saved_only=true, ?mastered_only=true).
    When caller is authenticated, 'saved' and 'mastered' flags reflect caller's state;
    otherwise they are null.
    """
    if (saved_only or mastered_only) and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to filter by saved or mastered signs.",
        )

    query = db.query(Sign)

    # Search filter
    if q and q.strip():
        term = f"%{q.strip()}%"
        query = query.filter(
            or_(
                Sign.label.ilike(term),
                Sign.description.ilike(term),
                Sign.hand_formation.ilike(term),
                Sign.mnemonic_tip.ilike(term),
            )
        )

    # Category filter
    if category and category.strip():
        query = query.filter(Sign.category == category.strip().lower())

    # Difficulty filter
    if difficulty and difficulty.strip():
        query = query.filter(Sign.difficulty.ilike(difficulty.strip()))

    # Per-user saved/mastered filtering
    if current_user and (saved_only or mastered_only):
        query = query.join(
            UserProgress,
            (UserProgress.sign_id == Sign.id) & (UserProgress.user_id == current_user.id),
        )
        if saved_only:
            query = query.filter(UserProgress.is_saved.is_(True))
        if mastered_only:
            query = query.filter(UserProgress.is_mastered.is_(True))

    signs = query.order_by(Sign.id.asc()).offset(skip).limit(limit).all()
    user_id = current_user.id if current_user else None
    return _attach_user_progress(signs, db, user_id)


@router.get(
    "/{sign_id}",
    response_model=SignResponse,
    summary="Get single ASL sign by ID",
)
def get_sign(
    sign_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    sign = db.query(Sign).filter(Sign.id == sign_id).first()
    if not sign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sign with ID {sign_id} not found.",
        )

    user_id = current_user.id if current_user else None
    results = _attach_user_progress([sign], db, user_id)
    return results[0]


@router.post(
    "",
    response_model=SignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new sign entry",
)
def create_sign(
    sign_in: SignCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = db.query(Sign).filter(Sign.label == sign_in.label).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A sign with label '{sign_in.label}' already exists.",
        )

    sign = Sign(
        label=sign_in.label,
        category=sign_in.category.lower(),
        difficulty=sign_in.difficulty,
        description=sign_in.description,
        hand_formation=sign_in.hand_formation,
        mnemonic_tip=sign_in.mnemonic_tip,
        icon_key=sign_in.icon_key,
    )
    db.add(sign)
    db.commit()
    db.refresh(sign)
    return _attach_user_progress([sign], db, current_user.id)[0]


@router.put(
    "/{sign_id}",
    response_model=SignResponse,
    summary="Update an existing sign entry",
)
def update_sign(
    sign_id: int,
    sign_in: SignUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sign = db.query(Sign).filter(Sign.id == sign_id).first()
    if not sign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sign with ID {sign_id} not found.",
        )

    update_data = sign_in.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        if field == "category" and val:
            setattr(sign, field, val.lower())
        else:
            setattr(sign, field, val)

    db.commit()
    db.refresh(sign)
    return _attach_user_progress([sign], db, current_user.id)[0]


@router.delete(
    "/{sign_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a sign entry",
)
def delete_sign(
    sign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sign = db.query(Sign).filter(Sign.id == sign_id).first()
    if not sign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sign with ID {sign_id} not found.",
        )

    db.delete(sign)
    db.commit()
    return None

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.database import Base


class UserProgress(Base):
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    sign_id = Column(Integer, ForeignKey("signs.id", ondelete="CASCADE"), nullable=False, index=True)
    is_saved = Column(Boolean, default=False, nullable=False)
    is_mastered = Column(Boolean, default=False, nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "sign_id", name="uq_user_sign_progress"),
    )

    # Relationships
    user = relationship("User", back_populates="progress_records")
    sign = relationship("Sign", back_populates="progress_records")

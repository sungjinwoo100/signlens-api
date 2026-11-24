from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship
from app.db.database import Base


class Sign(Base):
    __tablename__ = "signs"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(100), unique=True, index=True, nullable=False)
    category = Column(String(50), index=True, nullable=False)  # alphabet, numbers, phrases
    difficulty = Column(String(50), default="Beginner", nullable=False)
    description = Column(Text, nullable=False)
    hand_formation = Column(Text, nullable=False)
    mnemonic_tip = Column(Text, nullable=False)
    icon_key = Column(String(100), nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    progress_records = relationship(
        "UserProgress",
        back_populates="sign",
        cascade="all, delete-orphan",
    )

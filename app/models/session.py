from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.db.database import Base


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_name = Column(String(100), default="Live Detection Session", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    total_predictions = Column(Integer, default=0, nullable=False)
    average_confidence = Column(Float, default=0.0, nullable=False)
    spelled_text = Column(Text, default="", nullable=False)
    started_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    ended_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

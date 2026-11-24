"""Shared MediaPipe HandLandmarker wrapper used by training extraction and FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from .normalize import normalize_landmarks

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "ml" / "models" / "hand_landmarker.task"
)


class HandLandmarkDetector:
    """Thread-local / per-process MediaPipe HandLandmarker in static IMAGE mode."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_detection_confidence: float = 0.5,
        max_num_hands: int = 1,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand_landmarker.task not found at {self.model_path}"
            )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract_from_rgb(
        self, rgb_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Detect a single hand in an RGB uint8 image and return (normalized_63, handedness).

        Returns (None, None) if no hand is detected or landmarks are degenerate.
        """
        features, handedness, _ = self.extract_with_overlay_from_rgb(rgb_image)
        return features, handedness

    def extract_with_overlay_from_rgb(
        self, rgb_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[str], Optional[list[list[float]]]]:
        """Detect a single hand and return (normalized_63, handedness, 21 image-space [x, y] points)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._detector.detect(mp_image)
        if not result.hand_landmarks:
            return None, None, None

        landmarks = result.hand_landmarks[0]
        handedness: Optional[str] = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name

        overlay_xy = [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in landmarks]

        try:
            features = normalize_landmarks(landmarks, handedness=handedness, as_list=False)
        except ValueError:
            return None, handedness, None

        return features, handedness, overlay_xy

    def extract_from_bgr(
        self, bgr_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Convenience wrapper for OpenCV BGR images."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return self.extract_from_rgb(rgb)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

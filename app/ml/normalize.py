"""Shared hand-landmark normalization used by both training and FastAPI inference.

Coordinate & Handedness Convention
----------------------------------
- Input: 21 MediaPipe hand landmarks with (x, y, z) coordinates, provided as:
  * a flat sequence of 63 floats [x0, y0, z0, ..., x20, y20, z20], OR
  * a (21, 3) array-like of (x, y, z) coordinates, OR
  * a sequence of 21 landmark objects exposing `.x`, `.y`, `.z` attributes.
- Step 1 (Translation invariance): Subtract the wrist (landmark 0) from all 21 points
  so landmark 0 sits at the origin (0.0, 0.0, 0.0).
- Step 2 (Handedness canonicalization): Mirror Left hands onto the canonical Right-hand
  coordinate frame by negating the translated x-coordinates (`x = -x`).
  (Equivalently, flipping `x -> 1 - x` before wrist subtraction yields
  `(1 - x_i) - (1 - x_0) = -(x_i - x_0)`.)
  If `handedness` is None or "Right", coordinates are left unmirrored.
- Step 3 (Scale invariance): Divide all 21 (x, y, z) coordinates by the Euclidean
  distance between the wrist (landmark 0) and the middle-finger MCP joint (landmark 9).
- Step 4 (Fixed ordering): Flatten to a 1D list/array of 63 floats in fixed order:
  [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURE_COUNT = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
MIN_SCALE_EPS = 1e-6


def _to_landmark_array(landmarks: Any) -> np.ndarray:
    """Convert supported landmark representations into a (21, 3) float64 numpy array."""
    if isinstance(landmarks, np.ndarray):
        arr = landmarks.astype(np.float64, copy=True)
        if arr.shape == (FEATURE_COUNT,):
            return arr.reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)
        if arr.shape == (NUM_LANDMARKS, COORDS_PER_LANDMARK):
            return arr
        raise ValueError(
            f"Expected shape (63,) or (21, 3), received {arr.shape}"
        )

    if not isinstance(landmarks, Sequence) or isinstance(landmarks, (str, bytes)):
        raise ValueError("landmarks must be a sequence or numpy array")

    n = len(landmarks)
    if n == FEATURE_COUNT:
        arr = np.asarray(landmarks, dtype=np.float64)
        if arr.shape != (FEATURE_COUNT,):
            raise ValueError(f"Expected 63 numeric values, received shape {arr.shape}")
        return arr.reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)

    if n == NUM_LANDMARKS:
        first = landmarks[0]
        if hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "z"):
            return np.array(
                [[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks],
                dtype=np.float64,
            )
        arr = np.asarray(landmarks, dtype=np.float64)
        if arr.shape != (NUM_LANDMARKS, COORDS_PER_LANDMARK):
            raise ValueError(f"Expected 21 (x, y, z) points, received shape {arr.shape}")
        return arr

    raise ValueError(
        f"Expected 63 floats or 21 (x, y, z) landmarks, received length {n}"
    )


def normalize_landmarks(
    landmarks: Any,
    handedness: Optional[str] = None,
    as_list: bool = False,
) -> np.ndarray | list[float]:
    """Normalize 21 3D hand landmarks into a 63-feature vector.

    Args:
        landmarks: 21 (x, y, z) landmarks (MediaPipe objects, (21, 3) array, or 63 flat floats).
        handedness: Optional hand label ("Left" or "Right"). If "Left" (case-insensitive),
            mirrors the x-axis after wrist centering to match canonical Right-hand orientation.
        as_list: If True, return a Python list[float] of length 63; otherwise return np.ndarray (float32).

    Returns:
        Normalized 63-element feature vector in fixed [x0, y0, z0, ..., x20, y20, z20] order.

    Raises:
        ValueError: If input shape is invalid, contains non-finite values, or wrist-to-middle-MCP
            distance is degenerate (< 1e-6).
    """
    pts = _to_landmark_array(landmarks)

    if not np.all(np.isfinite(pts)):
        raise ValueError("Landmarks contain NaN or infinite values")

    # 1. Subtract wrist (landmark 0) from all points
    pts = pts - pts[WRIST_INDEX]

    # 2. Consistent left/right hand handling: mirror Left -> Right
    if handedness is not None and str(handedness).strip().lower() == "left":
        pts[:, 0] = -pts[:, 0]

    # 3. Divide by Euclidean distance from wrist (0) to middle-finger MCP (9)
    scale = float(np.linalg.norm(pts[MIDDLE_MCP_INDEX]))
    if scale < MIN_SCALE_EPS:
        raise ValueError(
            f"Degenerate hand landmarks: wrist-to-middle-MCP distance is {scale:.2e} (< {MIN_SCALE_EPS})"
        )
    pts = pts / scale

    # 4. Flatten to 63 features in fixed order
    flat = pts.reshape(FEATURE_COUNT).astype(np.float32)

    if as_list:
        return [float(v) for v in flat]
    return flat

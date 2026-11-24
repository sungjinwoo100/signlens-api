"""Phase 6 Unit Tests for app.ml.normalize.normalize_landmarks."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
import numpy as np

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.ml.normalize import FEATURE_COUNT, normalize_landmarks


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float


def _make_known_hand() -> np.ndarray:
    """Create a known 21x3 hand where wrist=(1, 2, 3) and middle MCP (9)=(1, 4, 3) -> scale=2.0."""
    pts = np.zeros((21, 3), dtype=np.float64)
    for i in range(21):
        pts[i] = [1.0 + i * 0.2, 2.0 + i * 0.1, 3.0 - i * 0.05]
    # Explicitly set wrist (0) and middle MCP (9) for exact analytical verification
    pts[0] = [1.0, 2.0, 3.0]
    pts[9] = [1.0, 4.0, 3.0]  # distance from wrist = exactly 2.0
    pts[8] = [3.0, 2.0, 5.0]  # index tip: offset = (+2, 0, +2) -> normalized = (+1, 0, +1)
    return pts


class TestNormalizeLandmarks(unittest.TestCase):
    def test_wrist_centering_and_mcp_scaling(self):
        pts = _make_known_hand()
        out = normalize_landmarks(pts)
        self.assertEqual(out.shape, (FEATURE_COUNT,))
        self.assertEqual(out.dtype, np.float32)

        # Landmark 0 (wrist) must be at origin (0, 0, 0)
        np.testing.assert_allclose(out[0:3], [0.0, 0.0, 0.0], atol=1e-6)
        # Landmark 9 (middle MCP) must be at unit distance (0, 1, 0)
        np.testing.assert_allclose(out[9 * 3 : 9 * 3 + 3], [0.0, 1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(out[9 * 3 : 9 * 3 + 3])), 1.0, places=6)
        # Landmark 8 (index tip): (+2, 0, +2) / 2.0 = (+1, 0, +1)
        np.testing.assert_allclose(out[8 * 3 : 8 * 3 + 3], [1.0, 0.0, 1.0], atol=1e-6)

    def test_translation_and_scale_invariance(self):
        pts = _make_known_hand()
        shifted_scaled = (pts * 3.7) + np.array([-12.5, 44.0, 8.2])
        out_orig = normalize_landmarks(pts)
        out_trans = normalize_landmarks(shifted_scaled)
        np.testing.assert_allclose(out_orig, out_trans, atol=1e-5)

    def test_left_hand_mirroring(self):
        pts = _make_known_hand()
        right = normalize_landmarks(pts, handedness="Right")
        left = normalize_landmarks(pts, handedness="Left")

        # X coordinates (indices 0, 3, 6, ...) are negated; Y and Z are unchanged
        np.testing.assert_allclose(left[0::3], -right[0::3], atol=1e-6)
        np.testing.assert_allclose(left[1::3], right[1::3], atol=1e-6)
        np.testing.assert_allclose(left[2::3], right[2::3], atol=1e-6)

        # Index tip (landmark 8) becomes (-1, 0, +1) when mirrored
        np.testing.assert_allclose(left[8 * 3 : 8 * 3 + 3], [-1.0, 0.0, 1.0], atol=1e-6)

    def test_input_formats_produce_identical_order(self):
        pts = _make_known_hand()
        flat_list = pts.reshape(-1).tolist()
        obj_list = [_FakeLandmark(float(r[0]), float(r[1]), float(r[2])) for r in pts]

        out_arr = normalize_landmarks(pts)
        out_flat = normalize_landmarks(flat_list)
        out_obj = normalize_landmarks(obj_list)
        out_pylist = normalize_landmarks(pts, as_list=True)

        np.testing.assert_allclose(out_arr, out_flat, atol=1e-6)
        np.testing.assert_allclose(out_arr, out_obj, atol=1e-6)
        self.assertIsInstance(out_pylist, list)
        self.assertEqual(len(out_pylist), 63)

    def test_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            normalize_landmarks([0.0] * 60)  # wrong length

        pts_zero_scale = np.ones((21, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            normalize_landmarks(pts_zero_scale)  # wrist == middle MCP

        pts_nan = _make_known_hand()
        pts_nan[3, 1] = float("nan")
        with self.assertRaises(ValueError):
            normalize_landmarks(pts_nan)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""ML inference service: loads the trained ASL classifier and MediaPipe HandLandmarker once at startup."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
from fastapi import HTTPException, status

from app.core.config import settings
from app.ml.hand_detector import HandLandmarkDetector
from app.ml.mlp_model import ASLLandmarkMLP
from app.ml.normalize import normalize_landmarks

logger = logging.getLogger(__name__)


class MLService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        mediapipe_model_path: Optional[str] = None,
        max_image_dim: Optional[int] = None,
    ):
        api_root = Path(__file__).resolve().parents[2]
        raw_model_path = model_path or settings.ML_MODEL_PATH
        raw_mp_path = mediapipe_model_path or settings.MEDIAPIPE_MODEL_PATH

        self.model_path = (
            raw_model_path
            if os.path.isabs(raw_model_path)
            else str(api_root / raw_model_path)
        )
        self.mediapipe_model_path = (
            raw_mp_path
            if os.path.isabs(raw_mp_path)
            else str(api_root / raw_mp_path)
        )
        self.max_image_dim = max_image_dim or settings.MAX_IMAGE_DIM

        self.model = None
        self.model_type: Optional[str] = None  # "pytorch" or "sklearn"
        self.labels: List[str] = []
        self.is_loaded: bool = False

        self.hand_detector: Optional[HandLandmarkDetector] = None
        self.mediapipe_loaded: bool = False
        self._mp_lock = threading.Lock()

    def load_model(self) -> bool:
        """Load both the gesture classifier and MediaPipe HandLandmarker once at startup."""
        clf_ok = self._load_classifier()
        self._load_mediapipe()
        return clf_ok

    def _load_classifier(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.warning(
                "ML model file not found at '%s'. Prediction endpoint will return HTTP 503.",
                self.model_path,
            )
            self.model = None
            self.is_loaded = False
            return False

        try:
            if self.model_path.endswith(".pt"):
                ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
                mlp = ASLLandmarkMLP(
                    input_dim=int(ckpt.get("input_dim", 63)),
                    num_classes=int(ckpt["num_classes"]),
                    hidden_dims=tuple(ckpt.get("hidden_dims", [128, 64])),
                    dropout=float(ckpt.get("dropout", 0.2)),
                )
                mlp.load_state_dict(ckpt["model_state_dict"])
                mlp.eval()
                self.model = mlp
                self.labels = list(ckpt["labels"])
                self.model_type = "pytorch"
            else:
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self.model = loaded["model"]
                    self.labels = list(loaded.get("labels", []))
                else:
                    self.model = loaded
                    self.labels = (
                        [str(c) for c in loaded.classes_]
                        if hasattr(loaded, "classes_")
                        else []
                    )
                self.model_type = "sklearn"

            self.is_loaded = True
            logger.info(
                "Loaded %s ASL classifier from '%s' (%d classes).",
                self.model_type,
                self.model_path,
                len(self.labels),
            )
            return True
        except Exception as e:
            logger.error("Failed to load ML model from '%s': %s", self.model_path, e)
            self.model = None
            self.is_loaded = False
            return False

    def _load_mediapipe(self) -> bool:
        if not os.path.exists(self.mediapipe_model_path):
            logger.warning(
                "MediaPipe hand_landmarker.task not found at '%s'. Image frame prediction unavailable.",
                self.mediapipe_model_path,
            )
            self.hand_detector = None
            self.mediapipe_loaded = False
            return False

        try:
            self.hand_detector = HandLandmarkDetector(
                model_path=self.mediapipe_model_path,
                min_detection_confidence=0.5,
                max_num_hands=1,
            )
            self.mediapipe_loaded = True
            logger.info(
                "Loaded MediaPipe HandLandmarker from '%s'.",
                self.mediapipe_model_path,
            )
            return True
        except Exception as e:
            logger.error("Failed to load MediaPipe HandLandmarker: %s", e)
            self.hand_detector = None
            self.mediapipe_loaded = False
            return False

    def close(self) -> None:
        if self.hand_detector is not None:
            self.hand_detector.close()
            self.hand_detector = None
            self.mediapipe_loaded = False

    def _ensure_classifier_ready(self) -> None:
        if not self.is_loaded or self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"SignLens detection model is currently unavailable. "
                    f"No trained model was found at '{self.model_path}'."
                ),
            )

    def _classify_normalized(
        self, features: np.ndarray, top_k: int = 3
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Run classifier inference on a normalized (63,) float32 vector.

        Returns (top1_label, top1_confidence, top_k_list).
        """
        self._ensure_classifier_ready()
        if self.model_type == "pytorch":
            with torch.no_grad():
                tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()
            k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:k]
            top_list = [
                (self.labels[int(i)], round(float(probs[int(i)]), 4))
                for i in top_indices
            ]
            return top_list[0][0], top_list[0][1], top_list

        arr = features.astype(np.float32).reshape(1, -1)
        prediction = self.model.predict(arr)
        pred_val = prediction[0]
        if self.labels and isinstance(pred_val, (int, np.integer)):
            label = self.labels[int(pred_val)]
        else:
            label = str(pred_val)
        confidence = 1.0
        top_list: List[Tuple[str, float]] = [(label, 1.0)]
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(arr)[0]
            k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:k]
            classes = (
                self.labels
                if self.labels
                else [str(c) for c in getattr(self.model, "classes_", range(len(probs)))]
            )
            top_list = [
                (classes[int(i)], round(float(probs[int(i)]), 4))
                for i in top_indices
            ]
            label, confidence = top_list[0]
        return label, round(confidence, 4), top_list

    def predict(
        self,
        landmarks: List[float],
        handedness: Optional[str] = None,
        already_normalized: bool = False,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Classify 63 landmark floats (normalizes with normalize_landmarks unless already_normalized)."""
        self._ensure_classifier_ready()
        try:
            if already_normalized:
                features = np.asarray(landmarks, dtype=np.float32)
            else:
                features = normalize_landmarks(
                    landmarks, handedness=handedness, as_list=False
                )
            return self._classify_normalized(features)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid landmarks: {e}",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error during landmark inference: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {e}",
            )

    def _decode_and_resize_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 JPEG/PNG into an RGB uint8 numpy array, resizing down for low latency."""
        payload = image_base64.strip()
        if "," in payload and payload.startswith("data:"):
            payload = payload.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(payload, validate=False)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 image payload: {e}",
            )

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image bytes (expected JPEG or PNG).",
            )

        h, w = bgr.shape[:2]
        max_side = max(h, w)
        if max_side > self.max_image_dim:
            scale = self.max_image_dim / float(max_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def predict_from_image_base64(
        self, image_base64: str
    ) -> Tuple[str, float, bool, float, List[Tuple[str, float]], Optional[List[List[float]]]]:
        """Run MediaPipe + normalize_landmarks + classifier on a base64 camera frame.

        Returns:
            (label, confidence, hand_detected, latency_ms, top_predictions, hand_landmarks)
        """
        t0 = time.perf_counter()
        self._ensure_classifier_ready()
        if not self.mediapipe_loaded or self.hand_detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MediaPipe HandLandmarker is not loaded on the server.",
            )

        rgb = self._decode_and_resize_image(image_base64)

        with self._mp_lock:
            features, _handedness, overlay_xy = (
                self.hand_detector.extract_with_overlay_from_rgb(rgb)
            )

        if features is None:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return "nothing", 1.0, False, latency_ms, [], None

        label, confidence, top_list = self._classify_normalized(features, top_k=3)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return label, confidence, True, latency_ms, top_list, overlay_xy


# Global singleton instance loaded once at application startup
ml_service = MLService()

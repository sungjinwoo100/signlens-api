"""Webcam utility to live-test the trained ASL model or collect real-world webcam samples.

Usage:
  Live test on webcam:
    python ml/collect_webcam_samples.py --mode test

  Collect 50 labeled samples for letter 'A' (press SPACE to start/pause, Q to quit):
    python ml/collect_webcam_samples.py --mode collect --label A --count 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.ml.hand_detector import HandLandmarkDetector
from app.ml.mlp_model import ASLLandmarkMLP

MODELS_DIR = Path(__file__).resolve().parent / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
WEBCAM_CSV = ARTIFACTS_DIR / "webcam_samples.csv"
FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def load_best_classifier():
    labels_info = json.loads((MODELS_DIR / "labels.json").read_text())
    labels = labels_info["labels"]
    model_type = labels_info["best_model_type"]

    if model_type == "pytorch":
        ckpt = torch.load(MODELS_DIR / "asl_mlp.pt", map_location="cpu", weights_only=True)
        model = ASLLandmarkMLP(
            input_dim=ckpt["input_dim"],
            num_classes=ckpt["num_classes"],
            hidden_dims=tuple(ckpt["hidden_dims"]),
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        def predict_fn(features: np.ndarray):
            with torch.no_grad():
                logits = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0].numpy()
            idx = int(np.argmax(probs))
            return labels[idx], float(probs[idx])

        return predict_fn
    else:
        bundle = joblib.load(MODELS_DIR / "asl_rf.joblib")
        rf = bundle["model"]

        def predict_fn(features: np.ndarray):
            probs = rf.predict_proba(features.reshape(1, -1))[0]
            idx = int(np.argmax(probs))
            return labels[idx], float(probs[idx])

        return predict_fn


def run_live_test(camera_index: int = 0) -> None:
    predict_fn = load_best_classifier()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    print("Live webcam test started. Press 'q' to quit.")
    with HandLandmarkDetector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Selfie mirror view
            features, handedness = detector.extract_from_bgr(frame)

            if features is None:
                text = "No hand detected (nothing)"
                color = (0, 165, 255)
            else:
                label, conf = predict_fn(features)
                text = f"Sign: {label} ({conf*100:.1f}%) [{handedness}]"
                color = (0, 220, 0) if conf >= 0.7 else (0, 215, 255)

            cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("SignLens - Live ASL Test (press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_collect(label: str, count: int, camera_index: int = 0) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not WEBCAM_CSV.exists()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    collected = 0
    recording = False
    print(f"Collecting {count} webcam samples for label '{label}'.")
    print("Press SPACE to start/pause recording, Q to quit.")

    with HandLandmarkDetector() as detector, WEBCAM_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["label", "handedness", "source_file"] + FEATURE_COLS)

        while collected < count:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            features, handedness = detector.extract_from_bgr(frame)

            if recording and features is not None:
                writer.writerow(
                    [label, handedness or "Unknown", f"webcam_{label}_{collected}"]
                    + [round(float(v), 6) for v in features]
                )
                collected += 1

            status = f"REC {collected}/{count}" if recording else f"PAUSED {collected}/{count} (SPACE=rec)"
            cv2.putText(
                frame,
                f"Label: {label} | {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if recording else (255, 255, 0),
                2,
            )
            cv2.imshow("SignLens - Webcam Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {collected} samples for '{label}' to {WEBCAM_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignLens webcam testing & sample collector")
    parser.add_argument("--mode", choices=["test", "collect"], default="test")
    parser.add_argument("--label", type=str, default="A")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "test":
        run_live_test(args.camera)
    else:
        run_collect(args.label.upper(), args.count, args.camera)


if __name__ == "__main__":
    main()

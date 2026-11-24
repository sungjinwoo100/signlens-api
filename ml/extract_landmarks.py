"""Phase 2: Batch, resume-able MediaPipe hand landmark extraction into landmarks.csv.

Strategy for 'nothing' class:
    Uses the `hand_detected=false` rule rather than synthetic all-zero rows.
    MediaPipe's hand detector runs on the 'nothing' folder to verify its true-negative
    (no-hand) rate, while only active sign classes (24 static ASL letters A-Y excl. J, Z)
    are written to landmarks.csv for the landmark classifier.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm

# Ensure signlens_api is importable
API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.ml.hand_detector import DEFAULT_MODEL_PATH, HandLandmarkDetector
from app.ml.normalize import FEATURE_COUNT

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PHASE1_SUMMARY = ARTIFACTS_DIR / "phase1_summary.json"
CSV_PATH = ARTIFACTS_DIR / "landmarks.csv"
CHECKPOINT_PATH = ARTIFACTS_DIR / "extraction_checkpoint.json"
REPORT_PATH = ARTIFACTS_DIR / "phase2_extraction_report.json"

FEATURE_COLS = [
    f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
]
CSV_HEADER = ["label", "handedness", "source_file"] + FEATURE_COLS

# Worker-local detector singleton
_WORKER_DETECTOR: HandLandmarkDetector | None = None


def _init_worker(model_path: str) -> None:
    global _WORKER_DETECTOR
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    _WORKER_DETECTOR = HandLandmarkDetector(
        model_path=model_path,
        min_detection_confidence=0.5,
        max_num_hands=1,
    )


def _process_batch(
    args: Tuple[str, List[str]]
) -> Tuple[str, List[List[object]], int, int, Dict[str, int]]:
    """Process a batch of image paths for a single class."""
    cls_name, paths = args
    assert _WORKER_DETECTOR is not None
    rows: List[List[object]] = []
    skipped = 0
    handedness_counts: Dict[str, int] = {"Left": 0, "Right": 0, "Unknown": 0}

    for img_path in paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            skipped += 1
            continue
        features, handedness = _WORKER_DETECTOR.extract_from_bgr(bgr)
        if features is None:
            skipped += 1
            continue
        hand_key = handedness if handedness in ("Left", "Right") else "Unknown"
        handedness_counts[hand_key] += 1
        rows.append([cls_name, hand_key, os.path.basename(img_path)] + [round(float(v), 6) for v in features])

    return cls_name, rows, skipped, len(paths), handedness_counts


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {"completed_classes": {}, "nothing_eval": None}


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2))


def main() -> None:
    t0 = time.time()
    if not PHASE1_SUMMARY.exists():
        raise FileNotFoundError(
            f"Run Phase 1 first: {PHASE1_SUMMARY} not found."
        )

    phase1 = json.loads(PHASE1_SUMMARY.read_text())
    train_dir = Path(phase1["train_dir"])
    included_classes: List[str] = phase1["included_classes"]
    active_classes = [c for c in included_classes if c != "nothing"]

    checkpoint = load_checkpoint()
    completed = checkpoint["completed_classes"]

    # Initialize CSV with header if starting fresh
    if not CSV_PATH.exists() or not completed:
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        completed = {}
        checkpoint["completed_classes"] = completed

    num_workers = max(1, min((os.cpu_count() or 4) - 1, 6))
    batch_size = 150
    print(f"Using {num_workers} worker processes (batch_size={batch_size}).")
    print(f"Active sign classes: {len(active_classes)} | Already completed: {len(completed)}")

    remaining_classes = [c for c in active_classes if c not in completed]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(str(DEFAULT_MODEL_PATH),),
    ) as executor:
        # 1. Extract active sign classes (A-Y excluding J, Z)
        for cls in remaining_classes:
            cls_dir = train_dir / cls
            img_paths = sorted(str(p) for p in cls_dir.glob("*.jpg"))
            batches = [
                (cls, img_paths[i : i + batch_size])
                for i in range(0, len(img_paths), batch_size)
            ]

            cls_rows: List[List[object]] = []
            cls_skipped = 0
            cls_total = 0
            cls_hand_counts = {"Left": 0, "Right": 0, "Unknown": 0}

            futures = [executor.submit(_process_batch, b) for b in batches]
            with tqdm(total=len(img_paths), desc=f"Class {cls:>2}", unit="img", leave=False) as pbar:
                for fut in as_completed(futures):
                    _, rows, skipped, count, h_counts = fut.result()
                    cls_rows.extend(rows)
                    cls_skipped += skipped
                    cls_total += count
                    for k, v in h_counts.items():
                        cls_hand_counts[k] += v
                    pbar.update(count)

            # Append completed class rows atomically to CSV
            with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(cls_rows)

            skip_rate = (cls_skipped / cls_total * 100.0) if cls_total else 0.0
            warn_flag = skip_rate > 30.0
            completed[cls] = {
                "total": cls_total,
                "extracted": len(cls_rows),
                "skipped": cls_skipped,
                "skip_rate_pct": round(skip_rate, 2),
                "warning_over_30pct": warn_flag,
                "handedness": cls_hand_counts,
            }
            save_checkpoint(checkpoint)

            warn_tag = " [WARNING >30% SKIPPED!]" if warn_flag else ""
            print(
                f"  Class {cls:>2}: extracted={len(cls_rows):>4}/{cls_total} "
                f"| skipped={cls_skipped:>4} ({skip_rate:5.1f}%) "
                f"| L/R={cls_hand_counts['Left']}/{cls_hand_counts['Right']}{warn_tag}"
            )

        # 2. Verify 'nothing' class true-negative rate (hand_detected=false rule)
        if checkpoint.get("nothing_eval") is None and "nothing" in included_classes:
            nothing_dir = train_dir / "nothing"
            nothing_paths = sorted(str(p) for p in nothing_dir.glob("*.jpg"))
            batches = [
                ("nothing", nothing_paths[i : i + batch_size])
                for i in range(0, len(nothing_paths), batch_size)
            ]
            false_positives = 0
            total_nothing = 0
            futures = [executor.submit(_process_batch, b) for b in batches]
            with tqdm(total=len(nothing_paths), desc="Class nothing", unit="img", leave=False) as pbar:
                for fut in as_completed(futures):
                    _, rows, _, count, _ = fut.result()
                    false_positives += len(rows)
                    total_nothing += count
                    pbar.update(count)

            no_hand_count = total_nothing - false_positives
            checkpoint["nothing_eval"] = {
                "strategy": "hand_detected=false rule (excluded from landmarks.csv)",
                "total_images": total_nothing,
                "correctly_no_hand": no_hand_count,
                "false_positive_hands": false_positives,
                "true_negative_rate_pct": round(no_hand_count / total_nothing * 100.0, 2) if total_nothing else 0.0,
            }
            save_checkpoint(checkpoint)
            print(
                f"  Class nothing: correctly rejected (no hand) = {no_hand_count}/{total_nothing} "
                f"({checkpoint['nothing_eval']['true_negative_rate_pct']:.2f}%)"
            )

    # Final summary report
    total_images = sum(v["total"] for v in completed.values())
    total_extracted = sum(v["extracted"] for v in completed.values())
    total_skipped = sum(v["skipped"] for v in completed.values())
    warned_classes = [c for c, v in completed.items() if v["warning_over_30pct"]]
    elapsed = time.time() - t0

    report = {
        "csv_path": str(CSV_PATH),
        "feature_count": FEATURE_COUNT,
        "active_classes_count": len(completed),
        "total_active_images": total_images,
        "total_extracted_rows": total_extracted,
        "total_skipped": total_skipped,
        "overall_skip_rate_pct": round(total_skipped / total_images * 100.0, 2) if total_images else 0.0,
        "classes_over_30pct_loss": warned_classes,
        "per_class": completed,
        "nothing_evaluation": checkpoint.get("nothing_eval"),
        "elapsed_seconds": round(elapsed, 1),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== PHASE 2 EXTRACTION SUMMARY ===")
    print(f"CSV saved to          : {CSV_PATH}")
    print(f"Total active images   : {total_images:,}")
    print(f"Total extracted rows  : {total_extracted:,}")
    print(f"Total skipped         : {total_skipped:,} ({report['overall_skip_rate_pct']:.2f}%)")
    print(f"Classes >30% loss     : {warned_classes if warned_classes else 'None'}")
    if checkpoint.get("nothing_eval"):
        ne = checkpoint["nothing_eval"]
        print(
            f"'nothing' no-hand rate: {ne['correctly_no_hand']}/{ne['total_images']} "
            f"({ne['true_negative_rate_pct']:.2f}%)"
        )
    print(f"Elapsed time          : {elapsed:.1f}s")
    print(f"Report saved to       : {REPORT_PATH}")


if __name__ == "__main__":
    main()

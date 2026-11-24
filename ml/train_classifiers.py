"""Phase 3: Train and compare Random Forest vs PyTorch MLP (with & without landmark augmentation).

Outputs:
  - ml/models/best_model.pt (or .joblib) + ml/models/labels.json
  - ml/artifacts/confusion_matrix.png
  - ml/artifacts/phase3_training_report.json
  - ml/artifacts/phase3_training_report.md
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.ml.mlp_model import ASLLandmarkMLP

SEED = 42
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODELS_DIR = Path(__file__).resolve().parent / "models"
CSV_PATH = ARTIFACTS_DIR / "landmarks.csv"
FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_landmarks_batch(
    X: np.ndarray,
    rng: np.random.Generator,
    max_rot_deg: float = 12.0,
    scale_range: Tuple[float, float] = (0.90, 1.10),
    jitter_std: float = 0.015,
) -> np.ndarray:
    """Apply small 2D in-plane rotation, isotropic scale, and Gaussian jitter to (N, 63) landmarks.

    Wrist (landmark 0) stays anchored at (0, 0, 0) since rotations/scales are about the origin.
    """
    pts = X.reshape(-1, 21, 3).copy()
    n = pts.shape[0]

    # Random in-plane rotation in XY around wrist
    angles = rng.uniform(-max_rot_deg, max_rot_deg, size=n) * (math.pi / 180.0)
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    pts[:, :, 0] = x * cos_a - y * sin_a
    pts[:, :, 1] = x * sin_a + y * cos_a

    # Random small scale factor
    scales = rng.uniform(scale_range[0], scale_range[1], size=(n, 1, 1))
    pts = pts * scales

    # Small Gaussian jitter on non-wrist joints
    noise = rng.normal(0.0, jitter_std, size=pts.shape)
    noise[:, 0, :] = 0.0
    pts = pts + noise

    return pts.reshape(n, 63).astype(np.float32)


def make_augmented_dataset(
    X: np.ndarray, y: np.ndarray, copies: int = 1, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_list = [X]
    y_list = [y]
    for _ in range(copies):
        X_list.append(augment_landmarks_batch(X, rng))
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[RandomForestClassifier, dict, np.ndarray]:
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)
    elapsed = time.time() - t0
    metrics = {
        "val_accuracy": round(float(accuracy_score(y_val, val_pred)), 4),
        "val_macro_f1": round(float(f1_score(y_val, val_pred, average="macro")), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
        "test_macro_f1": round(float(f1_score(y_test, test_pred, average="macro")), 4),
        "test_weighted_f1": round(float(f1_score(y_test, test_pred, average="weighted")), 4),
        "train_seconds": round(elapsed, 2),
    }
    return rf, metrics, test_pred


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    use_augmentation: bool = False,
    epochs: int = 45,
    batch_size: int = 256,
) -> Tuple[ASLLandmarkMLP, dict, np.ndarray]:
    set_seed(SEED)
    t0 = time.time()
    rng = np.random.default_rng(SEED + (100 if use_augmentation else 0))

    # Inverse-frequency class weights to handle uneven extraction counts (e.g. M, N)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = ASLLandmarkMLP(input_dim=63, num_classes=num_classes, hidden_dims=(128, 64), dropout=0.2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    early_stop_patience = 8

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            if use_augmentation:
                # Apply on-the-fly augmentation to 50% of batch
                xb_np = xb.numpy()
                mask = rng.random(len(xb_np)) < 0.5
                if mask.any():
                    xb_np[mask] = augment_landmarks_batch(xb_np[mask], rng)
                xb = torch.tensor(xb_np, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(dim=1).numpy()
        val_acc = float(accuracy_score(y_val, val_preds))
        scheduler.step(val_acc)

        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(dim=1).numpy()
        test_preds = model(X_test_t).argmax(dim=1).numpy()

    elapsed = time.time() - t0
    metrics = {
        "val_accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
        "val_macro_f1": round(float(f1_score(y_val, val_preds, average="macro")), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "test_macro_f1": round(float(f1_score(y_test, test_preds, average="macro")), 4),
        "test_weighted_f1": round(float(f1_score(y_test, test_preds, average="weighted")), 4),
        "epochs_run": epoch,
        "train_seconds": round(elapsed, 2),
    }
    return model, metrics, test_preds


def save_confusion_matrix_plot(
    cm: np.ndarray, labels: List[str], out_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > thresh else "black",
                )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, labels: List[str], top_k: int = 12) -> List[dict]:
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": labels[i],
                        "predicted": labels[j],
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top_k]


def main() -> None:
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    labels = sorted(df["label"].unique().tolist())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)
    print(f"Loaded {len(X):,} samples across {len(labels)} classes.")

    # Stratified 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(
        f"Stratified split (seed={SEED}): "
        f"train={len(X_train):,} (70%), val={len(X_val):,} (15%), test={len(X_test):,} (15%)"
    )

    # 1. Random Forest (baseline)
    print("\n[1/4] Training Random Forest (baseline)...")
    rf_base, rf_base_metrics, rf_base_preds = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print("  RF baseline:", rf_base_metrics)

    # 2. Random Forest (+ landmark augmentation)
    print("\n[2/4] Training Random Forest (+ landmark augmentation)...")
    X_train_aug, y_train_aug = make_augmented_dataset(X_train, y_train, copies=1, seed=SEED)
    rf_aug, rf_aug_metrics, rf_aug_preds = train_random_forest(
        X_train_aug, y_train_aug, X_val, y_val, X_test, y_test
    )
    print("  RF + aug   :", rf_aug_metrics)

    # 3. PyTorch MLP (baseline)
    print("\n[3/4] Training PyTorch MLP (baseline)...")
    mlp_base, mlp_base_metrics, mlp_base_preds = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=len(labels), use_augmentation=False,
    )
    print("  MLP base   :", mlp_base_metrics)

    # 4. PyTorch MLP (+ on-the-fly landmark augmentation)
    print("\n[4/4] Training PyTorch MLP (+ landmark augmentation)...")
    mlp_aug, mlp_aug_metrics, mlp_aug_preds = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=len(labels), use_augmentation=True,
    )
    print("  MLP + aug  :", mlp_aug_metrics)

    # Select best model by validation macro-F1 (tie-break by validation accuracy)
    candidates = [
        ("rf_baseline", "sklearn", rf_base, rf_base_metrics, rf_base_preds),
        ("rf_augmented", "sklearn", rf_aug, rf_aug_metrics, rf_aug_preds),
        ("mlp_baseline", "pytorch", mlp_base, mlp_base_metrics, mlp_base_preds),
        ("mlp_augmented", "pytorch", mlp_aug, mlp_aug_metrics, mlp_aug_preds),
    ]
    candidates.sort(
        key=lambda item: (item[3]["val_macro_f1"], item[3]["val_accuracy"]),
        reverse=True,
    )
    best_name, best_kind, best_model, best_metrics, best_test_preds = candidates[0]
    print(f"\nBest model selected: {best_name} ({best_kind}) -> test_acc={best_metrics['test_accuracy']:.4f}")

    # Save labels.json
    labels_path = MODELS_DIR / "labels.json"
    labels_payload = {
        "labels": labels,
        "excluded_classes": ["J", "Z", "del", "space"],
        "nothing_strategy": "hand_detected=false",
        "best_model_name": best_name,
        "best_model_type": best_kind,
    }
    labels_path.write_text(json.dumps(labels_payload, indent=2))

    # Save best model artifact
    if best_kind == "pytorch":
        best_model_path = MODELS_DIR / "asl_mlp.pt"
        torch.save(
            {
                "model_state_dict": best_model.state_dict(),
                "input_dim": 63,
                "num_classes": len(labels),
                "hidden_dims": [128, 64],
                "dropout": 0.2,
                "labels": labels,
                "model_name": best_name,
            },
            best_model_path,
        )
    else:
        best_model_path = MODELS_DIR / "asl_rf.joblib"
        joblib.dump({"model": best_model, "labels": labels, "model_name": best_name}, best_model_path)

    # Also save the best sklearn RF for quick comparison/fallback
    best_rf_obj = rf_aug if rf_aug_metrics["val_macro_f1"] >= rf_base_metrics["val_macro_f1"] else rf_base
    joblib.dump({"model": best_rf_obj, "labels": labels}, MODELS_DIR / "asl_rf_backup.joblib")

    # Detailed evaluation on test set
    cls_report = classification_report(
        y_test, best_test_preds, target_names=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, best_test_preds)
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    save_confusion_matrix_plot(
        cm, labels, cm_path, title=f"SignLens ASL Confusion Matrix ({best_name})"
    )
    confusions = top_confusions(cm, labels, top_k=12)

    report = {
        "seed": SEED,
        "split_counts": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "models_comparison": {
            "rf_baseline": rf_base_metrics,
            "rf_augmented": rf_aug_metrics,
            "mlp_baseline": mlp_base_metrics,
            "mlp_augmented": mlp_aug_metrics,
        },
        "best_model": {
            "name": best_name,
            "type": best_kind,
            "path": str(best_model_path),
            "labels_path": str(labels_path),
            "metrics": best_metrics,
        },
        "per_class_report": {
            lbl: {
                "precision": round(cls_report[lbl]["precision"], 4),
                "recall": round(cls_report[lbl]["recall"], 4),
                "f1": round(cls_report[lbl]["f1-score"], 4),
                "support": int(cls_report[lbl]["support"]),
            }
            for lbl in labels
        },
        "top_confusions": confusions,
        "confusion_matrix_image": str(cm_path),
        "domain_shift_warning": (
            "The grassknoted/asl-alphabet dataset was captured in controlled indoor lighting "
            "with a uniform background and a single signer. Expect real-world webcam accuracy "
            "to be lower than the held-out test accuracy until fine-tuned with diverse webcam samples."
        ),
    }

    report_json_path = ARTIFACTS_DIR / "phase3_training_report.json"
    report_json_path.write_text(json.dumps(report, indent=2))

    # Write concise human-readable markdown report
    md_lines = [
        "# SignLens Phase 3 Training Report",
        "",
        f"- **Dataset split (stratified, seed={SEED})**: Train `{len(X_train):,}` (70%) / Val `{len(X_val):,}` (15%) / Test `{len(X_test):,}` (15%)",
        f"- **Best Model**: `{best_name}` (`{best_kind}`) saved to `{best_model_path}`",
        "",
        "## Model Comparison",
        "| Model | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, _, _, m, _ in candidates:
        md_lines.append(
            f"| `{name}` | {m['val_accuracy']*100:.2f}% | {m['val_macro_f1']*100:.2f}% | "
            f"{m['test_accuracy']*100:.2f}% | {m['test_macro_f1']*100:.2f}% | {m['train_seconds']}s |"
        )
    md_lines += [
        "",
        "## Top Confused Letter Pairs (Test Set)",
        "| True | Predicted | Count |",
        "| :---: | :---: | :---: |",
    ]
    for pair in confusions:
        md_lines.append(f"| {pair['true']} | {pair['predicted']} | {pair['count']} |")

    md_lines += [
        "",
        "## Per-Class Test Precision / Recall / F1 (Best Model)",
        "| Class | Precision | Recall | F1 | Support |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for lbl in labels:
        r = report["per_class_report"][lbl]
        md_lines.append(
            f"| {lbl} | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% | {r['f1']*100:.2f}% | {r['support']} |"
        )

    md_lines += [
        "",
        "## Domain-Shift Warning",
        report["domain_shift_warning"],
    ]
    report_md_path = ARTIFACTS_DIR / "phase3_training_report.md"
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nSaved best model : {best_model_path}")
    print(f"Saved labels     : {labels_path}")
    print(f"Saved CM plot    : {cm_path}")
    print(f"Saved report     : {report_json_path}")
    print(f"Top confusions   : {confusions[:6]}")


if __name__ == "__main__":
    main()

"""Phase 6 API Test: Test POST /api/v1/predict across all 25 classes (1 sample image per class).

Reports per-class result, overall accuracy, and latency statistics (mean, median, p95).
Run with FastAPI server listening on http://127.0.0.1:8000.
"""

from __future__ import annotations

import base64
import json
import statistics
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
PHASE1_SUMMARY = (
    Path(__file__).resolve().parents[1] / "ml" / "artifacts" / "phase1_summary.json"
)


def post_predict(payload: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    client_ms = (time.perf_counter() - t0) * 1000.0
    return body, client_ms


def main() -> None:
    phase1 = json.loads(PHASE1_SUMMARY.read_text())
    train_dir = Path(phase1["train_dir"])
    classes = phase1["included_classes"]  # 24 static letters + 'nothing' = 25 classes

    # Warm-up request so cold-start JIT doesn't skew latency stats
    warmup_img = base64.b64encode((train_dir / "A" / "A100.jpg").read_bytes()).decode("ascii")
    post_predict({"image_base64": warmup_img})

    correct = 0
    total = 0
    server_latencies = []
    client_latencies = []
    mismatches = []

    print(f"{'Class':<8} {'Pred':<8} {'Conf':>7} {'Hand':<6} {'Server ms':>10} {'Client ms':>10}  Status")
    print("-" * 68)

    for cls in classes:
        img_path = train_dir / cls / f"{cls}1500.jpg"
        if not img_path.exists():
            imgs = sorted((train_dir / cls).glob("*.jpg"))
            img_path = imgs[len(imgs) // 2]

        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        res, client_ms = post_predict({"image_base64": b64})

        pred = res["label"]
        conf = res["confidence"]
        hand = res["hand_detected"]
        srv_ms = res.get("latency_ms", client_ms)

        if cls == "nothing":
            is_ok = (not hand) and (pred == "nothing")
        else:
            is_ok = hand and (pred == cls)

        total += 1
        if is_ok:
            correct += 1
        else:
            mismatches.append({"true": cls, "pred": pred, "conf": conf, "hand": hand})

        server_latencies.append(srv_ms)
        client_latencies.append(client_ms)

        status = "PASS" if is_ok else "MISS"
        print(
            f"{cls:<8} {pred:<8} {conf*100:>6.1f}% {str(hand):<6} {srv_ms:>9.1f} {client_ms:>9.1f}  {status}"
        )

    sorted_srv = sorted(server_latencies)
    p95_idx = min(len(sorted_srv) - 1, int(round(0.95 * (len(sorted_srv) - 1))))

    print("\n=== PHASE 6 /predict 25-CLASS SUMMARY ===")
    print(f"Overall Accuracy : {correct}/{total} ({correct/total*100:.1f}%)")
    print(
        f"Server Latency   : mean={statistics.mean(server_latencies):.1f} ms | "
        f"median={statistics.median(server_latencies):.1f} ms | "
        f"p95={sorted_srv[p95_idx]:.1f} ms"
    )
    print(
        f"End-to-End (HTTP): mean={statistics.mean(client_latencies):.1f} ms | "
        f"median={statistics.median(client_latencies):.1f} ms"
    )
    if mismatches:
        print(f"Mismatches ({len(mismatches)})  : {mismatches}")


if __name__ == "__main__":
    main()

"""Phase 1: Download and verify the grassknoted/asl-alphabet dataset from Kaggle."""

import json
import os
import random
from pathlib import Path
from PIL import Image
import kagglehub

EXCLUDED_CLASSES = {"J", "Z", "del", "space"}


def find_train_dir(dataset_root: Path) -> Path:
    """Locate asl_alphabet_train inside the downloaded kagglehub directory."""
    candidates = [
        dataset_root / "asl_alphabet_train" / "asl_alphabet_train",
        dataset_root / "asl_alphabet_train",
        dataset_root,
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "A").is_dir() and (cand / "nothing").is_dir():
            return cand
    raise FileNotFoundError(f"Could not locate asl_alphabet_train under {dataset_root}")


def main():
    print("Downloading grassknoted/asl-alphabet via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    print(f"Dataset root: {dataset_path}")

    train_dir = find_train_dir(dataset_path)
    print(f"Train directory: {train_dir}")

    all_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    included_classes = [c for c in all_classes if c not in EXCLUDED_CLASSES]
    excluded_found = [c for c in all_classes if c in EXCLUDED_CLASSES]

    class_counts = {}
    sizes_seen = set()
    modes_seen = set()
    total_all = 0
    total_included = 0
    rng = random.Random(42)

    for cls in all_classes:
        cls_dir = train_dir / cls
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        count = len(imgs)
        class_counts[cls] = count
        total_all += count
        if cls in included_classes:
            total_included += count

        # Sample up to 10 images per class to verify image dimensions and color mode
        for sample_path in rng.sample(imgs, min(10, count)):
            with Image.open(sample_path) as im:
                sizes_seen.add(im.size)
                modes_seen.add(im.mode)

    # Build a 4x4 visual inspection grid from included classes
    sample_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "O", "V", "Y", "nothing"]
    grid = Image.new("RGB", (4 * 200, 4 * 200))
    for idx, cls in enumerate(sample_classes):
        cls_dir = train_dir / cls
        if not cls_dir.is_dir():
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))
        if imgs:
            # Pick a middle image (e.g. index 500) to avoid edge artifacts
            pick = imgs[min(500, len(imgs) - 1)]
            with Image.open(pick) as im:
                im_rgb = im.convert("RGB").resize((200, 200))
                r, c = divmod(idx, 4)
                grid.paste(im_rgb, (c * 200, r * 200))

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "phase1_sample_grid.jpg"
    grid.save(grid_path, quality=90)

    summary = {
        "dataset_root": str(dataset_path),
        "train_dir": str(train_dir),
        "total_classes_found": len(all_classes),
        "total_images_all": total_all,
        "included_classes_count": len(included_classes),
        "included_classes": included_classes,
        "total_images_included": total_included,
        "excluded_classes": excluded_found,
        "image_sizes_sampled": [list(s) for s in sorted(sizes_seen)],
        "image_modes_sampled": sorted(modes_seen),
        "class_counts": class_counts,
        "sample_grid_path": str(grid_path),
    }

    summary_path = out_dir / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== PHASE 1 DATASET SUMMARY ===")
    print(f"Total classes found   : {len(all_classes)} ({total_all:,} images)")
    print(f"Included classes ({len(included_classes)}) : {', '.join(included_classes)}")
    print(f"Included images       : {total_included:,}")
    print(f"Excluded classes ({len(excluded_found)})  : {', '.join(excluded_found)}")
    print(f"Sampled image sizes   : {sorted(sizes_seen)}")
    print(f"Sampled image modes   : {sorted(modes_seen)}")
    print(f"Min/Max per class     : {min(class_counts.values())} / {max(class_counts.values())}")
    print(f"Sample grid saved to  : {grid_path}")
    print(f"Summary JSON saved to : {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract CTX-UXO bounding-box crops from the COCO annotations into
``dataset/instances/{split}/{class_name}/``.

Keeps the 9 target classes in ``TARGET_CLASS_MAP``; Cartridge Magazine, Fuse
and Sea Mine are excluded.

Usage:
    python dataset/extract_instances.py
"""

import json
import shutil
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "dataset" else SCRIPT_DIR

DATASET_ROOT = PROJECT_ROOT / "dataset" / "ctxuxo_Dataset"
INSTANCES_ROOT = PROJECT_ROOT / "dataset" / "instances"

# Source images (nested: images/{split}/images/*.jpg)
IMAGE_DIRS = {
    "train": DATASET_ROOT / "images" / "train" / "images",
    "valid": DATASET_ROOT / "images" / "valid" / "images",
    "test":  DATASET_ROOT / "images" / "test"  / "images",
}

# COCO annotation files
COCO_FILES = {
    "train": DATASET_ROOT / "coco_labels" / "coco_train.json",
    "valid": DATASET_ROOT / "coco_labels" / "coco_val.json",
    "test":  DATASET_ROOT / "coco_labels" / "coco_test.json",
}

# The dataset has 12 classes; we keep only the 9 target UXO subtypes.
# COCO name -> canonical class name.

TARGET_CLASS_MAP: dict[str, str] = {
    "Mortar Bomb":        "Mortar_Bomb",
    "Projectile":         "Projectile",
    "Grenade":            "Grenade",
    "Aviation Bomb":      "Aviation_Bomb",
    "RPG":                "RPG",
    "LandMine":           "Landmine",
    "Rocket":             "Rockets",
    "AntiSubmarine Bomb": "Anti-Submarine",
    "Cartridge":          "Cartridge",
}

# Excluded classes (present in dataset but not part of the benchmark)
EXCLUDED_CLASSES = {"Cartridge Magazine", "Fuse", "Sea Mine"}

# Minimum crop size in pixels; smaller boxes are skipped.
MIN_CROP_SIZE = 4


def extract_instances_for_split(
    split: str,
    coco_path: Path,
    image_dir: Path,
    output_root: Path,
) -> dict[str, int]:
    """
    Extract crop instances for a single split.

    Returns:
        Canonical class name -> number of extracted crops.
    """
    print(f"\nProcessing split: {split.upper()}")

    with open(coco_path) as f:
        coco = json.load(f)

    cat_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco["categories"]
    }
    img_id_to_info: dict[int, dict] = {
        img["id"]: img for img in coco["images"]
    }

    for canonical_name in TARGET_CLASS_MAP.values():
        out_dir = output_root / split / canonical_name
        out_dir.mkdir(parents=True, exist_ok=True)

    extracted: Counter = Counter()
    skipped_excluded: Counter = Counter()
    skipped_bad_crop: int = 0
    skipped_missing_img: int = 0

    # Sorted by image_id so each source image is opened once per groupby group.
    annotations_sorted = sorted(coco["annotations"], key=lambda a: a["image_id"])
    total_annots = len(annotations_sorted)
    processed = 0

    for img_id, annots_group in groupby(annotations_sorted, key=lambda a: a["image_id"]):
        annots_list = list(annots_group)
        img_info = img_id_to_info.get(img_id)
        if img_info is None:
            skipped_missing_img += len(annots_list)
            continue

        img_filename = img_info["file_name"]
        img_path = image_dir / img_filename

        if not img_path.exists():
            skipped_missing_img += len(annots_list)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  WARNING: Cannot open {img_path}: {e}")
            skipped_missing_img += len(annots_list)
            continue

        img_w, img_h = img.size

        for ann in annots_list:
            processed += 1
            cat_name = cat_id_to_name.get(ann["category_id"], "UNKNOWN")

            if cat_name in EXCLUDED_CLASSES or cat_name not in TARGET_CLASS_MAP:
                skipped_excluded[cat_name] += 1
                continue

            canonical_name = TARGET_CLASS_MAP[cat_name]

            bx, by, bw, bh = ann["bbox"]  # COCO: [x, y, width, height]
            x1 = max(0, int(round(bx)))
            y1 = max(0, int(round(by)))
            x2 = min(img_w, int(round(bx + bw)))
            y2 = min(img_h, int(round(by + bh)))

            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                skipped_bad_crop += 1
                continue

            crop = img.crop((x1, y1, x2, y2))
            extracted[canonical_name] += 1
            count = extracted[canonical_name]
            out_path = output_root / split / canonical_name / f"{canonical_name}_{count:05d}.jpg"
            crop.save(out_path, "JPEG", quality=95)

        if processed % 2000 == 0:
            print(f"  ... processed {processed}/{total_annots} annotations")

    print(f"\n  {split.upper()} results")
    print(f"  Target instances extracted: {sum(extracted.values())}")
    if skipped_excluded:
        print(f"  Excluded (non-target classes): {sum(skipped_excluded.values())}")
        for name, cnt in sorted(skipped_excluded.items()):
            print(f"    {name}: {cnt}")
    if skipped_bad_crop:
        print(f"  Skipped (degenerate bbox < {MIN_CROP_SIZE}px): {skipped_bad_crop}")
    if skipped_missing_img:
        print(f"  Skipped (missing source image): {skipped_missing_img}")

    print(f"\n  Per-class breakdown ({split}):")
    for name in sorted(TARGET_CLASS_MAP.values()):
        cnt = extracted.get(name, 0)
        print(f"    {name:20s}: {cnt:5d}")

    return dict(extracted)


def main():
    for path in COCO_FILES.values():
        if not path.exists():
            print(f"ERROR: COCO file not found: {path}")
            sys.exit(1)
    for path in IMAGE_DIRS.values():
        if not path.exists():
            print(f"ERROR: Image directory not found: {path}")
            sys.exit(1)

    if INSTANCES_ROOT.exists():
        print(f"\nWARNING: Removing existing instances directory: {INSTANCES_ROOT}")
        shutil.rmtree(INSTANCES_ROOT)

    INSTANCES_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, int]] = {}
    for split in ["train", "valid", "test"]:
        results = extract_instances_for_split(
            split=split,
            coco_path=COCO_FILES[split],
            image_dir=IMAGE_DIRS[split],
            output_root=INSTANCES_ROOT,
        )
        all_results[split] = results

    print("CTX-UXO instance extraction summary")

    class_order = sorted(TARGET_CLASS_MAP.values())
    header = f"{'Class':20s} | {'train':>7s} | {'valid':>7s} | {'test':>7s} | {'TOTAL':>7s}"
    print(header)
    print("-" * len(header))

    grand_total = 0
    for cls in class_order:
        tr = all_results.get("train", {}).get(cls, 0)
        va = all_results.get("valid", {}).get(cls, 0)
        te = all_results.get("test", {}).get(cls, 0)
        total = tr + va + te
        grand_total += total
        print(f"{cls:20s} | {tr:7d} | {va:7d} | {te:7d} | {total:7d}")

    print("-" * len(header))
    tr_total = sum(all_results.get("train", {}).values())
    va_total = sum(all_results.get("valid", {}).values())
    te_total = sum(all_results.get("test", {}).values())
    print(f"{'TOTAL':20s} | {tr_total:7d} | {va_total:7d} | {te_total:7d} | {grand_total:7d}")

    print(f"\nExcluded classes: {', '.join(sorted(EXCLUDED_CLASSES))}")
    print(f"{grand_total} crops written to {INSTANCES_ROOT}")

    return grand_total


if __name__ == "__main__":
    main()

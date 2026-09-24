"""FGVC-Aircraft Dataset Loader."""

import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fewshot_common import ClassPool, SplitIndex

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "dataset" else _SCRIPT_DIR
DEFAULT_AIRCRAFT_ROOT: Path = REPO_ROOT / "dataset" / "others" / "fgvc aircraft"
_IMAGES_SUBDIR: Path = Path("fgvc-aircraft-2013b") / "fgvc-aircraft-2013b" / "data" / "images"

_SPLIT_TO_CSV: Dict[str, str] = {"train": "train.csv", "valid": "val.csv", "test": "test.csv"}


def _load_split_csv(csv_path: Path, images_dir: Path) -> ClassPool:
    pool: Dict[str, List[Path]] = {}
    if not csv_path.is_file():
        raise FileNotFoundError(f"FGVC-Aircraft split CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "Classes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing column(s) {sorted(missing)}.")
        for row in reader:
            class_name = row["Classes"]
            image_path = images_dir / row["filename"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{csv_path} references '{row['filename']}' but no such file exists "
                    f"under {images_dir}."
                )
            pool.setdefault(class_name, []).append(image_path)
    for class_name in pool:
        pool[class_name].sort()
    return pool


def load_index(root: Path = DEFAULT_AIRCRAFT_ROOT) -> SplitIndex:
    """Build the FGVC-Aircraft ``SplitIndex`` from the pre-made
    train/val/test CSVs, verifying every referenced image exists on disk.

    Args:
        root: Path to the ``fgvc aircraft`` directory (containing
            ``train.csv``, ``val.csv``, ``test.csv``, and the
            ``fgvc-aircraft-2013b/`` image bundle).

    Returns:
        A ``SplitIndex`` with ``pools["train"|"valid"|"test"]`` (100
        classes each, identical class set across splits, verified below),
        ``shared_classes_across_splits=True``.
    """
    images_dir = root / _IMAGES_SUBDIR
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"FGVC-Aircraft image directory not found: {images_dir}. Expected the "
            "official fgvc-aircraft-2013b bundle under dataset/others/fgvc aircraft/."
        )

    pools: Dict[str, ClassPool] = {}
    for split, csv_name in _SPLIT_TO_CSV.items():
        pools[split] = _load_split_csv(root / csv_name, images_dir)

    class_sets = {split: frozenset(pool.keys()) for split, pool in pools.items()}
    if len(set(class_sets.values())) != 1:
        raise ValueError(
            "FGVC-Aircraft train/val/test CSVs do not share the same class set "
            f"(sizes: { {s: len(c) for s, c in class_sets.items()} }); "
            "SplitIndex.shared_classes_across_splits=True would be a false claim."
        )
    class_names: Tuple[str, ...] = tuple(sorted(class_sets["train"]))

    return SplitIndex(
        name="fgvc_aircraft",
        pools=pools,
        class_names_per_split={s: class_names for s in pools},
        shared_classes_across_splits=True,
    )


if __name__ == "__main__":
    index = load_index()
    print(f"Dataset: {index.name}, classes: {len(index.class_names_per_split['train'])}")

    counts = index.counts()
    grand_total = 0
    for split in ("train", "valid", "test"):
        n = sum(counts[split].values())
        grand_total += n
        n_classes_present = sum(1 for c in counts[split].values() if c > 0)
        print(f"split={split:5s} total_images={n:6d}  classes_with_>=1_image={n_classes_present}")
    print(f"TOTAL images across splits: {grand_total}")

    min_train = min(counts["train"].values())
    min_valid = min(counts["valid"].values())
    min_test = min(counts["test"].values())
    print(f"\nMin per-class counts: train={min_train}  valid={min_valid}  test={min_test}")

    print("\nsample episode (5-way, 5-shot, 15-query, support=train, query=test)")
    from dataset.fewshot_common import load_episode_tensors_generic, sample_episode_generic

    episode = sample_episode_generic(
        n_way=5,
        n_shot=5,
        n_query=15,
        support_pool=index.pools["train"],
        query_pool=index.pools["test"],
        rng=random.Random(0),
    )
    for cls in episode.way_classes:
        print(f"  {cls:20s} S_a={episode.shots_per_class[cls]}")
    support_images, support_labels, query_images, query_labels = load_episode_tensors_generic(episode)
    print(
        f"Loaded {len(support_images)} support / {len(query_images)} query images "
        f"(sample support shape={tuple(support_images[0].shape)})."
    )

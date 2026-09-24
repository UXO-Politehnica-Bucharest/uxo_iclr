"""CUB-200-2011 Dataset Loader."""

import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fewshot_common import ClassPool, SplitIndex

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "dataset" else _SCRIPT_DIR
DEFAULT_CUB_ROOT: Path = REPO_ROOT / "dataset" / "others" / "cub_200_2011" / "CUB_200_2011"

DEFAULT_VALID_FRACTION: float = 0.2
"""Fraction of the official train-flagged images per class carved out as
our "valid" split; the rest stays "train". Official test images are never
touched by this fraction."""
DEFAULT_SPLIT_SEED: int = 42

_CLASS_NAME_STRIP_RE = re.compile(r"^\d+\.")


def _strip_class_index_prefix(folder_name: str) -> str:
    """"001.Black_footed_Albatross" -> "Black_footed_Albatross"."""
    return _CLASS_NAME_STRIP_RE.sub("", folder_name)


def load_index(
    root: Path = DEFAULT_CUB_ROOT,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> SplitIndex:
    """Build the CUB-200-2011 ``SplitIndex`` from the official annotation
    files, verifying internal consistency (every image referenced by the
    annotation files must exist on disk) rather than assuming it.

    Args:
        root: Path to the ``CUB_200_2011`` directory (containing
            ``images.txt``, ``images/``, etc.).
        valid_fraction: Fraction of official train-flagged images per
            class held out as "valid" (see ``DEFAULT_VALID_FRACTION``).
        split_seed: Seed for the deterministic per-class train/valid carve.

    Returns:
        A ``SplitIndex`` with ``pools["train"|"valid"|"test"]`` and
        ``class_names_per_split`` (identical 200-class tuple for all three
        splits), ``shared_classes_across_splits=True``.
    """
    images_txt = root / "images.txt"
    labels_txt = root / "image_class_labels.txt"
    classes_txt = root / "classes.txt"
    split_txt = root / "train_test_split.txt"
    images_dir = root / "images"
    for required in (images_txt, labels_txt, classes_txt, split_txt, images_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"CUB-200-2011 file/dir not found: {required}. Expected the official "
                "raw distribution under dataset/others/cub_200_2011/CUB_200_2011/."
            )

    class_id_to_name: Dict[int, str] = {}
    with open(classes_txt) as f:
        for line in f:
            class_id, folder_name = line.strip().split(" ", 1)
            class_id_to_name[int(class_id)] = _strip_class_index_prefix(folder_name)

    image_paths: Dict[int, str] = {}
    with open(images_txt) as f:
        for line in f:
            image_id, rel_path = line.strip().split(" ", 1)
            image_paths[int(image_id)] = rel_path

    image_labels: Dict[int, int] = {}
    with open(labels_txt) as f:
        for line in f:
            image_id, class_id = line.strip().split()
            image_labels[int(image_id)] = int(class_id)

    is_official_train: Dict[int, bool] = {}
    with open(split_txt) as f:
        for line in f:
            image_id, flag = line.strip().split()
            is_official_train[int(image_id)] = bool(int(flag))

    if not (image_paths.keys() == image_labels.keys() == is_official_train.keys()):
        raise ValueError(
            "CUB-200-2011 annotation files disagree on which image IDs exist "
            f"(images.txt={len(image_paths)}, image_class_labels.txt="
            f"{len(image_labels)}, train_test_split.txt={len(is_official_train)})."
        )

    missing_on_disk = [
        image_id for image_id, rel in image_paths.items() if not (images_dir / rel).is_file()
    ]
    if missing_on_disk:
        raise FileNotFoundError(
            f"{len(missing_on_disk)} image(s) listed in images.txt are missing on disk "
            f"under {images_dir} (e.g. image_id={missing_on_disk[0]} -> "
            f"{image_paths[missing_on_disk[0]]}). The CUB-200-2011 download under "
            "dataset/others/cub_200_2011/ looks incomplete."
        )

    official_train_by_class: Dict[int, List[int]] = {cid: [] for cid in class_id_to_name}
    official_test_by_class: Dict[int, List[int]] = {cid: [] for cid in class_id_to_name}
    for image_id, class_id in image_labels.items():
        bucket = official_train_by_class if is_official_train[image_id] else official_test_by_class
        bucket[class_id].append(image_id)

    rng = random.Random(split_seed)
    train_pool: ClassPool = {}
    valid_pool: ClassPool = {}
    test_pool: ClassPool = {}
    for class_id, class_name in sorted(class_id_to_name.items()):
        train_ids = sorted(official_train_by_class[class_id])
        n_valid = max(1, round(len(train_ids) * valid_fraction)) if train_ids else 0
        valid_ids = set(rng.sample(train_ids, k=n_valid)) if n_valid > 0 else set()
        our_train_ids = [i for i in train_ids if i not in valid_ids]

        train_pool[class_name] = sorted(images_dir / image_paths[i] for i in our_train_ids)
        valid_pool[class_name] = sorted(images_dir / image_paths[i] for i in valid_ids)
        test_pool[class_name] = sorted(
            images_dir / image_paths[i] for i in sorted(official_test_by_class[class_id])
        )

    class_names: Tuple[str, ...] = tuple(
        name for _cid, name in sorted(class_id_to_name.items())
    )
    return SplitIndex(
        name="cub200",
        pools={"train": train_pool, "valid": valid_pool, "test": test_pool},
        class_names_per_split={"train": class_names, "valid": class_names, "test": class_names},
        shared_classes_across_splits=True,
    )


if __name__ == "__main__":
    index = load_index()
    print(f"Dataset: {index.name}, classes: {len(index.class_names_per_split['train'])}")

    counts = index.counts()
    grand = {"train": 0, "valid": 0, "test": 0}
    for split in ("train", "valid", "test"):
        n = sum(counts[split].values())
        grand[split] = n
        n_classes_present = sum(1 for c in counts[split].values() if c > 0)
        print(f"split={split:5s} total_images={n:6d}  classes_with_>=1_image={n_classes_present}")
    print(f"TOTAL images across splits: {sum(grand.values())} (expect 11788)")

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
        print(f"  {cls:30s} S_a={episode.shots_per_class[cls]}")
    support_images, support_labels, query_images, query_labels = load_episode_tensors_generic(episode)
    print(
        f"Loaded {len(support_images)} support / {len(query_images)} query images "
        f"(sample support shape={tuple(support_images[0].shape)})."
    )

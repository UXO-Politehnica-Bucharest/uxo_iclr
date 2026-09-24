"""tieredImageNet Dataset Loader."""

import random
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fewshot_common import ClassPool, SplitIndex

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "dataset" else _SCRIPT_DIR
DEFAULT_TIERED_ROOT: Path = REPO_ROOT / "dataset" / "others" / "tiered_imagenet"

_IMAGE_GLOB: str = "*.jpg"
# On-disk folder name -> this project's Split naming convention
# (dataset/ctx_uxo.py uses "valid", this distribution uses "val").
_SPLIT_TO_DIR: Dict[str, str] = {"train": "train", "valid": "val", "test": "test"}


def load_index(root: Path = DEFAULT_TIERED_ROOT) -> SplitIndex:
    """Build the tieredImageNet ``SplitIndex`` by walking
    ``{split_dir}/{synset}/*.jpg``, verifying no synset (class) appears in
    more than one split (the property the whole benchmark depends on).

    Args:
        root: Path to the ``tiered_imagenet`` directory (containing
            ``train/``, ``val/``, ``test/``).

    Returns:
        A ``SplitIndex`` with ``pools["train"|"valid"|"test"]`` over
        disjoint 351/97/160-class pools,
        ``shared_classes_across_splits=False``.
    """
    pools: Dict[str, ClassPool] = {}
    class_names_per_split: Dict[str, Tuple[str, ...]] = {}

    for split, dir_name in _SPLIT_TO_DIR.items():
        split_dir = root / dir_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"tieredImageNet split directory not found: {split_dir}. Expected "
                "dataset/others/tiered_imagenet/{train,val,test}/{synset}/*.jpg."
            )
        pool: ClassPool = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(class_dir.glob(_IMAGE_GLOB))
            if images:
                pool[class_dir.name] = images
        if not pool:
            raise RuntimeError(f"No class subdirectories with images found under {split_dir}.")
        pools[split] = pool
        class_names_per_split[split] = tuple(sorted(pool.keys()))

    overlaps = []
    splits = list(pools.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            shared = set(class_names_per_split[splits[i]]) & set(class_names_per_split[splits[j]])
            if shared:
                overlaps.append((splits[i], splits[j], len(shared)))
    if overlaps:
        raise ValueError(
            f"tieredImageNet splits are supposed to use disjoint classes, but found "
            f"overlap(s): {overlaps}. Refusing to build an index that would leak "
            "meta-train/meta-val/meta-test classes into each other."
        )

    return SplitIndex(
        name="tiered_imagenet",
        pools=pools,
        class_names_per_split=class_names_per_split,
        shared_classes_across_splits=False,
    )


if __name__ == "__main__":
    print("(Indexing ~780K files -- this takes a few seconds.)")
    index = load_index()

    counts = index.counts()
    grand_total = 0
    for split in ("train", "valid", "test"):
        n_classes = len(index.class_names_per_split[split])
        n_images = sum(counts[split].values())
        grand_total += n_images
        print(f"split={split:5s} classes={n_classes:4d}  total_images={n_images:7d}")
    print(f"TOTAL images across splits: {grand_total} (expect 779165)")

    print("\ndisjointness check")
    tr = set(index.class_names_per_split["train"])
    va = set(index.class_names_per_split["valid"])
    te = set(index.class_names_per_split["test"])
    print(f"train & valid overlap: {len(tr & va)} (expect 0)")
    print(f"train & test overlap:  {len(tr & te)} (expect 0)")
    print(f"valid & test overlap:  {len(va & te)} (expect 0)")

    print("\nsample episode (5-way, 5-shot, 15-query, support=query=test split)")
    from dataset.fewshot_common import load_episode_tensors_generic, sample_episode_generic

    episode = sample_episode_generic(
        n_way=5,
        n_shot=5,
        n_query=15,
        support_pool=index.pools["test"],
        query_pool=index.pools["test"],
        rng=random.Random(0),
    )
    for cls in episode.way_classes:
        print(f"  {cls:15s} S_a={episode.shots_per_class[cls]}")
    overlap_check = set(episode.support_paths) & set(episode.query_paths)
    print(f"support/query image overlap within episode: {len(overlap_check)} (expect 0)")

    support_images, support_labels, query_images, query_labels = load_episode_tensors_generic(episode)
    print(
        f"Loaded {len(support_images)} support / {len(query_images)} query images "
        f"(sample support shape={tuple(support_images[0].shape)})."
    )

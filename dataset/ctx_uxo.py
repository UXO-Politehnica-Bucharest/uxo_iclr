"""CTX-UXO Dataset Loader & Episodic Sampler

Loads the CTX-UXO crop instances produced by ``dataset/extract_instances.py``
from ``dataset/instances/{split}/{class_name}/*.jpg``.
"""

import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "dataset" else _SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


Split = Literal["train", "valid", "test"]

DEFAULT_INSTANCES_ROOT: Path = REPO_ROOT / "dataset" / "instances"

CLASS_NAMES: Tuple[str, ...] = (
    "Aviation_Bomb",
    "Grenade",
    "Mortar_Bomb",
    "Projectile",
    "RPG",
)
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES: int = len(CLASS_NAMES)

_IMAGE_GLOB: str = "*.jpg"

LONG_TAIL_CLASSES: Tuple[str, ...] = ("RPG",)


def list_class_instances(
    split: Split, class_name: str, root: Path = DEFAULT_INSTANCES_ROOT
) -> List[Path]:
    """Sorted image paths for one (split, class); empty if the directory is missing."""
    if class_name not in CLASS_TO_IDX:
        raise ValueError(f"Unknown UXO class '{class_name}'. Expected one of {CLASS_NAMES}.")
    class_dir = root / split / class_name
    if not class_dir.is_dir():
        return []
    return sorted(class_dir.glob(_IMAGE_GLOB))


def build_split_index(
    split: Split, root: Path = DEFAULT_INSTANCES_ROOT
) -> List[Tuple[Path, int]]:
    """Build the full (path, label_index) index for a given split.

    Iterates over the fixed CLASS_NAMES order so label indices are identical
    and stable across train/valid/test (never remapped per split).
    """
    index: List[Tuple[Path, int]] = []
    for class_name in CLASS_NAMES:
        label = CLASS_TO_IDX[class_name]
        for path in list_class_instances(split, class_name, root=root):
            index.append((path, label))
    return index


def count_instances(
    root: Path = DEFAULT_INSTANCES_ROOT,
) -> Dict[str, Dict[str, int]]:
    """Count on-disk instances per class per split.

    Returns:
        Mapping class_name -> {"train": n, "valid": n, "test": n, "total": n}.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for class_name in CLASS_NAMES:
        per_split = {
            split: len(list_class_instances(split, class_name, root=root))
            for split in ("train", "valid", "test")
        }
        per_split["total"] = sum(per_split.values())
        counts[class_name] = per_split
    return counts


class CTXUXODataset(Dataset):
    """Raw CTX-UXO crops as float [0, 1] (3, H, W) tensors at native resolution.

    No augmentation, resizing or normalization is applied; feature extractors do
    their own preprocessing.
    """

    def __init__(
        self,
        split: Split,
        root: Path = DEFAULT_INSTANCES_ROOT,
        as_tensor: bool = True,
    ) -> None:
        """
        Args:
            split: One of "train", "valid", "test".
            root: Root of the ``dataset/instances`` directory tree.
            as_tensor: If True, __getitem__ returns a float32 CHW tensor in
                [0, 1]. If False, returns the raw PIL.Image.Image crop.
        """
        if split not in ("train", "valid", "test"):
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'valid', or 'test'.")
        self.split: Split = split
        self.root: Path = root
        self.as_tensor: bool = as_tensor
        self.samples: List[Tuple[Path, int]] = build_split_index(split, root=root)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No instances found for split='{split}' under {root}. "
                "Did you run dataset/extract_instances.py?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor | Image.Image, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if not self.as_tensor:
            return image, label
        return pil_to_raw_tensor(image), label

    def class_name(self, label: int) -> str:
        """Map an integer label back to its canonical class name."""
        return IDX_TO_CLASS[label]

    def per_class_indices(self) -> Dict[int, List[int]]:
        """Return dataset-position indices grouped by class label."""
        grouped: Dict[int, List[int]] = {label: [] for label in range(NUM_CLASSES)}
        for position, (_, label) in enumerate(self.samples):
            grouped[label].append(position)
        return grouped


def pil_to_raw_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL RGB image to a float32 CHW tensor in [0, 1].

    This is a pure format conversion (no resize, no crop, no normalization,
    no color jitter), so it is not data augmentation.
    """
    array = torch.from_numpy(np.array(image, copy=True))  # (H, W, 3), uint8
    return array.permute(2, 0, 1).contiguous().float() / 255.0


def load_images_parallel(
    paths: Sequence[Path],
    max_workers: Optional[int] = None,
) -> Dict[Path, Tensor]:
    """Load ``paths`` (deduplicated) as raw [0, 1] CHW tensors using a thread pool.

    Output is identical to loading sequentially. Threads suffice because PIL
    decoding and file reads release the GIL; the default pool size
    ``min(32, 4 * cpu_count)`` exceeds the core count because the work is
    I/O-bound.
    """
    unique_paths: List[Path] = list(dict.fromkeys(paths))
    if not unique_paths:
        return {}
    workers = max_workers or min(32, 4 * (os.cpu_count() or 4))

    def _load_one(p: Path) -> Tensor:
        return pil_to_raw_tensor(Image.open(p).convert("RGB"))

    result: Dict[Path, Tensor] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for p, tensor in zip(unique_paths, executor.map(_load_one, unique_paths), strict=True):
            result[p] = tensor
    return result


def variable_size_collate(
    batch: Sequence[Tuple[Tensor, int]]
) -> Tuple[List[Tensor], Tensor]:
    """Collate function for crops of heterogeneous spatial resolution.

    Raw un-augmented crops are not resized, so they cannot be stacked into
    a single dense tensor. Returns a list of per-sample tensors plus a
    stacked label tensor.
    """
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def _default_batched_feature_fn(images: Sequence[Tensor]) -> Tensor:
    """Default batched `feature_fn`: a lazily constructed, memoized frozen
    `algorithm.features.ClipFeatureExtractor`.

    The import is local so dataset I/O and episodic sampling keep working
    where `transformers`/CLIP weights are unavailable; the instance is
    cached on this function so the checkpoint is not reloaded per batch.
    """
    if not hasattr(_default_batched_feature_fn, "_extractor"):
        from algorithm.features import ClipFeatureExtractor

        _default_batched_feature_fn._extractor = ClipFeatureExtractor(device="cpu")
    return _default_batched_feature_fn._extractor.extract(list(images))


def batched_feature_mean(
    paths: Sequence[Path],
    feature_fn: Callable[[Sequence[Tensor]], Tensor],
    batch_size: int = 32,
) -> Tensor:
    """Mean feature vector over ``paths``, extracted ``batch_size`` images at
    a time and accumulated in double precision; returned as float32."""
    running_sum: Optional[Tensor] = None
    for start in range(0, len(paths), batch_size):
        chunk_paths = paths[start : start + batch_size]
        image_map = load_images_parallel(chunk_paths)
        features = feature_fn([image_map[p] for p in chunk_paths]).double()
        batch_sum = features.sum(dim=0)
        running_sum = batch_sum if running_sum is None else running_sum + batch_sum
    return (running_sum / len(paths)).float()


def compute_base_split_mean(
    root: Path = DEFAULT_INSTANCES_ROOT,
    feature_fn: Optional[Callable[[Sequence[Tensor]], Tensor]] = None,
    batch_size: int = 32,
) -> Tensor:
    """Computes the base-split centering statistic f_bar_base from the train split."""
    train_paths = [path for path, _label in build_split_index("train", root=root)]
    if not train_paths:
        raise RuntimeError(
            f"No instances found for split='train' under {root}. "
            "Did you run dataset/extract_instances.py?"
        )
    extractor = feature_fn if feature_fn is not None else _default_batched_feature_fn
    return batched_feature_mean(train_paths, extractor, batch_size)


@dataclass
class Episode:
    """A single C-way S-shot episode."""

    way_classes: List[int]
    support_paths: List[Path]
    support_labels: List[int]
    query_paths: List[Path]
    query_labels: List[int]
    shots_per_class: Dict[int, int] = field(default_factory=dict)


def sample_episode(
    n_way: int,
    n_shot: int,
    n_query: int,
    query_split: Split = "test",
    support_split: Split = "train",
    root: Path = DEFAULT_INSTANCES_ROOT,
    rng: Optional[random.Random] = None,
) -> Episode:
    """Samples one C-way S-shot episode with per-class support capping."""
    if not (1 <= n_way <= NUM_CLASSES):
        raise ValueError(f"n_way must be in [1, {NUM_CLASSES}], got {n_way}.")
    if n_shot < 1 or n_query < 1:
        raise ValueError("n_shot and n_query must both be >= 1.")

    rng = rng if rng is not None else random.Random()

    sampled_classes = rng.sample(range(NUM_CLASSES), k=n_way)

    support_paths: List[Path] = []
    support_labels: List[int] = []
    query_paths: List[Path] = []
    query_labels: List[int] = []
    shots_per_class: Dict[int, int] = {}

    for label in sampled_classes:
        class_name = IDX_TO_CLASS[label]

        support_pool = list_class_instances(support_split, class_name, root=root)
        query_pool = list_class_instances(query_split, class_name, root=root)

        s_a = min(n_shot, len(support_pool))
        q_a = min(n_query, len(query_pool))
        if s_a == 0:
            raise RuntimeError(
                f"Class '{class_name}' has zero instances in support_split="
                f"'{support_split}'; cannot form an episode."
            )
        if q_a == 0:
            raise RuntimeError(
                f"Class '{class_name}' has zero instances in query_split="
                f"'{query_split}'; cannot form an episode."
            )

        shots_per_class[label] = s_a

        chosen_support = rng.sample(support_pool, k=s_a)
        chosen_query = rng.sample(query_pool, k=q_a)

        support_paths.extend(chosen_support)
        support_labels.extend([label] * s_a)
        query_paths.extend(chosen_query)
        query_labels.extend([label] * q_a)

    return Episode(
        way_classes=sampled_classes,
        support_paths=support_paths,
        support_labels=support_labels,
        query_paths=query_paths,
        query_labels=query_labels,
        shots_per_class=shots_per_class,
    )


def load_episode_tensors(
    episode: Episode,
) -> Tuple[List[Tensor], Tensor, List[Tensor], Tensor]:
    """Load raw, un-augmented image tensors for an already-sampled Episode.

    Returns:
        (support_images, support_labels, query_images, query_labels), where
        the image lists contain variable-resolution (3, H, W) float tensors
        in [0, 1] (no resizing/augmentation applied), and the label tensors
        are torch.long.
    """
    image_map = load_images_parallel(list(episode.support_paths) + list(episode.query_paths))
    support_images = [image_map[p] for p in episode.support_paths]
    query_images = [image_map[p] for p in episode.query_paths]
    support_labels = torch.tensor(episode.support_labels, dtype=torch.long)
    query_labels = torch.tensor(episode.query_labels, dtype=torch.long)
    return support_images, support_labels, query_images, query_labels


if __name__ == "__main__":
    print(f"Repo root:        {REPO_ROOT}")
    print(f"Instances root:   {DEFAULT_INSTANCES_ROOT}")
    print(f"Num classes:      {NUM_CLASSES}")
    print(f"Class names:      {CLASS_NAMES}")
    print(f"Long-tail flag:   {LONG_TAIL_CLASSES}")

    print("\nPer-class / per-split instance counts (on-disk)")
    counts = count_instances()
    header = f"{'Class':16s} | {'train':>7s} | {'valid':>7s} | {'test':>7s} | {'total':>7s}"
    print(header)
    print("-" * len(header))
    grand_train = grand_valid = grand_test = grand_total = 0
    for class_name in CLASS_NAMES:
        c = counts[class_name]
        grand_train += c["train"]
        grand_valid += c["valid"]
        grand_test += c["test"]
        grand_total += c["total"]
        print(
            f"{class_name:16s} | {c['train']:7d} | {c['valid']:7d} | "
            f"{c['test']:7d} | {c['total']:7d}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':16s} | {grand_train:7d} | {grand_valid:7d} | "
        f"{grand_test:7d} | {grand_total:7d}"
    )

    print("\nDataset objects")
    for split in ("train", "valid", "test"):
        ds = CTXUXODataset(split=split)  # type: ignore[arg-type]
        image, label = ds[0]
        print(
            f"split={split:5s} len={len(ds):6d}  "
            f"sample0: tensor_shape={tuple(image.shape)} "
            f"label={label} ({ds.class_name(label)})"
        )

    print("\nBase-split mean (train)")
    f_base = compute_base_split_mean()
    print(f"f_bar_base shape: {tuple(f_base.shape)}  dtype: {f_base.dtype}")
    print(f"f_bar_base[:5]:   {f_base[:5].tolist()}")

    print("\nSample episode")
    episode_rng = random.Random(42)
    episode = sample_episode(
        n_way=NUM_CLASSES,
        n_shot=10,
        n_query=5,
        support_split="train",
        query_split="test",
        rng=episode_rng,
    )
    print(f"Sampled {len(episode.way_classes)}-way episode, requested S=10, Q=5")
    for label in episode.way_classes:
        class_name = IDX_TO_CLASS[label]
        s_a = episode.shots_per_class[label]
        capped_flag = " (CAPPED)" if s_a < 10 else ""
        print(f"  {class_name:16s}: S_a = {s_a:2d}{capped_flag}")

    support_images, support_labels, query_images, query_labels = load_episode_tensors(
        episode
    )
    print(f"\nLoaded {len(support_images)} support / {len(query_images)} query images")
    print(f"support_labels shape: {tuple(support_labels.shape)}")
    print(f"query_labels shape:   {tuple(query_labels.shape)}")

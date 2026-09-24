"""Dataset-agnostic episodic few-shot utilities."""

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from torch import Tensor

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "dataset" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.ctx_uxo import batched_feature_mean, load_episode_tensors
from dataset.ctx_uxo import load_images_parallel  # noqa: F401  (re-exported)

# {class_name: [image_path, ...]}, all images belonging to one split.
ClassPool = Dict[str, List[Path]]


@dataclass(frozen=True)
class SplitIndex:
    """Per-split ``{class_name: [image paths]}`` pools for one dataset.

    Attributes:
        name: Dataset identifier.
        pools: ``{split: {class_name: [Path, ...]}}`` for "train"/"valid"/"test".
        class_names_per_split: Class names per split. CUB-200-2011, FGVC-Aircraft
            and CTX-UXO share one class set; tieredImageNet's splits are
            disjoint, so the pools' keys may differ between splits.
        shared_classes_across_splits: Whether all splits use the same classes.
    """

    name: str
    pools: Dict[str, ClassPool]
    class_names_per_split: Dict[str, Tuple[str, ...]]
    shared_classes_across_splits: bool

    def counts(self) -> Dict[str, Dict[str, int]]:
        """On-disk instance counts per split per class, mirroring
        ``dataset.ctx_uxo.count_instances``'s shape."""
        return {
            split: {cls: len(paths) for cls, paths in pool.items()}
            for split, pool in self.pools.items()
        }


@dataclass
class Episode:
    """One C-way episode; mirrors ``dataset.ctx_uxo.Episode``.

    ``way_classes`` holds class names in sampling order, and the integer labels
    index into it, since a fixed global label mapping does not exist across
    datasets with disjoint splits.
    """

    way_classes: List[str]
    support_paths: List[Path]
    support_labels: List[int]
    query_paths: List[Path]
    query_labels: List[int]
    shots_per_class: Dict[str, int] = field(default_factory=dict)


def sample_episode_generic(
    n_way: int,
    n_shot: int,
    n_query: int,
    support_pool: ClassPool,
    query_pool: ClassPool,
    rng: Optional[random.Random] = None,
) -> Episode:
    """Sample one C-way episode from explicit support and query pools.

    Shared-class datasets pass different splits (e.g. train/test);
    tieredImageNet passes the same split for both. A class's query candidates
    exclude its chosen support images, so no image appears in both sets even
    when ``support_pool is query_pool``. Per-class counts are capped at the
    available images (S_a = min(n_shot, |pool|), likewise for queries).

    Args:
        n_way: Number of classes, drawn from those with >= 1 image in both pools.
        n_shot: Requested shots per class.
        n_query: Requested queries per class.
        support_pool: ``{class_name: [Path, ...]}`` for support.
        query_pool: ``{class_name: [Path, ...]}`` for queries.
        rng: Optional seeded ``random.Random``.
    """
    if n_shot < 1 or n_query < 1:
        raise ValueError("n_shot and n_query must both be >= 1.")

    eligible = sorted(set(support_pool.keys()) & set(query_pool.keys()))
    eligible = [c for c in eligible if len(support_pool[c]) > 0 and len(query_pool[c]) > 0]
    if n_way > len(eligible):
        raise ValueError(
            f"n_way={n_way} exceeds the number of classes with >= 1 image in "
            f"BOTH the support and query pools ({len(eligible)} eligible)."
        )

    rng = rng if rng is not None else random.Random()
    sampled_classes = rng.sample(eligible, k=n_way)

    support_paths: List[Path] = []
    support_labels: List[int] = []
    query_paths: List[Path] = []
    query_labels: List[int] = []
    shots_per_class: Dict[str, int] = {}

    for label_idx, class_name in enumerate(sampled_classes):
        support_candidates = support_pool[class_name]
        s_a = min(n_shot, len(support_candidates))
        chosen_support = rng.sample(support_candidates, k=s_a)

        chosen_support_set = set(chosen_support)
        query_candidates = [p for p in query_pool[class_name] if p not in chosen_support_set]
        q_a = min(n_query, len(query_candidates))
        if q_a == 0:
            raise RuntimeError(
                f"Class '{class_name}' has zero query-eligible instances left "
                f"after excluding its {s_a} chosen support image(s); cannot "
                "form an episode. This can happen with support_pool is "
                "query_pool and a very small per-class pool -- lower n_shot "
                "or n_query, or use a larger class."
            )
        chosen_query = rng.sample(query_candidates, k=q_a)

        shots_per_class[class_name] = s_a
        support_paths.extend(chosen_support)
        support_labels.extend([label_idx] * s_a)
        query_paths.extend(chosen_query)
        query_labels.extend([label_idx] * q_a)

    return Episode(
        way_classes=sampled_classes,
        support_paths=support_paths,
        support_labels=support_labels,
        query_paths=query_paths,
        query_labels=query_labels,
        shots_per_class=shots_per_class,
    )


def load_episode_tensors_generic(
    episode: Episode,
) -> Tuple[List[Tensor], Tensor, List[Tensor], Tensor]:
    """Load raw, un-augmented image tensors for an already-sampled Episode
    (identical contract to ``dataset.ctx_uxo.load_episode_tensors``)."""
    return load_episode_tensors(episode)


def compute_base_mean_generic(
    paths: Sequence[Path],
    feature_fn: Callable[[Sequence[Tensor]], Tensor],
    batch_size: int = 32,
) -> Tensor:
    """CL2N centering statistic (mean feature) over an explicit list of base-split
    image paths. Unlike ``ctx_uxo.compute_base_split_mean``, ``feature_fn`` is
    required so the backbone is always explicit.
    """
    if len(paths) == 0:
        raise RuntimeError("compute_base_mean_generic: the base/train pool is empty.")
    return batched_feature_mean(list(paths), feature_fn, batch_size)

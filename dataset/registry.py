"""Dataset registry providing a unified loader interface across benchmarks."""

import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fewshot_common import ClassPool, SplitIndex

DATASET_NAMES: Tuple[str, ...] = ("ctx_uxo", "cub200", "fgvc_aircraft", "tiered_imagenet")

# Every cross-domain dataset except tieredImageNet shares classes across
# splits, so support_split="train" / query_split="test" (or "valid") is the
# natural default pairing there; tieredImageNet needs support_split ==
# query_split (see dataset.fewshot_common.sample_episode_generic).
DEFAULT_SUPPORT_SPLIT: Dict[str, str] = {
    "ctx_uxo": "train",
    "cub200": "train",
    "fgvc_aircraft": "train",
    "tiered_imagenet": "test",
}
DEFAULT_BENCHMARK_QUERY_SPLIT: Dict[str, str] = {
    "ctx_uxo": "test",
    "cub200": "test",
    "fgvc_aircraft": "test",
    "tiered_imagenet": "test",
}
DEFAULT_VAL_QUERY_SPLIT: Dict[str, str] = {
    "ctx_uxo": "valid",
    "cub200": "valid",
    "fgvc_aircraft": "valid",
    "tiered_imagenet": "test",
}
"""tieredImageNet's "valid" split is its own disjoint 97-class pool used
for hyperparameter search directly (matching its official protocol);
there is no cross-split pairing to make there, unlike the other three."""

BASE_MEAN_SPLIT: Dict[str, str] = {
    "ctx_uxo": "train",
    "cub200": "train",
    "fgvc_aircraft": "train",
    "tiered_imagenet": "train",
}


def _ctx_uxo_index() -> SplitIndex:
    """Wrap ``dataset.ctx_uxo``'s existing loader into a ``SplitIndex``,
    without duplicating or modifying any of its logic."""
    from dataset.ctx_uxo import CLASS_NAMES, list_class_instances

    pools: Dict[str, ClassPool] = {}
    for split in ("train", "valid", "test"):
        pools[split] = {
            class_name: list_class_instances(split, class_name)  # type: ignore[arg-type]
            for class_name in CLASS_NAMES
        }
    class_names = tuple(CLASS_NAMES)
    return SplitIndex(
        name="ctx_uxo",
        pools=pools,
        class_names_per_split={"train": class_names, "valid": class_names, "test": class_names},
        shared_classes_across_splits=True,
    )


def get_dataset_index(name: str, **loader_kwargs) -> SplitIndex:
    """Return the ``SplitIndex`` for one of ``DATASET_NAMES``.

    Args:
        name: One of "ctx_uxo", "cub200", "fgvc_aircraft", "tiered_imagenet".
        **loader_kwargs: Forwarded to the underlying per-dataset
            ``load_index`` (e.g. ``root=...``, or CUB-200-2011's
            ``valid_fraction=``/``split_seed=``). Not accepted for
            "ctx_uxo" (its loader takes no arguments here).

    Returns:
        The dataset's ``SplitIndex``.
    """
    if name == "ctx_uxo":
        if loader_kwargs:
            raise TypeError(f"get_dataset_index('ctx_uxo') takes no extra kwargs, got {loader_kwargs}.")
        return _ctx_uxo_index()
    if name == "cub200":
        from dataset.cub200 import load_index as _load

        return _load(**loader_kwargs)
    if name == "fgvc_aircraft":
        from dataset.fgvc_aircraft import load_index as _load

        return _load(**loader_kwargs)
    if name == "tiered_imagenet":
        from dataset.tiered_imagenet import load_index as _load

        return _load(**loader_kwargs)
    raise ValueError(f"Unknown dataset '{name}'. Expected one of {DATASET_NAMES}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print split statistics of a registered dataset.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="ctx_uxo")
    args = parser.parse_args()

    index = get_dataset_index(args.dataset)
    counts = index.counts()
    for split in index.pools:
        n_classes = len(index.class_names_per_split[split])
        n_images = sum(counts[split].values())
        print(f"split={split:5s} classes={n_classes:4d}  total_images={n_images:7d}")
    print(f"shared_classes_across_splits: {index.shared_classes_across_splits}")
    print(f"support_split default:      {DEFAULT_SUPPORT_SPLIT[args.dataset]}")
    print(f"benchmark query_split:      {DEFAULT_BENCHMARK_QUERY_SPLIT[args.dataset]}")
    print(f"hyperparam-search val split: {DEFAULT_VAL_QUERY_SPLIT[args.dataset]}")
    print(f"base-mean split:            {BASE_MEAN_SPLIT[args.dataset]}")

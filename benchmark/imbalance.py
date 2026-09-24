"""Query-set imbalance robustness (Table 4): balanced vs. Dirichlet-imbalanced
query distributions across the baseline methods.
"""

import dataclasses
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "benchmark" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithm.diagnostics import DiagnosticsConfig, run_diagnostics
from algorithm.features import ClipFeatureExtractor
from algorithm.rtim import HTIMConfig, cl2n_condition
from benchmark.baselines import run_all_methods
from benchmark.episode_generator import (
    ClassEpisodeEntry,
    EpisodeSpec,
    ScaleConfig,
    SMALL_SCALE_CONFIG,
    _build_eligible_pools,
    generate_episodes,
)
from dataset.ctx_uxo import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_INSTANCES_ROOT,
    NUM_CLASSES,
    load_images_parallel,
)

__all__ = [
    "BALANCED_ALPHA",
    "METHOD_ORDER",
    "sample_dirichlet_query_counts",
    "generate_imbalanced_episodes",
    "run_imbalance_comparison",
]


# Sentinel for the ordinary balanced protocol (fixed n_query per class); +inf
# is the balanced limit of Dirichlet(alpha).
BALANCED_ALPHA: float = float("inf")

# Methods returned by benchmark.baselines.run_all_methods, in report order.
METHOD_ORDER: Tuple[str, ...] = (
    "SimpleShot",
    "TIM (Euclidean)",
    "Hyp-SimpleShot",
    "LorentzTIM",
)

# Negative scalar curvature K < 0.
K_CURVATURE: float = -1.0

# d_eff fallback and upper bound for the small-scale run in __main__.
SMALL_SCALE_D_EFF_FALLBACK: int = 8
SMALL_SCALE_D_EFF_DIAG_CAP: int = 20


def sample_dirichlet_query_counts(
    num_classes: int,
    total_queries: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[int]:
    """Per-class query counts drawn from a symmetric Dirichlet(alpha).

    Proportions are scaled by ``total_queries`` and rounded with the
    largest-remainder method, so the counts sum exactly to ``total_queries``
    (per-element rounding would change the query budget between alpha levels).
    Ties are broken by class index via a stable sort.

    Args:
        num_classes: Number of classes C (>= 1).
        total_queries: Query budget to distribute (>= 0).
        alpha: Concentration (> 0); smaller means more skewed.
        rng: Seeded generator.

    Returns:
        ``num_classes`` non-negative ints summing to ``total_queries``.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    if total_queries < 0:
        raise ValueError(f"total_queries must be >= 0, got {total_queries}.")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")

    if total_queries == 0:
        return [0] * num_classes

    proportions = rng.dirichlet(np.full(num_classes, alpha, dtype=np.float64))
    raw = proportions * total_queries
    floor_counts = np.floor(raw).astype(np.int64)
    remainder = int(total_queries - int(floor_counts.sum()))

    if remainder > 0:
        fractional = raw - floor_counts
        # Largest-remainder allocation of the leftover units; stable sort
        # so ties resolve deterministically by (original) class index.
        order = np.argsort(-fractional, kind="stable")
        for i in range(remainder):
            floor_counts[order[i]] += 1

    counts = floor_counts.tolist()
    assert sum(counts) == total_queries, (
        f"Largest-remainder rounding invariant violated: sum(counts)="
        f"{sum(counts)} != total_queries={total_queries}."
    )
    assert all(c >= 0 for c in counts)
    return counts


def generate_imbalanced_episodes(
    split: str,
    n_shot: int,
    num_episodes: int,
    seed: int,
    alpha: float,
    total_queries_per_episode: int,
    max_per_class: Optional[int] = None,
    support_split: str = "train",
    root: Path = DEFAULT_INSTANCES_ROOT,
) -> List[EpisodeSpec]:
    """Paired NUM_CLASSES-way episodes whose query side is Dirichlet-imbalanced.

    Support sampling and capping are identical to
    ``episode_generator.generate_episodes``. Each episode's query counts come
    from :func:`sample_dirichlet_query_counts` on an independent RNG stream, so
    the Dirichlet draw never perturbs the support/query index draws. Requests
    larger than a class's eligible pool are capped, not redistributed.

    Args:
        split: Query split ("valid" or "test").
        n_shot: Requested shots per class.
        num_episodes: Number of episodes M.
        seed: Master seed; all randomness derives from it via SeedSequence.
        alpha: Dirichlet concentration (> 0).
        total_queries_per_episode: Query budget per episode, summed over classes.
        max_per_class: Optional cap on both eligible pools per class.
        support_split: Support split ("train" by default).
        root: Root of ``dataset/instances``.

    Returns:
        ``EpisodeSpec`` list. ``n_query`` holds the per-episode total; a class
        may have no queries in an episode, which is expected under imbalance.

    Raises:
        ValueError: On out-of-range arguments.
        RuntimeError: If a class has no eligible support instances.
    """
    if n_shot < 1:
        raise ValueError(f"n_shot must be >= 1, got {n_shot}.")
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}.")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if total_queries_per_episode < 1:
        raise ValueError(
            f"total_queries_per_episode must be >= 1, got {total_queries_per_episode}."
        )

    master_seq = np.random.SeedSequence(seed)
    pool_seq, episodes_root_seq = master_seq.spawn(2)
    pool_rng = np.random.default_rng(pool_seq)

    support_pools, query_pools = _build_eligible_pools(
        support_split=support_split,  # type: ignore[arg-type]
        query_split=split,  # type: ignore[arg-type]
        support_max_per_class=max_per_class,
        query_max_per_class=max_per_class,
        root=root,
        rng=pool_rng,
    )

    episode_seqs = episodes_root_seq.spawn(num_episodes)

    episodes: List[EpisodeSpec] = []
    # (episode, class) pairs where the Dirichlet request exceeded the
    # eligible query pool and was capped.
    shortfall_events = 0

    for ep_idx, ep_seq in enumerate(episode_seqs):
        # Two independent child streams so the Dirichlet draw's own
        # randomness consumption never perturbs the support/query index
        # draw stream.
        dirichlet_seq, sampling_seq = ep_seq.spawn(2)
        dirichlet_rng = np.random.default_rng(dirichlet_seq)
        sampling_rng = np.random.default_rng(sampling_seq)
        provenance_seed = int(ep_seq.generate_state(1, dtype=np.uint32)[0])

        requested_counts = sample_dirichlet_query_counts(
            NUM_CLASSES, total_queries_per_episode, alpha, dirichlet_rng
        )

        entries: List[ClassEpisodeEntry] = []
        for class_name, requested_q in zip(CLASS_NAMES, requested_counts, strict=True):
            support_pool = support_pools[class_name]
            query_pool = query_pools[class_name]

            s_a = min(n_shot, len(support_pool))
            if s_a == 0:
                raise RuntimeError(
                    f"Class '{class_name}' has zero eligible instances in "
                    f"support_split='{support_split}' (after max_per_class="
                    f"{max_per_class} restriction); cannot form an episode."
                )

            # Cap by the eligible pool; the shortfall is not redistributed.
            q_a = min(requested_q, len(query_pool))
            if q_a < requested_q:
                shortfall_events += 1

            support_idx = sampling_rng.choice(len(support_pool), size=s_a, replace=False)
            support_paths = tuple(support_pool[i] for i in support_idx)

            if q_a > 0:
                query_idx = sampling_rng.choice(len(query_pool), size=q_a, replace=False)
                query_paths = tuple(query_pool[i] for i in query_idx)
            else:
                query_paths = ()

            entries.append(
                ClassEpisodeEntry(
                    class_name=class_name,
                    support_paths=support_paths,
                    query_paths=query_paths,
                )
            )

        episodes.append(
            EpisodeSpec(
                index=ep_idx,
                n_shot=n_shot,
                n_query=total_queries_per_episode,
                seed=provenance_seed,
                classes=tuple(entries),
            )
        )

    if shortfall_events > 0:
        warnings.warn(
            f"generate_imbalanced_episodes(alpha={alpha}): {shortfall_events} "
            "(episode, class) draws requested more queries than the eligible "
            "pool contained and were capped without redistribution, so "
            "those episodes have fewer than total_queries_per_episode queries.",
            stacklevel=2,
        )

    return episodes


def run_imbalance_comparison(
    alpha_values: List[float],
    scale_config: ScaleConfig,
    htim_config: HTIMConfig,
    feature_cache: Dict[Path, Tensor],
    base_mean: Tensor,
    extractor: ClipFeatureExtractor,
    return_per_episode: bool = False,
):
    """Balanced-vs-Dirichlet query-imbalance comparison (Table 4).

    ``BALANCED_ALPHA`` uses ``generate_episodes`` (fixed ``n_query`` per class);
    any finite alpha uses :func:`generate_imbalanced_episodes`. All alpha levels
    share the shot count ``scale_config.shots[0]`` and the same total query
    budget ``NUM_CLASSES * scale_config.n_query``, so only the class
    distribution of the queries changes.

    ``feature_cache`` is extended in place with features for any path not
    already cached.

    Args:
        alpha_values: Alpha levels, e.g. ``[BALANCED_ALPHA, 1.0, 0.5]``.
        scale_config: Episode-count / pool-cap preset.
        htim_config: Config passed unchanged to ``run_all_methods``.
        feature_cache: ``{path: feature}`` cache, extended in place.
        base_mean: CL2N base-split centering vector.
        extractor: Feature extractor for cache misses.
        return_per_episode: Also return per-episode score rows.

    Returns:
        ``{method: {alpha: mean_macro_f1}}`` (NaN if an alpha level realized no
        queries), or ``(results, per_episode_rows)`` if ``return_per_episode``.
    """
    shot = scale_config.shots[0]
    total_queries_per_episode = NUM_CLASSES * scale_config.n_query

    results: Dict[str, Dict[float, float]] = {m: {} for m in METHOD_ORDER}
    per_episode_rows: List[Dict] = []

    for alpha in alpha_values:
        if alpha == BALANCED_ALPHA:
            episodes = generate_episodes(
                split="test",
                n_shot=shot,
                n_query=scale_config.n_query,
                num_episodes=scale_config.num_episodes,
                seed=scale_config.seed,
                max_per_class=scale_config.support_max_per_class,
                query_max_per_class=scale_config.query_max_per_class,
            )
        else:
            episodes = generate_imbalanced_episodes(
                split="test",
                n_shot=shot,
                num_episodes=scale_config.num_episodes,
                seed=scale_config.seed,
                alpha=alpha,
                total_queries_per_episode=total_queries_per_episode,
                max_per_class=scale_config.support_max_per_class,
            )

        # Extend feature_cache in place for any new paths
        needed_paths: set = set()
        for episode in episodes:
            for entry in episode.classes:
                needed_paths.update(entry.support_paths)
                needed_paths.update(entry.query_paths)
        missing = sorted(p for p in needed_paths if p not in feature_cache)
        if missing:
            missing_image_map = load_images_parallel(missing)
            images = [missing_image_map[p] for p in missing]
            feats = extractor(images, batch_size=32)
            for p, f in zip(missing, feats, strict=True):
                feature_cache[p] = f

        # Per-episode evaluation
        per_method_scores: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}
        alpha_label = "balanced" if alpha == BALANCED_ALPHA else alpha
        for ep_idx, episode in enumerate(episodes):
            support_paths: List[Path] = []
            support_labels: List[int] = []
            query_paths: List[Path] = []
            query_labels: List[int] = []
            for entry in episode.classes:
                label = CLASS_TO_IDX[entry.class_name]
                support_paths.extend(entry.support_paths)
                support_labels.extend([label] * len(entry.support_paths))
                query_paths.extend(entry.query_paths)
                query_labels.extend([label] * len(entry.query_paths))

            if not query_paths:
                # Possible only under extreme imbalance with a tiny budget.
                continue

            support_feats = torch.stack([feature_cache[p] for p in support_paths], dim=0)
            query_feats = torch.stack([feature_cache[p] for p in query_paths], dim=0)
            support_y = torch.tensor(support_labels, dtype=torch.long, device=support_feats.device)
            query_y = torch.tensor(query_labels, dtype=torch.long, device=query_feats.device)

            episode_scores = run_all_methods(
                support_feats, support_y, query_feats, query_y, base_mean, htim_config
            )
            for method in METHOD_ORDER:
                per_method_scores[method].append(episode_scores[method])
                per_episode_rows.append(
                    {"shot": shot, "alpha": alpha_label, "method": method,
                     "episode_index": ep_idx, "macro_f1": episode_scores[method]}
                )

        for method in METHOD_ORDER:
            scores = per_method_scores[method]
            results[method][alpha] = float(np.mean(scores)) if scores else float("nan")

    if return_per_episode:
        return results, per_episode_rows
    return results


def _small_scale_base_mean(feature_cache: Dict[Path, Tensor], support_paths: set) -> Tensor:
    """Mean feature over the episodes' support images (train split only);
    stands in for ``dataset.ctx_uxo.compute_base_split_mean`` on CPU."""
    ordered = sorted(support_paths)
    stacked = torch.stack([feature_cache[p] for p in ordered], dim=0)
    return stacked.mean(dim=0)


def _select_small_scale_d_eff(
    feature_cache: Dict[Path, Tensor], support_paths: set, base_mean: Tensor
) -> int:
    """Diagnostic d_eff on the support pool. Falls back to
    SMALL_SCALE_D_EFF_FALLBACK when the diagnostics fail or the estimate is
    outside [2, SMALL_SCALE_D_EFF_DIAG_CAP]: with n << 768 points, d_PR
    tracks the sample size rather than the intrinsic dimension."""
    ordered = sorted(support_paths)
    raw = torch.stack([feature_cache[p] for p in ordered], dim=0)
    z = cl2n_condition(raw, base_mean).cpu().numpy().astype(np.float64)

    try:
        diag_out = run_diagnostics(z, config=DiagnosticsConfig(delta_n_batches=5))
        d_eff_diag = int(diag_out["Z"]["d_eff"])
    except Exception:
        return SMALL_SCALE_D_EFF_FALLBACK
    if 2 <= d_eff_diag <= SMALL_SCALE_D_EFF_DIAG_CAP:
        return d_eff_diag
    return SMALL_SCALE_D_EFF_FALLBACK


if __name__ == "__main__":
    # Half the episodes of SMALL_SCALE_CONFIG, since the loop runs once per alpha.
    scale_config = dataclasses.replace(SMALL_SCALE_CONFIG, num_episodes=10)
    shot = scale_config.shots[0]

    extractor = ClipFeatureExtractor(device="cpu")

    balanced_episodes = generate_episodes(
        split="test",
        n_shot=shot,
        n_query=scale_config.n_query,
        num_episodes=scale_config.num_episodes,
        seed=scale_config.seed,
        max_per_class=scale_config.support_max_per_class,
        query_max_per_class=scale_config.query_max_per_class,
    )
    support_paths: set = set()
    seed_paths: set = set()
    for ep in balanced_episodes:
        for entry in ep.classes:
            support_paths.update(entry.support_paths)
            seed_paths.update(entry.support_paths)
            seed_paths.update(entry.query_paths)

    ordered_seed = sorted(seed_paths)
    seed_image_map = load_images_parallel(ordered_seed)
    images = [seed_image_map[p] for p in ordered_seed]
    feats = extractor(images, batch_size=32)
    feature_cache: Dict[Path, Tensor] = {p: feats[i] for i, p in enumerate(ordered_seed)}

    base_mean = _small_scale_base_mean(feature_cache, support_paths)
    d_eff = _select_small_scale_d_eff(feature_cache, support_paths, base_mean)
    htim_config = HTIMConfig(K=K_CURVATURE, d_eff=d_eff)
    print(f"S={shot}, M={scale_config.num_episodes}, d_eff={d_eff}")

    alpha_values: List[float] = [BALANCED_ALPHA, 1.0, 0.5]
    t0 = time.perf_counter()
    results = run_imbalance_comparison(
        alpha_values=alpha_values,
        scale_config=scale_config,
        htim_config=htim_config,
        feature_cache=feature_cache,
        base_mean=base_mean,
        extractor=extractor,
    )
    print(f"run_imbalance_comparison: {time.perf_counter() - t0:.1f}s")

    print(f"{'Method':<18s}" + "".join(
        f"{('balanced' if a == BALANCED_ALPHA else f'alpha={a}'):>16s}" for a in alpha_values
    ))
    for method in METHOD_ORDER:
        print(f"{method:<18s}" + "".join(f"{results[method][a]:>16.4f}" for a in alpha_values))

"""Random search for the per-shot hyperparameter tuple
psi_S* = (T, beta, ce_weight, marginal_h_weight, cond_h_weight, omega, tau).

Evaluates on validation-split episodes to select hyperparameter configurations.
"""

import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.features import ClipFeatureExtractor
from algorithm.rtim import (
    HTIMConfig,
    cl2n_condition,
    fit_pca_projection,
    hyperbolic_lift,
    htim_adapt,
)
from benchmark.episode_generator import EpisodeSpec, generate_episodes
from benchmark.evaluator import macro_f1
from dataset.ctx_uxo import CLASS_TO_IDX, NUM_CLASSES, load_images_parallel
from properties.search_space import (
    SearchSpace,
    diagnostics_informed_search_space,
    measure_geodesic_distance_scale,
)

DEFAULT_N_TRIALS = 25
DEFAULT_N_VAL_EPISODES = 30


@dataclass(frozen=True)
class SearchTrialResult:
    """One candidate evaluation record."""

    trial_index: int
    candidate: Dict[str, float]
    mean_val_macro_f1: float


@dataclass(frozen=True)
class ShotSearchResult:
    shot: int
    best_config: HTIMConfig
    best_candidate: Dict[str, float]
    best_val_macro_f1: float
    search_space: SearchSpace
    d_bar: float
    trials: List[SearchTrialResult]
    n_val_episodes_used: int
    wall_clock_s: float


def _config_from_candidate(candidate: Dict[str, float], K: float, d_eff: int) -> HTIMConfig:
    """HTIMConfig for one sampled psi, with the query-only MI pool used by every search."""
    return HTIMConfig(
        K=K,
        d_eff=d_eff,
        T=int(candidate["T"]),
        beta=float(candidate["beta"]),
        omega=float(candidate["omega"]),
        tau=float(candidate["tau"]),
        use_query_in_mi=False,
        ce_weight=float(candidate["ce_weight"]),
        marginal_h_weight=float(candidate["marginal_h_weight"]),
        cond_h_weight=float(candidate["cond_h_weight"]),
    )


def _episode_image_paths(episodes: List[EpisodeSpec]) -> Set[Path]:
    paths: Set[Path] = set()
    for ep in episodes:
        for entry in ep.classes:
            paths.update(entry.support_paths)
            paths.update(entry.query_paths)
    return paths


def _extract_feature_cache(
    paths: Set[Path], extractor: ClipFeatureExtractor, batch_size: int = 32
) -> Dict[Path, Tensor]:
    ordered = sorted(paths)
    image_map = load_images_parallel(ordered)  # parallel I/O, same tensors
    images = [image_map[p] for p in ordered]
    feats = extractor(images, batch_size=batch_size)
    return {p: feats[i] for i, p in enumerate(ordered)}


def _episode_to_tensors(
    episode: EpisodeSpec, feature_cache: Dict[Path, Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    support_feats = torch.stack([feature_cache[p] for p in support_paths], dim=0)
    query_feats = torch.stack([feature_cache[p] for p in query_paths], dim=0)
    support_y = torch.tensor(support_labels, dtype=torch.long, device=support_feats.device)
    query_y = torch.tensor(query_labels, dtype=torch.long, device=query_feats.device)
    return support_feats, support_y, query_feats, query_y


def _evaluate_candidate(
    candidate: Dict[str, float],
    K: float,
    d_eff: int,
    episodes: List[EpisodeSpec],
    feature_cache: Dict[Path, Tensor],
    base_mean: Tensor,
) -> float:
    """Mean LorentzTIM macro-F1 of one candidate over the validation episodes."""
    config = _config_from_candidate(candidate, K, d_eff)
    scores: List[float] = []
    for episode in episodes:
        support_feats, support_y, query_feats, query_y = _episode_to_tensors(
            episode, feature_cache
        )
        z_support = cl2n_condition(support_feats, base_mean)
        z_query = cl2n_condition(query_feats, base_mean)
        u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)
        x_support = hyperbolic_lift(z_support, u_pr, K)
        x_query = hyperbolic_lift(z_query, u_pr, K)
        result = htim_adapt(x_support, support_y, x_query, config)
        scores.append(
            macro_f1(
                query_y.cpu().numpy(), result.predictions.cpu().numpy(), NUM_CLASSES
            )
        )
    return float(np.mean(scores))


def run_search_for_shot(
    shot: int,
    K: float,
    d_eff: int,
    base_mean: Tensor,
    extractor: ClipFeatureExtractor,
    n_trials: int = DEFAULT_N_TRIALS,
    n_val_episodes: int = DEFAULT_N_VAL_EPISODES,
    n_query: int = 15,
    max_per_class: Optional[int] = None,
    seed: int = 42,
    root: Optional[Path] = None,
    backbone: str = "",
) -> ShotSearchResult:
    """Random search for psi_S* at one shot count, on validation episodes only.

    K and d_eff are held fixed. ``backbone`` salts the search RNG so different
    backbones draw different candidates for the same ``seed``.

    Args:
        shot: Shot count S.
        K: Curvature.
        d_eff: Effective manifold dimension.
        base_mean: Base-split mean.
        extractor: Feature extractor.
        n_trials: Search budget.
        n_val_episodes: Validation episodes per trial.
        n_query: Queries per class in validation episodes.
        max_per_class: Optional eligible-pool cap.
        seed: Master seed for episodes and search.
        root: Optional ``dataset/instances`` root.
        backbone: Backbone name used as RNG salt.
    """
    t0 = time.perf_counter()
    kwargs = {"root": root} if root is not None else {}
    val_episodes = generate_episodes(
        split="valid",
        n_shot=shot,
        n_query=n_query,
        num_episodes=n_val_episodes,
        seed=seed,
        max_per_class=max_per_class,
        query_max_per_class=max_per_class,
        **kwargs,
    )
    paths = _episode_image_paths(val_episodes)
    feature_cache = _extract_feature_cache(paths, extractor)

    # Diagnostics-informed search space: measure the geodesic-distance
    # scale on a sample of lifted validation-episode points.
    sample_ep = val_episodes[0]
    s_feats, _, q_feats, _ = _episode_to_tensors(sample_ep, feature_cache)
    z_sample = cl2n_condition(torch.cat([s_feats, q_feats], dim=0), base_mean)
    u_pr_sample = fit_pca_projection(z_sample, d_eff)
    lifted_sample = hyperbolic_lift(z_sample, u_pr_sample, K)
    d_bar = measure_geodesic_distance_scale(lifted_sample, K, seed=seed)
    search_space = diagnostics_informed_search_space(d_bar)

    backbone_salt = zlib.crc32(backbone.encode("utf-8")) if backbone else 0
    rng = np.random.default_rng(seed + 10_000 + shot + backbone_salt)
    trials: List[SearchTrialResult] = []
    best_score = -1.0
    best_candidate: Optional[Dict[str, float]] = None
    for trial_idx in range(n_trials):
        candidate = search_space.sample(rng)
        score = _evaluate_candidate(candidate, K, d_eff, val_episodes, feature_cache, base_mean)
        trials.append(SearchTrialResult(trial_idx, candidate, score))
        if score > best_score:
            best_score = score
            best_candidate = candidate

    assert best_candidate is not None, "n_trials must be >= 1"
    elapsed = time.perf_counter() - t0
    return ShotSearchResult(
        shot=shot,
        best_config=_config_from_candidate(best_candidate, K, d_eff),
        best_candidate=best_candidate,
        best_val_macro_f1=best_score,
        search_space=search_space,
        d_bar=d_bar,
        trials=trials,
        n_val_episodes_used=len(val_episodes),
        wall_clock_s=elapsed,
    )


if __name__ == "__main__":
    # Small search (5 trials x 5 validation episodes, S=1); the full budget is
    # run by export_full_results.py.
    extractor = ClipFeatureExtractor(device="cpu")
    K_probe, d_eff_probe = -1.0, 8
    # Base mean from 5 train images per class instead of the full split.
    from dataset.ctx_uxo import list_class_instances, CLASS_NAMES

    sample_paths = []
    for cname in CLASS_NAMES:
        sample_paths.extend(list_class_instances("train", cname)[:5])
    sample_img_map = load_images_parallel(sample_paths)
    sample_imgs = [sample_img_map[p] for p in sample_paths]
    base_mean_probe = extractor(sample_imgs).mean(dim=0)

    result = run_search_for_shot(
        shot=1,
        K=K_probe,
        d_eff=d_eff_probe,
        base_mean=base_mean_probe,
        extractor=extractor,
        n_trials=5,
        n_val_episodes=5,
        n_query=5,
        max_per_class=10,
        seed=42,
    )

    print(f"d_bar={result.d_bar:.4f} tau_range={result.search_space.tau_range} "
          f"omega_range={result.search_space.omega_range}")
    for t in result.trials:
        print(f"trial {t.trial_index}: val_macro_f1={t.mean_val_macro_f1:.4f}  {t.candidate}")
    print(f"best psi_1* = {result.best_candidate} (val macro-F1 {result.best_val_macro_f1:.4f}, "
          f"{result.wall_clock_s:.1f}s)")

    assert 0.0 <= result.best_val_macro_f1 <= 1.0
    assert len(result.trials) == 5

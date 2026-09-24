#!/usr/bin/env python3
"""Pilot (not the official configuration): LorentzTIM with a TIM-structured
search space -- T up to 1000, beta from a discrete set, three independent
loss weights and a query-only MI pool -- so geometry is the main remaining
difference from TIM (Euclidean, original).

Evaluates only LorentzTIM on the same test episodes as
``export_full_results.py`` and writes to a separate output directory.

Usage:
    python3 benchmark/lorentztim_extended_pilot.py --device cuda --backbone dinov3 \\
        --num-episodes 100 --shots 1,3,5,10 --search-trials 25 --val-episodes 30 \\
        --output-dir results_run/lorentztim_extended_pilot_table3_dinov3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from algorithm.rtim import cl2n_condition, fit_pca_projection, hyperbolic_lift, htim_adapt
from benchmark.episode_generator import EpisodeSpec, FULL_SCALE_CONFIG, generate_episodes
from benchmark.evaluator import macro_f1
from benchmark.export_full_results import (
    K_CURVATURE,
    _collect_paths,
    _episode_to_tensors,
    _extract_feature_cache,
    _select_d_eff,
)
from benchmark.hyperparam_search import _config_from_candidate, _episode_image_paths
from benchmark.reporting import prediction_metric_rows, write_csv
from dataset.ctx_uxo import CLASS_NAMES, NUM_CLASSES, compute_base_split_mean
from properties.search_space import diagnostics_informed_extended_search_space, measure_geodesic_distance_scale

METHOD_NAME: str = "LorentzTIM (extended pilot)"


def _evaluate_candidate_extended(
    candidate: Dict[str, float], K: float, d_eff: int,
    episodes: List[EpisodeSpec], feature_cache: Dict[Path, Tensor], base_mean: Tensor,
) -> float:
    config = _config_from_candidate(candidate, K, d_eff)
    scores: List[float] = []
    for episode in episodes:
        support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
        z_support = cl2n_condition(support_feats, base_mean)
        z_query = cl2n_condition(query_feats, base_mean)
        u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)
        x_support = hyperbolic_lift(z_support, u_pr, K)
        x_query = hyperbolic_lift(z_query, u_pr, K)
        result = htim_adapt(x_support, support_y, x_query, config)
        scores.append(macro_f1(query_y.cpu().numpy(), result.predictions.cpu().numpy(), NUM_CLASSES))
    return float(np.mean(scores))


def run_extended_search_for_shot(
    shot: int, K: float, d_eff: int, base_mean: Tensor, extractor,
    n_trials: int, n_val_episodes: int, n_query: int, seed: int,
) -> Dict:
    t0 = time.perf_counter()
    val_episodes = generate_episodes(
        split="valid", n_shot=shot, n_query=n_query, num_episodes=n_val_episodes, seed=seed,
    )
    paths = _episode_image_paths(val_episodes)
    feature_cache = _extract_feature_cache(paths, extractor)

    sample_ep = val_episodes[0]
    s_feats, _, q_feats, _ = _episode_to_tensors(sample_ep, feature_cache)
    z_sample = cl2n_condition(torch.cat([s_feats, q_feats], dim=0), base_mean)
    u_pr_sample = fit_pca_projection(z_sample, d_eff)
    lifted_sample = hyperbolic_lift(z_sample, u_pr_sample, K)
    d_bar = measure_geodesic_distance_scale(lifted_sample, K, seed=seed)
    search_space = diagnostics_informed_extended_search_space(d_bar)

    rng = np.random.default_rng(seed + 20_000 + shot)
    trials = []
    best_score = -1.0
    best_candidate: Optional[Dict[str, float]] = None
    for trial_idx in range(n_trials):
        candidate = search_space.sample(rng)
        score = _evaluate_candidate_extended(candidate, K, d_eff, val_episodes, feature_cache, base_mean)
        trials.append({"trial_index": trial_idx, "candidate": candidate, "mean_val_macro_f1": score})
        if score > best_score:
            best_score = score
            best_candidate = candidate
        print(f"    trial {trial_idx+1}/{n_trials}: val_macro_f1={score:.4f} candidate={candidate}", flush=True)

    assert best_candidate is not None
    return {
        "shot": shot, "best_candidate": best_candidate, "best_val_macro_f1": best_score,
        "d_bar": d_bar, "n_trials": n_trials, "n_val_episodes": n_val_episodes,
        "wall_clock_s": time.perf_counter() - t0,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--shots", type=str, default="1,3,5,10")
    p.add_argument("--search-trials", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=30)
    p.add_argument("--n-query", type=int, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_deterministic(FULL_SCALE_CONFIG.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shots = tuple(int(s.strip()) for s in args.shots.split(",") if s.strip())
    n_query = args.n_query if args.n_query is not None else FULL_SCALE_CONFIG.n_query
    seed = FULL_SCALE_CONFIG.seed

    print(f"[LorentzTIM extended pilot] Generating {args.num_episodes} paired episodes per shot "
          f"for S in {shots} (identical seed/generator to export_full_results.py) ...", flush=True)
    episodes_by_shot = {
        shot: generate_episodes(split="test", n_shot=shot, n_query=n_query, num_episodes=args.num_episodes, seed=seed)
        for shot in shots
    }
    all_paths, support_paths = _collect_paths(episodes_by_shot.values())

    print(f"Loading feature extractor '{args.backbone}' on {args.device}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print(f"Extracting features for {len(all_paths)} distinct images...", flush=True)
    t0 = time.perf_counter()
    feature_cache = _extract_feature_cache(all_paths, extractor)
    print(f"Feature extraction took {time.perf_counter() - t0:.1f}s.", flush=True)

    print("Computing the base-split mean over the complete train split...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    print(f"Base-split mean computed in {time.perf_counter() - t0:.1f}s.", flush=True)

    d_eff = _select_d_eff(feature_cache, support_paths, base_mean)
    print(f"d_eff = {d_eff} (K={K_CURVATURE})", flush=True)

    search_reports: List[str] = []
    per_class_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []

    for shot in shots:
        print(f"\nS={shot}: extended search ({args.search_trials} trials, "
              f"{args.val_episodes} val episodes/trial)", flush=True)
        search_result = run_extended_search_for_shot(
            shot=shot, K=K_CURVATURE, d_eff=d_eff, base_mean=base_mean, extractor=extractor,
            n_trials=args.search_trials, n_val_episodes=args.val_episodes, n_query=n_query, seed=seed,
        )
        cand = search_result["best_candidate"]
        print(f"S={shot}: best candidate = {cand} (val Macro-F1={search_result['best_val_macro_f1']:.4f}, "
              f"{search_result['wall_clock_s']:.1f}s)", flush=True)
        search_reports.append(
            f"## S={shot}\n\n- d_bar: {search_result['d_bar']:.6f}\n"
            f"- best candidate = {cand}\n"
            f"- best validation Macro-F1: {search_result['best_val_macro_f1']:.4f}\n"
            f"- trials: {search_result['n_trials']}, val episodes/trial: {search_result['n_val_episodes']}, "
            f"wall-clock: {search_result['wall_clock_s']:.1f}s\n"
        )
        config = _config_from_candidate(cand, K_CURVATURE, d_eff)

        episodes = episodes_by_shot[shot]
        print(f"S={shot}: evaluating {len(episodes)} benchmark episodes with {METHOD_NAME} ...", flush=True)
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        t_shot = time.perf_counter()
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
            query_y_np = query_y.cpu().numpy()
            z_support = cl2n_condition(support_feats, base_mean)
            z_query = cl2n_condition(query_feats, base_mean)
            u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)
            x_support = hyperbolic_lift(z_support, u_pr, K_CURVATURE)
            x_query = hyperbolic_lift(z_query, u_pr, K_CURVATURE)
            y_pred = htim_adapt(x_support, support_y, x_query, config).predictions.cpu().numpy()
            y_true_all.extend(query_y_np.tolist())
            y_pred_all.extend(y_pred.tolist())
            ep_f1 = precision_recall_fscore_support(
                query_y_np, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0
            )[2]
            per_episode_rows.append({"shot": shot, "method": METHOD_NAME, "episode_index": ep_idx, "macro_f1": ep_f1})
            if (ep_idx + 1) % 25 == 0:
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({time.perf_counter() - t_shot:.1f}s)", flush=True)

        per_class, agg, confusion = prediction_metric_rows(
            shot, METHOD_NAME, np.array(y_true_all), np.array(y_pred_all), CLASS_NAMES
        )
        per_class_rows.extend(per_class)
        aggregate_rows.append(agg)
        confusion_rows.extend(confusion)
        print(f"  {METHOD_NAME:32s} macro_f1={agg['macro_f1']:.4f} micro_f1={agg['micro_f1']:.4f} "
              f"weighted_f1={agg['weighted_f1']:.4f} accuracy={agg['accuracy']:.4f}", flush=True)

    (args.output_dir / "hyperparameter_search_extended.md").write_text("\n".join(search_reports))

    write_csv(args.output_dir, "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir, "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir, "confusion_matrix.csv", confusion_rows)
    write_csv(args.output_dir, "per_episode_scores.csv", per_episode_rows)


if __name__ == "__main__":
    main()

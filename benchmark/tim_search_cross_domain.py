#!/usr/bin/env python3
"""TIM (Euclidean) with its own per-shot random search on the cross-domain
episodes of ``cross_domain_eval.py``; the cross-domain counterpart of
``tim_search_ctx_uxo.py``.

Usage:
    python3 benchmark/tim_search_cross_domain.py --device cuda \\
        --dataset fgvc_aircraft --backbone dinov3 --num-episodes 100 \\
        --shots 1,3,5,10 --search-trials 25 --val-episodes 30 \\
        --output-dir results_run/tim_search_crossdomain_fgvc_aircraft_dinov3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from algorithm.rtim import HTIMConfig, cl2n_condition, fit_pca_projection
from benchmark.baselines import _euclidean_tim_predict
from benchmark.cross_domain_eval import (
    BASE_MEAN_BATCH_SIZE,
    N_WAY,
    WAY_SLOT_LABELS,
    _capped_pool,
    _episode_paths,
    _episode_to_tensors,
    _extract_feature_cache,
    _sample_episodes,
    _select_d_eff,
)
from benchmark.evaluator import macro_f1
from benchmark.reporting import prediction_metric_rows, write_csv
from dataset.fewshot_common import compute_base_mean_generic
from dataset.registry import (
    BASE_MEAN_SPLIT,
    DEFAULT_BENCHMARK_QUERY_SPLIT,
    DEFAULT_SUPPORT_SPLIT,
    DEFAULT_VAL_QUERY_SPLIT,
    get_dataset_index,
)
from properties.search_space import diagnostics_informed_euclidean_search_space, measure_euclidean_distance_scale

METHOD_NAME: str = "TIM (Euclidean, searched)"
K_CURVATURE: float = -1.0


def _config_from_candidate(candidate: Dict[str, float]) -> HTIMConfig:
    return HTIMConfig(
        # K and d_eff are required by HTIMConfig; _euclidean_tim_predict ignores them.
        K=K_CURVATURE, d_eff=1,
        T=int(candidate["T"]), beta=float(candidate["beta"]), tau=float(candidate["tau"]),
        use_query_in_mi=False,
        ce_weight=float(candidate["ce_weight"]),
        marginal_h_weight=float(candidate["marginal_h_weight"]),
        cond_h_weight=float(candidate["cond_h_weight"]),
    )


def _evaluate_candidate_euclidean(
    candidate: Dict[str, float], d_eff: int, episodes, feature_cache, base_mean: torch.Tensor,
) -> float:
    config = _config_from_candidate(candidate)
    scores: List[float] = []
    for episode in episodes:
        support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
        z_support = cl2n_condition(support_feats, base_mean)
        z_query = cl2n_condition(query_feats, base_mean)
        u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)
        p_support = z_support @ u_pr
        p_query = z_query @ u_pr
        num_classes = int(support_y.max().item()) + 1
        y_pred = _euclidean_tim_predict(p_support, support_y, p_query, num_classes, config)
        scores.append(macro_f1(query_y.cpu().numpy(), y_pred.cpu().numpy(), num_classes))
    return float(np.mean(scores))


def run_euclidean_search_for_shot(
    shot: int, d_eff: int, base_mean: torch.Tensor, extractor,
    support_pool, query_pool, n_trials: int, n_val_episodes: int, n_query: int, seed: int,
) -> Dict:
    t0 = time.perf_counter()
    val_episodes = _sample_episodes(
        n_val_episodes, n_way=N_WAY, n_shot=shot, n_query=n_query,
        support_pool=support_pool, query_pool=query_pool, seed=seed + 5_000 + shot,
    )
    feature_cache = _extract_feature_cache(_episode_paths(val_episodes), extractor)

    sample_ep = val_episodes[0]
    s_feats, _, q_feats, _ = _episode_to_tensors(sample_ep, feature_cache)
    z_sample = cl2n_condition(torch.cat([s_feats, q_feats], dim=0), base_mean)
    u_pr_sample = fit_pca_projection(z_sample, d_eff)
    p_sample = z_sample @ u_pr_sample
    d_bar = measure_euclidean_distance_scale(p_sample, seed=seed)
    search_space = diagnostics_informed_euclidean_search_space(d_bar)

    rng = np.random.default_rng(seed + 40_000 + shot)
    trials = []
    best_score = -1.0
    best_candidate: Optional[Dict[str, float]] = None
    for trial_idx in range(n_trials):
        candidate = search_space.sample(rng)
        score = _evaluate_candidate_euclidean(candidate, d_eff, val_episodes, feature_cache, base_mean)
        trials.append({"trial_index": trial_idx, "candidate": candidate, "mean_val_macro_f1": score})
        if score > best_score:
            best_score = score
            best_candidate = candidate

    assert best_candidate is not None
    return {
        "shot": shot, "best_candidate": best_candidate, "best_val_macro_f1": best_score,
        "d_bar": d_bar, "n_trials": n_trials, "n_val_episodes": n_val_episodes,
        "wall_clock_s": time.perf_counter() - t0,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["cub200", "fgvc_aircraft", "tiered_imagenet", "ctx_uxo"], required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--shots", type=str, default="1,3,5,10")
    p.add_argument("--search-trials", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=30)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--base-mean-max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_deterministic(args.seed)
    shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TIM-searched] Dataset: {args.dataset}   Backbone: {args.backbone}   Device: {args.device}", flush=True)
    index = get_dataset_index(args.dataset)
    support_split = DEFAULT_SUPPORT_SPLIT[args.dataset]
    bench_query_split = DEFAULT_BENCHMARK_QUERY_SPLIT[args.dataset]
    val_query_split = DEFAULT_VAL_QUERY_SPLIT[args.dataset]
    base_split = BASE_MEAN_SPLIT[args.dataset]

    support_pool = _capped_pool(index.pools[support_split], args.max_per_class, args.seed + 1)
    bench_query_pool = _capped_pool(index.pools[bench_query_split], args.max_per_class, args.seed + 2)
    val_query_pool = _capped_pool(index.pools[val_query_split], args.max_per_class, args.seed + 3)
    base_pool = _capped_pool(index.pools[base_split], args.base_mean_max_per_class, args.seed + 4)
    base_pool_paths: List[Path] = sorted(p for paths in base_pool.values() for p in paths)

    print(f"Loading feature extractor '{args.backbone}' on {args.device!r}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print(f"Computing base-split mean over {len(base_pool_paths)} images...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_mean_generic(base_pool_paths, feature_fn=extractor, batch_size=BASE_MEAN_BATCH_SIZE)
    print(f"Base-mean computed in {time.perf_counter() - t0:.1f}s.", flush=True)

    base_feature_cache = _extract_feature_cache(set(base_pool_paths), extractor)
    raw_d_eff = _select_d_eff(base_pool_paths, base_feature_cache, base_mean)
    del base_feature_cache

    min_support = min(len(v) for v in support_pool.values())
    min_bench_query = min(len(v) for v in bench_query_pool.values())
    worst_shot = min(shots)
    max_feasible_rank = N_WAY * (min(worst_shot, min_support) + min(args.n_query, min_bench_query))
    d_eff = min(raw_d_eff, max_feasible_rank)
    print(f"Diagnostic-calibrated d_eff = {d_eff}. TIM (Euclidean, searched): own independent search, "
          f"no manifold.", flush=True)

    per_class_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []
    search_reports: List[str] = []

    for shot in shots:
        print(f"\nS={shot}", flush=True)
        print(f"Running TIM(Euclidean) psi_{shot}* independent search ({args.search_trials} trials, "
              f"{args.val_episodes} val episodes/trial)...", flush=True)
        search_result = run_euclidean_search_for_shot(
            shot=shot, d_eff=d_eff, base_mean=base_mean, extractor=extractor,
            support_pool=support_pool, query_pool=val_query_pool,
            n_trials=args.search_trials, n_val_episodes=args.val_episodes, n_query=args.n_query, seed=args.seed,
        )
        cand = search_result["best_candidate"]
        print(f"psi_{shot}* = {cand} (val Macro-F1={search_result['best_val_macro_f1']:.4f}, "
              f"{search_result['wall_clock_s']:.1f}s)", flush=True)
        search_reports.append(
            f"## S={shot}\n\n- d_bar (Euclidean): {search_result['d_bar']:.6f}\n"
            f"- best psi_{shot}* = {cand}\n- best validation Macro-F1: {search_result['best_val_macro_f1']:.4f}\n"
            f"- trials: {search_result['n_trials']}, val episodes/trial: {search_result['n_val_episodes']}, "
            f"wall-clock: {search_result['wall_clock_s']:.1f}s\n"
        )
        config = _config_from_candidate(cand)

        print(f"Sampling {args.num_episodes} paired benchmark episodes (identical seed to cross_domain_eval.py)...",
              flush=True)
        episodes = _sample_episodes(
            args.num_episodes, n_way=N_WAY, n_shot=shot, n_query=args.n_query,
            support_pool=support_pool, query_pool=bench_query_pool, seed=args.seed + shot,
        )
        feature_cache = _extract_feature_cache(_episode_paths(episodes), extractor)

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        t_shot = time.perf_counter()
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
            query_y_np = query_y.cpu().numpy()
            z_support = cl2n_condition(support_feats, base_mean)
            z_query = cl2n_condition(query_feats, base_mean)
            u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)
            p_support = z_support @ u_pr
            p_query = z_query @ u_pr
            num_classes = int(support_y.max().item()) + 1
            y_pred = _euclidean_tim_predict(p_support, support_y, p_query, num_classes, config).cpu().numpy()
            y_true_all.extend(query_y_np.tolist())
            y_pred_all.extend(y_pred.tolist())
            ep_f1 = precision_recall_fscore_support(
                query_y_np, y_pred, labels=range(N_WAY), average="macro", zero_division=0
            )[2]
            per_episode_rows.append({"shot": shot, "method": METHOD_NAME, "episode_index": ep_idx, "macro_f1": ep_f1})
            if (ep_idx + 1) % 100 == 0:
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({time.perf_counter() - t_shot:.1f}s elapsed)", flush=True)

        per_class, agg, confusion = prediction_metric_rows(
            shot, METHOD_NAME, np.asarray(y_true_all), np.asarray(y_pred_all), WAY_SLOT_LABELS,
            count_field="n_query_total",
        )
        per_class_rows.extend(per_class)
        aggregate_rows.append(agg)
        confusion_rows.extend(confusion)
        print(f"  {METHOD_NAME:28s} macro_f1={agg['macro_f1']:.4f} micro_f1={agg['micro_f1']:.4f}", flush=True)

    (args.output_dir / "hyperparameter_search_extended.md").write_text("\n".join(search_reports))

    write_csv(args.output_dir, "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir, "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir, "confusion_matrix.csv", confusion_rows)
    write_csv(args.output_dir, "per_episode_scores.csv", per_episode_rows)


if __name__ == "__main__":
    main()

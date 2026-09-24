#!/usr/bin/env python3
"""TIM (Euclidean, original TIM-GD hyperparameters) on the same cross-domain
episodes as ``cross_domain_eval.py`` (same seed and pool capping). No search.

Usage:
    python3 benchmark/tim_original_cross_domain.py --device cuda \\
        --dataset fgvc_aircraft --backbone dinov3 --num-episodes 1000 \\
        --shots 1,3,5,10 --output-dir results_run/tim_original_crossdomain_fgvc_aircraft_dinov3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from algorithm.rtim import cl2n_condition, fit_pca_projection
from benchmark.baselines import _euclidean_tim_original_predict
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
from benchmark.reporting import prediction_metric_rows, write_csv
from dataset.fewshot_common import compute_base_mean_generic
from dataset.registry import (
    BASE_MEAN_SPLIT,
    DEFAULT_BENCHMARK_QUERY_SPLIT,
    DEFAULT_SUPPORT_SPLIT,
    get_dataset_index,
)

METHOD_NAME: str = "TIM (Euclidean, original)"
K_CURVATURE: float = -1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["cub200", "fgvc_aircraft", "tiered_imagenet", "ctx_uxo"], required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=1000)
    p.add_argument("--shots", type=str, default="1,3,5,10")
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

    print(f"[TIM-original] Dataset: {args.dataset}   Backbone: {args.backbone}   Device: {args.device}", flush=True)
    index = get_dataset_index(args.dataset)
    support_split = DEFAULT_SUPPORT_SPLIT[args.dataset]
    bench_query_split = DEFAULT_BENCHMARK_QUERY_SPLIT[args.dataset]
    base_split = BASE_MEAN_SPLIT[args.dataset]

    support_pool = _capped_pool(index.pools[support_split], args.max_per_class, args.seed + 1)
    bench_query_pool = _capped_pool(index.pools[bench_query_split], args.max_per_class, args.seed + 2)
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
    print(f"Diagnostic-calibrated d_eff = {d_eff} (K fixed at {K_CURVATURE}). "
          f"TIM-original: T=1000, lr=1e-4, tau=7.5 -- no search.", flush=True)

    per_class_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []

    for shot in shots:
        print(f"\nS={shot}", flush=True)
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
            y_pred = _euclidean_tim_original_predict(p_support, support_y, p_query, num_classes).cpu().numpy()
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

    write_csv(args.output_dir, "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir, "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir, "confusion_matrix.csv", confusion_rows)
    write_csv(args.output_dir, "per_episode_scores.csv", per_episode_rows)


if __name__ == "__main__":
    main()

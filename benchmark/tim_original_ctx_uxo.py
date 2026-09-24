#!/usr/bin/env python3
"""TIM (Euclidean) with the original TIM-GD hyperparameters (Boudiaf et al.:
T=1000, lr=1e-4, tau=7.5, independently weighted losses) on the same paired
CTX-UXO test episodes as ``export_full_results.py``. Output uses the same
CSV schema.

Usage:
    python3 benchmark/tim_original_ctx_uxo.py --device cuda --backbone dinov3 \\
        --num-episodes 1000 --shots 1,3,5,10 --output-dir results_run/tim_original_table3_dinov3
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
from benchmark.episode_generator import FULL_SCALE_CONFIG, generate_episodes
from benchmark.export_full_results import (
    K_CURVATURE,
    _collect_paths,
    _episode_to_tensors,
    _extract_feature_cache,
    _select_d_eff,
)
from benchmark.reporting import prediction_metric_rows, write_csv
from dataset.ctx_uxo import CLASS_NAMES, NUM_CLASSES, compute_base_split_mean

METHOD_NAME: str = "TIM (Euclidean, original)"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=1000)
    p.add_argument("--shots", type=str, required=True, help="Comma-separated shot counts S.")
    p.add_argument("--n-query", type=int, default=None, help="Default: FULL_SCALE_CONFIG.n_query.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_deterministic(FULL_SCALE_CONFIG.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shots = tuple(int(s.strip()) for s in args.shots.split(",") if s.strip())
    n_query = args.n_query if args.n_query is not None else FULL_SCALE_CONFIG.n_query
    seed = FULL_SCALE_CONFIG.seed

    print(f"[TIM-original] Generating {args.num_episodes} paired episodes per shot for S in {shots} "
          f"(identical seed/generator to export_full_results.py) ...", flush=True)
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
    print(f"d_eff = {d_eff} (K={K_CURVATURE}, fixed TIM-original hyperparameters -- no search)", flush=True)

    per_class_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []

    for shot in shots:
        episodes = episodes_by_shot[shot]
        print(f"\nS={shot}: evaluating {len(episodes)} episodes with TIM-original (T=1000, lr=1e-4, tau=7.5) ...",
              flush=True)

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
            try:
                y_pred = _euclidean_tim_original_predict(
                    p_support, support_y, p_query, NUM_CLASSES
                ).cpu().numpy()
            except Exception as exc:  # skip and report
                print(f"  WARNING: episode {ep_idx} skipped ({type(exc).__name__}: {exc})", flush=True)
                continue
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
        print(f"  {METHOD_NAME:28s} macro_f1={agg['macro_f1']:.4f} micro_f1={agg['micro_f1']:.4f} "
              f"weighted_f1={agg['weighted_f1']:.4f} accuracy={agg['accuracy']:.4f}", flush=True)

    write_csv(args.output_dir, "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir, "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir, "confusion_matrix.csv", confusion_rows)
    write_csv(args.output_dir, "per_episode_scores.csv", per_episode_rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Curvature sweep following the (curv) ablation.

Scaling the tangent vector by s before the exponential map (K = -1 fixed) is
equivalent to using K_eff = -s^2, since the exp-map argument is
sqrt(-K) |u|. The sweep reports s and K_eff with psi*_S, CL2N and R-Adam held
fixed.

Usage:
    python3 benchmark/curvature_scale_sweep.py --device cuda --backbone dinov3 \\
        --shots 5 --num-episodes 1000 --scales 1,2,3,5,10,20 \\
        --output-dir results_run/curvature_scale_sweep_dinov3
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from algorithm.manifold import exp_map
from algorithm.rtim import (
    cl2n_condition,
    fit_pca_projection,
    htim_adapt,
    _origin,
)
from benchmark.episode_generator import FULL_SCALE_CONFIG, generate_episodes
from benchmark.evaluator import macro_f1
from benchmark.export_full_results import _collect_paths, _select_d_eff
from benchmark.hyperparam_search import (
    _episode_to_tensors,
    _extract_feature_cache,
    run_search_for_shot,
)
from dataset.ctx_uxo import compute_base_split_mean

K_CURVATURE = -1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=FULL_SCALE_CONFIG.num_episodes)
    p.add_argument("--shots", type=str, required=True, help="Comma-separated shot counts.")
    p.add_argument("--scales", type=str, default="1,2,3,5,10,20", help="Comma-separated tangent-vector scales.")
    p.add_argument("--search-trials", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=FULL_SCALE_CONFIG.seed)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _scaled_hyperbolic_lift(z: Tensor, U_pr: Tensor, K: float, scale: float) -> Tensor:
    """Same as :func:`algorithm.rtim.hyperbolic_lift`, but the tangent
    vector is multiplied by ``scale`` before the exponential map. K stays
    fixed; ``scale=1.0`` reproduces ``hyperbolic_lift``."""
    projected = z @ U_pr
    zero_time = torch.zeros_like(projected[..., :1])
    v = torch.cat([zero_time, projected], dim=-1) * scale
    d_eff = U_pr.shape[1]
    o = _origin(d_eff, K, dtype=z.dtype, device=z.device)
    o = o.expand(v.shape[0], -1)
    return exp_map(o, v, K)


def main() -> None:
    args = _parse_args()
    set_deterministic(args.seed)
    shots = [int(s) for s in args.shots.split(",")]
    scales = [float(s) for s in args.scales.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading feature extractor '{args.backbone}' on {args.device!r}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print("Computing the base-split mean over the complete train split...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    print(f"Base-split mean computed in {time.perf_counter() - t0:.1f}s.", flush=True)

    all_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []

    # Same paired test episodes, d_eff calibration and backbone-salted
    # psi_S* search as export_full_results.py --force-search.
    episodes_by_shot = {
        shot: generate_episodes(
            split="test", n_shot=shot, n_query=FULL_SCALE_CONFIG.n_query,
            num_episodes=args.num_episodes, seed=args.seed,
        )
        for shot in shots
    }
    all_paths, support_paths = _collect_paths(episodes_by_shot.values())
    print(f"Extracting features for {len(all_paths)} distinct images...", flush=True)
    feature_cache = _extract_feature_cache(all_paths, extractor)
    d_eff = _select_d_eff(feature_cache, support_paths, base_mean)
    print(f"d_eff={d_eff}", flush=True)

    for shot in shots:
        print(f"\nS={shot}", flush=True)
        episodes = episodes_by_shot[shot]
        print(f"Searching psi_{shot}* (R={args.search_trials}, val={args.val_episodes})...", flush=True)
        search_result = run_search_for_shot(
            shot=shot, K=K_CURVATURE, d_eff=d_eff, base_mean=base_mean, extractor=extractor,
            n_trials=args.search_trials, n_val_episodes=args.val_episodes, seed=args.seed,
            backbone=args.backbone,
        )
        base_config = search_result.best_config
        print(f"psi_{shot}* = {search_result.best_candidate}", flush=True)

        scale_scores: Dict[float, List[float]] = {s: [] for s in scales}
        t0 = time.perf_counter()
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
            query_y_np = query_y.cpu().numpy()
            num_classes = int(support_y.max().item()) + 1

            z_support = cl2n_condition(support_feats, base_mean)
            z_query = cl2n_condition(query_feats, base_mean)
            u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)

            for s in scales:
                support_h = _scaled_hyperbolic_lift(z_support, u_pr, K_CURVATURE, s)
                query_h = _scaled_hyperbolic_lift(z_query, u_pr, K_CURVATURE, s)
                result = htim_adapt(support_h, support_y, query_h, base_config)
                f1 = macro_f1(query_y_np, result.predictions.detach().cpu().numpy(), num_classes)
                scale_scores[s].append(f1)
                per_episode_rows.append({"shot": shot, "scale": s, "episode_index": ep_idx, "macro_f1": f1})

            if (ep_idx + 1) % 100 == 0:
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({time.perf_counter() - t0:.1f}s)", flush=True)

        base_f1 = float(np.mean(scale_scores[1.0])) if 1.0 in scale_scores else None
        for s in scales:
            mean_f1 = float(np.mean(scale_scores[s]))
            all_rows.append(
                {
                    "shot": shot, "scale": s, "K_eff": -(s ** 2), "mean_macro_f1": mean_f1,
                    "delta_vs_scale1": (float("nan") if base_f1 is None else mean_f1 - base_f1),
                    "n_episodes": len(scale_scores[s]),
                }
            )
            print(f"  scale={s:<6g} K_eff={-(s**2):<8g} macro_f1={mean_f1:.4f}", flush=True)

    def _write_csv(name: str, rows: List[Dict]) -> None:
        path = args.output_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}", flush=True)

    _write_csv("scale_sweep_summary.csv", all_rows)
    _write_csv("scale_sweep_per_episode.csv", per_episode_rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Table 5 (ablation suite) on CTX-UXO.

Uses the paired test episodes, d_eff calibration and backbone-salted psi_S*
search of ``export_full_results.py --force-search``, so for the same
--shots/--num-episodes/--backbone/--seed it reproduces Table 3's psi_S*.
Row (c) is reported as N/A (see ``benchmark.ablations._ROW_C_NOTE``).

Usage:
    python3 benchmark/run_ablations_full.py --device cuda --backbone dinov3 \\
        --num-episodes 1000 --output-dir results_ablation
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from algorithm.rtim import cl2n_condition, fit_pca_projection
from benchmark.ablations import run_ablation_suite
from benchmark.baselines import _euclidean_tim_predict
from benchmark.episode_generator import FULL_SCALE_CONFIG, generate_episodes
from benchmark.evaluator import macro_f1
from benchmark.export_full_results import _collect_paths, _select_d_eff
from benchmark.hyperparam_search import (
    _episode_to_tensors,
    _extract_feature_cache,
    run_search_for_shot,
)
from dataset.ctx_uxo import NUM_CLASSES, compute_base_split_mean

ABLATION_ROWS: List[str] = ["Full Model", "(a)", "(curv)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
K_CURVATURE = -1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=FULL_SCALE_CONFIG.num_episodes)
    p.add_argument("--shots", type=str, default=None, help="Comma-separated; default: FULL_SCALE_CONFIG.shots.")
    p.add_argument("--search-trials", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=FULL_SCALE_CONFIG.seed)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_deterministic(args.seed)
    shots = [int(s) for s in args.shots.split(",")] if args.shots else list(FULL_SCALE_CONFIG.shots)
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

        row_scores: Dict[str, List[float]] = {row: [] for row in ABLATION_ROWS}
        t0 = time.perf_counter()
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
            query_y_np = query_y.cpu().numpy()

            z_support = cl2n_condition(support_feats, base_mean)
            z_query = cl2n_condition(query_feats, base_mean)
            u_pr = fit_pca_projection(torch.cat([z_support, z_query], dim=0), d_eff)
            p_support, p_query = z_support @ u_pr, z_query @ u_pr
            euclidean_preds = _euclidean_tim_predict(p_support, support_y, p_query, NUM_CLASSES, base_config)
            euclidean_twin_f1 = macro_f1(query_y_np, euclidean_preds.cpu().numpy(), NUM_CLASSES)

            result = run_ablation_suite(
                support_feats, support_y, query_feats, query_y, base_mean, base_config, euclidean_twin_f1,
            )
            for row in ABLATION_ROWS:
                val = result[row]
                row_scores[row].append(val)
                per_episode_rows.append({"shot": shot, "row": row, "episode_index": ep_idx, "macro_f1": val})
            if (ep_idx + 1) % 100 == 0:
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({time.perf_counter() - t0:.1f}s)", flush=True)

        full_f1 = float(np.nanmean(row_scores["Full Model"]))
        for row in ABLATION_ROWS:
            mean_f1 = float("nan") if row == "(c)" else float(np.nanmean(row_scores[row]))
            all_rows.append(
                {
                    "shot": shot, "row": row, "mean_macro_f1": mean_f1,
                    "delta_vs_full": (float("nan") if row in ("Full Model", "(c)") else mean_f1 - full_f1),
                    "n_episodes": int(np.sum(~np.isnan(row_scores[row]))),
                }
            )
            print(f"  {row:12s} macro_f1={mean_f1:.4f}", flush=True)

    def _write_csv(name: str, rows: List[Dict]) -> None:
        path = args.output_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}", flush=True)

    _write_csv("ablation_summary.csv", all_rows)
    _write_csv("ablation_per_episode.csv", per_episode_rows)


if __name__ == "__main__":
    main()

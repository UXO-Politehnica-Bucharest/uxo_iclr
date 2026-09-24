#!/usr/bin/env python3
"""Full-scale driver for the query-imbalance study
(``benchmark.imbalance.run_imbalance_comparison``); every alpha level is
evaluated at the single shot count ``--shot``.

Usage:
    python3 benchmark/run_imbalance_full.py --device cuda --backbone dinov3 \\
        --num-episodes 1000 --shot 5 --output-dir results_imbalance
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.diagnostics import DiagnosticsConfig, run_diagnostics
from algorithm.features import get_feature_extractor
from algorithm.rtim import cl2n_condition
from benchmark.episode_generator import ScaleConfig, generate_episodes
from benchmark.hyperparam_search import run_search_for_shot
from benchmark.imbalance import BALANCED_ALPHA, run_imbalance_comparison
from dataset.ctx_uxo import compute_base_split_mean, load_images_parallel

D_EFF_SUBSAMPLE_N = 3000
D_EFF_SUBSAMPLE_SEED = 42
K_CURVATURE = -1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--num-episodes", type=int, default=1000)
    p.add_argument("--shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--alphas", type=str, default="1.0,0.5,0.1", help="Dirichlet alpha levels (excl. balanced).")
    p.add_argument("--search-trials", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_deterministic(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = [BALANCED_ALPHA] + [float(a) for a in args.alphas.split(",")]

    print(f"Loading feature extractor '{args.backbone}' on {args.device!r}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print("Computing the base-split mean over the complete train split...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    print(f"Base-split mean computed in {time.perf_counter() - t0:.1f}s.", flush=True)

    print("Sampling a diagnostic subsample to select d_eff...", flush=True)
    probe_episodes = generate_episodes(
        split="train", n_shot=args.shot, n_query=args.n_query, num_episodes=5, seed=args.seed,
    )
    probe_paths = sorted({p for ep in probe_episodes for e in ep.classes for p in (e.support_paths + e.query_paths)})
    rng = np.random.default_rng(D_EFF_SUBSAMPLE_SEED)
    if len(probe_paths) > D_EFF_SUBSAMPLE_N:
        idx = rng.choice(len(probe_paths), size=D_EFF_SUBSAMPLE_N, replace=False)
        probe_paths = [probe_paths[i] for i in idx]
    probe_map = load_images_parallel(probe_paths)
    probe_feats = extractor([probe_map[p] for p in probe_paths])
    z_probe = cl2n_condition(probe_feats, base_mean).cpu().numpy().astype(np.float64)
    d_eff = int(run_diagnostics(z_probe, config=DiagnosticsConfig(delta_n_batches=8))["Z"]["d_eff"])
    print(f"d_eff={d_eff}", flush=True)

    print(f"Searching psi_{args.shot}* (R={args.search_trials}, val={args.val_episodes})...", flush=True)
    search_result = run_search_for_shot(
        shot=args.shot, K=K_CURVATURE, d_eff=d_eff, base_mean=base_mean, extractor=extractor,
        n_trials=args.search_trials, n_val_episodes=args.val_episodes, seed=args.seed,
        backbone=args.backbone,
    )
    htim_config = search_result.best_config
    print(f"psi_{args.shot}* = {search_result.best_candidate}", flush=True)

    scale_config = ScaleConfig(
        name="imbalance_full_scale",
        support_max_per_class=None,
        query_max_per_class=None,
        num_episodes=args.num_episodes,
        shots=(args.shot,),
        n_query=args.n_query,
        seed=args.seed,
    )

    print(f"\nRunning run_imbalance_comparison at M={args.num_episodes}, alphas={alpha_values}...", flush=True)
    t0 = time.perf_counter()
    feature_cache: dict = {}
    results, per_episode_rows = run_imbalance_comparison(
        alpha_values=alpha_values, scale_config=scale_config, htim_config=htim_config,
        feature_cache=feature_cache, base_mean=base_mean, extractor=extractor,
        return_per_episode=True,
    )
    print(f"run_imbalance_comparison took {time.perf_counter() - t0:.1f}s.", flush=True)

    rows = []
    for method, per_alpha in results.items():
        for alpha, mean_f1 in per_alpha.items():
            alpha_label = "balanced" if alpha == BALANCED_ALPHA else alpha
            rows.append({"shot": args.shot, "method": method, "alpha": alpha_label, "mean_macro_f1": mean_f1})
            print(f"  {method:16s} alpha={alpha_label!s:>10s}  macro_f1={mean_f1:.4f}", flush=True)

    out_path = args.output_dir / "imbalance_full.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shot", "method", "alpha", "mean_macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}", flush=True)

    per_ep_path = args.output_dir / "imbalance_per_episode.csv"
    with open(per_ep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shot", "alpha", "method", "episode_index", "macro_f1"])
        writer.writeheader()
        writer.writerows(per_episode_rows)
    print(f"Wrote {per_ep_path}", flush=True)


if __name__ == "__main__":
    main()

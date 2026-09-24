import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.rtim import (
    HTIMConfig,
    cl2n_condition,
    fit_pca_projection,
    htim_adapt,
    hyperbolic_lift,
)
from benchmark.baselines import _euclidean_tim_predict
from benchmark.episode_generator import EpisodeSpec, FULL_SCALE_CONFIG, generate_episodes
from benchmark.evaluator import confidence_interval
from benchmark.export_full_results import (
    K_CURVATURE,
    SEARCHED_HYPERPARAMETERS,
    _collect_paths,
    _extract_feature_cache,
    _episode_to_tensors,
    _select_d_eff,
)
from benchmark.hyperparam_search import _config_from_candidate
from dataset.ctx_uxo import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, compute_base_split_mean

# Elongated munitions that are mostly confused with one another.
CONFUSABLE_CLASSES: Tuple[str, ...] = ("Aviation_Bomb", "Mortar_Bomb", "Projectile", "RPG")
# Small, rounded, visually distinct from the four above.
DISTINCT_CLASSES: Tuple[str, ...] = ("Grenade",)

assert set(CONFUSABLE_CLASSES) | set(DISTINCT_CLASSES) == set(CLASS_NAMES), (
    "CONFUSABLE_CLASSES + DISTINCT_CLASSES must partition CLASS_NAMES exactly "
    f"-- got {CONFUSABLE_CLASSES + DISTINCT_CLASSES} vs {CLASS_NAMES}"
)

DEFAULT_SHOTS: Tuple[int, ...] = (5, 10)
DEFAULT_NUM_EPISODES: int = 500


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--shots",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SHOTS),
        help=(
            "Comma-separated shot counts to analyze. Must each already have a "
            "hardcoded psi_S* for --backbone in export_full_results.SEARCHED_HYPERPARAMETERS "
            f"(currently: {sorted(SEARCHED_HYPERPARAMETERS)}). Default: "
            f"{','.join(str(s) for s in DEFAULT_SHOTS)} (the two shots where the "
            "main benchmark found a statistically significant LorentzTIM "
            "advantage)."
        ),
    )
    p.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov3",
        help="Feature extractor backbone: 'dinov3' (paper's primary backbone), 'clip' (Appendix cross-backbone check), or HuggingFace model ID (producing d=768 embeddings). Default: 'dinov3'.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("results_class_groups"))
    return p.parse_args()


def _predict_two_methods(
    support_raw: Tensor,
    support_y: Tensor,
    query_raw: Tensor,
    base_mean: Tensor,
    config: HTIMConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """LorentzTIM and TIM (Euclidean) query predictions for one episode."""
    num_classes = int(support_y.max().item()) + 1
    z_support = cl2n_condition(support_raw, base_mean)
    z_query = cl2n_condition(query_raw, base_mean)
    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, config.d_eff)
    p_support = z_support @ u_pr
    p_query = z_query @ u_pr

    tim_euclidean_pred = (
        _euclidean_tim_predict(p_support, support_y, p_query, num_classes, config).cpu().numpy()
    )

    support_h = hyperbolic_lift(z_support, u_pr, config.K)
    query_h = hyperbolic_lift(z_query, u_pr, config.K)
    lorentztim_pred = htim_adapt(support_h, support_y, query_h, config).predictions.cpu().numpy()

    return lorentztim_pred, tim_euclidean_pred


def _group_f1(y_true: np.ndarray, y_pred: np.ndarray, group_idxs: Sequence[int]) -> float:
    """Mean per-class F1 over ``group_idxs``, with precision computed over the
    full label space so false positives from outside the group count.
    """
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0
    )
    return float(np.mean([f1[i] for i in group_idxs]))


def _cohens_dz(delta: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(delta) / std(delta, ddof=1)."""
    sd = float(np.std(delta, ddof=1))
    return float(np.mean(delta) / sd) if sd > 0 else float("nan")


def main() -> None:
    args = _parse_args()
    set_deterministic(FULL_SCALE_CONFIG.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shots = tuple(int(s.strip()) for s in args.shots.split(",") if s.strip())
    for shot in shots:
        if (args.backbone, shot) not in SEARCHED_HYPERPARAMETERS:
            raise ValueError(
                f"S={shot} has no hardcoded psi_S* for backbone={args.backbone!r} in SEARCHED_HYPERPARAMETERS "
                f"(available: {sorted(SEARCHED_HYPERPARAMETERS)}). Run "
                "benchmark/export_full_results.py --shots <this shot> first "
                "(or hardcode the resulting psi_S*) before analyzing it here."
            )

    confusable_idxs = [CLASS_TO_IDX[c] for c in CONFUSABLE_CLASSES]
    distinct_idxs = [CLASS_TO_IDX[c] for c in DISTINCT_CLASSES]
    n_query = FULL_SCALE_CONFIG.n_query
    seed = FULL_SCALE_CONFIG.seed

    print(f"Generating {args.num_episodes} paired test episodes per shot for S in {shots}...", flush=True)
    episodes_by_shot: Dict[int, List[EpisodeSpec]] = {
        shot: generate_episodes(
            split="test", n_shot=shot, n_query=n_query, num_episodes=args.num_episodes, seed=seed
        )
        for shot in shots
    }
    all_paths, support_paths = _collect_paths(episodes_by_shot.values())

    print(f"Loading feature extractor '{args.backbone}' on {args.device}...", flush=True)
    from algorithm.features import get_feature_extractor

    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print(f"Extracting features for {len(all_paths)} distinct images...", flush=True)
    t0 = time.perf_counter()
    feature_cache = _extract_feature_cache(all_paths, extractor)
    print(f"Feature extraction took {time.perf_counter() - t0:.1f}s.", flush=True)

    print("Computing the base-split mean over the complete train split...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    print(f"Base-split mean computed in {time.perf_counter() - t0:.1f}s.", flush=True)

    print("Selecting the effective manifold dimension via diagnostics...", flush=True)
    d_eff = _select_d_eff(feature_cache, support_paths, base_mean)
    print(f"d_eff = {d_eff}", flush=True)

    per_episode_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for shot in shots:
        h = SEARCHED_HYPERPARAMETERS[(args.backbone, shot)]
        config = _config_from_candidate(h, K_CURVATURE, d_eff)
        episodes = episodes_by_shot[shot]
        print(f"\nS={shot}: evaluating {len(episodes)} episodes with psi_{shot}*={h} ...", flush=True)

        confusable_lorentztim: List[float] = []
        confusable_tim: List[float] = []
        distinct_lorentztim: List[float] = []
        distinct_tim: List[float] = []

        t_shot = time.perf_counter()
        n_skipped = 0
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
            query_y_np = query_y.cpu().numpy()
            try:
                lorentztim_pred, tim_pred = _predict_two_methods(
                    support_feats, support_y, query_feats, base_mean, config
                )
            except Exception as exc:  # skip and report
                n_skipped += 1
                print(f"  WARNING: episode {ep_idx} skipped ({type(exc).__name__}: {exc})", flush=True)
                continue

            confusable_lorentztim.append(_group_f1(query_y_np, lorentztim_pred, confusable_idxs))
            confusable_tim.append(_group_f1(query_y_np, tim_pred, confusable_idxs))
            distinct_lorentztim.append(_group_f1(query_y_np, lorentztim_pred, distinct_idxs))
            distinct_tim.append(_group_f1(query_y_np, tim_pred, distinct_idxs))

            per_episode_rows.append({
                "shot": shot, "episode_index": ep_idx,
                "confusable_lorentztim_f1": confusable_lorentztim[-1],
                "confusable_tim_euclidean_f1": confusable_tim[-1],
                "confusable_delta": confusable_lorentztim[-1] - confusable_tim[-1],
                "grenade_lorentztim_f1": distinct_lorentztim[-1],
                "grenade_tim_euclidean_f1": distinct_tim[-1],
                "grenade_delta": distinct_lorentztim[-1] - distinct_tim[-1],
            })

            if (ep_idx + 1) % 100 == 0:
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({time.perf_counter() - t_shot:.1f}s)", flush=True)

        confusable_delta = np.array(confusable_lorentztim) - np.array(confusable_tim)
        grenade_delta = np.array(distinct_lorentztim) - np.array(distinct_tim)
        cross_group_delta = confusable_delta - grenade_delta

        def _wilcoxon_p(x: np.ndarray) -> float:
            if np.allclose(x, 0.0):
                return 1.0
            return float(wilcoxon(x).pvalue)

        for group_name, delta in (
            ("confusable (Aviation_Bomb/Mortar_Bomb/Projectile/RPG)", confusable_delta),
            ("Grenade", grenade_delta),
            ("confusable_minus_grenade (cross-group)", cross_group_delta),
        ):
            ci_lo, ci_hi = confidence_interval(delta)
            summary_rows.append({
                "shot": shot,
                "group": group_name,
                "n_episodes": len(delta),
                "mean_delta_macro_f1": float(np.mean(delta)),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "wilcoxon_p": _wilcoxon_p(delta),
                "cohens_dz": _cohens_dz(delta),
            })
            print(
                f"  {group_name}: mean_delta={np.mean(delta):+.4f} "
                f"[{ci_lo:+.4f}, {ci_hi:+.4f}]  d_z={_cohens_dz(delta):+.3f}  "
                f"p={summary_rows[-1]['wilcoxon_p']:.4g}",
                flush=True,
            )
        if n_skipped:
            print(f"  ({n_skipped} episode(s) skipped at S={shot})", flush=True)

    def _write_csv(name: str, rows: List[Dict]) -> None:
        path = args.output_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)", flush=True)

    _write_csv("class_group_per_episode.csv", per_episode_rows)
    _write_csv("class_group_summary.csv", summary_rows)

    md_lines = [
        "# Confusable group vs. Grenade\n",
        f"Classes: confusable={CONFUSABLE_CLASSES}, distinct={DISTINCT_CLASSES}. "
        f"d_eff={d_eff}, K={K_CURVATURE}.\n",
        "| Shot | Group | N | Mean Delta (LorentzTIM - TIM Euclidean) | 95% CI | Cohen's d_z | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['shot']} | {row['group']} | {row['n_episodes']} | "
            f"{row['mean_delta_macro_f1']:+.4f} | [{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}] | "
            f"{row['cohens_dz']:+.3f} | {row['wilcoxon_p']:.4g} |"
        )
    (args.output_dir / "class_group_analysis.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {args.output_dir / 'class_group_analysis.md'}", flush=True)


if __name__ == "__main__":
    main()

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from algorithm.rtim import (
    HTIMConfig,
    cl2n_condition,
    fit_pca_projection,
    hyperbolic_lift,
    htim_adapt,
)
from benchmark.baselines import _class_means, _euclidean_tim_predict
from benchmark.baselines_extra_euclidean import (
    _dsn_predict,
    _laplacianshot_predict,
    _pt_map_predict,
)
from benchmark.baselines_extra_hyperbolic import (
    _busemann_predict,
    _hyp_protonet_predict,
    _taylor_hnn_predict,
    _tim_hyperbolic_predict,
)
from benchmark.episode_generator import EpisodeSpec, FULL_SCALE_CONFIG, generate_episodes
from benchmark.hyperparam_search import (
    DEFAULT_N_TRIALS,
    DEFAULT_N_VAL_EPISODES,
    ShotSearchResult,
    _config_from_candidate,
    _episode_to_tensors,
    _extract_feature_cache,
    run_search_for_shot,
)
from benchmark.reporting import prediction_metric_rows, write_csv
from dataset.ctx_uxo import CLASS_NAMES, NUM_CLASSES, compute_base_split_mean

METHOD_ORDER: List[str] = [
    "SimpleShot",
    "LaplacianShot",
    "PT-MAP",
    "DSN",
    "Hyp-SimpleShot",
    "Hyp-ProtoNet",
    "Taylor-HNN",
    "Hyp-Busemann",
    "TIM (Euclidean)",
    "TIM (hyperbolic)",
    "LorentzTIM",
]
"""DSN, Hyp-ProtoNet and Taylor-HNN are metric-based and need no
meta-training; they run as closed-form inference on the same frozen
CL2N+PCA features as the other baselines, with the curvature and
temperature of the original papers (not searched). Hyp-Busemann has no
few-shot protocol of its own and is adapted as described in
benchmark/baselines_extra_hyperbolic.py."""

# psi_S* per (backbone, shot) from benchmark/hyperparam_search.py (25 trials,
# 30 validation episodes per trial). K and d_eff are not searched. Shots
# missing here get a fresh search. benchmark/search_and_update_psi.py rewrites
# the entries of one backbone.
SEARCHED_HYPERPARAMETERS: Dict[tuple[str, int], Dict[str, float]] = {
    # Values behind Table 3 (DINOv3, d_eff=29) and Tables 7/8 (CLIP, d_eff=10).
    ("dinov3", 1): {"T": 76, "beta": 0.010059014134369089, "ce_weight": 0.4609520092598397, "marginal_h_weight": 1.0010246003534802, "cond_h_weight": 0.5139214522219316, "omega": 0.006493169809239729, "tau": 3.4295803752902847},
    ("dinov3", 3): {"T": 98, "beta": 0.03294689531968659, "ce_weight": 0.06402149556615813, "marginal_h_weight": 0.6364350507330307, "cond_h_weight": 0.37504712833311904, "omega": 0.1206842374756634, "tau": 8.655179127804484},
    ("dinov3", 5): {"T": 64, "beta": 0.05443786139495558, "ce_weight": 0.3280204901963535, "marginal_h_weight": 0.5380287206668769, "cond_h_weight": 0.08084577423274028, "omega": 0.9652538329167627, "tau": 10.883898499101031},
    ("dinov3", 10): {"T": 30, "beta": 0.06779679278725265, "ce_weight": 0.4211141582346084, "marginal_h_weight": 0.8388573087966457, "cond_h_weight": 0.05018839354232237, "omega": 0.15963695214574775, "tau": 2.5655393794255548},
    ("clip", 1): {"T": 16, "beta": 0.04198174113437759, "ce_weight": 0.09577049659293976, "marginal_h_weight": 2.045180219283288, "cond_h_weight": 0.043811089566499546, "omega": 0.08092329401004936, "tau": 5.62368791481595},
    ("clip", 3): {"T": 96, "beta": 0.34631272182323486, "ce_weight": 0.15546539183177352, "marginal_h_weight": 0.8720596724745044, "cond_h_weight": 0.030357486900388593, "omega": 0.0541004855732655, "tau": 2.476710302071261},
    ("clip", 5): {"T": 64, "beta": 0.05443786139495558, "ce_weight": 0.3280204901963535, "marginal_h_weight": 0.5380287206668769, "cond_h_weight": 0.08084577423274028, "omega": 1.343868626596482, "tau": 15.153039780017858},
    ("clip", 10): {"T": 62, "beta": 0.0293263231465368, "ce_weight": 0.11441784582587886, "marginal_h_weight": 0.2421075299950291, "cond_h_weight": 0.029091179615315755, "omega": 0.08214462207939642, "tau": 16.11469734618602},
}
K_CURVATURE = -1.0
D_EFF_SUBSAMPLE_N = 3000
D_EFF_SUBSAMPLE_SEED = 42


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--output-dir", type=Path, default=Path("results_final"))
    p.add_argument(
        "--shots",
        type=str,
        default=None,
        help=(
            "Comma-separated shot counts to evaluate, e.g. '15,20,25,30,35,40'. "
            f"Omit to use the main benchmark grid {FULL_SCALE_CONFIG.shots}. Any "
            "shot not present in SEARCHED_HYPERPARAMETERS gets its own fresh "
            "random hyperparameter search (see --search-trials/--val-episodes)."
        ),
    )
    p.add_argument(
        "--force-search",
        action="store_true",
        help=(
            "Run the random search even for a shot that already has a hardcoded "
            "psi_S* in SEARCHED_HYPERPARAMETERS (e.g. to verify/refine it with a "
            "larger --search-trials budget). Without this flag, hardcoded shots "
            "always reuse the existing psi_S* and skip the search entirely."
        ),
    )
    p.add_argument(
        "--search-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Random search budget R for any shot requiring a fresh search. Default: {DEFAULT_N_TRIALS}.",
    )
    p.add_argument(
        "--val-episodes",
        type=int,
        default=DEFAULT_N_VAL_EPISODES,
        help=f"Validation episodes per search trial. Default: {DEFAULT_N_VAL_EPISODES}.",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov3",
        help="Backbone: 'dinov3', 'clip', or a HuggingFace model ID with d=768 embeddings. Default: 'dinov3'.",
    )
    p.add_argument(
        "--search-report",
        type=Path,
        default=None,
        help=(
            "Per-trial search report for the shots searched in this run. "
            "Default: <output-dir>/hyperparameter_search_extended.md; not "
            "written if no shot was searched."
        ),
    )
    return p.parse_args()


def _collect_paths(episode_lists: Iterable[List[EpisodeSpec]]) -> Tuple[Set[Path], Set[Path]]:
    """All image paths, and support-only paths, across every given episode."""
    all_paths: Set[Path] = set()
    support_paths: Set[Path] = set()
    for episodes in episode_lists:
        for ep in episodes:
            for entry in ep.classes:
                all_paths.update(entry.support_paths)
                all_paths.update(entry.query_paths)
                support_paths.update(entry.support_paths)
    return all_paths, support_paths


def _select_d_eff(
    feature_cache: Dict[Path, Tensor], support_paths: Set[Path], base_mean: Tensor
) -> int:
    """d_eff = round(participation ratio) of the CL2N-conditioned features of a
    seeded random subsample (``D_EFF_SUBSAMPLE_N``) of the given support paths.
    The subsample keeps the O(n^2) diagnostics tractable.
    """
    from algorithm.diagnostics import DiagnosticsConfig, run_diagnostics

    ordered = sorted(support_paths)
    rng = np.random.default_rng(D_EFF_SUBSAMPLE_SEED)
    if len(ordered) > D_EFF_SUBSAMPLE_N:
        idx = rng.choice(len(ordered), size=D_EFF_SUBSAMPLE_N, replace=False)
        sampled = [ordered[i] for i in idx]
    else:
        sampled = ordered
    raw = torch.stack([feature_cache[p] for p in sampled], dim=0)
    z = cl2n_condition(raw, base_mean).cpu().numpy().astype(np.float64)
    diag = run_diagnostics(z, config=DiagnosticsConfig(delta_n_batches=8))
    return int(diag["Z"]["d_eff"])


def _predict_all_methods(
    support_raw: Tensor,
    support_y: Tensor,
    query_raw: Tensor,
    base_mean: Tensor,
    config: HTIMConfig,
) -> Dict[str, np.ndarray]:
    """Query predictions of every method on one episode, using the same
    conditioning, PCA and lift as the per-method runners.
    """
    num_classes = int(support_y.max().item()) + 1
    preds: Dict[str, np.ndarray] = {}

    z_support = cl2n_condition(support_raw, base_mean)
    z_query = cl2n_condition(query_raw, base_mean)
    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, config.d_eff)
    p_support = z_support @ u_pr
    p_query = z_query @ u_pr

    proto = _class_means(p_support, support_y, num_classes)
    preds["SimpleShot"] = torch.cdist(p_query, proto).argmin(dim=-1).cpu().numpy()

    preds["LaplacianShot"] = _laplacianshot_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()
    preds["PT-MAP"] = _pt_map_predict(p_support, support_y, p_query, num_classes).cpu().numpy()
    preds["DSN"] = _dsn_predict(p_support, support_y, p_query, num_classes).cpu().numpy()

    preds["TIM (Euclidean)"] = _euclidean_tim_predict(
        p_support, support_y, p_query, num_classes, config
    ).cpu().numpy()

    support_h = hyperbolic_lift(z_support, u_pr, config.K)
    query_h = hyperbolic_lift(z_query, u_pr, config.K)
    hyp_simpleshot_cfg = dataclasses.replace(config, T=0)
    preds["Hyp-SimpleShot"] = htim_adapt(
        support_h, support_y, query_h, hyp_simpleshot_cfg
    ).predictions.cpu().numpy()
    preds["LorentzTIM"] = htim_adapt(
        support_h, support_y, query_h, config
    ).predictions.cpu().numpy()

    preds["Hyp-ProtoNet"] = _hyp_protonet_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()
    preds["Taylor-HNN"] = _taylor_hnn_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()
    preds["Hyp-Busemann"] = _busemann_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()

    preds["TIM (hyperbolic)"] = _tim_hyperbolic_predict(
        support_raw, support_y, query_raw, config
    ).cpu().numpy()

    return preds


def format_searched_entry(backbone: str, shot: int, candidate: Dict[str, float]) -> str:
    """One ``SEARCHED_HYPERPARAMETERS`` line, in the dict's source format."""
    return (
        f'    ("{backbone}", {shot}): {{"T": {int(candidate["T"])}, "beta": {candidate["beta"]!r}, '
        f'"ce_weight": {candidate["ce_weight"]!r}, "marginal_h_weight": {candidate["marginal_h_weight"]!r}, '
        f'"cond_h_weight": {candidate["cond_h_weight"]!r}, "omega": {candidate["omega"]!r}, '
        f'"tau": {candidate["tau"]!r}}},'
    )


def _build_search_report(shot_search_results: Dict[int, ShotSearchResult], backbone: str) -> str:
    """Per-trial report for the shots searched in this run."""
    lines: List[str] = [
        "# Hyperparameter search: psi_S* = (T, beta, ce_weight, "
        "marginal_h_weight, cond_h_weight, omega, tau)\n",
        "Random search on validation-split episodes, for shot counts not in "
        "SEARCHED_HYPERPARAMETERS (or all shots with --force-search).\n",
    ]
    for shot, sr in shot_search_results.items():
        lines.append(f"\n## S={shot}\n")
        lines.append(f"- measured geodesic-distance scale d_bar = {sr.d_bar:.6f}")
        lines.append(f"- search trial budget R = {len(sr.trials)}")
        lines.append(f"- validation episodes per trial = {sr.n_val_episodes_used}")
        lines.append(f"- search wall-clock time: {sr.wall_clock_s:.1f} s")
        lines.append(
            f"\n**Best psi_{shot}\\* = {sr.best_candidate}** "
            f"(validation Macro-F1 = {sr.best_val_macro_f1:.4f})\n"
        )
        lines.append("| Trial | T | beta | ce_weight | marginal_h_weight | cond_h_weight | omega | tau | val Macro-F1 |")
        lines.append("| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |")
        for t in sr.trials:
            c = t.candidate
            lines.append(
                f"| {t.trial_index} | {c['T']} | {c['beta']:.5f} | {c['ce_weight']:.5f} | "
                f"{c['marginal_h_weight']:.5f} | {c['cond_h_weight']:.5f} | "
                f"{c['omega']:.5f} | {c['tau']:.5f} | {t.mean_val_macro_f1:.4f} |"
            )
        lines.append(
            f"\nSEARCHED_HYPERPARAMETERS entry for S={shot}:\n"
            f"```python\n{format_searched_entry(backbone, shot, sr.best_candidate).strip()}\n```\n"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    set_deterministic(FULL_SCALE_CONFIG.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shots is not None:
        shots = tuple(int(s.strip()) for s in args.shots.split(",") if s.strip())
        if not shots:
            raise ValueError(f"--shots produced an empty shot list from {args.shots!r}")
    else:
        shots = FULL_SCALE_CONFIG.shots
    n_query = FULL_SCALE_CONFIG.n_query
    seed = FULL_SCALE_CONFIG.seed

    print(f"Generating {args.num_episodes} paired episodes per shot for S in {shots}...", flush=True)
    episodes_by_shot = {
        shot: generate_episodes(
            split="test", n_shot=shot, n_query=n_query, num_episodes=args.num_episodes, seed=seed
        )
        for shot in shots
    }
    all_paths, support_paths = _collect_paths(episodes_by_shot.values())

    print(f"Loading feature extractor '{args.backbone}' on {args.device}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print(f"Extracting features for {len(all_paths)} distinct images...", flush=True)
    t0 = time.perf_counter()
    feature_cache = _extract_feature_cache(all_paths, extractor)
    dt = time.perf_counter() - t0
    print(f"Feature extraction took {dt:.1f}s.", flush=True)

    print("Computing the base-split mean over the complete train split...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    dt = time.perf_counter() - t0
    print(f"Base-split mean computed in {dt:.1f}s.", flush=True)

    print("Selecting the effective manifold dimension via diagnostics...", flush=True)
    d_eff = _select_d_eff(feature_cache, support_paths, base_mean)
    print(f"d_eff = {d_eff}", flush=True)

    shot_search_results: Dict[int, ShotSearchResult] = {}
    configs_by_shot: Dict[int, HTIMConfig] = {}
    for shot in shots:
        hp_key = (args.backbone, shot)
        needs_search = args.force_search or hp_key not in SEARCHED_HYPERPARAMETERS
        if not needs_search:
            h = SEARCHED_HYPERPARAMETERS[hp_key]
            configs_by_shot[shot] = _config_from_candidate(h, K_CURVATURE, d_eff)
            print(f"S={shot}: using hardcoded psi_{shot}* = {h}", flush=True)
            continue
        print(
            f"S={shot}: no hardcoded psi_{shot}* (or --force-search set) -- running "
            f"a fresh random search (R={args.search_trials} trials, "
            f"{args.val_episodes} validation episodes/trial)...",
            flush=True,
        )
        t_search = time.perf_counter()
        search_result = run_search_for_shot(
            shot=shot,
            K=K_CURVATURE,
            d_eff=d_eff,
            base_mean=base_mean,
            extractor=extractor,
            n_trials=args.search_trials,
            n_val_episodes=args.val_episodes,
            n_query=n_query,
            seed=seed,
            backbone=args.backbone,
        )
        shot_search_results[shot] = search_result
        configs_by_shot[shot] = search_result.best_config
        print(
            f"S={shot}: best psi_{shot}* = {search_result.best_candidate} "
            f"(val Macro-F1={search_result.best_val_macro_f1:.4f}, "
            f"{time.perf_counter() - t_search:.1f}s)",
            flush=True,
        )

    if shot_search_results:
        report_path = args.search_report or (args.output_dir / "hyperparameter_search_extended.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_build_search_report(shot_search_results, args.backbone))
        print(f"Wrote {report_path}", flush=True)

    per_class_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []

    for shot in shots:
        config = configs_by_shot[shot]
        episodes = episodes_by_shot[shot]
        print(f"\nS={shot}: evaluating {len(episodes)} episodes with {config} ...", flush=True)

        pooled_true: Dict[str, List[int]] = {m: [] for m in METHOD_ORDER}
        pooled_pred: Dict[str, List[int]] = {m: [] for m in METHOD_ORDER}

        t_shot = time.perf_counter()
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(
                episode, feature_cache
            )
            query_y_np = query_y.cpu().numpy()
            try:
                preds = _predict_all_methods(
                    support_feats, support_y, query_feats, base_mean, config
                )
            except Exception as exc:  # skip and report
                print(f"  WARNING: episode {ep_idx} skipped ({type(exc).__name__}: {exc})", flush=True)
                continue
            for method, y_pred in preds.items():
                pooled_true[method].extend(query_y_np.tolist())
                pooled_pred[method].extend(y_pred.tolist())
                ep_f1 = precision_recall_fscore_support(
                    query_y_np, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0
                )[2]
                per_episode_rows.append(
                    {"shot": shot, "method": method, "episode_index": ep_idx, "macro_f1": ep_f1}
                )
            if (ep_idx + 1) % 25 == 0:
                dt_shot = time.perf_counter() - t_shot
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({dt_shot:.1f}s)", flush=True)

        for method in METHOD_ORDER:
            per_class, agg, confusion = prediction_metric_rows(
                shot, method, np.array(pooled_true[method]), np.array(pooled_pred[method]), CLASS_NAMES
            )
            per_class_rows.extend(per_class)
            aggregate_rows.append(agg)
            confusion_rows.extend(confusion)
            print(
                f"  {method:20s} macro_f1={agg['macro_f1']:.4f} micro_f1={agg['micro_f1']:.4f} "
                f"weighted_f1={agg['weighted_f1']:.4f} accuracy={agg['accuracy']:.4f}",
                flush=True,
            )

    write_csv(args.output_dir, "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir, "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir, "confusion_matrix.csv", confusion_rows)
    write_csv(args.output_dir, "per_episode_scores.csv", per_episode_rows)


if __name__ == "__main__":
    main()

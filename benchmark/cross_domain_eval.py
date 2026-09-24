import argparse
import csv
import dataclasses
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.diagnostics import DiagnosticsConfig, run_diagnostics
from algorithm.features import get_feature_extractor
from algorithm.rtim import (
    HTIMConfig,
    cl2n_condition,
    fit_pca_projection,
    hyperbolic_lift,
    htim_adapt,
)
from benchmark.baselines import _class_means, _euclidean_tim_predict
from benchmark.baselines_extra_euclidean import _dsn_predict
from benchmark.baselines_extra_hyperbolic import (
    _busemann_predict,
    _hyp_protonet_predict,
    _taylor_hnn_predict,
)
from benchmark.evaluator import macro_f1
from dataset.fewshot_common import (
    ClassPool,
    Episode,
    compute_base_mean_generic,
    load_images_parallel,
    sample_episode_generic,
)
from dataset.registry import (
    BASE_MEAN_SPLIT,
    DEFAULT_BENCHMARK_QUERY_SPLIT,
    DEFAULT_SUPPORT_SPLIT,
    DEFAULT_VAL_QUERY_SPLIT,
    get_dataset_index,
)
from properties.search_space import (
    SearchSpace,
    diagnostics_informed_search_space,
    measure_geodesic_distance_scale,
)

METHOD_ORDER: List[str] = [
    "SimpleShot",
    "DSN",
    "Hyp-ProtoNet",
    "Taylor-HNN",
    "Hyp-Busemann",
    "TIM (Euclidean)",
    "LorentzTIM",
]

D_EFF_SUBSAMPLE_N: int = 3000
D_EFF_SUBSAMPLE_SEED: int = 42
BASE_MEAN_BATCH_SIZE: int = 32
K_CURVATURE: float = -1.0
"""Fixed project-wide curvature."""
N_WAY: int = 5
WAY_SLOT_LABELS = tuple(f"way_slot_{i}" for i in range(N_WAY))
"""Labels are positions within each episode's sampled class subset, not
fixed global class identities."""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset",
        choices=["cub200", "fgvc_aircraft", "tiered_imagenet", "ctx_uxo"],
        required=True,
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov3",
        help="'dinov3' (primary) or 'clip' (Appendix cross-backbone check). Default: 'dinov3'.",
    )
    p.add_argument("--num-episodes", type=int, default=1000, help="M paired benchmark episodes per shot.")
    p.add_argument("--shots", type=str, default="1,3,5,10", help="Comma-separated shot counts S.")
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--search-trials", type=int, default=25, help="Random-search budget R per shot.")
    p.add_argument("--val-episodes", type=int, default=30, help="Validation episodes per search trial.")
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on eligible per-class pool size when sampling episodes "
        "(None = use the full split pool for every class).",
    )
    p.add_argument(
        "--base-mean-max-per-class",
        type=int,
        default=None,
        help="Optional per-class cap on the base/train pool used for the CL2N "
        "centering statistic (None = full base split). Recommended for "
        "tiered_imagenet, whose 'train' split has ~1300 images/class across 351 "
        "classes (448K images total); document the value used in the run's "
        "methods/reproducibility notes.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


LOAD_CHUNK_SIZE: int = 512
"""Raw-image decode chunk size for :func:`_extract_feature_cache`. Native-
resolution images (e.g. FGVC-Aircraft's ~971x740 originals, ~8.5 MB/image as
a raw float32 CHW tensor) can, at high shots with many benchmark episodes,
touch a large fraction of the whole dataset; decoding all of them into RAM
at once (as raw tensors, before any extraction) can exceed available memory.
Chunking the decode+extract loop bounds peak RAM to O(LOAD_CHUNK_SIZE) raw
images regardless of dataset size or resolution, with byte-identical output
(frozen backbones are per-sample deterministic, not batch-order-dependent)."""


def _extract_feature_cache(paths: Set[Path], extractor, batch_size: int = 32) -> Dict[Path, Tensor]:
    ordered = sorted(paths)
    feats_chunks: List[Tensor] = []
    for start in range(0, len(ordered), LOAD_CHUNK_SIZE):
        chunk_paths = ordered[start : start + LOAD_CHUNK_SIZE]
        image_map = load_images_parallel(chunk_paths)  # parallel I/O, same tensors
        images = [image_map[p] for p in chunk_paths]
        feats_chunks.append(extractor(images, batch_size=batch_size))
        del image_map, images
    feats = torch.cat(feats_chunks, dim=0) if feats_chunks else torch.empty((0,))
    return {p: feats[i] for i, p in enumerate(ordered)}


def _episode_to_tensors(
    episode: Episode, feature_cache: Dict[Path, Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    support_feats = torch.stack([feature_cache[p] for p in episode.support_paths], dim=0)
    query_feats = torch.stack([feature_cache[p] for p in episode.query_paths], dim=0)
    support_y = torch.tensor(episode.support_labels, dtype=torch.long, device=support_feats.device)
    query_y = torch.tensor(episode.query_labels, dtype=torch.long, device=query_feats.device)
    return support_feats, support_y, query_feats, query_y


def _episode_paths(episodes: List[Episode]) -> Set[Path]:
    paths: Set[Path] = set()
    for ep in episodes:
        paths.update(ep.support_paths)
        paths.update(ep.query_paths)
    return paths


def _capped_pool(pool: ClassPool, cap: Optional[int], seed: int) -> ClassPool:
    """Seeded per-class subsample of at most ``cap`` paths (``None`` = no cap)."""
    if cap is None:
        return pool
    rng = random.Random(seed)
    return {
        cls: (paths if len(paths) <= cap else sorted(rng.sample(paths, k=cap)))
        for cls, paths in pool.items()
    }


def _config_from_candidate(candidate: Dict[str, float], K: float, d_eff: int) -> HTIMConfig:
    return HTIMConfig(
        K=K, d_eff=d_eff, T=int(candidate["T"]), beta=float(candidate["beta"]),
        omega=float(candidate["omega"]), tau=float(candidate["tau"]),
        use_query_in_mi=False,
        ce_weight=float(candidate["ce_weight"]),
        marginal_h_weight=float(candidate["marginal_h_weight"]),
        cond_h_weight=float(candidate["cond_h_weight"]),
    )


def _sample_episodes(
    n: int,
    n_way: int,
    n_shot: int,
    n_query: int,
    support_pool: ClassPool,
    query_pool: ClassPool,
    seed: int,
) -> List[Episode]:
    rng = random.Random(seed)
    return [
        sample_episode_generic(
            n_way=n_way, n_shot=n_shot, n_query=n_query,
            support_pool=support_pool, query_pool=query_pool, rng=rng,
        )
        for _ in range(n)
    ]


def _select_d_eff(base_paths: List[Path], feature_cache: Dict[Path, Tensor], base_mean: Tensor) -> int:
    rng = np.random.default_rng(D_EFF_SUBSAMPLE_SEED)
    ordered = sorted(base_paths)
    if len(ordered) > D_EFF_SUBSAMPLE_N:
        idx = rng.choice(len(ordered), size=D_EFF_SUBSAMPLE_N, replace=False)
        sampled = [ordered[i] for i in idx]
    else:
        sampled = ordered
    raw = torch.stack([feature_cache[p] for p in sampled], dim=0)
    z = cl2n_condition(raw, base_mean).cpu().numpy().astype(np.float64)
    diag = run_diagnostics(z, config=DiagnosticsConfig(delta_n_batches=8))
    return int(diag["Z"]["d_eff"])


def _predict_methods(
    support_raw: Tensor, support_y: Tensor, query_raw: Tensor, base_mean: Tensor, config: HTIMConfig,
) -> Dict[str, np.ndarray]:
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

    preds["DSN"] = _dsn_predict(p_support, support_y, p_query, num_classes).cpu().numpy()
    preds["Hyp-ProtoNet"] = _hyp_protonet_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()
    preds["Taylor-HNN"] = _taylor_hnn_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()
    preds["Hyp-Busemann"] = _busemann_predict(
        p_support, support_y, p_query, num_classes
    ).cpu().numpy()

    preds["TIM (Euclidean)"] = _euclidean_tim_predict(
        p_support, support_y, p_query, num_classes, config
    ).cpu().numpy()

    support_h = hyperbolic_lift(z_support, u_pr, config.K)
    query_h = hyperbolic_lift(z_query, u_pr, config.K)
    preds["LorentzTIM"] = htim_adapt(support_h, support_y, query_h, config).predictions.cpu().numpy()

    return preds


@dataclasses.dataclass
class SearchTrialResult:
    trial_index: int
    candidate: Dict[str, float]
    mean_val_macro_f1: float


@dataclasses.dataclass
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


def _evaluate_candidate(
    candidate: Dict[str, float], K: float, d_eff: int,
    episodes: List[Episode], feature_cache: Dict[Path, Tensor], base_mean: Tensor,
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
        n_way = int(support_y.max().item()) + 1
        scores.append(macro_f1(query_y.cpu().numpy(), result.predictions.cpu().numpy(), n_way))
    return float(np.mean(scores))


def run_search_for_shot(
    shot: int, K: float, d_eff: int, base_mean: Tensor, extractor,
    support_pool: ClassPool, query_pool: ClassPool, n_way: int = N_WAY,
    n_trials: int = 25, n_val_episodes: int = 30, n_query: int = 15, seed: int = 42,
    tag: str = "",
) -> ShotSearchResult:
    """Random search for psi_S* on this dataset's validation episodes.

    ``tag`` (e.g. "cub200:dinov3") salts the search RNG so that different
    dataset/backbone combinations draw different candidates for the same seed.
    """
    t0 = time.perf_counter()
    val_episodes = _sample_episodes(
        n_val_episodes, n_way, shot, n_query, support_pool, query_pool, seed=seed + 5_000 + shot
    )
    feature_cache = _extract_feature_cache(_episode_paths(val_episodes), extractor)

    sample_ep = val_episodes[0]
    s_feats, _, q_feats, _ = _episode_to_tensors(sample_ep, feature_cache)
    z_sample = cl2n_condition(torch.cat([s_feats, q_feats], dim=0), base_mean)
    u_pr_sample = fit_pca_projection(z_sample, d_eff)
    lifted_sample = hyperbolic_lift(z_sample, u_pr_sample, K)
    d_bar = measure_geodesic_distance_scale(lifted_sample, K, seed=seed)
    search_space = diagnostics_informed_search_space(d_bar)

    tag_salt = zlib.crc32(tag.encode("utf-8")) if tag else 0
    rng = np.random.default_rng(seed + 10_000 + shot + tag_salt)
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

    assert best_candidate is not None
    return ShotSearchResult(
        shot=shot, best_config=_config_from_candidate(best_candidate, K, d_eff), best_candidate=best_candidate,
        best_val_macro_f1=best_score, search_space=search_space, d_bar=d_bar,
        trials=trials, n_val_episodes_used=len(val_episodes), wall_clock_s=time.perf_counter() - t0,
    )


def main() -> None:
    args = _parse_args()
    set_deterministic(args.seed)
    shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}   Backbone: {args.backbone}   Device: {args.device}", flush=True)
    index = get_dataset_index(args.dataset)
    support_split = DEFAULT_SUPPORT_SPLIT[args.dataset]
    bench_query_split = DEFAULT_BENCHMARK_QUERY_SPLIT[args.dataset]
    val_query_split = DEFAULT_VAL_QUERY_SPLIT[args.dataset]
    base_split = BASE_MEAN_SPLIT[args.dataset]
    print(
        f"Splits -> support:{support_split}  benchmark-query:{bench_query_split}  "
        f"val-query:{val_query_split}  base-mean:{base_split}  "
        f"shared_classes_across_splits={index.shared_classes_across_splits}",
        flush=True,
    )

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
    del base_feature_cache  # only needed for the d_eff subsample above

    # Each per-episode PCA fit has only N_WAY * (shot + n_query) points, which
    # can be fewer than the diagnostic d_eff for small per-class pools (e.g.
    # CUB-200's ~6-image valid split). Cap d_eff at the smallest rank any
    # episode in this run can support.
    min_support = min(len(v) for v in support_pool.values())
    min_val_query = min(len(v) for v in val_query_pool.values())
    min_bench_query = min(len(v) for v in bench_query_pool.values())
    worst_shot = min(shots)
    max_feasible_rank = N_WAY * (
        min(worst_shot, min_support) + min(args.n_query, min_val_query, min_bench_query)
    )
    d_eff = min(raw_d_eff, max_feasible_rank)
    if d_eff < raw_d_eff:
        print(
            f"d_eff={raw_d_eff} exceeds the maximum PCA rank {max_feasible_rank} "
            f"at shot={worst_shot} (n_way={N_WAY}, min support={min_support}, "
            f"min query={min(min_val_query, min_bench_query)}); using d_eff={d_eff}.",
            flush=True,
        )
    print(f"Diagnostic-calibrated d_eff = {d_eff} (K fixed at {K_CURVATURE}).", flush=True)

    per_class_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    per_episode_rows: List[Dict] = []
    search_reports: List[str] = []

    for shot in shots:
        print(f"\nS={shot}", flush=True)
        print(f"Running psi_{shot}* random search ({args.search_trials} trials, "
              f"{args.val_episodes} val episodes/trial)...", flush=True)
        search_result = run_search_for_shot(
            shot=shot, K=K_CURVATURE, d_eff=d_eff, base_mean=base_mean, extractor=extractor,
            support_pool=support_pool, query_pool=val_query_pool,
            n_trials=args.search_trials, n_val_episodes=args.val_episodes,
            n_query=args.n_query, seed=args.seed,
            tag=f"{args.dataset}:{args.backbone}",
        )
        config = search_result.best_config
        print(
            f"psi_{shot}* = T={config.T} beta={config.beta:.4f} "
            f"ce_weight={config.ce_weight:.4f} marginal_h_weight={config.marginal_h_weight:.4f} "
            f"cond_h_weight={config.cond_h_weight:.4f} "
            f"omega={config.omega:.4f} tau={config.tau:.4f} "
            f"(val Macro-F1={search_result.best_val_macro_f1:.4f}, "
            f"{search_result.wall_clock_s:.1f}s)",
            flush=True,
        )
        search_reports.append(
            f"## S={shot}\n\n"
            f"- d_bar (median geodesic dist^2): {search_result.d_bar:.6f}\n"
            f"- best psi_{shot}* = {search_result.best_candidate}\n"
            f"- best validation Macro-F1: {search_result.best_val_macro_f1:.4f}\n"
            f"- trials: {args.search_trials}, val episodes/trial: {args.val_episodes}, "
            f"wall-clock: {search_result.wall_clock_s:.1f}s\n"
        )

        print(f"Sampling {args.num_episodes} paired benchmark episodes...", flush=True)
        episodes = _sample_episodes(
            args.num_episodes, n_way=N_WAY, n_shot=shot, n_query=args.n_query,
            support_pool=support_pool, query_pool=bench_query_pool, seed=args.seed + shot,
        )
        feature_cache = _extract_feature_cache(_episode_paths(episodes), extractor)

        pooled_true: Dict[str, List[int]] = {m: [] for m in METHOD_ORDER}
        pooled_pred: Dict[str, List[int]] = {m: [] for m in METHOD_ORDER}
        t_shot = time.perf_counter()
        for ep_idx, episode in enumerate(episodes):
            support_feats, support_y, query_feats, query_y = _episode_to_tensors(episode, feature_cache)
            query_y_np = query_y.cpu().numpy()
            preds = _predict_methods(support_feats, support_y, query_feats, base_mean, config)
            for method, y_pred in preds.items():
                pooled_true[method].extend(query_y_np.tolist())
                pooled_pred[method].extend(y_pred.tolist())
                ep_f1 = precision_recall_fscore_support(
                    query_y_np, y_pred, labels=range(N_WAY), average="macro", zero_division=0
                )[2]
                per_episode_rows.append(
                    {"shot": shot, "method": method, "episode_index": ep_idx, "macro_f1": ep_f1}
                )
            if (ep_idx + 1) % 100 == 0:
                dt = time.perf_counter() - t_shot
                print(f"  {ep_idx + 1}/{len(episodes)} episodes ({dt:.1f}s elapsed)", flush=True)

        for method in METHOD_ORDER:
            y_true = np.asarray(pooled_true[method])
            y_pred = np.asarray(pooled_pred[method])
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=range(N_WAY), average=None, zero_division=0
            )
            for class_idx, slot in enumerate(WAY_SLOT_LABELS):
                per_class_rows.append(
                    {
                        "shot": shot, "method": method, "class": slot,
                        "precision": precision[class_idx], "recall": recall[class_idx],
                        "f1": f1[class_idx], "support": int(support[class_idx]),
                    }
                )
            macro = precision_recall_fscore_support(
                y_true, y_pred, labels=range(N_WAY), average="macro", zero_division=0
            )[2]
            micro = precision_recall_fscore_support(
                y_true, y_pred, labels=range(N_WAY), average="micro", zero_division=0
            )[2]
            weighted = precision_recall_fscore_support(
                y_true, y_pred, labels=range(N_WAY), average="weighted", zero_division=0
            )[2]
            accuracy = float((y_true == y_pred).mean())
            aggregate_rows.append(
                {
                    "shot": shot, "method": method, "macro_f1": macro, "micro_f1": micro,
                    "weighted_f1": weighted, "accuracy": accuracy, "n_query_total": len(y_true),
                }
            )
            print(f"  {method:16s} macro_f1={macro:.4f} micro_f1={micro:.4f}", flush=True)

            cm = confusion_matrix(y_true, y_pred, labels=range(N_WAY))
            for i in range(N_WAY):
                for j in range(N_WAY):
                    confusion_rows.append(
                        {"shot": shot, "method": method, "true_way_slot": i, "pred_way_slot": j, "count": int(cm[i, j])}
                    )

    def _write_csv(name: str, rows: List[Dict]) -> None:
        if not rows:
            return
        path = args.output_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}", flush=True)

    # Class columns are way slots (0..4) within each episode's sampled
    # classes, not global class identities.
    _write_csv("per_class_metrics.csv", per_class_rows)
    _write_csv("aggregate_metrics.csv", aggregate_rows)
    _write_csv("confusion_matrix.csv", confusion_rows)
    _write_csv("per_episode_scores.csv", per_episode_rows)

    report_path = args.output_dir / "hyperparameter_search_report.md"
    report_path.write_text(
        f"# Hyperparameter search -- dataset={args.dataset} backbone={args.backbone}\n\n"
        f"d_eff={d_eff}, K={K_CURVATURE}\n\n" + "\n".join(search_reports)
    )
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()

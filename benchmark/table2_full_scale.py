#!/usr/bin/env python3
"""Table 2 (pre-transduction geometric diagnostics) on the full CTX-UXO train
split.

Runs ``algorithm.diagnostics.run_diagnostics`` with the spectrum-matched null
on a seeded global random subsample (``--max-points``, default 3000, to keep
the O(n^2) diagnostics tractable) of CL2N-conditioned features, using the
full base-split mean.

Usage:
    python3 benchmark/table2_full_scale.py --device cuda --max-points 3000 \\
        --output-dir results
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.diagnostics import (
    DiagnosticsConfig,
    generate_null_control_full_spectrum,
    run_diagnostics,
)
from algorithm.rtim import cl2n_condition
from dataset.ctx_uxo import (
    CLASS_NAMES,
    DEFAULT_INSTANCES_ROOT,
    compute_base_split_mean,
    list_class_instances,
    load_images_parallel,
)

DEFAULT_MAX_POINTS: int = 3000
"""Matches export_full_results.py's D_EFF_SUBSAMPLE_N -- the memory-safe
cap used throughout this project for O(n^2) diagnostics on full-scale data."""

RANDOM_SEED: int = 42
K_CURVATURE: float = -1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov3",
        help="Feature extractor backbone: 'dinov3' (paper's primary backbone), 'clip' (Appendix cross-backbone check), or HuggingFace model ID (producing d=768 embeddings). Default: 'dinov3'.",
    )
    p.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def _collect_full_pool() -> Tuple[List[Path], List[int]]:
    """Every train-split image path across all NUM_CLASSES classes,
    with its integer class index -- the full eligible pool, no per-class
    truncation (unlike benchmark/tables.py's ``collect_class_samples``)."""
    paths: List[Path] = []
    labels: List[int] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_paths = list_class_instances("train", class_name, root=DEFAULT_INSTANCES_ROOT)
        if not class_paths:
            raise RuntimeError(f"No train-split instances found for class '{class_name}'.")
        paths.extend(class_paths)
        labels.extend([class_idx] * len(class_paths))
    return paths, labels


def _random_subsample(
    paths: List[Path], labels: List[int], max_points: int, seed: int
) -> Tuple[List[Path], List[int]]:
    """Seeded random subsample over the pooled classes."""
    n = len(paths)
    if n <= max_points:
        return paths, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    return [paths[i] for i in idx], [labels[i] for i in idx]


def _fmt_ci(value: float, ci95: Tuple[float, float]) -> str:
    return f"{value:.4f} [{ci95[0]:.4f}, {ci95[1]:.4f}]"


def _build_markdown(
    diagnostics: Dict[str, object],
    n_samples: int,
    per_class_counts: Dict[str, int],
    extraction_time_s: float,
    base_mean_time_s: float,
    full_pool_size: int,
    backbone_label: str,
) -> str:
    z_res = diagnostics["Z"]
    zt_res = diagnostics["Z_tilde"]
    gaps = diagnostics["gaps"]
    delta_z, delta_zt = z_res["delta_rel"], zt_res["delta_rel"]
    coph_z, coph_zt = z_res["rho_coph"], zt_res["rho_coph"]
    counts_line = ", ".join(f"{k}={v}" for k, v in per_class_counts.items())

    lines: List[str] = []
    lines.append("# Table 2 - Pre-Transduction Geometric Diagnostics")
    lines.append("")
    lines.append(
        f"`algorithm.diagnostics.run_diagnostics` on n={n_samples} "
        f"{backbone_label} CL2N-conditioned embeddings sampled uniformly (seeded) "
        f"from the {full_pool_size}-image train split ({counts_line}). CL2N uses "
        "the base-split mean over the full train split."
    )
    lines.append("")
    lines.append(f"- Ambient dimension: {int(diagnostics['ambient_dim'])} (frozen {backbone_label}).")
    lines.append(
        f"- Base-split mean time: {base_mean_time_s:.2f} s."
    )
    lines.append(f"- Feature extraction time: {extraction_time_s:.2f} s.")
    lines.append(
        "- Null control $\\tilde{Z}$: Gaussian with the same covariance spectrum "
        "(`algorithm.diagnostics.generate_null_control_full_spectrum`)."
    )
    lines.append(f"- Random seed: {diagnostics['config'].random_seed}")
    lines.append("")
    lines.append(f"| Diagnostic | Equation | Z ({backbone_label}) | $\\tilde{{Z}}$ (Null Control) | Target / Reference |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    lines.append(
        "| Global tree-likeness ($\\hat{\\delta}_{\\text{rel}}$) | Eq. (4) | "
        f"{_fmt_ci(delta_z['value'], delta_z['ci95'])} | "
        f"{_fmt_ci(delta_zt['value'], delta_zt['ci95'])} | "
        f"Gap $\\Delta\\hat{{\\delta}}_{{\\text{{rel}}}}$ = {gaps['delta_delta_rel']:+.4f} "
        f"{'< 0 (tree-like)' if gaps['delta_delta_rel'] < 0 else '>= 0'} |"
    )
    lines.append(
        "| Intrinsic Dimension ($d_{\\text{TwoNN}}$) | Eq. (5) | "
        f"{z_res['d_TwoNN']:.2f} | {zt_res['d_TwoNN']:.2f} | - |"
    )
    lines.append(
        "| Intrinsic Dimension ($d_{\\text{MLE}}$) | Eq. (6) | "
        f"{z_res['d_MLE']:.2f} | {zt_res['d_MLE']:.2f} | - |"
    )
    lines.append(
        "| Intrinsic Dimension ($d_{\\text{PR}}$) | Eq. (7) | "
        f"{z_res['d_PR']:.2f} | {zt_res['d_PR']:.2f} | "
        f"$d_{{\\text{{eff}}}}$ = round($d_{{\\text{{PR}}}}$) = {z_res['d_eff']} |"
    )
    lines.append(
        "| Cophenetic Alignment ($\\rho_{\\text{coph}}$) | Eq. (10) | "
        f"{coph_z['pearson']:.4f} (Pearson) / {coph_z['spearman']:.4f} (Spearman) | "
        f"{coph_zt['pearson']:.4f} (Pearson) / {coph_zt['spearman']:.4f} (Spearman) | "
        "$\\rho_{\\text{coph}} \\in [-1, 1]$ (High agreement) |"
    )
    lines.append("")
    lines.append(
        f"$d_{{\\text{{eff}}}}={z_res['d_eff']}$ here is measured on this Table 2 "
        f"diagnostic sample specifically (n={n_samples}); the main benchmark's own "
        "d_eff (computed independently on its own random subsample inside "
        "export_full_results.py) may differ slightly by sampling "
        "variance."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    set_deterministic(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting the full train-split pool across all classes...", flush=True)
    all_paths, all_labels = _collect_full_pool()
    full_pool_size = len(all_paths)
    print(f"Full pool: {full_pool_size} images across {len(CLASS_NAMES)} classes.", flush=True)

    sample_paths, sample_labels = _random_subsample(all_paths, all_labels, args.max_points, args.seed)
    per_class_counts = {name: sample_labels.count(i) for i, name in enumerate(CLASS_NAMES)}
    print(
        f"Diagnostic sample: n={len(sample_paths)} (seed={args.seed}), "
        f"counts={per_class_counts}",
        flush=True,
    )

    print(f"Loading feature extractor '{args.backbone}' on {args.device!r}...", flush=True)
    from algorithm.features import get_feature_extractor

    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print(f"Computing the base-split mean over all {full_pool_size} train images...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    base_mean_time_s = time.perf_counter() - t0
    print(f"Base-split mean computed in {base_mean_time_s:.2f}s.", flush=True)

    print(f"Extracting {args.backbone} features for the {len(sample_paths)}-image diagnostic sample...", flush=True)
    sample_image_map = load_images_parallel(sample_paths)
    images = [sample_image_map[p] for p in sample_paths]
    t0 = time.perf_counter()
    raw_features = extractor(images, batch_size=32)
    extraction_time_s = time.perf_counter() - t0
    print(f"Feature extraction took {extraction_time_s:.2f}s.", flush=True)

    z = cl2n_condition(raw_features, base_mean).double().cpu().numpy()
    print(f"Ambient embedding matrix Z: shape={z.shape}", flush=True)

    print("Running run_diagnostics(Z) with the spectrum-matched null...", flush=True)
    t0 = time.perf_counter()
    diag_config = DiagnosticsConfig(random_seed=args.seed, delta_n_batches=8)
    diagnostics = run_diagnostics(z, config=diag_config, null_generator=generate_null_control_full_spectrum)
    print(f"Diagnostics wall-clock time: {time.perf_counter() - t0:.2f}s.", flush=True)

    md = _build_markdown(
        diagnostics, len(sample_paths), per_class_counts, extraction_time_s, base_mean_time_s,
        full_pool_size, backbone_label=args.backbone,
    )
    md_path = args.output_dir / "table2_diagnostics_full.md"
    md_path.write_text(md)
    print(f"Wrote {md_path}", flush=True)

    z_res, zt_res, gaps = diagnostics["Z"], diagnostics["Z_tilde"], diagnostics["gaps"]
    csv_rows = [
        {"quantity": "n_samples", "Z": len(sample_paths), "Z_tilde": len(sample_paths)},
        {"quantity": "ambient_dim", "Z": diagnostics["ambient_dim"], "Z_tilde": diagnostics["ambient_dim"]},
        {"quantity": "delta_rel", "Z": z_res["delta_rel"]["value"], "Z_tilde": zt_res["delta_rel"]["value"]},
        {"quantity": "delta_rel_ci95_lo", "Z": z_res["delta_rel"]["ci95"][0], "Z_tilde": zt_res["delta_rel"]["ci95"][0]},
        {"quantity": "delta_rel_ci95_hi", "Z": z_res["delta_rel"]["ci95"][1], "Z_tilde": zt_res["delta_rel"]["ci95"][1]},
        {"quantity": "d_TwoNN", "Z": z_res["d_TwoNN"], "Z_tilde": zt_res["d_TwoNN"]},
        {"quantity": "d_MLE", "Z": z_res["d_MLE"], "Z_tilde": zt_res["d_MLE"]},
        {"quantity": "d_PR", "Z": z_res["d_PR"], "Z_tilde": zt_res["d_PR"]},
        {"quantity": "d_eff", "Z": z_res["d_eff"], "Z_tilde": zt_res["d_eff"]},
        {"quantity": "rho_coph_pearson", "Z": z_res["rho_coph"]["pearson"], "Z_tilde": zt_res["rho_coph"]["pearson"]},
        {"quantity": "rho_coph_spearman", "Z": z_res["rho_coph"]["spearman"], "Z_tilde": zt_res["rho_coph"]["spearman"]},
        {"quantity": "gap_delta_delta_rel", "Z": gaps["delta_delta_rel"], "Z_tilde": ""},
    ]
    csv_path = args.output_dir / "table2_diagnostics_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quantity", "Z", "Z_tilde"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()

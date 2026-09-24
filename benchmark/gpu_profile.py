"""GPU latency (RT_ext, RT_ad, RT) and peak-VRAM profiling."""

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "benchmark" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
from dataset.ctx_uxo import (
    CLASS_NAMES,
    DEFAULT_INSTANCES_ROOT,
    list_class_instances,
    load_images_parallel,
)

MAX_PER_CLASS: int = 15
RANDOM_SEED: int = 42
RT_EXT_BATCH_SIZE: int = 16
RT_EXT_WARMUP_IMAGES: int = 1
RT_AD_N_SHOT: int = 5
RT_AD_T: int = 50
RT_AD_N_REPEATS: int = 5
K_CURVATURE: float = -1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def collect_class_samples(max_per_class: int = MAX_PER_CLASS) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    labels: List[int] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_paths = list_class_instances("train", class_name, root=DEFAULT_INSTANCES_ROOT)
        chosen = class_paths[:max_per_class]
        if not chosen:
            raise RuntimeError(f"No train-split instances found for class '{class_name}'.")
        paths.extend(chosen)
        labels.extend([class_idx] * len(chosen))
    return paths, labels


def gpu_description(device: torch.device) -> str:
    if device.type != "cuda":
        return f"{platform.machine()} CPU (no CUDA device; --device cpu was requested)"
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    return f"{name} ({total_gb:.1f} GB VRAM)"


def measure_rt_ext(extractor, images: Sequence[Tensor], batch_size: int, device: torch.device) -> Tuple[float, int, float]:
    if len(images) < batch_size:
        raise ValueError(f"Need at least {batch_size} images to time RT_ext, got {len(images)}.")
    batch = list(images[:batch_size])
    extractor.extract(list(images[:RT_EXT_WARMUP_IMAGES]), batch_size=RT_EXT_WARMUP_IMAGES)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    extractor.extract(batch, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
    per_image_ms = (t1 - t0) / len(batch) * 1000.0
    return per_image_ms, len(batch), peak_mb


def build_latency_episode(
    raw_features: Tensor, base_mean: Tensor, labels: Sequence[int], d_eff: int, n_shot: int, device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    z_all = cl2n_condition(raw_features, base_mean)
    support_idx: List[int] = []
    query_idx: List[int] = []
    for class_label in sorted(set(labels)):
        class_positions = [i for i, lbl in enumerate(labels) if lbl == class_label]
        support_positions = class_positions[:n_shot]
        query_positions = class_positions[n_shot:] or class_positions[:1]
        support_idx.extend(support_positions)
        query_idx.extend(query_positions)
    support_z = z_all[support_idx]
    support_y = torch.tensor([labels[i] for i in support_idx], dtype=torch.long, device=device)
    query_z = z_all[query_idx]
    u_pr = fit_pca_projection(support_z, d_eff=d_eff)
    support_x = hyperbolic_lift(support_z, u_pr, K=K_CURVATURE)
    query_x = hyperbolic_lift(query_z, u_pr, K=K_CURVATURE)
    return support_x, support_y, query_x


def measure_rt_ad(
    support_x: Tensor, support_y: Tensor, query_x: Tensor, d_eff: int, n_repeats: int, device: torch.device,
) -> Tuple[float, float, float]:
    config = HTIMConfig(K=K_CURVATURE, d_eff=d_eff, T=RT_AD_T)
    htim_adapt(support_x, support_y, query_x, config)  # untimed warm-up
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times_ms: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        htim_adapt(support_x, support_y, query_x, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
    return float(np.mean(times_ms)), float(np.std(times_ms)), peak_mb


def build_markdown(
    hardware: str, backbone: str,
    rt_ext_ms: float, rt_ext_batch: int, rt_ext_peak_mb: float,
    rt_ad_mean_ms: float, rt_ad_std_ms: float, rt_ad_repeats: int, rt_ad_peak_mb: float,
    n_shot: int, d_eff: int,
) -> str:
    rt_total_ms = rt_ext_ms + rt_ad_mean_ms
    peak_mb = max(rt_ext_peak_mb, rt_ad_peak_mb) if not np.isnan(rt_ext_peak_mb) else float("nan")
    lines: List[str] = []
    lines.append(f"# Table 6 (GPU) - Per-Query Inference Latency and Peak VRAM ({backbone})")
    lines.append("")
    lines.append(
        f"Measured on {hardware}: CTX-UXO train-split images, frozen "
        f"`{backbone}` feature extractor, `htim_adapt` with R-Adam."
    )
    lines.append("")
    lines.append(
        f"- $RT_{{\\text{{ext}}}}$: one batched `{backbone}` forward pass over "
        f"{rt_ext_batch} CTX-UXO train-split images (batch_size={rt_ext_batch}), "
        "CUDA-synchronized wall-clock time divided by the batch size, after one "
        "untimed 1-image warm-up call. Peak VRAM from "
        "`torch.cuda.max_memory_allocated`, reset before the timed call."
    )
    lines.append(
        f"- $RT_{{\\text{{ad}}}}$: one `algorithm.rtim.htim_adapt` call, "
        f"$T={RT_AD_T}$ R-Adam iterations, on a {n_shot}-shot, "
        f"{len(CLASS_NAMES)}-way episode, $K={K_CURVATURE}$, $d_{{\\text{{eff}}}}={d_eff}$ "
        f"(diagnostic-calibrated on the same sample). PCA and the lift are "
        f"not timed; mean $\\pm$ std over {rt_ad_repeats} CUDA-synchronized "
        "repeats after one warm-up call. Peak VRAM reset before the repeats."
    )
    lines.append("")
    lines.append(
        "| Platform | Backbone | Feature Extraction (ms) | Adaptation (ms) | "
        "Total Latency (ms) | Peak VRAM (MB) |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    lines.append(
        f"| {hardware} | {backbone} | {rt_ext_ms:.2f} | "
        f"{rt_ad_mean_ms:.2f} $\\pm$ {rt_ad_std_ms:.2f} | {rt_total_ms:.2f} | "
        f"{peak_mb:.1f} |"
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    set_deterministic(RANDOM_SEED)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hardware = gpu_description(device)
    print(f"Hardware: {hardware}", flush=True)

    print(f"Loading feature extractor '{args.backbone}' on {args.device!r}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    paths, labels = collect_class_samples(MAX_PER_CLASS)
    print(f"Collected {len(paths)} train-split images across {len(CLASS_NAMES)} classes.", flush=True)
    image_map = load_images_parallel(paths)
    images = [image_map[p] for p in paths]

    print("Extracting features for the full sample (untimed, batch_size=32)...", flush=True)
    raw_features = extractor(images, batch_size=32)
    base_mean = raw_features.mean(dim=0)
    z = cl2n_condition(raw_features, base_mean).double().cpu().numpy()

    print("Running run_diagnostics(Z) for a diagnostic-calibrated d_eff...", flush=True)
    diag = run_diagnostics(z, config=DiagnosticsConfig(random_seed=RANDOM_SEED))
    d_eff = int(diag["Z"]["d_eff"])
    print(f"d_eff={d_eff}", flush=True)

    print(f"Measuring RT_ext (batch_size={RT_EXT_BATCH_SIZE})...", flush=True)
    rt_ext_ms, rt_ext_batch, rt_ext_peak_mb = measure_rt_ext(extractor, images, RT_EXT_BATCH_SIZE, device)
    print(f"RT_ext = {rt_ext_ms:.2f} ms/image, peak VRAM = {rt_ext_peak_mb:.1f} MB", flush=True)

    support_x, support_y, query_x = build_latency_episode(
        raw_features, base_mean, labels, d_eff=d_eff, n_shot=RT_AD_N_SHOT, device=device,
    )
    print(f"Measuring RT_ad (T={RT_AD_T}, {RT_AD_N_REPEATS} repeats)...", flush=True)
    rt_ad_mean_ms, rt_ad_std_ms, rt_ad_peak_mb = measure_rt_ad(
        support_x, support_y, query_x, d_eff=d_eff, n_repeats=RT_AD_N_REPEATS, device=device,
    )
    print(f"RT_ad = {rt_ad_mean_ms:.2f} +/- {rt_ad_std_ms:.2f} ms, peak VRAM = {rt_ad_peak_mb:.1f} MB", flush=True)

    md = build_markdown(
        hardware=hardware, backbone=args.backbone,
        rt_ext_ms=rt_ext_ms, rt_ext_batch=rt_ext_batch, rt_ext_peak_mb=rt_ext_peak_mb,
        rt_ad_mean_ms=rt_ad_mean_ms, rt_ad_std_ms=rt_ad_std_ms, rt_ad_repeats=RT_AD_N_REPEATS, rt_ad_peak_mb=rt_ad_peak_mb,
        n_shot=RT_AD_N_SHOT, d_eff=d_eff,
    )
    out_path = args.output_dir / f"table6_latency_gpu_{args.backbone}.md"
    out_path.write_text(md)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

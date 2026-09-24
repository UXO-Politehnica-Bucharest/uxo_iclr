"""Table 6 (per-query inference latency) on a CPU machine; writes
results/table6_latency.md.
"""

import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "benchmark" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithm.diagnostics import DiagnosticsConfig, run_diagnostics
from algorithm.features import ClipFeatureExtractor
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


RESULTS_DIR: Path = _REPO_ROOT / "results"
TABLE6_PATH: Path = RESULTS_DIR / "table6_latency.md"

MAX_PER_CLASS: int = 15
"""Train-split images per class used for the d_eff calibration and the
latency episode."""

RANDOM_SEED: int = 42

RT_EXT_BATCH_SIZE: int = 16
"""Episode-sized batch used to time RT_ext (CLIP extraction)."""
RT_EXT_WARMUP_IMAGES: int = 1
RT_AD_N_SHOT: int = 5
RT_AD_T: int = 50
RT_AD_N_REPEATS: int = 5
K_CURVATURE: float = -1.0
"""Fixed Lorentz curvature; K is not searched (only psi_S* is)."""


@dataclass
class ClassSample:
    """A deterministic small sample of (path, label) pairs across all
    CTX-UXO classes, drawn exclusively from the train split."""

    paths: List[Path]
    labels: List[int]


def collect_class_samples(max_per_class: int = MAX_PER_CLASS) -> ClassSample:
    """First ``max_per_class`` train-split images of each class (sorted, so the
    sample is reproducible); smaller classes contribute all their images.
    """
    paths: List[Path] = []
    labels: List[int] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_paths = list_class_instances("train", class_name, root=DEFAULT_INSTANCES_ROOT)
        chosen = class_paths[:max_per_class]
        if not chosen:
            raise RuntimeError(
                f"No train-split instances found for class '{class_name}' under "
                f"{DEFAULT_INSTANCES_ROOT}; cannot build the diagnostic/latency sample."
            )
        paths.extend(chosen)
        labels.extend([class_idx] * len(chosen))
    return ClassSample(paths=paths, labels=labels)


def load_raw_tensors(paths: Sequence[Path]) -> List[Tensor]:
    """Load images as raw [0, 1] CHW tensors (no augmentation), via the
    exact ``pil_to_raw_tensor`` conversion used by the rest of the project,
    loaded in parallel via ``load_images_parallel``.
    """
    image_map = load_images_parallel(list(paths))
    return [image_map[p] for p in paths]


def cpu_description() -> str:
    """CPU model string and logical core count (from /proc/cpuinfo when available)."""
    model: str | None = None
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        for line in cpuinfo_path.read_text().splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    if not model:
        model = platform.processor() or f"{platform.machine()} CPU"
    n_cores = os.cpu_count() or 1
    return f"{model}, {n_cores} logical cores"


def build_ambient_embeddings(
    extractor: ClipFeatureExtractor, sample: ClassSample
) -> Tuple[np.ndarray, Tensor, float]:
    """CLIP features of ``sample`` and their CL2N-conditioned float64 matrix.

    The base-split mean is approximated by the mean of this sample's features.

    Returns:
        ``(Z, raw_features, extraction_time_s)``.
    """
    images = load_raw_tensors(sample.paths)
    t0 = time.perf_counter()
    raw_features = extractor(images, batch_size=32)
    t1 = time.perf_counter()
    extraction_time_s = t1 - t0

    base_mean = raw_features.mean(dim=0)
    z = cl2n_condition(raw_features, base_mean)
    Z = z.double().cpu().numpy()
    return Z, raw_features, extraction_time_s


def measure_rt_ext(
    extractor: ClipFeatureExtractor, images: Sequence[Tensor], batch_size: int
) -> Tuple[float, int]:
    """Per-image extraction latency (ms) of one batched forward pass, after an
    untimed warm-up call.
    """
    if len(images) < batch_size:
        raise ValueError(
            f"Need at least {batch_size} images to time RT_ext, got {len(images)}."
        )
    batch = list(images[:batch_size])
    extractor.extract(list(images[:RT_EXT_WARMUP_IMAGES]), batch_size=RT_EXT_WARMUP_IMAGES)

    t0 = time.perf_counter()
    extractor.extract(batch, batch_size=batch_size)
    t1 = time.perf_counter()
    total_s = t1 - t0
    per_image_ms = (total_s / len(batch)) * 1000.0
    return per_image_ms, len(batch)


def build_latency_episode(
    raw_features: Tensor, base_mean: Tensor, labels: Sequence[int], d_eff: int, n_shot: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Lifted support/query tensors for the latency episode (untimed).

    The first ``n_shot`` images of each class are support and the rest are
    queries; a class with no remaining image reuses one support image as a
    query so the episode is valid. PCA is fitted on the support set.
    """
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
    support_y = torch.tensor(
        [labels[i] for i in support_idx], dtype=torch.long, device=support_z.device
    )
    query_z = z_all[query_idx]

    u_pr = fit_pca_projection(support_z, d_eff=d_eff)
    support_x = hyperbolic_lift(support_z, u_pr, K=K_CURVATURE)
    query_x = hyperbolic_lift(query_z, u_pr, K=K_CURVATURE)
    return support_x, support_y, query_x


def measure_rt_ad(
    support_x: Tensor, support_y: Tensor, query_x: Tensor, d_eff: int, n_repeats: int
) -> Tuple[float, float]:
    """Mean and std (ms) of ``n_repeats`` timed ``htim_adapt`` calls, after one
    untimed warm-up call.
    """
    config = HTIMConfig(K=K_CURVATURE, d_eff=d_eff, T=RT_AD_T)

    htim_adapt(support_x, support_y, query_x, config)  # untimed warm-up

    times_ms: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        htim_adapt(support_x, support_y, query_x, config)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return float(np.mean(times_ms)), float(np.std(times_ms))


def build_table6_markdown(
    rt_ext_ms: float,
    rt_ext_batch: int,
    rt_ad_mean_ms: float,
    rt_ad_std_ms: float,
    rt_ad_repeats: int,
    n_shot: int,
    d_eff: int,
    hardware: str,
) -> str:
    """Table 6 markdown: Platform | RT_ext | RT_ad | Total RT."""
    rt_total_ms = rt_ext_ms + rt_ad_mean_ms
    lines: List[str] = []
    lines.append("# Table 6 - Per-Query Inference Latency (CPU)")
    lines.append("")
    lines.append(
        f"Measured on {hardware}. GPU latency: benchmark/gpu_profile.py."
    )
    lines.append("")
    lines.append(
        f"- $RT_{{\\text{{ext}}}}$: one batched CLIP ViT-L/14 forward pass over "
        f"{rt_ext_batch} CTX-UXO train-split images, divided by the batch size, "
        "after one untimed 1-image warm-up call."
    )
    lines.append(
        f"- $RT_{{\\text{{ad}}}}$: one `htim_adapt` call, $T={RT_AD_T}$ R-Adam "
        f"iterations, {n_shot}-shot {len(CLASS_NAMES)}-way episode on "
        f"$\\mathcal{{L}}^{{{d_eff}}}_K$ ($K={K_CURVATURE}$). PCA and the lift "
        f"are not timed; mean $\\pm$ std over {rt_ad_repeats} repeats after one "
        "warm-up call."
    )
    lines.append("- $RT = RT_{\\text{ext}} + RT_{\\text{ad}}$.")
    lines.append("")
    lines.append(
        "| Platform | $RT_{\\text{ext}}$ (ms) | $RT_{\\text{ad}}$ (ms) | Total $RT$ (ms) |"
    )
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(
        f"| {hardware} | {rt_ext_ms:.2f} | {rt_ad_mean_ms:.2f} $\\pm$ {rt_ad_std_ms:.2f} | "
        f"{rt_total_ms:.2f} |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    hardware = cpu_description()
    print(f"{hardware}, torch threads: {torch.get_num_threads()}")

    sample = collect_class_samples(MAX_PER_CLASS)
    extractor = ClipFeatureExtractor(device="cpu")
    Z, raw_features, extraction_time_s = build_ambient_embeddings(extractor, sample)
    print(f"Extracted {Z.shape[0]} features in {extraction_time_s:.1f}s")

    diagnostics = run_diagnostics(Z, config=DiagnosticsConfig(random_seed=RANDOM_SEED))
    d_eff = int(diagnostics["Z"]["d_eff"])
    print(f"d_eff={d_eff}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    images = load_raw_tensors(sample.paths)
    rt_ext_ms, rt_ext_batch = measure_rt_ext(extractor, images, RT_EXT_BATCH_SIZE)
    print(f"RT_ext = {rt_ext_ms:.2f} ms/image (batch_size={rt_ext_batch})")

    base_mean = raw_features.mean(dim=0)
    support_x, support_y, query_x = build_latency_episode(
        raw_features, base_mean, sample.labels, d_eff=d_eff, n_shot=RT_AD_N_SHOT
    )
    rt_ad_mean_ms, rt_ad_std_ms = measure_rt_ad(
        support_x, support_y, query_x, d_eff=d_eff, n_repeats=RT_AD_N_REPEATS
    )
    print(f"RT_ad = {rt_ad_mean_ms:.2f} +/- {rt_ad_std_ms:.2f} ms")

    table6_md = build_table6_markdown(
        rt_ext_ms=rt_ext_ms,
        rt_ext_batch=rt_ext_batch,
        rt_ad_mean_ms=rt_ad_mean_ms,
        rt_ad_std_ms=rt_ad_std_ms,
        rt_ad_repeats=RT_AD_N_REPEATS,
        n_shot=RT_AD_N_SHOT,
        d_eff=d_eff,
        hardware=hardware,
    )
    TABLE6_PATH.write_text(table6_md)
    print(f"Wrote {TABLE6_PATH}")

if __name__ == "__main__":
    main()

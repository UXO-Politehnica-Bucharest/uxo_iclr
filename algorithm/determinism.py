#!/usr/bin/env python3
"""Configures process-wide deterministic execution and random seeds."""

import os
import random

import torch


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    set_deterministic(seed=42)

    import torch
    from algorithm.manifold import exp_map
    from algorithm.rtim import HTIMConfig, _origin, htim_adapt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_eff, n_way, shot, n_query = 16, 5, 5, 15
    g = torch.Generator(device="cpu").manual_seed(123)
    support_z = torch.randn(n_way * shot, d_eff, generator=g).to(device)
    query_z = torch.randn(n_way * n_query, d_eff, generator=g).to(device)
    support_y = torch.arange(n_way, device=device).repeat_interleave(shot)

    def _lift(z: torch.Tensor, K: float) -> torch.Tensor:
        v = torch.cat([torch.zeros_like(z[..., :1]), z], dim=-1)
        o = _origin(d_eff, K, dtype=z.dtype, device=z.device).expand(v.shape[0], -1)
        return exp_map(o, v, K)

    K = -1.0
    support_x = _lift(support_z, K)
    query_x = _lift(query_z, K)
    config = HTIMConfig(K=K, d_eff=d_eff, T=60)
    result = htim_adapt(support_x, support_y, query_x, config)
    print(f"device={device}")
    print(f"predictions={result.predictions.cpu().tolist()}")
    print(f"final_loss={result.loss_trajectory[-1]!r}")
    print(f"prototypes_sum={result.prototypes.sum().item()!r}")
    print(f"prototypes_full={result.prototypes.cpu().tolist()}")

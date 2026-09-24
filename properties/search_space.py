import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.manifold import lorentz_distance

_T_RANGE: Tuple[int, int] = (10, 100)
_BETA_RANGE: Tuple[float, float] = (0.005, 0.5)

_CE_WEIGHT_RANGE: Tuple[float, float] = (0.01, 1.0)
_MARGINAL_H_WEIGHT_RANGE: Tuple[float, float] = (0.1, 3.0)
_COND_H_WEIGHT_RANGE: Tuple[float, float] = (0.01, 1.0)


_TAU_LOG_RANGE_DECADES: Tuple[float, float] = (-1.0, 1.5)
_OMEGA_LOG_RANGE_DECADES: Tuple[float, float] = (-2.0, 0.5)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def _centered_range(center: float, decades: Tuple[float, float]) -> Tuple[float, float]:
    return (center * (10.0 ** decades[0]), center * (10.0 ** decades[1]))


def _check_d_bar(d_bar: float) -> None:
    if not (d_bar > 0) or not math.isfinite(d_bar):
        raise ValueError(f"d_bar must be a finite positive number, got {d_bar}.")


def _sample_distinct_pairs(n: int, n_pairs: int, seed: int) -> Tuple[Tensor, Tensor]:
    """Up to ``n_pairs`` random (i, j) index pairs with i != j."""
    if n < 2:
        raise ValueError("Need at least 2 points to measure a distance scale.")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = min(n_pairs, n * (n - 1))
    idx_a = torch.randint(0, n, (k * 2,), generator=gen)
    idx_b = torch.randint(0, n, (k * 2,), generator=gen)
    mask = idx_a != idx_b
    return idx_a[mask][:k], idx_b[mask][:k]


@dataclass(frozen=True)
class SearchSpace:
    """Ranges for psi_S*. T is sampled as a uniform integer, every other field
    log-uniformly.
    """

    T_range: Tuple[int, int]
    beta_range: Tuple[float, float]
    ce_weight_range: Tuple[float, float]
    marginal_h_weight_range: Tuple[float, float]
    cond_h_weight_range: Tuple[float, float]
    omega_range: Tuple[float, float]
    tau_range: Tuple[float, float]

    def sample(self, rng: np.random.Generator) -> Dict[str, float]:
        """Draw one random candidate psi from Omega."""

        return {
            "T": int(rng.integers(self.T_range[0], self.T_range[1] + 1)),
            "beta": _log_uniform(rng, *self.beta_range),
            "ce_weight": _log_uniform(rng, *self.ce_weight_range),
            "marginal_h_weight": _log_uniform(rng, *self.marginal_h_weight_range),
            "cond_h_weight": _log_uniform(rng, *self.cond_h_weight_range),
            "omega": _log_uniform(rng, *self.omega_range),
            "tau": _log_uniform(rng, *self.tau_range),
        }


def measure_geodesic_distance_scale(
    lifted_points: Tensor,
    K: float,
    n_pairs: int = 4000,
    seed: int = 0,
) -> float:
    """Median squared geodesic distance over ``n_pairs`` random pairs of lifted
    points (pair sampling seeded by ``seed``).
    """
    idx_a, idx_b = _sample_distinct_pairs(lifted_points.shape[0], n_pairs, seed)
    d2 = lorentz_distance(lifted_points[idx_a], lifted_points[idx_b], K) ** 2
    return float(d2.median().item())


def measure_euclidean_distance_scale(
    points: Tensor,
    n_pairs: int = 4000,
    seed: int = 0,
) -> float:
    """Median squared Euclidean distance over ``n_pairs`` random pairs of
    PCA-projected (unlifted) points; the scale used by TIM (Euclidean).
    """
    idx_a, idx_b = _sample_distinct_pairs(points.shape[0], n_pairs, seed)
    d2 = ((points[idx_a] - points[idx_b]) ** 2).sum(dim=-1)
    return float(d2.median().item())


@dataclass(frozen=True)
class EuclideanSearchSpace:
    """Search space for TIM (Euclidean)'s own search: psi_S* without omega, since
    the Euclidean objective has no origin regularizer.
    """

    T_range: Tuple[int, int]
    beta_range: Tuple[float, float]
    ce_weight_range: Tuple[float, float]
    marginal_h_weight_range: Tuple[float, float]
    cond_h_weight_range: Tuple[float, float]
    tau_range: Tuple[float, float]

    def sample(self, rng: np.random.Generator) -> Dict[str, float]:
        """Draw one random candidate psi_E from Omega_E."""

        return {
            "T": int(rng.integers(self.T_range[0], self.T_range[1] + 1)),
            "beta": _log_uniform(rng, *self.beta_range),
            "ce_weight": _log_uniform(rng, *self.ce_weight_range),
            "marginal_h_weight": _log_uniform(rng, *self.marginal_h_weight_range),
            "cond_h_weight": _log_uniform(rng, *self.cond_h_weight_range),
            "tau": _log_uniform(rng, *self.tau_range),
        }


def diagnostics_informed_euclidean_search_space(d_bar: float) -> EuclideanSearchSpace:
    """TIM (Euclidean) search space with tau centred on 1/d_bar, where d_bar is
    the Euclidean scale from :func:`measure_euclidean_distance_scale`.
    """
    _check_d_bar(d_bar)
    return EuclideanSearchSpace(
        T_range=_T_RANGE,
        beta_range=_BETA_RANGE,
        ce_weight_range=_CE_WEIGHT_RANGE,
        marginal_h_weight_range=_MARGINAL_H_WEIGHT_RANGE,
        cond_h_weight_range=_COND_H_WEIGHT_RANGE,
        tau_range=_centered_range(1.0 / d_bar, _TAU_LOG_RANGE_DECADES),
    )


def diagnostics_informed_search_space(d_bar: float) -> SearchSpace:
    """psi_S* search space with tau and omega centred on 1/d_bar (median squared
    geodesic distance, > 0).
    """
    _check_d_bar(d_bar)
    center = 1.0 / d_bar
    return SearchSpace(
        T_range=_T_RANGE,
        beta_range=_BETA_RANGE,
        ce_weight_range=_CE_WEIGHT_RANGE,
        marginal_h_weight_range=_MARGINAL_H_WEIGHT_RANGE,
        cond_h_weight_range=_COND_H_WEIGHT_RANGE,
        omega_range=_centered_range(center, _OMEGA_LOG_RANGE_DECADES),
        tau_range=_centered_range(center, _TAU_LOG_RANGE_DECADES),
    )


_EXTENDED_T_RANGE: Tuple[int, int] = (10, 1000)
_EXTENDED_BETA_CHOICES: Tuple[float, ...] = (
    1e-5, 2e-5, 4e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2,
)
"""Discrete learning-rate candidates, sampled uniformly (the set is already
roughly log-spaced), unlike the continuous _BETA_RANGE."""


@dataclass(frozen=True)
class ExtendedSearchSpace:
    """psi_S* search space with T up to 1000 and beta from a discrete set. Callers
    must also set ``use_query_in_mi=False`` (query-only MI pool, as in TIM-GD).
    """

    T_range: Tuple[int, int]
    beta_choices: Tuple[float, ...]
    ce_weight_range: Tuple[float, float]
    marginal_h_weight_range: Tuple[float, float]
    cond_h_weight_range: Tuple[float, float]
    omega_range: Tuple[float, float]
    tau_range: Tuple[float, float]

    def sample(self, rng: np.random.Generator) -> Dict[str, float]:
        """Draw one random candidate psi' from Omega'."""

        return {
            "T": int(rng.integers(self.T_range[0], self.T_range[1] + 1)),
            "beta": float(rng.choice(self.beta_choices)),
            "ce_weight": _log_uniform(rng, *self.ce_weight_range),
            "marginal_h_weight": _log_uniform(rng, *self.marginal_h_weight_range),
            "cond_h_weight": _log_uniform(rng, *self.cond_h_weight_range),
            "omega": _log_uniform(rng, *self.omega_range),
            "tau": _log_uniform(rng, *self.tau_range),
        }


def diagnostics_informed_extended_search_space(d_bar: float) -> ExtendedSearchSpace:
    """Extended search space with tau and omega centred on 1/d_bar, as in
    :func:`diagnostics_informed_search_space`.
    """
    _check_d_bar(d_bar)
    center = 1.0 / d_bar
    return ExtendedSearchSpace(
        T_range=_EXTENDED_T_RANGE,
        beta_choices=_EXTENDED_BETA_CHOICES,
        ce_weight_range=_CE_WEIGHT_RANGE,
        marginal_h_weight_range=_MARGINAL_H_WEIGHT_RANGE,
        cond_h_weight_range=_COND_H_WEIGHT_RANGE,
        omega_range=_centered_range(center, _OMEGA_LOG_RANGE_DECADES),
        tau_range=_centered_range(center, _TAU_LOG_RANGE_DECADES),
    )


if __name__ == "__main__":
    from algorithm.manifold import exp_map

    torch.manual_seed(0)
    K = -1.0
    d_eff = 8
    n_points = 200
    o = torch.zeros(d_eff + 1)
    o[0] = 1.0 / math.sqrt(-K)
    tangent = torch.randn(n_points, d_eff) * 0.6
    tangent_full = torch.cat([torch.zeros(n_points, 1), tangent], dim=-1)
    lifted = exp_map(o.expand(n_points, -1), tangent_full, K)

    d_bar = measure_geodesic_distance_scale(lifted, K, n_pairs=2000, seed=42)
    assert d_bar > 0 and math.isfinite(d_bar)

    space = diagnostics_informed_search_space(d_bar)
    print(f"d_bar={d_bar:.6f} tau_range={space.tau_range} omega_range={space.omega_range}")

    tau_center_product = (space.tau_range[0] * space.tau_range[1]) ** 0.5 * d_bar
    assert 0.1 < tau_center_product < 10.0, "tau range is not centered on 1/d_bar"

    rng = np.random.default_rng(123)
    for _ in range(5):
        cand = space.sample(rng)
        assert space.T_range[0] <= cand["T"] <= space.T_range[1]
        assert space.beta_range[0] <= cand["beta"] <= space.beta_range[1]
        assert space.ce_weight_range[0] <= cand["ce_weight"] <= space.ce_weight_range[1]
        assert space.marginal_h_weight_range[0] <= cand["marginal_h_weight"] <= space.marginal_h_weight_range[1]
        assert space.cond_h_weight_range[0] <= cand["cond_h_weight"] <= space.cond_h_weight_range[1]
        assert space.omega_range[0] <= cand["omega"] <= space.omega_range[1]
        assert space.tau_range[0] <= cand["tau"] <= space.tau_range[1]

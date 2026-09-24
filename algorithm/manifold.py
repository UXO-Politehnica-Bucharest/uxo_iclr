import math
from typing import Optional

import torch
from torch import Tensor

_ARCOSH_MIN: float = 1.0 + 1e-7
_NORM_MIN: float = 1e-7
_TAYLOR_THRESHOLD: float = 1e-4


def minkowski_dot(x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
    """Computes the Minkowski (Lorentzian) inner product <x, y>_L."""
    time = -x[..., 0:1] * y[..., 0:1]
    space = torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
    result = time + space
    if not keepdim:
        result = result.squeeze(-1)
    return result


def lorentz_distance(x: Tensor, y: Tensor, K: float) -> Tensor:
    """Computes the geodesic distance on the Lorentz hyperboloid."""
    sqrt_neg_k = math.sqrt(-K)
    dot = minkowski_dot(x, y)
    zeta = torch.clamp(K * dot, min=_ARCOSH_MIN)
    return torch.arccosh(zeta) / sqrt_neg_k


def _lorentz_norm(u: Tensor) -> Tensor:
    """Computes the Lorentzian norm ||u||_L of a tangent vector."""
    sq = torch.clamp(minkowski_dot(u, u), min=0.0)
    return torch.sqrt(sq)


def exp_map(x: Tensor, u: Tensor, K: float) -> Tensor:
    """Exponential map from tangent space onto the hyperboloid manifold."""
    sqrt_neg_k = math.sqrt(-K)
    norm_u = _lorentz_norm(u)
    z = sqrt_neg_k * norm_u

    cosh_z = torch.cosh(z)
    z_safe = torch.where(z.abs() < _TAYLOR_THRESHOLD, torch.ones_like(z), z)
    sinh_over_z_exact = torch.sinh(z_safe) / z_safe
    sinh_over_z_taylor = 1.0 + (z * z) / 6.0
    sinh_over_z = torch.where(
        z.abs() < _TAYLOR_THRESHOLD, sinh_over_z_taylor, sinh_over_z_exact
    )

    return cosh_z * x + sinh_over_z * u


def proj_tangent(x: Tensor, v: Tensor, K: float) -> Tensor:
    """Orthogonal projection of an ambient vector onto the tangent space at x."""
    coeff = K * minkowski_dot(x, v)
    return v - coeff * x


def log_map(x: Tensor, y: Tensor, K: float) -> Tensor:
    """Logarithmic map from the hyperboloid manifold into tangent space at x."""
    dot = minkowski_dot(x, y)
    zeta = torch.clamp(K * dot, min=_ARCOSH_MIN)
    theta = torch.arccosh(zeta)

    theta_safe = torch.where(theta < _TAYLOR_THRESHOLD, torch.ones_like(theta), theta)
    coeff_exact = theta_safe / torch.sinh(theta_safe)
    coeff_taylor = 1.0 - (theta * theta) / 6.0
    coeff = torch.where(theta < _TAYLOR_THRESHOLD, coeff_taylor, coeff_exact)

    proj = proj_tangent(x, y, K)
    return coeff * proj


def parallel_transport(x: Tensor, y: Tensor, u: Tensor, K: float) -> Tensor:
    """Parallel transports vector u from T_x to T_y on the hyperboloid."""
    denom = 1.0 + K * minkowski_dot(x, y)
    coeff = (K * minkowski_dot(y, u)) / denom
    return u - coeff * (x + y)


def apply_eta(v: Tensor) -> Tensor:
    """Applies the Minkowski metric eta = diag(-1, 1, ..., 1) to an ambient vector."""
    eta_v = v.clone()
    eta_v[..., 0] = -eta_v[..., 0]
    return eta_v


def lorentz_centroid(points: Tensor, K: float, dim: int = -2) -> Tensor:
    """Computes the closed-form Lorentzian centroid of a point set."""
    mean_point = torch.mean(points, dim=dim)
    denom_sq = K * minkowski_dot(mean_point, mean_point)
    denom = torch.sqrt(torch.clamp(denom_sq, min=_NORM_MIN))
    return mean_point / denom


def reproject(x: Tensor, K: float) -> Tensor:
    """Projects a point back onto the Lorentz hyperboloid manifold."""
    denom_sq = K * minkowski_dot(x, x)
    denom = torch.sqrt(torch.clamp(denom_sq, min=_NORM_MIN))
    return x / denom


class RiemannianAdam(torch.optim.Optimizer):
    """Riemannian-Adam (R-Adam) optimizer on the Lorentz hyperboloid manifold."""

    def __init__(
        self,
        params,
        K: float,
        beta: float = 1e-2,
        rho1: float = 0.9,
        rho2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if K >= 0:
            raise ValueError(f"Curvature K must be negative, got {K}.")
        if not 0.0 <= rho1 < 1.0:
            raise ValueError(f"rho1 must be in [0, 1), got {rho1}.")
        if not 0.0 <= rho2 < 1.0:
            raise ValueError(f"rho2 must be in [0, 1), got {rho2}.")
        defaults = dict(K=K, beta=beta, rho1=rho1, rho2=rho2, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[Tensor]:
        """Performs a single R-Adam optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            K = group["K"]
            beta = group["beta"]
            rho1 = group["rho1"]
            rho2 = group["rho2"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                riemannian_grad = proj_tangent(p, apply_eta(p.grad), K)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros(p.shape[:-1] + (1,), dtype=p.dtype, device=p.device)
                    state["prev_point"] = p.clone()

                m_prev = state["m"]
                v_prev = state["v"]
                prev_point = state["prev_point"]
                step_count = state["step"] + 1
                state["step"] = step_count

                m_transported = parallel_transport(prev_point, p, m_prev, K)
                m_new = rho1 * m_transported + (1.0 - rho1) * riemannian_grad

                grad_norm_sq = torch.clamp(
                    minkowski_dot(riemannian_grad, riemannian_grad), min=0.0
                )
                v_new = rho2 * v_prev + (1.0 - rho2) * grad_norm_sq

                bias_correction1 = 1.0 - rho1**step_count
                bias_correction2 = 1.0 - rho2**step_count
                m_hat = m_new / bias_correction1
                v_hat = v_new / bias_correction2

                tangent_step = -(beta / (torch.sqrt(v_hat) + eps)) * m_hat
                new_point = exp_map(p, tangent_step, K)
                new_point = reproject(new_point, K)

                state["prev_point"] = p.clone()
                state["m"] = m_new
                state["v"] = v_new
                p.copy_(new_point)

        return loss


import dataclasses
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from algorithm.rtim import (
    HTIMConfig,
    cl2n_condition,
    fit_pca_projection,
    htim_adapt,
    hyperbolic_lift,
    plain_l2_normalize,
)
from benchmark.evaluator import macro_f1

__all__ = [
    "run_extra_hyperbolic_methods",
    "auto_select_c",
]


_ARCCOSH_MIN: float = 1.0 + 1e-7
_DENOM_EPS: float = 1e-7
_NORM_EPS: float = 1e-12

_BALL_EPS: float = 1e-5


def _safe_norm(v: Tensor) -> Tensor:
    """Euclidean norm along the last dimension, floored at ``_NORM_EPS`` to
    avoid ``0/0`` when normalizing a (near-)zero vector."""
    return v.norm(p=2, dim=-1, keepdim=True).clamp_min(_NORM_EPS)


def _clamp_ball_norm(x: Tensor, max_norm: float) -> Tensor:
    """Radially shrink rows of ``x`` with norm above ``max_norm`` to ``max_norm``."""
    norm = _safe_norm(x)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def _poincare_exp0(v: Tensor, c: float) -> Tensor:
    """Poincare-ball exponential map at the origin, curvature -c:
    exp0(v) = tanh(sqrt(c) |v|) v / (sqrt(c) |v|). Maps into the ball of radius
    1/sqrt(c).
    """
    sqrt_c = math.sqrt(c)
    norm = _safe_norm(v)
    return torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm)


def _poincare_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    """Poincare distance at curvature -c:
    arccosh(1 + 2c|x - y|^2 / ((1 - c|x|^2)(1 - c|y|^2))) / sqrt(c).

    The arccosh argument and the denominator are clamped; callers should also
    keep points strictly inside the ball (see :func:`_clamp_ball_norm`).
    Returns distances with the last dimension reduced.
    """
    diff2 = torch.sum((x - y) ** 2, dim=-1)
    x2 = torch.sum(x * x, dim=-1)
    y2 = torch.sum(y * y, dim=-1)
    denom = torch.clamp((1.0 - c * x2) * (1.0 - c * y2), min=_DENOM_EPS)
    argument = torch.clamp(1.0 + 2.0 * c * diff2 / denom, min=_ARCCOSH_MIN)
    return torch.arccosh(argument) / math.sqrt(c)


def auto_select_c(d: int) -> float:
    """Curvature heuristic from the official hyptorch code (``pmath.auto_select_c``):
    the c for which the d-dimensional ball has volume pi. Used because our
    d_eff differs from the 512/1600-dim embeddings the published fixed
    curvatures were chosen for.
    """
    dim2 = d / 2.0
    radius = (math.gamma(dim2 + 1) / (math.pi ** (dim2 - 1))) ** (1.0 / d)
    return 1.0 / (radius ** 2)


def _mobius_add(x: Tensor, y: Tensor, c: float) -> Tensor:
    """Mobius addition x (+)_c y on the Poincare ball."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denom = torch.clamp(1.0 + 2.0 * c * xy + c * c * x2 * y2, min=_DENOM_EPS)
    return num / denom


def _poincare_p2k(x: Tensor, c: float) -> Tensor:
    """Poincare ball -> Klein model, curvature ``c``."""
    denom = torch.clamp(1.0 + c * (x * x).sum(dim=-1, keepdim=True), min=_DENOM_EPS)
    return 2.0 * x / denom


def _poincare_k2p(x: Tensor, c: float) -> Tensor:
    """Klein model -> Poincare ball, curvature ``c``."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    denom = torch.clamp(1.0 + torch.sqrt(torch.clamp(1.0 - c * x2, min=_DENOM_EPS)), min=_DENOM_EPS)
    return x / denom


def _lorenz_factor(x: Tensor, c: float) -> Tensor:
    """Lorentz factor in the Klein model, used to weight points in the
    Einstein-midpoint average: $\\lambda_x = 1/\\sqrt{1-c\\|x\\|^2}$."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 1.0 / torch.sqrt(torch.clamp(1.0 - c * x2, min=_DENOM_EPS))


def _poincare_mean(x: Tensor, c: float, dim: int = 0) -> Tensor:
    """Einstein midpoint of ``x`` along ``dim`` (hyptorch ``pmath.poincare_mean``):
    Poincare -> Klein, Lorentz-factor-weighted average, Klein -> Poincare. A
    closed-form substitute for the Frechet mean, as in the official code.
    """
    xk = _poincare_p2k(x, c)
    lamb = _lorenz_factor(xk, c)
    mean = (lamb * xk).sum(dim=dim, keepdim=True) / lamb.sum(dim=dim, keepdim=True).clamp_min(_DENOM_EPS)
    mean = _poincare_k2p(mean, c)
    return mean.squeeze(dim)


def _poincare_prototypes(lifted: Tensor, labels: Tensor, num_classes: int, c: float) -> Tensor:
    """Per-class Einstein-midpoint prototypes of ball points, shape ``(C, d)``."""
    prototypes = []
    for a in range(num_classes):
        mask = labels == a
        if not mask.any():
            raise ValueError(
                f"No support samples for class {a}; every class in "
                f"[0, num_classes) must have >= 1 support point."
            )
        prototypes.append(_poincare_mean(lifted[mask], c, dim=0))
    return torch.stack(prototypes, dim=0)


def _hyp_protonet_predict(
    p_support: Tensor,
    support_y: Tensor,
    p_query: Tensor,
    num_classes: int,
) -> Tensor:

    d_eff = p_support.shape[-1]
    c = auto_select_c(d_eff)
    max_norm = (1.0 - _BALL_EPS) / math.sqrt(c)
    temperature = 1.0  # official train_protonet.py default, never overridden

    support_lifted = _clamp_ball_norm(_poincare_exp0(p_support, c), max_norm)  # (n_s, d_eff)
    query_lifted = _clamp_ball_norm(_poincare_exp0(p_query, c), max_norm)  # (n_q, d_eff)

    prototypes = _poincare_prototypes(support_lifted, support_y, num_classes, c)

    d2 = _poincare_distance(
        query_lifted.unsqueeze(1), prototypes.unsqueeze(0), c
    ) ** 2  # (n_q, C)
    posteriors = F.softmax(-d2 / temperature, dim=-1)
    return posteriors.argmax(dim=-1)


def _taylor_tanh(x: Tensor) -> Tensor:
    """Order-5 Taylor truncation of tanh, as in ``ts_pmath.tanh``."""
    x2 = x * x
    x3 = x2 * x
    return x - x3 / 3.0 + (2.0 / 15.0) * x3 * x2


def _taylor_artanh(x: Tensor) -> Tensor:
    """Order-5 Taylor truncation of artanh, as in ``ts_pmath.artanh``."""
    x2 = x * x
    x3 = x2 * x
    return x + x3 / 3.0 + x3 * x2 / 5.0


def _poincare_exp0_taylor(v: Tensor, c: float) -> Tensor:
    """:func:`_poincare_exp0` with tanh replaced by :func:`_taylor_tanh`."""
    sqrt_c = math.sqrt(c)
    norm = _safe_norm(v)
    return _taylor_tanh(sqrt_c * norm) * v / (sqrt_c * norm)


def _poincare_distance_taylor(x: Tensor, y: Tensor, c: float) -> Tensor:
    """Taylor-HNN distance (``ts_pmath.dist``): exact Mobius addition followed by
    the truncated artanh, 2/sqrt(c) * artanh_T(sqrt(c) |(-x) (+)_c y|).
    """
    sqrt_c = math.sqrt(c)
    diff = _mobius_add(-x, y, c)
    norm_diff = _safe_norm(diff).squeeze(-1)
    arg = torch.clamp(sqrt_c * norm_diff, max=1.0 - 1e-5)
    return (2.0 / sqrt_c) * _taylor_artanh(arg)


def _taylor_hnn_predict(
    p_support: Tensor,
    support_y: Tensor,
    p_query: Tensor,
    num_classes: int,
) -> Tensor:

    d_eff = p_support.shape[-1]
    c = auto_select_c(d_eff)
    max_norm = (1.0 - _BALL_EPS) / math.sqrt(c)
    temperature = 1.0

    support_lifted = _clamp_ball_norm(_poincare_exp0_taylor(p_support, c), max_norm)
    query_lifted = _clamp_ball_norm(_poincare_exp0_taylor(p_query, c), max_norm)

    prototypes = _poincare_prototypes(support_lifted, support_y, num_classes, c)

    d2 = _poincare_distance_taylor(
        query_lifted.unsqueeze(1), prototypes.unsqueeze(0), c
    ) ** 2  # (n_q, C)
    posteriors = F.softmax(-d2 / temperature, dim=-1)
    return posteriors.argmax(dim=-1)


_BUSEMANN_CURVATURE: float = 1.0  # official HBL.py --curv default


def _buse_distance(query: Tensor, prototypes: Tensor) -> Tensor:
    """Busemann distance log(|g - p|^2 / (1 - |p|^2)) from each query p to each
    ideal point g (``hyperbolicLoss.buse_distance_array``), shape ``(n_q, C)``.
    """
    data_norm2 = (query * query).sum(dim=-1, keepdim=True)  # (n_q, 1)
    denom = torch.clamp(1.0 - data_norm2, min=_DENOM_EPS)
    diff2 = torch.cdist(query, prototypes) ** 2  # (n_q, C)
    return torch.log(diff2 / denom)


def _busemann_predict(
    p_support: Tensor,
    support_y: Tensor,
    p_query: Tensor,
    num_classes: int,
) -> Tensor:

    c = _BUSEMANN_CURVATURE
    max_norm = 1.0 - _BALL_EPS  # radius 1/sqrt(c) = 1 at c=1.0

    support_lifted = _clamp_ball_norm(_poincare_exp0(p_support, c), max_norm)
    query_lifted = _clamp_ball_norm(_poincare_exp0(p_query, c), max_norm)

    prototypes = _poincare_prototypes(support_lifted, support_y, num_classes, c)

    buse_d = _buse_distance(query_lifted, prototypes)  # (n_q, C)
    return buse_d.argmin(dim=-1)


def _tim_hyperbolic_predict(
    support_raw_feats: Tensor,
    support_y: Tensor,
    query_raw_feats: Tensor,
    config: HTIMConfig,
) -> Tensor:

    z_support = plain_l2_normalize(support_raw_feats)
    z_query = plain_l2_normalize(query_raw_feats)

    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, config.d_eff)

    support_h = hyperbolic_lift(z_support, u_pr, config.K)
    query_h = hyperbolic_lift(z_query, u_pr, config.K)

    cfg_rsgd = dataclasses.replace(config, optimizer="rsgd")
    result = htim_adapt(support_h, support_y, query_h, cfg_rsgd)
    return result.predictions


def run_extra_hyperbolic_methods(
    support_raw_feats: Tensor,
    support_y: Tensor,
    query_raw_feats: Tensor,
    query_y: Tensor,
    base_mean: Tensor,
    config: HTIMConfig,
) -> Dict[str, float]:

    num_classes = int(support_y.max().item()) + 1
    query_y_np = query_y.detach().cpu().numpy()

    # Shared CL2N conditioning + PCA fit for methods 1-3 (once)
    z_support = cl2n_condition(support_raw_feats, base_mean)
    z_query = cl2n_condition(query_raw_feats, base_mean)
    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, config.d_eff)

    p_support = z_support @ u_pr  # (n_s, d_eff), Euclidean, pre-lift
    p_query = z_query @ u_pr  # (n_q, d_eff), Euclidean, pre-lift

    results: Dict[str, float] = {}

    # 1. Hyp-ProtoNet
    preds_protonet = _hyp_protonet_predict(p_support, support_y, p_query, num_classes)
    results["Hyp-ProtoNet"] = macro_f1(
        query_y_np, preds_protonet.detach().cpu().numpy(), num_classes
    )

    # 2. Taylor-HNN
    preds_taylor = _taylor_hnn_predict(p_support, support_y, p_query, num_classes)
    results["Taylor-HNN"] = macro_f1(
        query_y_np, preds_taylor.detach().cpu().numpy(), num_classes
    )

    # 3. Hyp-Busemann (adapted, see _busemann_predict)
    preds_busemann = _busemann_predict(p_support, support_y, p_query, num_classes)
    results["Hyp-Busemann"] = macro_f1(
        query_y_np, preds_busemann.detach().cpu().numpy(), num_classes
    )

    # 4. TIM (hyperbolic): own plain-L2 conditioning/PCA, R-SGD
    preds_tim_hyp = _tim_hyperbolic_predict(
        support_raw_feats, support_y, query_raw_feats, config
    )
    results["TIM (hyperbolic)"] = macro_f1(
        query_y_np, preds_tim_hyp.detach().cpu().numpy(), num_classes
    )

    return results


if __name__ == "__main__":
    torch.manual_seed(0)

    D_AMBIENT = 64
    N_CLASSES = 5
    D_EFF = 8
    SHOTS = 5
    QUERY_PER_CLASS = 15
    CLUSTER_SEPARATION = 8.0
    CLUSTER_STD = 1.0
    K = -1.0

    class_centers = torch.randn(N_CLASSES, D_AMBIENT) * CLUSTER_SEPARATION
    support_feats_list: List[Tensor] = []
    support_labels_list: List[Tensor] = []
    query_feats_list: List[Tensor] = []
    query_labels_list: List[Tensor] = []
    for c in range(N_CLASSES):
        s = class_centers[c] + CLUSTER_STD * torch.randn(SHOTS, D_AMBIENT)
        q = class_centers[c] + CLUSTER_STD * torch.randn(QUERY_PER_CLASS, D_AMBIENT)
        support_feats_list.append(s)
        support_labels_list.append(torch.full((SHOTS,), c, dtype=torch.long))
        query_feats_list.append(q)
        query_labels_list.append(torch.full((QUERY_PER_CLASS,), c, dtype=torch.long))
    support_feats = torch.cat(support_feats_list, dim=0)
    support_labels = torch.cat(support_labels_list, dim=0)
    query_feats = torch.cat(query_feats_list, dim=0)
    query_labels = torch.cat(query_labels_list, dim=0)

    base_pool = torch.cat(
        [class_centers[c] + CLUSTER_STD * torch.randn(200, D_AMBIENT) for c in range(N_CLASSES)],
        dim=0,
    )
    base_mean = base_pool.mean(dim=0)

    cfg = HTIMConfig(K=K, d_eff=D_EFF, T=50, beta=0.05, xi=0.1, omega=0.01, tau=10.0, optimizer="radam")

    scores = run_extra_hyperbolic_methods(
        support_feats, support_labels, query_feats, query_labels, base_mean, cfg
    )
    for method, score in scores.items():
        print(f"{method:<18s} macro-F1 = {score:.4f}")
        assert math.isfinite(score) and 0.0 <= score <= 1.0, (method, score)

    # Taylor-HNN truncation error of exp_0 and the distance to the origin.
    c_probe = auto_select_c(D_EFF)
    rand_dir = torch.randn(D_EFF)
    rand_dir = rand_dir / rand_dir.norm()
    origin = torch.zeros(1, D_EFF)

    print(f"{'r':>6s}  {'exp0 gap':>10s}  {'dist rel err %':>14s}")
    for r in [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
        v = (rand_dir * r).unsqueeze(0)
        exact_pt = _poincare_exp0(v, c_probe)
        gap = (exact_pt - _poincare_exp0_taylor(v, c_probe)).norm(p=2).item()
        exact_d = _poincare_distance(origin, exact_pt, c_probe).item()
        taylor_d = _poincare_distance_taylor(origin, exact_pt, c_probe).item()
        rel_err = abs(exact_d - taylor_d) / max(exact_d, 1e-12)
        print(f"{r:6.2f}  {gap:10.2e}  {100.0 * rel_err:14.3f}")
        if r == 0.01:
            assert rel_err < 0.01, f"Taylor distance rel. error {rel_err:.4f} at r={r}"

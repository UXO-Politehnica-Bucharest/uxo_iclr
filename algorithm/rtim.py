import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from algorithm.manifold import (
    apply_eta,
    exp_map,
    lorentz_centroid,
    lorentz_distance,
    minkowski_dot,
    proj_tangent,
    reproject,
    RiemannianAdam,
)

__all__ = [
    "HTIMConfig",
    "HTIMResult",
    "cl2n_condition",
    "plain_l2_normalize",
    "fit_pca_projection",
    "fit_pca_projection_support_only",
    "hyperbolic_lift",
    "htim_adapt",
]

# Manifold-constraint residual tolerance |<w_a, w_a>_L - 1/K|.
_RESIDUAL_TOLERANCE: float = 1e-6
_COMPUTE_DTYPE: torch.dtype = torch.float64
_LOG_EPS: float = 1e-12


@dataclass
class HTIMConfig:
    """Structured hyperparameters for the H-TIM adaptation engine."""

    K: float
    d_eff: int
    T: int = 50
    beta: float = 0.05
    xi: float = 0.1
    omega: float = 0.01
    tau: float = 10.0
    rho1: float = 0.9
    rho2: float = 0.999
    use_query_in_mi: bool = True
    optimizer: Literal["radam", "rsgd"] = "radam"
    include_marginal_entropy: bool = True
    ce_weight: Optional[float] = None
    marginal_h_weight: Optional[float] = None
    cond_h_weight: Optional[float] = None

    def __post_init__(self) -> None:
        if self.K >= 0:
            raise ValueError(f"Curvature K must be negative, got {self.K}.")
        if self.d_eff < 1:
            raise ValueError(f"d_eff must be >= 1, got {self.d_eff}.")
        if self.T < 0:
            raise ValueError(f"T must be >= 0, got {self.T}.")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}.")
        if self.optimizer not in ("radam", "rsgd"):
            raise ValueError(f"optimizer must be 'radam' or 'rsgd', got {self.optimizer!r}.")
        independent_weights = (self.ce_weight, self.marginal_h_weight, self.cond_h_weight)
        n_set = sum(w is not None for w in independent_weights)
        if n_set not in (0, 3):
            raise ValueError(
                "ce_weight/marginal_h_weight/cond_h_weight must be set together "
                f"(all-or-nothing independent-weight mode), got {n_set}/3 set."
            )


@dataclass
class HTIMResult:
    """Output of :func:`htim_adapt`."""

    predictions: Tensor
    posteriors: Tensor
    loss_trajectory: List[float]
    max_residual: float
    prototypes: Tensor


def plain_l2_normalize(raw_features: Tensor) -> Tensor:
    """Plain L2 normalization without base-mean centering: z_i = f_i / ||f_i||_2."""
    norm = torch.norm(raw_features, p=2, dim=-1, keepdim=True)
    return raw_features / torch.clamp(norm, min=1e-12)


def cl2n_condition(raw_features: Tensor, base_mean: Tensor) -> Tensor:
    """Centred-L2-Normalization (CL2N): z_i = (f_i - base_mean) / ||f_i - base_mean||_2."""
    return plain_l2_normalize(raw_features - base_mean)


def fit_pca_projection(features: Tensor, d_eff: int) -> Tensor:
    """Fits PCA projection matrix U_PR of top d_eff components via SVD."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (n, d), got shape {tuple(features.shape)}.")
    n, d = features.shape
    max_rank = min(n, d)
    if d_eff > max_rank:
        raise ValueError(
            f"Requested d_eff={d_eff} exceeds the pool's available rank "
            f"min(n={n}, d={d})={max_rank}."
        )
    centred = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return vh[:d_eff, :].transpose(0, 1).contiguous()


def fit_pca_projection_support_only(support_features: Tensor, d_eff: int) -> Tensor:
    return fit_pca_projection(support_features, d_eff)


def _origin(d_eff: int, K: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Manifold origin $\\mathbf{o} = (1/\\sqrt{-K}, 0, \\dots, 0) \\in
    \\mathcal{L}^{d_{eff}}_K$, shape ``(d_eff + 1,)``.
    """
    o = torch.zeros(d_eff + 1, dtype=dtype, device=device)
    o[0] = 1.0 / (-K) ** 0.5
    return o


def hyperbolic_lift(z: Tensor, U_pr: Tensor, K: float) -> Tensor:
    """Lift CL2N-conditioned ambient features onto $\\mathcal{L}^{d_{eff}}_K$.
    """
    projected = z @ U_pr  # (n, d_eff)
    zero_time = torch.zeros_like(projected[..., :1])
    v = torch.cat([zero_time, projected], dim=-1)  # (n, d_eff + 1), tangent at o
    d_eff = U_pr.shape[1]
    o = _origin(d_eff, K, dtype=z.dtype, device=z.device)
    o = o.expand(v.shape[0], -1)
    return exp_map(o, v, K)


def _squared_geodesic_distances(points: Tensor, prototypes: Tensor, K: float) -> Tensor:
    """Vectorized pairwise squared geodesic distances $d_L^2(z_i, w_a)$.

    Args:
        points: ``(n, d_eff + 1)`` points on the manifold.
        prototypes: ``(C, d_eff + 1)`` prototypes on the manifold.
        K: Negative scalar curvature.

    Returns:
        Tensor of shape ``(n, C)``.
    """
    p = points.unsqueeze(1)  # (n, 1, d_eff+1)
    w = prototypes.unsqueeze(0)  # (1, C, d_eff+1)
    dist = lorentz_distance(p, w, K).squeeze(-1)  # (n, C)
    return dist * dist


def _posteriors_from_prototypes(points: Tensor, prototypes: Tensor, tau: float, K: float) -> Tensor:
    """Geodesic softmax posteriors $p_{ia} = \\mathrm{softmax}_a(-\\tau
    d_L^2(z_i, w_a))$, shape ``(n, C)``.
    """
    d2 = _squared_geodesic_distances(points, prototypes, K)
    return F.softmax(-tau * d2, dim=-1)


def _init_prototypes(support_x: Tensor, support_y: Tensor, num_classes: int, K: float) -> Tensor:
    """Closed-form Lorentzian-centroid prototype initialization."""
    prototypes = []
    for a in range(num_classes):
        mask = support_y == a
        if not mask.any():
            raise ValueError(
                f"No support samples for class {a}; every class in "
                f"[0, num_classes) must have >= 1 support point."
            )
        prototypes.append(lorentz_centroid(support_x[mask], K, dim=-2))
    return torch.stack(prototypes, dim=0)


def _max_manifold_residual(W: Tensor, K: float) -> float:
    """Max absolute manifold-constraint residual |<w_a, w_a>_L - 1/K| across prototypes."""
    dot = minkowski_dot(W.detach(), W.detach(), keepdim=False)
    residual = (dot - 1.0 / K).abs()
    return float(residual.max().item())


def _total_loss(
    W: Tensor,
    support_x: Tensor,
    support_y: Tensor,
    pool_x: Tensor,
    config: HTIMConfig,
) -> Tensor:
    """Total H-TIM transductive objective function."""
    K = config.K
    tau = config.tau

    # L_CE on S only
    d2_support = _squared_geodesic_distances(support_x, W, K)
    log_p_support = F.log_softmax(-tau * d2_support, dim=-1)
    ce = F.nll_loss(log_p_support, support_y, reduction="mean")

    # L_TIM on U
    d2_pool = _squared_geodesic_distances(pool_x, W, K)
    log_p_pool = F.log_softmax(-tau * d2_pool, dim=-1)
    p_pool = log_p_pool.exp()
    h_cond = -(p_pool * log_p_pool).sum(dim=-1).mean()
    if config.include_marginal_entropy:
        p_hat = p_pool.mean(dim=0)
        h_marg = -(p_hat * torch.log(torch.clamp(p_hat, min=_LOG_EPS))).sum()
    else:
        h_marg = None

    # Origin-centering regularizer
    o = _origin(W.shape[-1] - 1, K, dtype=W.dtype, device=W.device).expand_as(W)
    d2_reg = lorentz_distance(W, o, K).squeeze(-1) ** 2
    reg = 0.5 * config.omega * d2_reg.sum()

    if config.ce_weight is not None:
        tim_term = config.cond_h_weight * h_cond
        if h_marg is not None:
            tim_term = tim_term - config.marginal_h_weight * h_marg
        return config.ce_weight * ce + tim_term + reg

    l_tim = h_cond if h_marg is None else h_cond - h_marg
    return ce + config.xi * l_tim + reg


def htim_adapt(
    support_x: Tensor,
    support_y: Tensor,
    query_x: Tensor,
    config: HTIMConfig,
) -> HTIMResult:
    """H-TIM transductive adaptation: fits class prototypes W and classifies query set."""
    if support_x.shape[-1] != config.d_eff + 1:
        raise ValueError(
            f"support_x last dim {support_x.shape[-1]} != config.d_eff+1={config.d_eff + 1}."
        )
    if query_x.shape[-1] != config.d_eff + 1:
        raise ValueError(
            f"query_x last dim {query_x.shape[-1]} != config.d_eff+1={config.d_eff + 1}."
        )
    input_dtype = support_x.dtype
    support_x = support_x.detach().to(_COMPUTE_DTYPE)
    query_x = query_x.detach().to(_COMPUTE_DTYPE)
    support_y = support_y.detach().to(torch.long)

    num_classes = int(support_y.max().item()) + 1
    K = config.K

    W0 = _init_prototypes(support_x, support_y, num_classes, K)

    if config.T == 0:
        posteriors = _posteriors_from_prototypes(query_x, W0, config.tau, K)
        predictions = posteriors.argmax(dim=-1)
        residual = _max_manifold_residual(W0, K)
        if residual >= _RESIDUAL_TOLERANCE:
            raise RuntimeError(
                f"Manifold constraint residual {residual:.3e} on the initial "
                f"(T=0) prototypes exceeds tolerance {_RESIDUAL_TOLERANCE:.1e}; "
                f"this indicates a bug in lorentz_centroid initialization, "
                f"not a recoverable numerical fluctuation."
            )
        return HTIMResult(
            predictions=predictions,
            posteriors=posteriors.to(input_dtype),
            loss_trajectory=[],
            max_residual=residual,
            prototypes=W0.to(input_dtype),
        )

    pool_x = torch.cat([support_x, query_x], dim=0) if config.use_query_in_mi else query_x

    W = W0.clone().detach().requires_grad_(True)

    optimizer = (
        RiemannianAdam([W], K=K, beta=config.beta, rho1=config.rho1, rho2=config.rho2)
        if config.optimizer == "radam"
        else None
    )

    loss_trajectory: List[float] = []
    for _ in range(config.T):
        W.grad = None
        loss = _total_loss(W, support_x, support_y, pool_x, config)
        loss.backward()

        if optimizer is not None:
            optimizer.step()
        else:
            # Trivial Riemannian-SGD fallback (ablation row (b)): a single
            # tangent step u = -beta * grad_w L, no momentum / 2nd moment.
            with torch.no_grad():
                riemannian_grad = proj_tangent(W, apply_eta(W.grad), K)
                W.copy_(reproject(exp_map(W, -config.beta * riemannian_grad, K), K))

        loss_trajectory.append(float(loss.item()))

    W_final = W.detach()
    residual = _max_manifold_residual(W_final, K)
    if residual >= _RESIDUAL_TOLERANCE:
        raise RuntimeError(
            f"Manifold constraint residual {residual:.3e} after T={config.T} "
            f"adaptation steps (optimizer={config.optimizer!r}) exceeds "
            f"tolerance {_RESIDUAL_TOLERANCE:.1e}; refusing to return "
            f"predictions from off-manifold prototypes. This indicates a "
            f"numerical bug in the adaptation loop (e.g. a missing "
            f"reproject step), not expected optimizer drift."
        )

    posteriors = _posteriors_from_prototypes(query_x, W_final, config.tau, K)
    predictions = posteriors.argmax(dim=-1)

    return HTIMResult(
        predictions=predictions,
        posteriors=posteriors.to(input_dtype),
        loss_trajectory=loss_trajectory,
        max_residual=residual,
        prototypes=W_final.to(input_dtype),
    )


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
    support_feats, support_labels = [], []
    query_feats, query_labels = [], []
    for c in range(N_CLASSES):
        s = class_centers[c] + CLUSTER_STD * torch.randn(SHOTS, D_AMBIENT)
        q = class_centers[c] + CLUSTER_STD * torch.randn(QUERY_PER_CLASS, D_AMBIENT)
        support_feats.append(s)
        support_labels.append(torch.full((SHOTS,), c, dtype=torch.long))
        query_feats.append(q)
        query_labels.append(torch.full((QUERY_PER_CLASS,), c, dtype=torch.long))
    support_feats = torch.cat(support_feats, dim=0)
    support_labels = torch.cat(support_labels, dim=0)
    query_feats = torch.cat(query_feats, dim=0)
    query_labels = torch.cat(query_labels, dim=0)

    # Base mean from a separately drawn pool of the same mixture.
    base_pool = torch.cat(
        [class_centers[c] + CLUSTER_STD * torch.randn(200, D_AMBIENT) for c in range(N_CLASSES)],
        dim=0,
    )
    base_mean = base_pool.mean(dim=0)

    all_feats = torch.cat([support_feats, query_feats], dim=0)
    z_all = cl2n_condition(all_feats, base_mean)
    U_pr = fit_pca_projection(z_all, d_eff=D_EFF)
    lifted_all = hyperbolic_lift(z_all, U_pr, K=K)
    n_support = support_feats.shape[0]
    support_x = lifted_all[:n_support]
    query_x = lifted_all[n_support:]

    def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
        return float((preds == labels).float().mean().item())

    def run(name: str, cfg: HTIMConfig) -> HTIMResult:
        result = htim_adapt(support_x, support_labels, query_x, cfg)
        acc = accuracy(result.predictions, query_labels)
        print(f"{name:24s} acc={acc:.4f}  residual={result.max_residual:.2e}")
        return result

    adapt_kw = dict(K=K, d_eff=D_EFF, T=50, beta=0.05, xi=0.1, omega=0.01, tau=10.0)

    result0 = run("T=0", HTIMConfig(K=K, d_eff=D_EFF, T=0))
    acc0 = accuracy(result0.predictions, query_labels)
    assert acc0 > 0.70, f"T=0 accuracy {acc0:.4f} on well-separated clusters"

    result1 = run("T=50 radam", HTIMConfig(**adapt_kw, optimizer="radam"))
    acc1 = accuracy(result1.predictions, query_labels)
    traj1 = result1.loss_trajectory
    dot_final = minkowski_dot(result1.prototypes, result1.prototypes, keepdim=False)
    manual_residual = float((dot_final - 1.0 / K).abs().max().item())
    assert acc1 >= acc0, f"T=50 acc={acc1:.4f} < T=0 acc={acc0:.4f}"
    assert manual_residual < _RESIDUAL_TOLERANCE, f"manifold residual {manual_residual:.3e}"
    assert traj1[-1] <= traj1[0] + 1e-6, "loss increased over adaptation"

    run("T=50 rsgd", HTIMConfig(**adapt_kw, optimizer="rsgd"))

    result3 = run("T=50 query-only MI", HTIMConfig(**adapt_kw, optimizer="radam", use_query_in_mi=False))
    assert any(abs(a - b) > 1e-9 for a, b in zip(traj1, result3.loss_trajectory, strict=True))

    result4 = run(
        "T=50 no marginal H",
        HTIMConfig(**adapt_kw, optimizer="radam", include_marginal_entropy=False),
    )
    assert any(abs(a - b) > 1e-9 for a, b in zip(traj1, result4.loss_trajectory, strict=True))

    z_plain_l2 = plain_l2_normalize(all_feats)
    plain_l2_norms = z_plain_l2.norm(dim=-1)
    assert torch.allclose(plain_l2_norms, torch.ones_like(plain_l2_norms), atol=1e-5)
    assert not torch.allclose(z_plain_l2, z_all)

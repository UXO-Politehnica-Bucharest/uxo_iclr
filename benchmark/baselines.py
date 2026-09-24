import dataclasses
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
)
from benchmark.evaluator import macro_f1

__all__ = ["run_all_methods"]

# log(0) guard for the marginal-entropy term hat{p}_a log hat{p}_a in the
# Euclidean TIM objective (mirrors algorithm/rtim.py's own _LOG_EPS).
_LOG_EPS: float = 1e-12


def _class_means(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    """Per-class mean feature vector.

    Args:
        features: ``(n, d)`` feature pool.
        labels: ``(n,)`` int64 contiguous zero-indexed class labels.
        num_classes: Number of classes C.

    Returns:
        Tensor of shape ``(C, d)``.
    """
    means = []
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            raise ValueError(
                f"No support samples for class {c}; every class in "
                f"[0, num_classes) must have >= 1 support point."
            )
        means.append(features[mask].mean(dim=0))
    return torch.stack(means, dim=0)


def _euclidean_tim_predict(
    support_z: Tensor,
    support_y: Tensor,
    query_z: Tensor,
    num_classes: int,
    config: HTIMConfig,
) -> Tensor:
    """TIM (Euclidean) baseline: flat/Euclidean transductive Information Maximization."""
    W = _class_means(support_z, support_y, num_classes).clone().detach().requires_grad_(True)

    if config.T > 0:
        pool_z = torch.cat([support_z, query_z], dim=0) if config.use_query_in_mi else query_z
        optimizer = torch.optim.Adam([W], lr=config.beta)

        for _ in range(config.T):
            optimizer.zero_grad()

            d2_support = torch.cdist(support_z, W) ** 2  # (n_s, C)
            log_p_support = F.log_softmax(-config.tau * d2_support, dim=-1)
            ce = F.nll_loss(log_p_support, support_y, reduction="mean")

            d2_pool = torch.cdist(pool_z, W) ** 2  # (n_u, C)
            log_p_pool = F.log_softmax(-config.tau * d2_pool, dim=-1)
            p_pool = log_p_pool.exp()
            h_cond = -(p_pool * log_p_pool).sum(dim=-1).mean()
            p_hat = p_pool.mean(dim=0)
            h_marg = -(p_hat * torch.log(torch.clamp(p_hat, min=_LOG_EPS))).sum()

            if config.ce_weight is not None:
                loss = config.ce_weight * ce + config.cond_h_weight * h_cond - config.marginal_h_weight * h_marg
            else:
                loss = ce + config.xi * (h_cond - h_marg)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        d2_query = torch.cdist(query_z, W) ** 2  # (n_q, C)
        predictions = (-config.tau * d2_query).argmax(dim=-1)
    return predictions


_TIM_ORIGINAL_TAU: float = 7.5
_TIM_ORIGINAL_LR: float = 1e-4
_TIM_ORIGINAL_ITER: int = 1000
_TIM_ORIGINAL_CE_WEIGHT: float = 0.1
_TIM_ORIGINAL_MARGINAL_H_WEIGHT: float = 1.0
_TIM_ORIGINAL_COND_H_WEIGHT: float = 0.1


def _euclidean_tim_original_predict(
    support_z: Tensor,
    support_y: Tensor,
    query_z: Tensor,
    num_classes: int,
) -> Tensor:
    """TIM (Euclidean, original hyperparameters) from Boudiaf et al. (NeurIPS 2020)."""
    W = _class_means(support_z, support_y, num_classes).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([W], lr=_TIM_ORIGINAL_LR)
    y_s_one_hot = F.one_hot(support_y, num_classes).float()

    for _ in range(_TIM_ORIGINAL_ITER):
        optimizer.zero_grad()

        d2_support = torch.cdist(support_z, W) ** 2
        log_p_support = F.log_softmax(-_TIM_ORIGINAL_TAU * d2_support, dim=-1)
        ce = -(y_s_one_hot * log_p_support).sum(dim=-1).mean()

        d2_query = torch.cdist(query_z, W) ** 2
        log_p_query = F.log_softmax(-_TIM_ORIGINAL_TAU * d2_query, dim=-1)
        p_query = log_p_query.exp()
        h_cond = -(p_query * log_p_query).sum(dim=-1).mean()
        p_hat = p_query.mean(dim=0)
        h_marg = -(p_hat * torch.log(torch.clamp(p_hat, min=_LOG_EPS))).sum()

        loss = _TIM_ORIGINAL_CE_WEIGHT * ce - (
            _TIM_ORIGINAL_MARGINAL_H_WEIGHT * h_marg - _TIM_ORIGINAL_COND_H_WEIGHT * h_cond
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        d2_query = torch.cdist(query_z, W) ** 2  # (n_q, C)
        predictions = (-_TIM_ORIGINAL_TAU * d2_query).argmax(dim=-1)
    return predictions


def run_all_methods(
    support_raw_feats: Tensor,
    support_y: Tensor,
    query_raw_feats: Tensor,
    query_y: Tensor,
    base_mean: Tensor,
    config: HTIMConfig,
) -> Dict[str, float]:
    """Run all 4 baselines on one episode's raw features and return per-method macro-F1."""
    num_classes = int(support_y.max().item()) + 1
    query_y_np = query_y.detach().cpu().numpy()

    # Shared CL2N conditioning
    z_support = cl2n_condition(support_raw_feats, base_mean)
    z_query = cl2n_condition(query_raw_feats, base_mean)

    # Shared PCA fit on the episode pool U = S union Q (once)
    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, config.d_eff)

    # Euclidean PCA-projected features, shared by methods 1-2.
    p_support = z_support @ u_pr  # (n_s, d_eff)
    p_query = z_query @ u_pr  # (n_q, d_eff)

    results: Dict[str, float] = {}

    # 1. SimpleShot: Euclidean nearest-class-centroid, no transduction
    proto_ss = _class_means(p_support, support_y, num_classes)
    d2_ss = torch.cdist(p_query, proto_ss) ** 2
    preds_ss = d2_ss.argmin(dim=-1)
    results["SimpleShot"] = macro_f1(
        query_y_np, preds_ss.detach().cpu().numpy(), num_classes
    )

    # 2. TIM (Euclidean): flat transductive Information Maximization
    preds_tim = _euclidean_tim_predict(p_support, support_y, p_query, num_classes, config)
    results["TIM (Euclidean)"] = macro_f1(
        query_y_np, preds_tim.detach().cpu().numpy(), num_classes
    )

    # Hyperbolic lift, shared by methods 3-4
    support_h = hyperbolic_lift(z_support, u_pr, config.K)
    query_h = hyperbolic_lift(z_query, u_pr, config.K)

    # 3. Hyp-SimpleShot: closed-form Lorentzian centroid, T=0
    hyp_simpleshot_cfg = dataclasses.replace(config, T=0)
    result_hss = htim_adapt(support_h, support_y, query_h, hyp_simpleshot_cfg)
    results["Hyp-SimpleShot"] = macro_f1(
        query_y_np, result_hss.predictions.detach().cpu().numpy(), num_classes
    )

    # 4. LorentzTIM: full proposed method (T=50, R-Adam by default)
    result_full = htim_adapt(support_h, support_y, query_h, config)
    results["LorentzTIM"] = macro_f1(
        query_y_np, result_full.predictions.detach().cpu().numpy(), num_classes
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

    scores = run_all_methods(support_feats, support_labels, query_feats, query_labels, base_mean, cfg)
    for method, score in scores.items():
        print(f"{method:<18s} macro-F1 = {score:.4f}")

    assert scores["LorentzTIM"] > 0.70, f"LorentzTIM macro-F1 {scores['LorentzTIM']:.4f}"
    assert scores["SimpleShot"] > 0.70, f"SimpleShot macro-F1 {scores['SimpleShot']:.4f}"

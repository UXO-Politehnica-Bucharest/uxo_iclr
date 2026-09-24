"""Ablation suite for LorentzTIM (Table 5): CL2N vs. plain L2, curvature
K -> 0, R-Adam vs. R-SGD, MI-pool and entropy variants, T=0, and the
Euclidean twin.
"""

import dataclasses
import sys
from pathlib import Path
from typing import Dict

import torch
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

__all__ = ["run_ablation_suite"]

_ROW_C_NOTE: str = "N/A: requires a hyperparameter search per shot count."

# Curvature for row "(curv)" (K -> 0). htim_adapt is numerically stable down
# to K=-1e-5 but diverges by K=-1e-7 (measured on synthetic data).
K_NEAR_ZERO: float = -1e-5


def _macro_f1_from_config(
    support_h: Tensor,
    support_y: Tensor,
    query_h: Tensor,
    query_y_np,
    num_classes: int,
    config: HTIMConfig,
) -> float:
    """Query macro-F1 of ``htim_adapt`` with ``config`` on already-lifted points."""
    result = htim_adapt(support_h, support_y, query_h, config)
    return macro_f1(query_y_np, result.predictions.detach().cpu().numpy(), num_classes)


def run_ablation_suite(
    support_raw_feats: Tensor,
    support_y: Tensor,
    query_raw_feats: Tensor,
    query_y: Tensor,
    base_mean: Tensor,
    base_config: HTIMConfig,
    euclidean_twin_f1: float,
) -> Dict[str, float]:
    """Table 5 ablation rows for one episode: per-row macro-F1 and deltas.

    Each row changes one thing relative to ``base_config`` (the full
    LorentzTIM config): row (a) swaps CL2N for plain L2 and refits PCA, the
    others vary one config field via ``dataclasses.replace``. Row (c) has no
    defined baseline and is NaN; row (h) is the caller-supplied TIM (Euclidean)
    score for the same episode.

    Returns:
        Macro-F1 per row key ("Full Model", "(a)", "(curv)", "(b)" ... "(h)"),
        plus "(c)_note" and a "deltas" dict of ``full_model_f1 - row_f1`` for
        every row except (c).
    """
    num_classes = int(support_y.max().item()) + 1
    query_y_np = query_y.detach().cpu().numpy()

    # Conditioning, PCA and lift shared by the Full Model and rows (b), (d)-(g).
    z_support = cl2n_condition(support_raw_feats, base_mean)
    z_query = cl2n_condition(query_raw_feats, base_mean)
    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, base_config.d_eff)
    support_h = hyperbolic_lift(z_support, u_pr, base_config.K)
    query_h = hyperbolic_lift(z_query, u_pr, base_config.K)

    results: Dict[str, float] = {}

    # Full Model (reference point)
    results["Full Model"] = _macro_f1_from_config(
        support_h, support_y, query_h, query_y_np, num_classes, base_config
    )

    # (a) CL2N -> plain L2: PCA and lift are refit on the new features.
    z_support_a = plain_l2_normalize(support_raw_feats)
    z_query_a = plain_l2_normalize(query_raw_feats)
    pool_z_a = torch.cat([z_support_a, z_query_a], dim=0)
    u_pr_a = fit_pca_projection(pool_z_a, base_config.d_eff)
    support_h_a = hyperbolic_lift(z_support_a, u_pr_a, base_config.K)
    query_h_a = hyperbolic_lift(z_query_a, u_pr_a, base_config.K)
    results["(a)"] = _macro_f1_from_config(
        support_h_a, support_y, query_h_a, query_y_np, num_classes, base_config
    )

    # (curv) K -> 0 with psi*_S fixed; CL2N and PCA do not depend on K.
    support_h_curv = hyperbolic_lift(z_support, u_pr, K_NEAR_ZERO)
    query_h_curv = hyperbolic_lift(z_query, u_pr, K_NEAR_ZERO)
    cfg_curv = dataclasses.replace(base_config, K=K_NEAR_ZERO)
    results["(curv)"] = _macro_f1_from_config(
        support_h_curv, support_y, query_h_curv, query_y_np, num_classes, cfg_curv
    )

    # (b) R-Adam -> R-SGD
    cfg_b = dataclasses.replace(base_config, optimizer="rsgd")
    results["(b)"] = _macro_f1_from_config(
        support_h, support_y, query_h, query_y_np, num_classes, cfg_b
    )

    # (c) per-shot psi_S* -> fixed psi: not executable, see _ROW_C_NOTE.
    results["(c)"] = float("nan")
    results["(c)_note"] = _ROW_C_NOTE  # type: ignore[assignment]

    # (d) MI pool Q -> S union Q (the searched configs always use Q only).
    cfg_d = dataclasses.replace(base_config, use_query_in_mi=True)
    results["(d)"] = _macro_f1_from_config(
        support_h, support_y, query_h, query_y_np, num_classes, cfg_d
    )

    # (e) w/o marginal entropy hat H(Y)
    cfg_e = dataclasses.replace(base_config, include_marginal_entropy=False)
    results["(e)"] = _macro_f1_from_config(
        support_h, support_y, query_h, query_y_np, num_classes, cfg_e
    )

    # (f) cross-entropy only. In independent-weight mode xi is ignored, so the
    # MI terms are removed by zeroing their weights.
    cfg_f = dataclasses.replace(base_config, marginal_h_weight=0.0, cond_h_weight=0.0)
    results["(f)"] = _macro_f1_from_config(
        support_h, support_y, query_h, query_y_np, num_classes, cfg_f
    )

    # (g) T=0 (Hyp-SimpleShot, no adaptation)
    cfg_g = dataclasses.replace(base_config, T=0)
    results["(g)"] = _macro_f1_from_config(
        support_h, support_y, query_h, query_y_np, num_classes, cfg_g
    )

    # (h) Euclidean twin, supplied by the caller.
    results["(h)"] = float(euclidean_twin_f1)

    # Deltas relative to the Full Model, for every row except (c).
    full_f1 = results["Full Model"]
    deltas: Dict[str, float] = {
        key: full_f1 - value
        for key, value in results.items()
        if key not in ("Full Model", "(c)", "(c)_note")
    }
    results["deltas"] = deltas  # type: ignore[assignment]

    return results


if __name__ == "__main__":
    import math

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
    support_feats_list = []
    support_labels_list = []
    query_feats_list = []
    query_labels_list = []
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

    # Independent-weight config, as produced by the hyperparameter search.
    base_cfg = HTIMConfig(
        K=K, d_eff=D_EFF, T=50, beta=0.05, omega=0.01, tau=10.0, optimizer="radam",
        use_query_in_mi=False, ce_weight=1.0, marginal_h_weight=0.1, cond_h_weight=0.1,
    )

    # Row (h) normally comes from the "TIM (Euclidean)" score of
    # benchmark.baselines.run_all_methods; here a flat TIM is run directly.
    import torch.nn.functional as F

    def _euclidean_twin_f1() -> float:
        """Flat-geometry TIM macro-F1 on the synthetic episode."""
        z_s = cl2n_condition(support_feats, base_mean)
        z_q = cl2n_condition(query_feats, base_mean)
        pool_z = torch.cat([z_s, z_q], dim=0)
        u_pr = fit_pca_projection(pool_z, D_EFF)
        p_s = z_s @ u_pr
        p_q = z_q @ u_pr
        means = []
        for c in range(N_CLASSES):
            means.append(p_s[support_labels == c].mean(dim=0))
        W = torch.stack(means, dim=0).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([W], lr=base_cfg.beta)
        pool_p = torch.cat([p_s, p_q], dim=0)
        for _ in range(base_cfg.T):
            optimizer.zero_grad()
            d2_s = torch.cdist(p_s, W) ** 2
            log_p_s = F.log_softmax(-base_cfg.tau * d2_s, dim=-1)
            ce = F.nll_loss(log_p_s, support_labels, reduction="mean")
            d2_pool = torch.cdist(pool_p, W) ** 2
            log_p_pool = F.log_softmax(-base_cfg.tau * d2_pool, dim=-1)
            p_pool = log_p_pool.exp()
            h_cond = -(p_pool * log_p_pool).sum(dim=-1).mean()
            p_hat = p_pool.mean(dim=0)
            h_marg = -(p_hat * torch.log(torch.clamp(p_hat, min=1e-12))).sum()
            loss = ce + base_cfg.xi * (h_cond - h_marg)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = (-base_cfg.tau * (torch.cdist(p_q, W) ** 2)).argmax(dim=-1)
        return macro_f1(query_labels.cpu().numpy(), preds.cpu().numpy(), N_CLASSES)

    euclidean_twin_f1 = _euclidean_twin_f1()

    results = run_ablation_suite(
        support_feats, support_labels, query_feats, query_labels,
        base_mean, base_cfg, euclidean_twin_f1,
    )

    row_labels = {
        "Full Model": "Full Model (LorentzTIM)",
        "(a)": "(a) CL2N -> plain L2",
        "(curv)": "(curv) Curvature K -> 0 (psi*_S fixed)",
        "(b)": "(b) R-Adam -> R-SGD",
        "(c)": "(c) per-shot psi_S* -> fixed psi",
        "(d)": "(d) MI pool U -> Q only",
        "(e)": "(e) w/o marginal entropy hat H(Y)",
        "(f)": "(f) w/o MI entirely (xi=0)",
        "(g)": "(g) T=0 (Hyp-SimpleShot)",
        "(h)": "(h) Euclidean twin",
    }
    deltas = results["deltas"]

    print(f"{'Row':<38s} {'macro-F1':>10s} {'Delta':>10s}")
    print("-" * 60)
    for key, label in row_labels.items():
        f1 = results[key]
        if key == "(c)":
            print(f"{label:<38s} {'N/A':>10s} {'N/A':>10s}")
        elif key == "Full Model":
            print(f"{label:<38s} {f1:>10.4f} {'(ref)':>10s}")
        else:
            delta = deltas[key]
            print(f"{label:<38s} {f1:>10.4f} {delta:>+10.4f}")

    for key in row_labels:
        value = results[key]
        if key == "(c)":
            assert math.isnan(value)
        else:
            assert math.isfinite(value) and 0.0 <= value <= 1.0, (key, value)
            if key != "Full Model":
                assert math.isfinite(deltas[key]), (key, deltas[key])

    assert set(deltas.keys()) == {"(a)", "(curv)", "(b)", "(d)", "(e)", "(f)", "(g)", "(h)"}

import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from algorithm.rtim import HTIMConfig, cl2n_condition, fit_pca_projection
from benchmark.baselines import _class_means
from benchmark.evaluator import macro_f1

__all__ = ["run_extra_euclidean_methods"]


_LAPLACIAN_KNN_K: int = 5

_LAPLACIAN_LAMBDA: float = 0.7

_LAPLACIAN_ITERS: int = 20

_LOG_EPS: float = 1e-12

_PT_MAP_BETA: float = 0.5

_PT_MAP_SHIFT_EPS: float = 1e-6

_PT_MAP_SINKHORN_ITERS: int = 10

_PT_MAP_SINKHORN_REG_SCALE: float = 1.0


def _protonet_predict(
    support_z: Tensor, support_y: Tensor, query_z: Tensor, num_classes: int
) -> Tensor:
    """Prototypical Networks: nearest class mean under squared Euclidean distance.

    Prototypes are support means in the shared CL2N+PCA space; queries are
    classified by argmax of softmax(-||z - c_a||^2). Purely inductive.

    Returns:
        ``(n_q,)`` predicted labels.
    """
    prototypes = _class_means(support_z, support_y, num_classes)  # (C, d_eff)
    d2_query = torch.cdist(query_z, prototypes) ** 2  # (n_q, C)
    posteriors = F.softmax(-d2_query, dim=-1)
    return posteriors.argmax(dim=-1)


def _laplacianshot_predict(
    support_z: Tensor,
    support_y: Tensor,
    query_z: Tensor,
    num_classes: int,
    knn_k: int = _LAPLACIAN_KNN_K,
    lam: float = _LAPLACIAN_LAMBDA,
    n_iters: int = _LAPLACIAN_ITERS,
) -> Tensor:

    n_query = query_z.shape[0]
    effective_k = min(knn_k, n_query - 1)

    prototypes = _class_means(support_z, support_y, num_classes)  # (C, d_eff)
    unary = torch.cdist(query_z, prototypes) ** 2  # (n_q, C)
    z = F.softmax(-unary, dim=-1)  # (n_q, C)

    if effective_k >= 1:
        d2_qq = torch.cdist(query_z, query_z) ** 2  # (n_q, n_q)
        d2_qq_no_self = d2_qq + torch.eye(
            n_query, device=query_z.device, dtype=d2_qq.dtype
        ) * torch.finfo(d2_qq.dtype).max
        knn_dists, knn_idx = torch.topk(d2_qq_no_self, k=effective_k, dim=-1, largest=False)

        sigma2 = torch.clamp(knn_dists.mean(), min=_LOG_EPS)
        knn_weights = torch.exp(-knn_dists / sigma2)  # (n_q, k)

        affinity = torch.zeros(n_query, n_query, device=query_z.device, dtype=knn_weights.dtype)
        affinity.scatter_(dim=1, index=knn_idx, src=knn_weights)
        affinity = (affinity + affinity.t()) / 2.0  # symmetrize
    else:
        # Degenerate 1-query-point episode: no meaningful graph.
        affinity = torch.zeros(n_query, n_query, device=query_z.device)

    for _ in range(n_iters):
        z = F.softmax(-unary + lam * (affinity @ z), dim=-1)

    return z.argmax(dim=-1)


def _pt_map_predict(
    support_z: Tensor,
    support_y: Tensor,
    query_z: Tensor,
    num_classes: int,
    beta_pt: float = _PT_MAP_BETA,
    sinkhorn_iters: int = _PT_MAP_SINKHORN_ITERS,
) -> Tensor:

    n_query = query_z.shape[0]

    pool = torch.cat([support_z, query_z], dim=0)
    shift = pool.min()
    support_pt = (support_z - shift + _PT_MAP_SHIFT_EPS).clamp(min=_PT_MAP_SHIFT_EPS) ** beta_pt
    query_pt = (query_z - shift + _PT_MAP_SHIFT_EPS).clamp(min=_PT_MAP_SHIFT_EPS) ** beta_pt

    prototypes = _class_means(support_pt, support_y, num_classes)  # (C, d_eff)
    cost = torch.cdist(query_pt, prototypes) ** 2  # (n_q, C)

    reg = torch.clamp(cost.mean(), min=_LOG_EPS) * _PT_MAP_SINKHORN_REG_SCALE
    kernel = torch.exp(-cost / reg)  # (n_q, C)

    r = torch.full((n_query, 1), 1.0 / n_query, device=query_z.device, dtype=kernel.dtype)
    c = torch.full(
        (num_classes, 1), 1.0 / num_classes, device=query_z.device, dtype=kernel.dtype
    )  # target column (class) marginal
    u = torch.ones(n_query, 1, device=query_z.device, dtype=kernel.dtype)
    v = torch.ones(num_classes, 1, device=query_z.device, dtype=kernel.dtype)
    for _ in range(sinkhorn_iters):
        u = r / torch.clamp(kernel @ v, min=_LOG_EPS)
        v = c / torch.clamp(kernel.t() @ u, min=_LOG_EPS)

    p_assign = u * kernel * v.t()  # (n_q, C), broadcasting diag(u) K diag(v)
    return p_assign.argmax(dim=-1)


def _dsn_predict(
    support_z: Tensor, support_y: Tensor, query_z: Tensor, num_classes: int
) -> Tensor:

    query_z64 = query_z.double()
    dists = torch.empty(query_z.shape[0], num_classes, dtype=torch.float64, device=query_z.device)

    for a in range(num_classes):
        mask = support_y == a
        n_shot_a = int(mask.sum().item())
        if n_shot_a == 0:
            raise ValueError(
                f"No support samples for class {a}; every class in "
                f"[0, num_classes) must have >= 1 support point."
            )
        class_support = support_z[mask].double()  # (n_shot_a, d_eff)
        mean_a = class_support.mean(dim=0, keepdim=True)  # (1, d_eff)
        centered = class_support - mean_a
        query_centered = query_z64 - mean_a  # (n_q, d_eff)

        subspace_dim = min(n_shot_a - 1, centered.shape[-1])
        if subspace_dim <= 0:
            # n_shot_a == 1: rank-0 subspace: projection is the zero vector,
            # distance reduces to nearest-centroid.
            dists[:, a] = (query_centered ** 2).sum(dim=-1)
            continue

        uu, _, _ = torch.svd(centered.t())  # (d_eff, n_shot_a) -> uu: (d_eff, k)
        basis = uu[:, :subspace_dim]  # (d_eff, subspace_dim), orthonormal
        projection = query_centered @ basis @ basis.t()  # (n_q, d_eff)
        dists[:, a] = ((query_centered - projection) ** 2).sum(dim=-1)

    return dists.argmin(dim=-1)


def run_extra_euclidean_methods(
    support_raw_feats: Tensor,
    support_y: Tensor,
    query_raw_feats: Tensor,
    query_y: Tensor,
    base_mean: Tensor,
    config: HTIMConfig,
) -> Dict[str, float]:
   
    num_classes = int(support_y.max().item()) + 1
    query_y_np = query_y.detach().cpu().numpy()

    z_support = cl2n_condition(support_raw_feats, base_mean)
    z_query = cl2n_condition(query_raw_feats, base_mean)

    # Shared PCA fit on the episode pool U = S union Q.
    pool_z = torch.cat([z_support, z_query], dim=0)
    u_pr = fit_pca_projection(pool_z, config.d_eff)
    p_support = z_support @ u_pr  # (n_s, d_eff)
    p_query = z_query @ u_pr  # (n_q, d_eff)

    results: Dict[str, float] = {}

    # 1. ProtoNet
    preds_pn = _protonet_predict(p_support, support_y, p_query, num_classes)
    results["ProtoNet"] = macro_f1(query_y_np, preds_pn.detach().cpu().numpy(), num_classes)

    # 2. LaplacianShot
    preds_ls = _laplacianshot_predict(p_support, support_y, p_query, num_classes)
    results["LaplacianShot"] = macro_f1(query_y_np, preds_ls.detach().cpu().numpy(), num_classes)

    # 3. PT-MAP
    preds_ptmap = _pt_map_predict(p_support, support_y, p_query, num_classes)
    results["PT-MAP"] = macro_f1(query_y_np, preds_ptmap.detach().cpu().numpy(), num_classes)

    # 4. DSN
    preds_dsn = _dsn_predict(p_support, support_y, p_query, num_classes)
    results["DSN"] = macro_f1(query_y_np, preds_dsn.detach().cpu().numpy(), num_classes)

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

    scores = run_extra_euclidean_methods(
        support_feats, support_labels, query_feats, query_labels, base_mean, cfg
    )
    for method, score in scores.items():
        print(f"{method:<18s} macro-F1 = {score:.4f}")

    for method_name in ("ProtoNet", "LaplacianShot", "PT-MAP", "DSN"):
        assert scores[method_name] > 0.70, f"{method_name} macro-F1 {scores[method_name]:.4f}"

"""Pre-transduction geometric diagnostics for embeddings.

Implements three geometric diagnostics evaluated on ambient embeddings:
    1. Relative delta-hyperbolicity (delta_rel)
    2. Intrinsic dimension (d_TwoNN, d_MLE, d_PR)
    3. Cophenetic alignment (rho_coph)
along with a spectrum-matched Gaussian null control generator.
"""

import dataclasses

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "DiagnosticsConfig",
    "cophenetic_alignment",
    "generate_null_control",
    "generate_null_control_full_spectrum",
    "intrinsic_dim_mle",
    "intrinsic_dim_pr",
    "intrinsic_dim_twonn",
    "pairwise_euclidean",
    "relative_delta_hyperbolicity",
    "run_diagnostics",
]


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """Structured hyperparameters for the diagnostics suite."""

    delta_sample_size: int = 60
    delta_n_batches: int = 25
    delta_ci: float = 0.95
    mle_k: int = 10

    linkage_method: str = "average"

    null_control_l2_normalize: bool = True

    random_seed: int = 0


def pairwise_euclidean(Z: np.ndarray) -> np.ndarray:
    """Dense symmetric Euclidean pairwise-distance matrix."""
    Z = np.asarray(Z, dtype=np.float64)
    return squareform(pdist(Z, metric="euclidean"))


def _gromov_product_matrix(D_sub: np.ndarray, w_idx: int) -> np.ndarray:
    """Gromov product matrix A_w(p, q) = 1/2 (D[w,p] + D[w,q] - D[p,q]) (Eq. 1).

    Fully vectorized via broadcasting; O(m^2) for an m x m distance submatrix.
    """
    row = D_sub[w_idx, :]
    return 0.5 * (row[:, None] + row[None, :] - D_sub)


def relative_delta_hyperbolicity(
    Z: np.ndarray | None = None,
    D: np.ndarray | None = None,
    config: DiagnosticsConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Computes relative Gromov delta-hyperbolicity delta_hat_rel(Z) in [0, 1]."""
    config = config or DiagnosticsConfig()
    rng = rng or np.random.default_rng(config.random_seed)
    if D is None:
        if Z is None:
            raise ValueError("Either Z or D must be provided.")
        D = pairwise_euclidean(Z)
    n = D.shape[0]
    diam = float(D.max())
    if diam <= 0.0:
        raise ValueError("Degenerate point set: diameter is zero.")

    m = min(config.delta_sample_size, n)
    with_replacement = n <= config.delta_sample_size

    batch_deltas = np.empty(config.delta_n_batches, dtype=np.float64)
    for b in range(config.delta_n_batches):
        idx = rng.choice(n, size=m, replace=with_replacement)
        D_sub = D[np.ix_(idx, idx)]

        w0 = 0
        z_star = int(np.argmax(D_sub[w0, :]))
        A = _gromov_product_matrix(D_sub, z_star)

        T = np.minimum(A[:, :, None], A[None, :, :])
        diff = T - A[:, None, :]
        delta_batch = float(diff.max())
        batch_deltas[b] = 2.0 * delta_batch / diam

    value = float(batch_deltas.mean())
    alpha = 1.0 - config.delta_ci
    lo, hi = np.percentile(batch_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "value": value,
        "ci95": (float(lo), float(hi)),
        "batch_values": batch_deltas,
        "sample_size": m,
        "n_batches": config.delta_n_batches,
        "bootstrap_only": with_replacement,
        "diam": diam,
    }


def intrinsic_dim_twonn(Z: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimator."""
    Z = np.asarray(Z, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=3).fit(Z)
    dist, _ = nn.kneighbors(Z)
    r1, r2 = dist[:, 1], dist[:, 2]
    valid = r1 > 1e-12
    eta = r2[valid] / r1[valid]
    eta = eta[eta > 1.0 + 1e-12]
    eta_sorted = np.sort(eta)
    m = eta_sorted.shape[0]
    if m < 2:
        raise ValueError("Not enough valid (non-degenerate) points for TwoNN.")
    F_emp = np.arange(1, m + 1) / (m + 1)
    x = np.log(eta_sorted)
    y = -np.log(1.0 - F_emp)
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        raise ValueError("Degenerate TwoNN regression (zero variance in log-ratios).")
    return float(np.sum(x * y) / denom)


def intrinsic_dim_mle(Z: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel MLE intrinsic-dimension estimator."""
    if k < 3:
        raise ValueError("k must be >= 3 for the (k-2)-corrected MLE estimator.")
    Z = np.asarray(Z, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, _ = nn.kneighbors(Z)
    r = dist[:, 1:]
    r = np.clip(r, 1e-12, None)
    r_k = r[:, -1:]
    r_j = r[:, :-1]
    log_ratios = np.log(r_k / r_j)
    sum_log_ratios = np.sum(log_ratios, axis=1)
    valid = sum_log_ratios > 1e-12
    d_hat_per_point = (k - 2) / sum_log_ratios[valid]
    return float(np.mean(d_hat_per_point))


def intrinsic_dim_pr(Z: np.ndarray) -> tuple[float, int]:
    """Participation-ratio intrinsic dimension d_PR and d_eff = round(d_PR)."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    n = Zc.shape[0]
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals**2))
    if s2 <= 0.0:
        raise ValueError("Degenerate covariance spectrum (all-zero eigenvalues).")
    d_pr = (s1**2) / s2
    return d_pr, round(d_pr)


def cophenetic_alignment(
    Z: np.ndarray | None = None,
    D: np.ndarray | None = None,
    config: DiagnosticsConfig | None = None,
) -> dict[str, float]:
    """Cophenetic alignment between pairwise distances and their dendrogram."""
    config = config or DiagnosticsConfig()
    if D is None:
        if Z is None:
            raise ValueError("Either Z or D must be provided.")
        D = pairwise_euclidean(Z)
    condensed = squareform(D, checks=False)
    Z_link = linkage(condensed, method=config.linkage_method)
    _, coph_condensed = cophenet(Z_link, condensed)
    rho_pearson, _ = pearsonr(condensed, coph_condensed)
    rho_spearman, _ = spearmanr(condensed, coph_condensed)
    return {"pearson": float(rho_pearson), "spearman": float(rho_spearman)}


def generate_null_control(
    Z: np.ndarray,
    config: DiagnosticsConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generates spectrum-matched Gaussian null control Z_tilde."""
    config = config or DiagnosticsConfig()
    rng = rng or np.random.default_rng(config.random_seed)
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape

    _, d_eff = intrinsic_dim_pr(Z)
    d_eff = max(2, min(d_eff, d))

    basis = rng.standard_normal(size=(d, d_eff))
    basis, _ = np.linalg.qr(basis)
    X_low = rng.standard_normal(size=(n, d_eff))
    Z_tilde = X_low @ basis.T

    if config.null_control_l2_normalize:
        Z_tilde = Z_tilde / np.linalg.norm(Z_tilde, axis=1, keepdims=True)
    return Z_tilde


def generate_null_control_full_spectrum(
    Z: np.ndarray,
    config: DiagnosticsConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Null control matching Z's covariance spectrum along its top d_eff directions.

    :func:`generate_null_control` uses an isotropic Gaussian, which cannot
    separate hierarchical structure from mere anisotropy (any anisotropic
    embedding beats it on delta_rel and cophenetic correlation). This variant
    samples independent Gaussians along Z's top-d_eff principal axes with Z's
    eigenvalues as variances, reproducing the anisotropy without any tree
    structure.
    """
    config = config or DiagnosticsConfig()
    rng = rng or np.random.default_rng(config.random_seed)
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape

    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    _, d_eff = intrinsic_dim_pr(Z)
    d_eff = max(2, min(d_eff, d))

    top_eigvals = np.clip(eigvals[:d_eff], 0.0, None)
    top_eigvecs = eigvecs[:, :d_eff]

    X_low = rng.standard_normal(size=(n, d_eff)) * np.sqrt(top_eigvals)[None, :]
    Z_tilde = X_low @ top_eigvecs.T

    if config.null_control_l2_normalize:
        Z_tilde = Z_tilde / np.linalg.norm(Z_tilde, axis=1, keepdims=True)
    return Z_tilde


def _run_single(
    Z: np.ndarray,
    config: DiagnosticsConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Compute all three diagnostics on a single point cloud Z."""
    D = pairwise_euclidean(Z)

    delta = relative_delta_hyperbolicity(D=D, config=config, rng=rng)
    d_twonn = intrinsic_dim_twonn(Z)
    d_mle = intrinsic_dim_mle(Z, k=config.mle_k)
    d_pr, d_eff = intrinsic_dim_pr(Z)
    coph = cophenetic_alignment(D=D, config=config)

    return {
        "delta_rel": delta,
        "d_TwoNN": d_twonn,
        "d_MLE": d_mle,
        "d_PR": d_pr,
        "d_eff": d_eff,
        "rho_coph": coph,
    }


def run_diagnostics(
    Z: np.ndarray,
    config: DiagnosticsConfig | None = None,
    null_generator=generate_null_control,
) -> dict[str, object]:
    """All three diagnostics on Z and on a null control Z_tilde.

    ``null_generator`` defaults to the isotropic :func:`generate_null_control`;
    pass :func:`generate_null_control_full_spectrum` to match Z's spectrum.
    """
    config = config or DiagnosticsConfig()
    rng = np.random.default_rng(config.random_seed)

    Z = np.asarray(Z, dtype=np.float64)
    Z_tilde = null_generator(Z, config=config, rng=rng)

    results_Z = _run_single(Z, config, rng)
    results_Zt = _run_single(Z_tilde, config, rng)

    gaps = {
        "delta_delta_rel": results_Z["delta_rel"]["value"] - results_Zt["delta_rel"]["value"],
    }

    return {
        "n_samples": Z.shape[0],
        "ambient_dim": Z.shape[1],
        "Z": results_Z,
        "Z_tilde": results_Zt,
        "gaps": gaps,
        "config": config,
    }


def _make_synthetic_hierarchical_data(
    rng: np.random.Generator,
    n_points: int,
    depth: int,
    ambient_dim: int,
    branch_scale: float,
    branch_decay: float,
    noise_std: float,
) -> np.ndarray:
    """Synthetic point cloud around the leaves of a random binary tree of depth
    ``depth`` (branch length decaying by ``branch_decay`` per level).
    """
    centers = [np.zeros(ambient_dim)]
    scale = branch_scale
    for _ in range(depth):
        new_centers = []
        for c in centers:
            for _ in range(2):
                direction = rng.standard_normal(ambient_dim)
                direction /= np.linalg.norm(direction)
                new_centers.append(c + scale * direction)
        centers = new_centers
        scale *= branch_decay
    leaves = np.stack(centers, axis=0)  # (2**depth, ambient_dim)

    leaf_idx = rng.integers(0, leaves.shape[0], size=n_points)
    noise = rng.normal(scale=noise_std, size=(n_points, ambient_dim))
    return leaves[leaf_idx] + noise


def _fmt_ci(d: dict[str, object]) -> str:
    lo, hi = d["ci95"]
    return (
        f"{d['value']:.4f}  [{lo:.4f}, {hi:.4f}]  (m={d['sample_size']}, B={d['n_batches']})"
    )


def _print_block(name: str, res: dict[str, object]) -> None:
    print(f"{name}")
    print(f"  delta_hat_rel        : {_fmt_ci(res['delta_rel'])}")
    print(f"  d_TwoNN              : {res['d_TwoNN']:.3f}")
    print(f"  d_MLE                : {res['d_MLE']:.3f}")
    print(f"  d_PR                 : {res['d_PR']:.3f}  ->  d_eff = {res['d_eff']}")
    coph = res["rho_coph"]
    print(f"  rho_coph (Pearson)   : {coph['pearson']:.4f}")
    print(f"  rho_coph (Spearman)  : {coph['spearman']:.4f}")


if __name__ == "__main__":
    SEED = 0
    N_POINTS = 200
    AMBIENT_DIM = 32

    master_rng = np.random.default_rng(SEED)
    cfg = DiagnosticsConfig(random_seed=SEED)

    Z_tree = _make_synthetic_hierarchical_data(
        rng=master_rng,
        n_points=N_POINTS,
        depth=4,
        ambient_dim=AMBIENT_DIM,
        branch_scale=3.0,
        branch_decay=0.55,
        noise_std=0.15,
    )
    Z_gauss = master_rng.standard_normal(size=(N_POINTS, AMBIENT_DIM))

    out_tree = run_diagnostics(Z_tree, config=cfg)
    out_gauss = run_diagnostics(Z_gauss, config=cfg)
    _print_block("Z_tree", out_tree["Z"])
    _print_block("Z_tree null", out_tree["Z_tilde"])
    _print_block("Z_gauss", out_gauss["Z"])
    print(f"delta_rel gap (tree - null): {out_tree['gaps']['delta_delta_rel']:+.4f}")

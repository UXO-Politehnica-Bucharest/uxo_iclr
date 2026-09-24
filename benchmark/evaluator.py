"""Evaluation statistics: macro-F1, paired Wilcoxon signed-rank tests with
Holm-Bonferroni correction, and bootstrap confidence intervals.
"""

import itertools
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-F1 over ``range(num_classes)`` (the primary metric).

    Every class has equal weight, which matters on the long-tailed CTX-UXO
    taxonomy. Classes with an undefined precision or recall get F1 = 0
    (scikit-learn ``average="macro"``, ``zero_division=0``).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(num_classes)
    return float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down correction.

    Sorted p-values are adjusted as ``max_{j<=i} (m - j + 1) p_(j)``, clipped to
    1, and returned in the input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        candidate = (m - i) * sorted_p[i]
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(running_max, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _signed_wilcoxon_z(diff: np.ndarray) -> float:
    """Signed Z of the Wilcoxon signed-rank test (normal approximation with tie
    correction, zero differences dropped as in ``zero_method="wilcox"``).

    ``Z > 0`` means the first method of ``diff = a - b`` scores higher. scipy
    only exposes the unsigned statistic, hence this computation. Returns 0.0
    when no non-zero differences remain or the variance is not positive.
    """
    d = diff[diff != 0]
    n = d.shape[0]
    if n == 0:
        return 0.0
    abs_d = np.abs(d)
    ranks = rankdata(abs_d, method="average")
    w_plus = ranks[d > 0].sum()
    mean_w = n * (n + 1) / 4.0
    _, tie_counts = np.unique(abs_d, return_counts=True)
    tie_correction = np.sum(tie_counts**3 - tie_counts) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var_w <= 0:
        return 0.0
    return float((w_plus - mean_w) / np.sqrt(var_w))


def _effect_size_category(r: float) -> str:
    """Categorize |r|: small (>=0.1), medium (>=0.3), large (>=0.5)."""
    if r >= 0.5:
        return "large"
    if r >= 0.3:
        return "medium"
    if r >= 0.1:
        return "small"
    return "negligible"


def _significance_marker(p_holm: float) -> str:
    """Significance marker computed from the Holm-corrected p-value."""
    if p_holm < 0.001:
        return "***"
    if p_holm < 0.01:
        return "**"
    if p_holm < 0.05:
        return "*"
    return "n.s."


def _corrected_stats(raw: dict[str, float], p_holm: float, r_effect: float, n_episodes: int) -> dict:
    return {
        "W": raw["W"],
        "p_raw": raw["p_raw"],
        "p_holm": float(p_holm),
        "Z": raw["Z"],
        "r": float(r_effect),
        "effect_size_category": _effect_size_category(r_effect),
        "significance": _significance_marker(float(p_holm)),
        "n_episodes": n_episodes,
    }


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Uncorrected paired Wilcoxon test of ``a`` vs. ``b``.

    Uses ``zero_method="wilcox"`` and ``method="approx"``, so the p-value is
    consistent with :func:`_signed_wilcoxon_z`.

    Returns:
        ``{"W", "p_raw", "Z"}``; ``Z > 0`` means ``a`` scores higher.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be paired (same shape); got {a.shape} vs {b.shape}.")
    diff = a - b
    if np.all(diff == 0):
        w_stat, p_raw = 0.0, 1.0
    else:
        try:
            w_stat, p_raw = wilcoxon(
                a,
                b,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )
        except ValueError:
            # All non-zero differences cancel out into a degenerate
            # case scipy refuses (e.g. n=0 after dropping ties).
            w_stat, p_raw = 0.0, 1.0
    z_score = _signed_wilcoxon_z(diff)
    return {"W": float(w_stat), "p_raw": float(p_raw), "Z": z_score}


def paired_episode_metrics(scores_by_method: dict[str, np.ndarray]) -> dict:
    """Paired Wilcoxon tests for every pair of methods, Holm-corrected over all pairs.

    All score arrays must come from the same M paired episodes; only their
    lengths can be checked here.

    Returns:
        ``{(method_a, method_b): {W, p_raw, p_holm, Z, r, effect_size_category,
        significance, n_episodes}}`` with ``r = |Z| / sqrt(M)`` and the
        significance marker taken from ``p_holm``.
    """
    methods = list(scores_by_method.keys())
    if len(methods) < 2:
        raise ValueError("Need at least two methods to compute pairwise statistics.")

    lengths = {len(np.asarray(v)) for v in scores_by_method.values()}
    if len(lengths) != 1:
        raise ValueError(
            "All methods must be evaluated on the same M paired episodes; "
            f"got mismatched lengths {lengths}."
        )
    m_episodes = lengths.pop()

    pairs = list(itertools.combinations(methods, 2))
    raw = {(a, b): paired_wilcoxon(scores_by_method[a], scores_by_method[b]) for a, b in pairs}
    p_holm_vector = holm_bonferroni([raw[pair]["p_raw"] for pair in pairs])

    return {
        pair: _corrected_stats(
            raw[pair], p_holm, abs(raw[pair]["Z"]) / np.sqrt(m_episodes), m_episodes
        )
        for pair, p_holm in zip(pairs, p_holm_vector, strict=True)
    }


def cross_configuration_significance(
    scores_by_config: dict[Any, dict[str, np.ndarray]],
    reference: str,
    candidates: Sequence[str] | None = None,
) -> dict[str, dict[Any, dict]]:

    config_keys = list(scores_by_config.keys())
    if not config_keys:
        raise ValueError("scores_by_config must contain at least one configuration.")

    first = scores_by_config[config_keys[0]]
    if reference not in first:
        raise KeyError(
            f"Reference method '{reference}' not found in configuration {config_keys[0]!r}."
        )
    if candidates is None:
        candidates = [name for name in first.keys() if name != reference]

    results: dict[str, dict[Any, dict]] = {}
    for candidate in candidates:
        raw: dict[Any, dict[str, float]] = {}
        n_by_cfg: dict[Any, int] = {}
        for cfg in config_keys:
            methods = scores_by_config[cfg]
            if reference not in methods or candidate not in methods:
                raise KeyError(
                    f"Configuration {cfg!r} must contain both '{reference}' "
                    f"and '{candidate}'."
                )
            xa = np.asarray(methods[candidate], dtype=float)
            xb = np.asarray(methods[reference], dtype=float)
            if xa.shape != xb.shape:
                raise ValueError(
                    f"Unpaired episode counts at configuration {cfg!r} for "
                    f"'{candidate}' vs '{reference}': {xa.shape} vs {xb.shape}."
                )
            raw[cfg] = paired_wilcoxon(xa, xb)
            n_by_cfg[cfg] = xa.shape[0]

        p_holm_vector = holm_bonferroni([raw[cfg]["p_raw"] for cfg in config_keys])

        per_candidate: dict[Any, dict] = {}
        for cfg, p_holm in zip(config_keys, p_holm_vector, strict=True):
            n_episodes = n_by_cfg[cfg]
            r_effect = abs(raw[cfg]["Z"]) / np.sqrt(n_episodes) if n_episodes > 0 else 0.0
            per_candidate[cfg] = _corrected_stats(raw[cfg], p_holm, r_effect, n_episodes)
        results[candidate] = per_candidate
    return results


def confidence_interval(
    scores: np.ndarray,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap (1 - alpha) confidence interval of the mean."""
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = scores[idx].mean(axis=1)
    lower = float(np.percentile(resample_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return lower, upper


def format_results_table(
    scores_by_method: dict[str, np.ndarray],
    pairwise: dict,
    reference: str,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = 0,
) -> str:
    """Markdown table: mean, bootstrap CI, and Holm-corrected significance vs.
    ``reference`` (``pairwise`` is the output of :func:`paired_episode_metrics`).
    """
    if reference not in scores_by_method:
        raise KeyError(f"Reference method '{reference}' not found in scores_by_method.")

    header = (
        f"| Method | Mean Macro-F1 | 95% CI (bootstrap, n={n_resamples}) "
        f"| Holm p vs. {reference} | Z | r (effect) | Sig. |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows: list[str] = []
    for method, scores in scores_by_method.items():
        scores = np.asarray(scores, dtype=float)
        mean_score = float(scores.mean())
        lower, upper = confidence_interval(
            scores, alpha=alpha, n_resamples=n_resamples, random_state=random_state
        )
        if method == reference:
            rows.append(
                f"| {method} (reference) | {mean_score:.4f} | "
                f"[{lower:.4f}, {upper:.4f}] | - | - | - | - |"
            )
            continue

        if (method, reference) in pairwise:
            stats = pairwise[(method, reference)]
            z_val = stats["Z"]
        elif (reference, method) in pairwise:
            stats = pairwise[(reference, method)]
            z_val = -stats["Z"]
        else:
            stats = None
            z_val = None

        if stats is None:
            rows.append(
                f"| {method} | {mean_score:.4f} | [{lower:.4f}, {upper:.4f}] "
                "| n/a | n/a | n/a | n/a |"
            )
        else:
            rows.append(
                f"| {method} | {mean_score:.4f} | [{lower:.4f}, {upper:.4f}] "
                f"| {stats['p_holm']:.4g} | {z_val:.3f} "
                f"| {stats['r']:.3f} ({stats['effect_size_category']}) "
                f"| {stats['significance']} |"
            )
    return header + "\n".join(rows) + "\n"


if __name__ == "__main__":
    y_true_toy = np.array([0, 0, 1, 1, 2, 2])
    y_pred_toy = np.array([0, 1, 1, 2, 2, 2])
    expected_macro_f1 = 59.0 / 90.0
    computed_macro_f1 = macro_f1(y_true_toy, y_pred_toy, num_classes=3)
    assert abs(computed_macro_f1 - expected_macro_f1) < 1e-9, computed_macro_f1

    rng = np.random.default_rng(42)
    m_episodes = 50
    episode_difficulty = rng.normal(0.0, 0.05, size=m_episodes)

    def _clip01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    scores_by_method = {
        "TIM (Euclidean)": _clip01(
            rng.beta(6.0, 4.0, size=m_episodes) + episode_difficulty
        ),
        "LorentzTIM": _clip01(
            rng.beta(9.0, 3.0, size=m_episodes) + episode_difficulty
        ),
        "Hyp-SimpleShot": _clip01(
            rng.beta(4.0, 6.0, size=m_episodes) + episode_difficulty
        ),
        "TIM (hyperbolic)": _clip01(
            rng.beta(6.3, 3.9, size=m_episodes) + episode_difficulty
        ),
    }

    pairwise = paired_episode_metrics(scores_by_method)

    print(
        f"{'Pair':<42s}{'W':>10s}{'p_raw':>10s}{'p_holm':>10s}"
        f"{'Z':>8s}{'r':>8s}{'effect':>12s}{'sig':>6s}"
    )
    for (a, b), stats in pairwise.items():
        pair_label = f"{a} vs {b}"
        print(
            f"{pair_label:<42s}{stats['W']:>10.3f}{stats['p_raw']:>10.4g}"
            f"{stats['p_holm']:>10.4g}{stats['Z']:>8.3f}{stats['r']:>8.3f}"
            f"{stats['effect_size_category']:>12s}{stats['significance']:>6s}"
        )
    print()

    table = format_results_table(
        scores_by_method, pairwise, reference="TIM (Euclidean)",
        alpha=0.05, n_resamples=10_000, random_state=0,
    )
    print(table)

    # Holm correction across shot counts for one candidate.
    scores_by_shot: dict[int, dict[str, np.ndarray]] = {}
    for shot, (ref_a, ref_b, hyp_a, hyp_b) in {
        1: (6.0, 4.0, 6.2, 3.9),
        3: (6.0, 4.0, 6.4, 3.8),
        5: (6.0, 4.0, 7.5, 3.2),
        10: (6.0, 4.0, 8.0, 3.0),
    }.items():
        shot_difficulty = rng.normal(0.0, 0.05, size=m_episodes)
        scores_by_shot[shot] = {
            "TIM (Euclidean)": _clip01(rng.beta(ref_a, ref_b, size=m_episodes) + shot_difficulty),
            "LorentzTIM": _clip01(rng.beta(hyp_a, hyp_b, size=m_episodes) + shot_difficulty),
        }

    cross_shot = cross_configuration_significance(
        scores_by_shot, reference="TIM (Euclidean)", candidates=["LorentzTIM"]
    )
    print(
        f"{'S':>4s}{'mean(LorentzTIM)':>18s}{'mean(TIM)':>12s}{'p_raw':>10s}"
        f"{'p_holm':>10s}{'Z':>8s}{'r':>8s}{'sig':>6s}"
    )
    for shot in scores_by_shot:
        stats = cross_shot["LorentzTIM"][shot]
        mean_lorentz = scores_by_shot[shot]["LorentzTIM"].mean()
        mean_tim = scores_by_shot[shot]["TIM (Euclidean)"].mean()
        print(
            f"{shot:>4d}{mean_lorentz:>18.4f}{mean_tim:>12.4f}{stats['p_raw']:>10.4g}"
            f"{stats['p_holm']:>10.4g}{stats['Z']:>8.3f}{stats['r']:>8.3f}"
            f"{stats['significance']:>6s}"
        )

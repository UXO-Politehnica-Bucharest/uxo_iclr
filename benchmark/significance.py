#!/usr/bin/env python3
"""Table 3 statistical-significance driver.

Reads per_episode_scores.csv and computes paired Wilcoxon signed-rank tests
with Holm-Bonferroni correction over shot configurations against a reference baseline.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.evaluator import (
    confidence_interval,
    cross_configuration_significance,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scores-csv",
        type=Path,
        required=True,
        help="Path to a per_episode_scores.csv (columns: shot, method, episode_index, macro_f1).",
    )
    p.add_argument(
        "--shots",
        type=str,
        required=True,
        help="Comma-separated shot counts forming ONE Holm family, e.g. '1,3,5,10' "
        "(Table 3's main grid) or '15,20,25,30,35,40' (extended-shot sweep). "
        "Every shot listed must be present in --scores-csv for every method tested.",
    )
    p.add_argument(
        "--reference",
        type=str,
        default="TIM (Euclidean)",
        help="Baseline every candidate method is compared against. Default: 'TIM (Euclidean)'.",
    )
    p.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated method names to test against --reference. Default: every "
        "other method present at all requested shots (one Holm family per candidate, "
        "across shots).",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level. Default: 0.05.")
    p.add_argument(
        "--n-resamples", type=int, default=10_000, help="Bootstrap resamples for the 95%% CI. Default: 10000."
    )
    p.add_argument(
        "--random-state", type=int, default=0, help="Seed for the percentile-bootstrap CI. Default: 0."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write significance_table3.md / significance_full.csv into. "
        "Default: --scores-csv's parent directory.",
    )
    return p.parse_args()


def _load_scores_by_shot(csv_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """Pivot per_episode_scores.csv into {shot: {method: scores}}, sorted by
    episode_index so that methods at the same shot are paired. Raises if the
    methods at a shot were not scored on the same episode indices.
    """
    raw: Dict[int, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"shot", "method", "episode_index", "macro_f1"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}."
            )
        for row in reader:
            shot = int(row["shot"])
            method = row["method"]
            ep_idx = int(row["episode_index"])
            score = float(row["macro_f1"])
            raw[shot][method].append((ep_idx, score))

    scores_by_shot: Dict[int, Dict[str, np.ndarray]] = {}
    for shot, by_method in raw.items():
        episode_index_sets = set()
        pivoted: Dict[str, np.ndarray] = {}
        for method, pairs in by_method.items():
            pairs.sort(key=lambda t: t[0])
            idxs = [p[0] for p in pairs]
            if len(idxs) != len(set(idxs)):
                raise ValueError(
                    f"Duplicate episode_index values for method '{method}' at shot={shot} "
                    f"in {csv_path}."
                )
            episode_index_sets.add(tuple(idxs))
            pivoted[method] = np.asarray([p[1] for p in pairs], dtype=float)
        if len(episode_index_sets) > 1:
            raise ValueError(
                f"Methods at shot={shot} in {csv_path} were scored on "
                f"{len(episode_index_sets)} different episode_index sequences; "
                "re-run export_full_results.py for this shot."
            )
        scores_by_shot[shot] = pivoted
    return scores_by_shot


def _fmt_ci_half_width(scores: np.ndarray, alpha: float, n_resamples: int, random_state: int) -> tuple[float, float]:
    """Mean and 95% CI half-width for Table 3's ``mean{\\tiny$\\pm$half-width}``
    notation. The percentile-bootstrap CI is asymmetric; the larger of the
    two mean-to-bound distances is reported."""
    mean = float(scores.mean())
    lo, hi = confidence_interval(scores, alpha=alpha, n_resamples=n_resamples, random_state=random_state)
    half_width = max(mean - lo, hi - mean)
    return mean, half_width


def main() -> None:
    args = _parse_args()
    shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]
    if not shots:
        raise ValueError("--shots must list at least one shot count.")

    scores_by_shot_all = _load_scores_by_shot(args.scores_csv)
    missing_shots = [s for s in shots if s not in scores_by_shot_all]
    if missing_shots:
        raise ValueError(f"{args.scores_csv} has no rows for shot(s) {missing_shots}.")
    scores_by_shot = {s: scores_by_shot_all[s] for s in shots}

    for s in shots:
        if args.reference not in scores_by_shot[s]:
            raise KeyError(f"Reference '{args.reference}' not found at shot={s} in {args.scores_csv}.")

    if args.candidates is not None:
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    else:
        common = set(scores_by_shot[shots[0]].keys())
        for s in shots[1:]:
            common &= set(scores_by_shot[s].keys())
        candidates = sorted(m for m in common if m != args.reference)
    if not candidates:
        raise ValueError("No candidate methods found (present at every requested shot, excluding --reference).")

    print(f"Reference: {args.reference}")
    print(f"Shots (one Holm family per candidate): {shots}")
    print(f"Candidates ({len(candidates)}): {candidates}")

    results = cross_configuration_significance(
        scores_by_shot, reference=args.reference, candidates=candidates
    )

    output_dir = args.output_dir or args.scores_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = []
    md_lines.append(f"# Significance vs. {args.reference} (Holm-corrected over S in {shots})")
    md_lines.append("")
    md_lines.append(
        f"Paired two-sided Wilcoxon signed-rank tests (scipy `method=\"approx\"`), "
        f"Holm-Bonferroni corrected within each candidate's own family of "
        f"{len(shots)} shot counts, alpha={args.alpha}. Source: `{args.scores_csv}`."
    )
    md_lines.append("")
    header = "| Method | " + " | ".join(f"S={s}" for s in shots) + " |"
    sep = "|---|" + "|".join(["---"] * len(shots)) + "|"
    md_lines.append(header)
    md_lines.append(sep)

    ref_cells = []
    for s in shots:
        mean, hw = _fmt_ci_half_width(
            scores_by_shot[s][args.reference], args.alpha, args.n_resamples, args.random_state
        )
        ref_cells.append(f"{mean:.4f}$\\pm${hw:.4f}")
    md_lines.append(f"| {args.reference} (reference) | " + " | ".join(ref_cells) + " |")

    for candidate in candidates:
        cells = []
        for s in shots:
            mean, hw = _fmt_ci_half_width(
                scores_by_shot[s][candidate], args.alpha, args.n_resamples, args.random_state
            )
            stats = results[candidate][s]
            marker = "$^{*}$" if stats["p_holm"] < args.alpha else ""
            cells.append(f"{mean:.4f}{marker}$\\pm${hw:.4f}")
        md_lines.append(f"| {candidate} | " + " | ".join(cells) + " |")

    md_lines.append("")
    md_lines.append("Significance detail (per candidate, per shot):")
    md_lines.append("")
    md_lines.append("| Method | S | mean | W | p_raw | p_holm | Z | r | effect | sig |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for candidate in candidates:
        for s in shots:
            stats = results[candidate][s]
            mean = float(scores_by_shot[s][candidate].mean())
            md_lines.append(
                f"| {candidate} | {s} | {mean:.4f} | {stats['W']:.3f} | {stats['p_raw']:.4g} "
                f"| {stats['p_holm']:.4g} | {stats['Z']:.3f} | {stats['r']:.3f} "
                f"| {stats['effect_size_category']} | {stats['significance']} |"
            )
    md_text = "\n".join(md_lines) + "\n"

    md_path = output_dir / "significance_table3.md"
    md_path.write_text(md_text)
    print(f"\nWrote {md_path}")

    csv_path = output_dir / "significance_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate", "reference", "shot", "mean_candidate", "mean_reference",
                "W", "p_raw", "p_holm", "Z", "r", "effect_size_category",
                "significance", "n_episodes",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            for s in shots:
                stats = results[candidate][s]
                writer.writerow(
                    {
                        "candidate": candidate,
                        "reference": args.reference,
                        "shot": s,
                        "mean_candidate": float(scores_by_shot[s][candidate].mean()),
                        "mean_reference": float(scores_by_shot[s][args.reference].mean()),
                        "W": stats["W"],
                        "p_raw": stats["p_raw"],
                        "p_holm": stats["p_holm"],
                        "Z": stats["Z"],
                        "r": stats["r"],
                        "effect_size_category": stats["effect_size_category"],
                        "significance": stats["significance"],
                        "n_episodes": stats["n_episodes"],
                    }
                )
    print(f"Wrote {csv_path}")

    print("\n" + md_text)


if __name__ == "__main__":
    main()

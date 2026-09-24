#!/usr/bin/env python3
"""benchmark/search_and_update_psi.py -- Search psi_S* for one backbone on
the main shot grid and write the result into
``export_full_results.SEARCHED_HYPERPARAMETERS`` in the source file.

Only the ``(backbone, shot)`` entries of the searched backbone are replaced
(or added); entries for every other backbone are left untouched.

The search is the one ``export_full_results.py --force-search`` performs
for the same ``--backbone``/``--seed``/``--num-episodes``: same paired test
episodes, same diagnostic-calibrated d_eff (``_select_d_eff`` over their
support images), same base-split mean, and the same backbone-salted
search RNG. Running this first lets ``export_full_results.py`` and
``class_group_analysis.py`` reuse one shared search pass per shot without
``--force-search``.

Usage:
    python3 benchmark/search_and_update_psi.py --device cuda --backbone dinov3
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.determinism import set_deterministic
from algorithm.features import get_feature_extractor
from benchmark.episode_generator import FULL_SCALE_CONFIG, generate_episodes
from benchmark.export_full_results import (
    K_CURVATURE,
    _collect_paths,
    _extract_feature_cache,
    _select_d_eff,
    format_searched_entry,
)
from benchmark.hyperparam_search import run_search_for_shot
from dataset.ctx_uxo import compute_base_split_mean

_TARGET_FILE = Path(__file__).resolve().parent / "export_full_results.py"
_DICT_START = "SEARCHED_HYPERPARAMETERS: Dict[tuple[str, int], Dict[str, float]] = {\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backbone", type=str, default="dinov3")
    p.add_argument("--shots", type=str, default=",".join(str(s) for s in FULL_SCALE_CONFIG.shots))
    p.add_argument(
        "--num-episodes",
        type=int,
        default=FULL_SCALE_CONFIG.num_episodes,
        help="Test episodes whose support images calibrate d_eff; use the same value "
        "as the export_full_results.py run that will consume the result.",
    )
    p.add_argument("--search-trials", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=30)
    return p.parse_args()


def update_searched_hyperparameters(text: str, backbone: str, psi_by_shot: Dict[int, Dict[str, float]]) -> str:
    """Return ``text`` (the source of export_full_results.py) with this
    backbone's entries replaced or appended inside the dict literal."""
    start = text.find(_DICT_START)
    if start < 0:
        raise RuntimeError(f"Could not find the SEARCHED_HYPERPARAMETERS block in {_TARGET_FILE}.")
    body_start = start + len(_DICT_START)
    body_end = text.index("\n}\n", body_start) + 1
    body = text[body_start:body_end]
    for shot in sorted(psi_by_shot):
        new_line = format_searched_entry(backbone, shot, psi_by_shot[shot])
        pattern = re.compile(rf'^    \("{re.escape(backbone)}", {shot}\): \{{.*\}},$', re.MULTILINE)
        body, n = pattern.subn(lambda _, line=new_line: line, body)
        if n == 0:
            body += new_line + "\n"
    return text[:body_start] + body + text[body_end:]


def main() -> None:
    args = _parse_args()
    seed = FULL_SCALE_CONFIG.seed
    set_deterministic(seed)
    shots = tuple(int(s.strip()) for s in args.shots.split(",") if s.strip())

    episodes_by_shot = {
        shot: generate_episodes(
            split="test", n_shot=shot, n_query=FULL_SCALE_CONFIG.n_query,
            num_episodes=args.num_episodes, seed=seed,
        )
        for shot in shots
    }
    _, support_paths = _collect_paths(episodes_by_shot.values())

    print(f"Loading feature extractor '{args.backbone}' on {args.device!r}...", flush=True)
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)

    print("Computing the base-split mean over the complete train split...", flush=True)
    t0 = time.perf_counter()
    base_mean = compute_base_split_mean(feature_fn=extractor)
    print(f"Base-split mean computed in {time.perf_counter() - t0:.1f}s.", flush=True)

    feature_cache = _extract_feature_cache(support_paths, extractor)
    d_eff = _select_d_eff(feature_cache, support_paths, base_mean)
    print(f"d_eff={d_eff} for backbone={args.backbone}", flush=True)

    psi_by_shot = {}
    for shot in shots:
        print(f"\nSearching psi_{shot}* (R={args.search_trials}, val={args.val_episodes})...", flush=True)
        t0 = time.perf_counter()
        result = run_search_for_shot(
            shot=shot, K=K_CURVATURE, d_eff=d_eff, base_mean=base_mean, extractor=extractor,
            n_trials=args.search_trials, n_val_episodes=args.val_episodes,
            n_query=FULL_SCALE_CONFIG.n_query, seed=seed, backbone=args.backbone,
        )
        psi_by_shot[shot] = result.best_candidate
        print(
            f"psi_{shot}* = {result.best_candidate} (val Macro-F1={result.best_val_macro_f1:.4f}, "
            f"{time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    _TARGET_FILE.write_text(update_searched_hyperparameters(_TARGET_FILE.read_text(), args.backbone, psi_by_shot))
    print(f"\nUpdated {_TARGET_FILE} with psi_S* for backbone={args.backbone}, d_eff={d_eff}.", flush=True)


if __name__ == "__main__":
    main()

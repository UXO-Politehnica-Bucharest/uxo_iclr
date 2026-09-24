"""Paired episode generator for CTX-UXO: file-path-level episode specs that
every method replays identically.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "benchmark" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.ctx_uxo import (
    CLASS_NAMES,
    DEFAULT_INSTANCES_ROOT,
    LONG_TAIL_CLASSES,
    NUM_CLASSES,
    Split,
    count_instances,
    list_class_instances,
)


@dataclass(frozen=True)
class ClassEpisodeEntry:
    """One class's support and query paths within an episode (possibly capped)."""

    class_name: str
    support_paths: Tuple[Path, ...]
    query_paths: Tuple[Path, ...]

    @property
    def n_support(self) -> int:
        """Actual (possibly capped) shot count S_a for this class."""
        return len(self.support_paths)

    @property
    def n_query_actual(self) -> int:
        """Actual (possibly capped) query count Q_a for this class."""
        return len(self.query_paths)


@dataclass(frozen=True)
class EpisodeSpec:


    index: int
    n_shot: int
    n_query: int
    seed: int
    classes: Tuple[ClassEpisodeEntry, ...]

    def shots_per_class(self) -> dict:
        """{class_name: S_a} for this episode."""
        return {entry.class_name: entry.n_support for entry in self.classes}

    def queries_per_class(self) -> dict:
        """{class_name: Q_a} for this episode."""
        return {entry.class_name: entry.n_query_actual for entry in self.classes}


def _restrict_pool(
    paths: Sequence[Path], max_count: Optional[int], rng: np.random.Generator
) -> List[Path]:
    """Seeded subsample of at most ``max_count`` paths, in ascending index order."""
    if max_count is None or max_count >= len(paths):
        return list(paths)
    if max_count < 0:
        raise ValueError(f"max_count must be >= 0 or None, got {max_count}.")
    chosen_idx = rng.choice(len(paths), size=max_count, replace=False)
    chosen_idx.sort()
    return [paths[i] for i in chosen_idx]


def _build_eligible_pools(
    support_split: Split,
    query_split: Split,
    support_max_per_class: Optional[int],
    query_max_per_class: Optional[int],
    root: Path,
    rng: np.random.Generator,
) -> Tuple[dict, dict]:

    support_pools: dict = {}
    query_pools: dict = {}
    for class_name in CLASS_NAMES:
        full_support = list_class_instances(support_split, class_name, root=root)
        full_query = list_class_instances(query_split, class_name, root=root)
        support_pools[class_name] = _restrict_pool(full_support, support_max_per_class, rng)
        query_pools[class_name] = _restrict_pool(full_query, query_max_per_class, rng)
    return support_pools, query_pools


def generate_episodes(
    split: str,
    n_shot: int,
    n_query: int,
    num_episodes: int,
    seed: int,
    max_per_class: Optional[int] = None,
    query_max_per_class: Optional[int] = None,
    support_split: str = "train",
    root: Path = DEFAULT_INSTANCES_ROOT,
) -> List[EpisodeSpec]:

    if n_shot < 1:
        raise ValueError(f"n_shot must be >= 1, got {n_shot}.")
    if n_query < 1:
        raise ValueError(f"n_query must be >= 1, got {n_query}.")
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}.")
    if query_max_per_class is None:
        query_max_per_class = max_per_class

    # Independent streams for the pool restriction and each episode.
    master_seq = np.random.SeedSequence(seed)
    pool_seq, episodes_root_seq = master_seq.spawn(2)
    pool_rng = np.random.default_rng(pool_seq)

    support_pools, query_pools = _build_eligible_pools(
        support_split=support_split,  # type: ignore[arg-type]
        query_split=split,  # type: ignore[arg-type]
        support_max_per_class=max_per_class,
        query_max_per_class=query_max_per_class,
        root=root,
        rng=pool_rng,
    )

    episode_seqs = episodes_root_seq.spawn(num_episodes)

    episodes: List[EpisodeSpec] = []
    for ep_idx, ep_seq in enumerate(episode_seqs):
        rng = np.random.default_rng(ep_seq)
        # Logged only; the file paths themselves define the pairing.
        provenance_seed = int(ep_seq.generate_state(1, dtype=np.uint32)[0])

        entries: List[ClassEpisodeEntry] = []
        for class_name in CLASS_NAMES:
            support_pool = support_pools[class_name]
            query_pool = query_pools[class_name]

            # Same per-class capping as ctx_uxo.sample_episode.
            s_a = min(n_shot, len(support_pool))
            q_a = min(n_query, len(query_pool))
            if s_a == 0:
                raise RuntimeError(
                    f"Class '{class_name}' has zero eligible instances in "
                    f"support_split='{support_split}' (after max_per_class="
                    f"{max_per_class} restriction); cannot form an episode."
                )
            if q_a == 0:
                raise RuntimeError(
                    f"Class '{class_name}' has zero eligible instances in "
                    f"query_split='{split}' (after query_max_per_class="
                    f"{query_max_per_class} restriction); cannot form an "
                    "episode."
                )

            support_idx = rng.choice(len(support_pool), size=s_a, replace=False)
            query_idx = rng.choice(len(query_pool), size=q_a, replace=False)
            entries.append(
                ClassEpisodeEntry(
                    class_name=class_name,
                    support_paths=tuple(support_pool[i] for i in support_idx),
                    query_paths=tuple(query_pool[i] for i in query_idx),
                )
            )

        episodes.append(
            EpisodeSpec(
                index=ep_idx,
                n_shot=n_shot,
                n_query=n_query,
                seed=provenance_seed,
                classes=tuple(entries),
            )
        )

    return episodes


@dataclass(frozen=True)
class ScaleConfig:
    """Named preset of the parameters that control run cost."""

    name: str
    support_max_per_class: Optional[int]
    query_max_per_class: Optional[int]
    num_episodes: int
    shots: Tuple[int, ...]
    n_query: int
    seed: int


SMALL_SCALE_CONFIG = ScaleConfig(
    name="small_scale",
    support_max_per_class=15,
    query_max_per_class=8,
    num_episodes=20,
    shots=(1, 5),
    n_query=5,
    seed=42,
)

FULL_SCALE_CONFIG = ScaleConfig(
    name="full_scale",
    support_max_per_class=None,
    query_max_per_class=None,
    num_episodes=1000,
    shots=(1, 3, 5, 10),
    n_query=15,
    seed=42,
)


if __name__ == "__main__":
    print(f"Instances root:        {DEFAULT_INSTANCES_ROOT}")
    print(f"Num classes:           {NUM_CLASSES}")
    print(f"Long-tail classes:     {LONG_TAIL_CLASSES}")
    print(f"SMALL_SCALE_CONFIG:    {SMALL_SCALE_CONFIG}")
    print(f"FULL_SCALE_CONFIG:     {FULL_SCALE_CONFIG}")

    print("\nInstance counts")
    counts = count_instances()
    for class_name in CLASS_NAMES:
        c = counts[class_name]
        print(
            f"  {class_name:16s} train={c['train']:5d}  valid={c['valid']:5d}  "
            f"test={c['test']:5d}"
        )

    n_shot = SMALL_SCALE_CONFIG.shots[0]
    n_query = SMALL_SCALE_CONFIG.n_query
    num_episodes = SMALL_SCALE_CONFIG.num_episodes
    seed_a = SMALL_SCALE_CONFIG.seed
    seed_b = seed_a + 1

    print(
        f"\nGenerating {num_episodes} episodes: split='test', "
        f"n_shot={n_shot}, n_query={n_query}, "
        f"support_max_per_class={SMALL_SCALE_CONFIG.support_max_per_class}, "
        f"query_max_per_class={SMALL_SCALE_CONFIG.query_max_per_class}, "
        f"seed={seed_a}"
    )
    episodes_a = generate_episodes(
        split="test",
        n_shot=n_shot,
        n_query=n_query,
        num_episodes=num_episodes,
        seed=seed_a,
        max_per_class=SMALL_SCALE_CONFIG.support_max_per_class,
        query_max_per_class=SMALL_SCALE_CONFIG.query_max_per_class,
    )
    assert all(
        {entry.class_name for entry in ep.classes} == set(CLASS_NAMES) for ep in episodes_a
    ), "episode missing a class"

    print(f"\nEpisode 0, S={n_shot}, Q={n_query}:")
    ep0 = episodes_a[0]
    for entry in ep0.classes:
        tail_flag = " (LONG-TAIL)" if entry.class_name in LONG_TAIL_CLASSES else ""
        capped_s = " CAPPED" if entry.n_support < n_shot else ""
        capped_q = " CAPPED" if entry.n_query_actual < n_query else ""
        print(
            f"  {entry.class_name:16s}: S_a={entry.n_support:2d}{capped_s:8s} "
            f"Q_a={entry.n_query_actual:2d}{capped_q:8s}{tail_flag}"
        )

    n_shot_5 = SMALL_SCALE_CONFIG.shots[1]
    episodes_5shot = generate_episodes(
        split="test",
        n_shot=n_shot_5,
        n_query=n_query,
        num_episodes=num_episodes,
        seed=seed_a,
        max_per_class=SMALL_SCALE_CONFIG.support_max_per_class,
        query_max_per_class=SMALL_SCALE_CONFIG.query_max_per_class,
    )
    print(f"\nEpisode 0, S={n_shot_5}, Q={n_query}:")
    for entry in episodes_5shot[0].classes:
        tail_flag = " (LONG-TAIL)" if entry.class_name in LONG_TAIL_CLASSES else ""
        capped_s = " CAPPED" if entry.n_support < n_shot_5 else ""
        capped_q = " CAPPED" if entry.n_query_actual < n_query else ""
        print(
            f"  {entry.class_name:16s}: S_a={entry.n_support:2d}{capped_s:8s} "
            f"Q_a={entry.n_query_actual:2d}{capped_q:8s}{tail_flag}"
        )

    episodes_a2 = generate_episodes(
        split="test",
        n_shot=n_shot,
        n_query=n_query,
        num_episodes=num_episodes,
        seed=seed_a,
        max_per_class=SMALL_SCALE_CONFIG.support_max_per_class,
        query_max_per_class=SMALL_SCALE_CONFIG.query_max_per_class,
    )
    assert all(
        tuple((e.class_name, e.support_paths, e.query_paths) for e in ep_a.classes)
        == tuple((e.class_name, e.support_paths, e.query_paths) for e in ep_a2.classes)
        for ep_a, ep_a2 in zip(episodes_a, episodes_a2, strict=True)
    ), "same seed produced different episodes"

    episodes_b = generate_episodes(
        split="test",
        n_shot=n_shot,
        n_query=n_query,
        num_episodes=num_episodes,
        seed=seed_b,
        max_per_class=SMALL_SCALE_CONFIG.support_max_per_class,
        query_max_per_class=SMALL_SCALE_CONFIG.query_max_per_class,
    )
    assert any(
        tuple((e.class_name, e.support_paths, e.query_paths) for e in ep_a.classes)
        != tuple((e.class_name, e.support_paths, e.query_paths) for e in ep_b.classes)
        for ep_a, ep_b in zip(episodes_a, episodes_b, strict=True)
    ), "different seeds produced identical episodes"

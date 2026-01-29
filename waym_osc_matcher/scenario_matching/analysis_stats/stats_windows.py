# scenario_matching/analysis_stats/stats_windows.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def max_possible_windows(T: int, minF: int, maxF: int) -> int:
    """Number of windows (t0,t1) in a length-T segment with length in [minF..maxF]."""
    if T <= 0:
        return 0
    minF = max(1, int(minF))
    maxF = max(minF, int(maxF))
    tot = 0
    for t0 in range(T):
        t1_min = t0 + minF - 1
        t1_max = min(T - 1, t0 + maxF - 1)
        if t1_min <= t1_max:
            tot += (t1_max - t1_min + 1)
    return int(tot)


def count_windows(windows_by_t0: Dict[int, Sequence[Tuple[int, int]]]) -> int:
    """Count windows represented as: t0 -> list of inclusive [t1_lo, t1_hi] ranges."""
    n = 0
    for ranges in (windows_by_t0 or {}).values():
        for lo, hi in ranges:
            n += int(hi) - int(lo) + 1
    return int(n)


def sample_window_from_W(
    rng: np.random.Generator, *, T: int, minF: int, maxF: int
) -> Optional[Tuple[int, int]]:
    """
    Sample uniformly over all possible windows W (t0,t1) with length in [minF..maxF].
    """
    if T <= 0:
        return None
    minF = max(1, int(minF))
    maxF = max(minF, int(maxF))

    t0s: List[int] = []
    counts: List[int] = []
    for t0 in range(T):
        t1_min = t0 + minF - 1
        t1_max = min(T - 1, t0 + maxF - 1)
        c = t1_max - t1_min + 1
        if c > 0:
            t0s.append(t0)
            counts.append(c)

    if not counts:
        return None

    w = np.asarray(counts, dtype=np.float64)
    p = w / w.sum()
    t0 = int(rng.choice(np.asarray(t0s, dtype=np.int64), p=p))

    t1_min = t0 + minF - 1
    t1_max = min(T - 1, t0 + maxF - 1)
    t1 = int(rng.integers(t1_min, t1_max + 1))
    return t0, t1


def sample_window_from_H(
    rng: np.random.Generator,
    windows_by_t0: Dict[int, Sequence[Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    """Sample uniformly over all hit windows represented in windows_by_t0."""
    if not windows_by_t0:
        return None

    t0s: List[int] = []
    weights: List[int] = []
    for t0, ranges in windows_by_t0.items():
        w = 0
        for lo, hi in ranges:
            w += int(hi) - int(lo) + 1
        if w > 0:
            t0s.append(int(t0))
            weights.append(int(w))

    if not weights:
        return None

    w = np.asarray(weights, dtype=np.float64)
    p = w / w.sum()
    t0 = int(rng.choice(np.asarray(t0s, dtype=np.int64), p=p))

    ranges = list(windows_by_t0[t0])
    lens = np.asarray([int(hi) - int(lo) + 1 for lo, hi in ranges], dtype=np.float64)
    ridx = int(rng.choice(np.arange(len(ranges)), p=lens / lens.sum()))
    lo, hi = ranges[ridx]
    t1 = int(rng.integers(int(lo), int(hi) + 1))
    return t0, t1


def _true_run_lengths(mask: np.ndarray) -> List[int]:
    """Return lengths of contiguous True-runs in a boolean mask."""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return [int(e - s) for s, e in zip(starts, ends)]


def count_windows_in_run(R: int, minF: int, maxF: int) -> int:
    """Number of windows of length in [minF..maxF] fully inside a run of length R."""
    if R <= 0:
        return 0
    a = max(1, int(minF))
    b = min(int(maxF), int(R))
    if a > b:
        return 0
    n = b - a + 1
    numerator = 2 * n * (R + 1) - n * (a + b)
    return int(numerator // 2)


def possible_windows_presence(feats, actor_ids: List[str], minF: int, maxF: int) -> int:
    """
    Count windows where ALL actor_ids are present for the whole window (presence-only).
    """
    if not actor_ids:
        return 0
    T = int(getattr(feats, "T", 0) or 0)
    if T <= 0:
        return 0

    mask = np.ones(T, dtype=bool)
    for a in actor_ids:
        p = feats.present.get(a)
        if p is None:
            return 0
        mask &= p.astype(bool)

    runs = _true_run_lengths(mask)
    tot = 0
    for R in runs:
        tot += count_windows_in_run(R, minF, maxF)
    return int(tot)

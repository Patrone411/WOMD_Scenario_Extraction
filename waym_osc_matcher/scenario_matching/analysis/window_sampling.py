"""scenario_matching.analysis.window_sampling

Uniform sampling of time windows:
- From W: all possible windows given (T, minF, maxF)
- From H: hit windows represented as windows_by_t0[t0] = [(lo,hi), ...]
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def sample_window_from_W(
    rng: np.random.Generator,
    *,
    T: int,
    minF: int,
    maxF: int,
) -> Optional[Tuple[int, int]]:
    """Sample (t0,t1) uniformly from all windows with length L in [minF..maxF]."""
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
    """Sample (t0,t1) uniformly from hit windows_by_t0 ranges."""
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


def max_possible_windows(T: int, minF: int, maxF: int) -> int:
    """Count all possible (t0,t1) windows for T and [minF..maxF]."""
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

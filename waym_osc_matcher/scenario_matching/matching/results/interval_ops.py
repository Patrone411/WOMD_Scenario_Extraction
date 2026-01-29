# scenario_matching/matching/results/interval_ops.py
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .types import Interval


def merge_windows_to_intervals(wins: Iterable[Tuple[int, int]]) -> List[Interval]:
    """
    Merge a list/iterable of inclusive windows (t_start, t_end) into coalesced intervals.

    NOTE:
    Interval is a typing alias (Tuple[int,int]) and must NOT be called like Interval(a,b).
    Always create intervals as plain tuples: (a, b).
    """
    ws: List[Tuple[int, int]] = []
    for a, b in wins:
        if a is None or b is None:
            continue
        a = int(a)
        b = int(b)
        if b < a:
            a, b = b, a
        ws.append((a, b))

    if not ws:
        return []

    ws.sort(key=lambda ab: (ab[0], ab[1]))

    merged: List[Interval] = [ws[0]]
    for a, b in ws[1:]:
        a0, b0 = merged[-1]
        if a <= b0 + 1:
            merged[-1] = (a0, max(b0, b))
        else:
            merged.append((a, b))
    return merged


def intervals_to_mask(intervals: List[Interval], T: int) -> np.ndarray:
    """
    Convert inclusive intervals to a boolean mask of length T.
    """
    T = int(T)
    m = np.zeros(T, dtype=bool)
    if T <= 0:
        return m
    for a, b in intervals:
        a = max(0, int(a))
        b = min(T - 1, int(b))
        if a <= b:
            m[a : b + 1] = True
    return m


def mask_to_intervals(mask: np.ndarray) -> List[Interval]:
    """
    Convert a boolean mask to inclusive intervals.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    idx = np.flatnonzero(m)
    out: List[Interval] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i != prev + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out
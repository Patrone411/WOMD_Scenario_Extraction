from __future__ import annotations
from typing import List, Iterable
import numpy as np
from .types import Interval

def intervals_to_mask(intervals: List[Interval], T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for iv in intervals:
        if iv.t0 <= iv.t1:
            m[iv.t0:iv.t1+1] = True
    return m

def intervals_from_mask(mask: np.ndarray) -> List[Interval]:
    if mask.size == 0:
        return []
    on = np.flatnonzero(mask)
    if on.size == 0:
        return []
    out: List[Interval] = []
    start = prev = int(on[0])
    for i in map(int, on[1:]):
        if i == prev + 1:
            prev = i
            continue
        out.append(Interval(start, prev))
        start = prev = i
    out.append(Interval(start, prev))
    return out

def coalesce_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    ivs = sorted(intervals, key=lambda x: (x.t0, x.t1))
    if not ivs:
        return []
    out = [ivs[0]]
    for iv in ivs[1:]:
        last = out[-1]
        if iv.t0 <= last.t1 + 1:
            out[-1] = Interval(last.t0, max(last.t1, iv.t1))
        else:
            out.append(iv)
    return out

def mask_from_intervals(intervals: List[Interval], T: int) -> np.ndarray:
    return intervals_to_mask(intervals, T)
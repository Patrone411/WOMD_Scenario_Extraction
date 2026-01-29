# osc_parser/matching/post/utils.py
from __future__ import annotations
from typing import List, Iterable, Optional, Dict, Tuple
import numpy as np
from dataclasses import dataclass

from scenario_matching.matching.results.types import Interval
#from osc_parser.matching.results.types import Interval

def mask_from_intervals(intervals: List[Interval], T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for iv in intervals:
        a = max(0, iv.t0); b = min(T-1, iv.t1)
        if a <= b:
            m[a:b+1] = True
    return m

def intervals_from_mask(mask: np.ndarray) -> List[Interval]:
    out: List[Interval] = []
    T = int(mask.shape[0])
    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j+1 < T and mask[j+1]:
                j += 1
            out.append(Interval(i, j))
            i = j + 1
        else:
            i += 1
    return out

def intersect_all(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros(0, dtype=bool)
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out

def union_all(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros(0, dtype=bool)
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out

def require_min_run(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only contiguous True runs with length >= min_len."""
    T = mask.shape[0]
    if min_len <= 1 or T == 0:
        return mask
    out = mask.copy()
    i = 0
    while i < T:
        if out[i]:
            j = i
            while j+1 < T and out[j+1]:
                j += 1
            if (j - i + 1) < min_len:
                out[i:j+1] = False
            i = j + 1
        else:
            i += 1
    return out

def coalesce_intervals(intervals: List[Interval], max_gap: int = 0) -> List[Interval]:
    """Merge touching/near intervals: [a,b] and [b+1 .. b+1+gap]."""
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda iv: (iv.t0, iv.t1))
    cur = ivs[0]
    out: List[Interval] = []
    for iv in ivs[1:]:
        if iv.t0 <= cur.t1 + 1 + max_gap:
            cur = Interval(cur.t0, max(cur.t1, iv.t1))
        else:
            out.append(cur)
            cur = iv
    out.append(cur)
    return out

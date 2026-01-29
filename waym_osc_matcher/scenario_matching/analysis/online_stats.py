"""scenario_matching.analysis.online_stats

Online aggregators for distributions:
- OnlineHist: numeric histogram with fixed bin edges
- OnlineCat: categorical counter
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import Counter
import numpy as np


@dataclass
class OnlineHist:
    bin_edges: np.ndarray
    counts: np.ndarray
    n: int = 0
    n_invalid: int = 0

    @classmethod
    def from_edges(cls, edges: List[float]) -> "OnlineHist":
        be = np.asarray(edges, dtype=np.float64)
        if be.ndim != 1 or be.size < 2:
            raise ValueError("bin_edges must be 1D with at least 2 entries")
        return cls(bin_edges=be, counts=np.zeros(be.size - 1, dtype=np.int64))

    def add(self, x: Any) -> None:
        if x is None:
            self.n_invalid += 1
            return
        try:
            xv = float(x)
        except Exception:
            self.n_invalid += 1
            return
        if not np.isfinite(xv):
            self.n_invalid += 1
            return
        i = int(np.searchsorted(self.bin_edges, xv, side="right") - 1)
        if 0 <= i < self.counts.size:
            self.counts[i] += 1
            self.n += 1
        else:
            self.n_invalid += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "hist",
            "bin_edges": self.bin_edges.tolist(),
            "counts": self.counts.tolist(),
            "n": int(self.n),
            "n_invalid": int(self.n_invalid),
        }


@dataclass
class OnlineCat:
    counts: Counter
    n: int = 0
    n_invalid: int = 0

    @classmethod
    def empty(cls) -> "OnlineCat":
        return cls(counts=Counter())

    def add(self, x: Any) -> None:
        if x is None:
            self.n_invalid += 1
            return
        self.counts[x] += 1
        self.n += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "cat",
            "counts": {str(k): int(v) for k, v in self.counts.items()},
            "n": int(self.n),
            "n_invalid": int(self.n_invalid),
        }

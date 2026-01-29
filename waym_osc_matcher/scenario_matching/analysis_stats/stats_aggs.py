# scenario_matching/analysis_stats/stats_aggs.py
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List
import numpy as np


@dataclass
class OnlineHist:
    """Simple mergeable histogram."""
    bin_edges: np.ndarray
    counts: np.ndarray
    n: int = 0
    n_invalid: int = 0

    @classmethod
    def from_edges(cls, bin_edges: List[float]) -> "OnlineHist":
        be = np.asarray(bin_edges, dtype=np.float64)
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

    def merge(self, other: "OnlineHist") -> None:
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise ValueError("Histogram bin_edges differ; cannot merge.")
        self.counts += other.counts
        self.n += int(other.n)
        self.n_invalid += int(other.n_invalid)

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
    """Simple mergeable categorical counter."""
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

    def merge(self, other: "OnlineCat") -> None:
        self.counts.update(other.counts)
        self.n += int(other.n)
        self.n_invalid += int(other.n_invalid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "cat",
            "counts": {str(k): int(v) for k, v in self.counts.items()},
            "n": int(self.n),
            "n_invalid": int(self.n_invalid),
        }

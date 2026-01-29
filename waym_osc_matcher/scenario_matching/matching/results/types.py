# scenario_matching/matching/results/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np

# -----------------------------
# Core aliases
# -----------------------------

# A closed interval on the timeline (inclusive indices)
Interval = Tuple[int, int]

# Compact representation of a set of windows:
#   windows_by_t0[t0] = [(t1_lo, t1_hi), ...]   (inclusive)
WindowsByT0 = Dict[int, List[Tuple[int, int]]]

# (block_label, call_index)
CallKey = Tuple[str, int]

# A single role assignment ("ego_vehicle" -> "vehicle_22")
RoleAssignment = Tuple[str, str]

# Stable dict key for a binding: sorted tuple of assignments
BindingKey = Tuple[RoleAssignment, ...]


# -----------------------------
# Atomic signals (per call)
# -----------------------------

@dataclass
class PerCallSignal:
    """
    Atomic match result for a SINGLE call under a SINGLE (segment, role-binding).
    `intervals/mask` represent union support across acceptable call windows.
    Optional extras are used for stats/debug.
    """
    segment_id: str
    roles: Dict[str, str]
    T: int
    intervals: List[Interval]
    mask: Optional[Any] = None  # typically np.ndarray[bool]

    # bookkeeping
    call_index: Optional[int] = None
    roles_used: Optional[Tuple[str, ...]] = None

    # optional: full set of acceptable call windows (compact form)
    windows_by_t0: Optional[WindowsByT0] = None

    # optional: per-check counters (SED)
    mod_stats: Optional[Dict[str, Any]] = None
    endframes: Optional[List[int]] = None


# -----------------------------
# Block signals (per block)
# -----------------------------

@dataclass
class BlockSignal:
    """
    Combined match result for a block under a SINGLE (segment, role-binding).

    `intervals/mask` are the union support across all acceptable block windows.
    `windows_by_t0` (if present) is the full set of acceptable block windows in compact form.

    NEW:
      - n_windows: number of acceptable block windows (t0,t1)
      - n_possible_windows: number of theoretical windows in [minF..maxF] for this segment
      - example_window: (t0_first, t1_first, t1_greedy)
    """
    segment_id: str
    roles: Dict[str, str]
    T: int
    intervals: List[Interval]
    mask: Optional[Any] = None  # typically np.ndarray[bool]

    # optional heavy detail
    windows_by_t0: Optional[WindowsByT0] = None

    # cheap counts (no need to store windows_by_t0)
    n_windows: Optional[int] = None
    n_possible_windows: Optional[int] = None

    # representative hit window per binding
    example_window: Optional[Tuple[int, int, int]] = None

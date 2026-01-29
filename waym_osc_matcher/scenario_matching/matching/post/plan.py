# osc_parser/matching/post/plan.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
@dataclass
class BlockPlan:
    label: str
    type: str
    indices: List[int]
    duration_min_frames: int | None = None
    duration_max_frames: int | None = None
    overlap: str = "any"   # default

def _dur_to_frames(dur: dict | None, fps: int) -> tuple[int | None, int | None]:
    if not dur:
        return None, None
    unit = dur.get("unit", "second")
    if unit != "second":
        return None, None  # oder raise

    if "value" in dur:
        lo = hi = float(dur["value"])
    else:
        lo, hi = map(float, dur.get("range", (None, None)))
        if lo is None or hi is None:
            return None, None

    minF = max(1, int(math.ceil(lo * fps)))
    maxF = max(minF, int(math.floor(hi * fps)))
    return minF, maxF

def build_block_plans(calls: List[dict], fps: int) -> Dict[str, BlockPlan]:
    by_label: Dict[str, List[int]] = {}
    type_of: Dict[str, str] = {}
    dur_of: Dict[str, tuple[int|None, int|None]] = {}
    overlap_of: Dict[str, str] = {}

    for i, c in enumerate(calls):
        lbl = c.get("block_label") or "<none>"
        by_label.setdefault(lbl, []).append(i)
        type_of.setdefault(lbl, c.get("block_type") or "serial")

        # <-- HIER greift dein block_duration
        if lbl not in dur_of:
            minF, maxF = _dur_to_frames(c.get("block_duration"), fps)
            dur_of[lbl] = (minF, maxF)

        if lbl not in overlap_of:
            overlap_of[lbl] = c.get("block_overlap") or "any"

    out: Dict[str, BlockPlan] = {}
    for lbl, idxs in by_label.items():
        minF, maxF = dur_of.get(lbl, (None, None))
        out[lbl] = BlockPlan(
            label=lbl,
            type=type_of[lbl],
            indices=idxs,
            duration_min_frames=minF,
            duration_max_frames=maxF,
            overlap=overlap_of.get(lbl, "any"),
        )
    return out

def block_sequence(calls: List[dict]) -> List[str]:
    """
    Return the sequence of block labels in the order they first appear
    (for top-level do-serial across blocks).
    """
    seen = set()
    seq: List[str] = []
    for c in calls:
        lbl = c.get("block_label") or "<none>"
        if lbl not in seen:
            seen.add(lbl)
            seq.append(lbl)
    return seq

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

from .spec import build_block_query
from .match_block import match_block

# ---------------------------
# Helpers
# ---------------------------
def _apply_group_duration(call: Dict[str, Any], fps: int, group_duration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure every call in a parallel group shares the same duration window.
    If group_duration is given, override/insert into call['action_args']['duration'].
    """
    call = deepcopy(call)
    if group_duration:
        aa = call.setdefault("action_args", {})
        aa["duration"] = deepcopy(group_duration)
    return call

# ---------------------------
# 1) Match a PARALLEL group
# ---------------------------
def match_parallel_group(
    feats,
    calls: List[Dict[str, Any]],
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    group_duration: Optional[Dict[str, Any]] = None,
    max_results_per_call: int = 5000,
) -> List[Dict[str, Any]]:
    """
    A 'parallel' block: every call must hold over the *same* inclusive window [t_start..t_end].

    Strategy:
      - Build each call into a BlockQuery (forcing same duration if group_duration is provided).
      - Run match_block for each call separately → hits_i = [{ego,npc,t_start,t_end}, ...].
      - Intersect on identical (t_start,t_end). A window is valid only if it appears in *all* calls.

    Returns:
      [
        {
          "t_start": int, "t_end": int,
          "members": [ { "ego":..., "npc":..., "t_start":..., "t_end":... } ... ]  # one per call
        }, ...
      ]
    """
    cfg = cfg or {}

    # Build and match each call
    per_call_hits: List[List[Dict[str, Any]]] = []
    for call in calls:
        c = _apply_group_duration(call, fps, group_duration)

        Q, candidate_pairs = build_block_query(c, fps=fps, cfg=cfg)
        hits = match_block(feats, Q, fps=fps, pairs=candidate_pairs, max_results=max_results_per_call)
        per_call_hits.append(hits)

    if not per_call_hits:
        return []

    # Build index of (t_start,t_end) → list of (call_index, hit)
    # Start with windows from the first call
    base = per_call_hits[0]
    if not base:
        return []

    out: List[Dict[str, Any]] = []

    # For fast intersection, make lookup tables per call: (t0,t1) -> [hits]
    lookups: List[Dict[Tuple[int,int], List[Dict[str, Any]]]] = []
    for hits in per_call_hits:
        lut: Dict[Tuple[int,int], List[Dict[str, Any]]] = {}
        for h in hits:
            key = (h["t_start"], h["t_end"])
            lut.setdefault(key, []).append(h)
        lookups.append(lut)

    # A window is valid if it's present in every lookup
    for key in lookups[0].keys():
        if all(key in L for L in lookups[1:]):
            # collect one hit per call for this window (if multiple per call, take them all as combinations or pick first)
            # Here we pick the first per-call hit for the shared window to keep output compact.
            members = [lookups[i][key][0] for i in range(len(lookups))]
            out.append({"t_start": key[0], "t_end": key[1], "members": members})

    # Sort by start time
    out.sort(key=lambda d: (d["t_start"], d["t_end"]))
    return out

# ---------------------------------
# 2) Stitch PARALLEL groups serially
# ---------------------------------
def stitch_serial_groups(
    group_windows: List[List[Dict[str, Any]]],
    *,
    contiguous: bool = True,
    min_gap: int = 0,
    max_gap: Optional[int] = None,
    max_sequences: int = 1000,
) -> List[List[Dict[str, Any]]]:
    """
    Given a list of group windows (each element is the result from match_parallel_group),
    find serial sequences G1 -> G2 -> ... -> Gk where time increases and gaps are respected.

    Gaps are defined between successive groups as:
      - contiguous=True  -> require t_start_next == t_end_prev + 1
      - contiguous=False -> require t_start_next >= t_end_prev + 1 + min_gap
                            and (if max_gap is not None) t_start_next <= t_end_prev + 1 + max_gap

    Returns: list of sequences; each sequence is a list of
      [{ "group_index": i, "t_start":..., "t_end":..., "members": [...] }, ...]
    """
    K = len(group_windows)
    if K == 0:
        return []

    # Quick check: any group with no windows => no sequences
    for gw in group_windows:
        if not gw:
            return []

    results: List[List[Dict[str, Any]]] = []

    def ok_gap(prev_end: int, next_start: int) -> bool:
        if contiguous:
            return next_start == (prev_end + 1)
        # non-contiguous
        lo = prev_end + 1 + int(min_gap)
        if next_start < lo:
            return False
        if max_gap is not None:
            hi = prev_end + 1 + int(max_gap)
            return next_start <= hi
        return True

    def dfs(i: int, chain: List[Dict[str, Any]]):
        if len(results) >= max_sequences:
            return
        if i == K:
            results.append(chain[:])
            return
        # where do we start?
        if i == 0:
            for win in group_windows[0]:
                node = {"group_index": 0, **win}
                chain.append(node)
                dfs(1, chain)
                chain.pop()
        else:
            prev = chain[-1]
            for win in group_windows[i]:
                if ok_gap(prev["t_end"], win["t_start"]):
                    node = {"group_index": i, **win}
                    chain.append(node)
                    dfs(i + 1, chain)
                    chain.pop()

    dfs(0, [])
    return results

# ---------------------------------
# 3) High-level convenience
# ---------------------------------
def match_serial_program(
    feats,
    serial_blocks: List[Dict[str, Any]],
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    contiguous: bool = True,
    min_gap: int = 0,
    max_gap: Optional[int] = None,
    max_results_per_call: int = 5000,
    max_sequences: int = 1000,
) -> List[List[Dict[str, Any]]]:
    """
    serial_blocks format (normalized from your OSC/IR), e.g.:

      serial_blocks = [
        { "type": "parallel", "duration": {"value": 15, "unit": "s"}, "calls": [call_ego, call_npc] },
        { "type": "parallel", "duration": {"value":  5, "unit": "s"}, "calls": [call_npc_change_lane] },
        { "type": "parallel", "duration": {"value": 20, "unit": "s"}, "calls": [call_npc_slow] },
      ]

    Returns a list of serial sequences; each sequence is a list of stitched group windows.
    """
    cfg = cfg or {}

    # 1) match each parallel block into windows
    group_windows: List[List[Dict[str, Any]]] = []
    for blk in serial_blocks:
        if str(blk.get("type", "parallel")).lower() != "parallel":
            # (you can extend to other group types later)
            continue
        calls = blk.get("calls", [])
        duration = blk.get("duration")  # will be pushed to each call
        wins = match_parallel_group(
            feats, calls, fps=fps, cfg=cfg, group_duration=duration, max_results_per_call=max_results_per_call
        )
        group_windows.append(wins)

    # 2) stitch windows in serial order
    sequences = stitch_serial_groups(
        group_windows,
        contiguous=contiguous, min_gap=min_gap, max_gap=max_gap, max_sequences=max_sequences
    )
    return sequences

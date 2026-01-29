from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import copy
from .spec import build_block_query, BlockQuery
from .match_block import match_block

def _force_block_duration(call: Dict[str, Any], duration_s: float) -> Dict[str, Any]:
    c = copy.deepcopy(call)
    aargs = c.setdefault("action_args", {})
    aargs["duration"] = {"value": float(duration_s), "unit": "s"}
    return c

def _role_map_from_hit(call: Dict[str, Any], Q: BlockQuery, hit: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a mapping {role_name -> concrete_actor_id} for this call's hit.
    - role for call actor is always call["actor"] -> hit["ego"]
    - if exactly one referenced role is present, map it to hit["npc"]
      (current BlockQuery supports one 'npc' slot per call)
    """
    role_map = {str(call["actor"]): str(hit["ego"])}
    if Q.npc_candidates and len(Q.npc_candidates) == 1 and hit.get("npc") is not None:
        role_map[str(Q.npc_candidates[0])] = str(hit["npc"])
    return role_map

def _join_role_maps(maps: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Merge role->actor maps; return None if any role maps to conflicting actors.
    """
    merged: Dict[str, str] = {}
    for m in maps:
        for role, actor in m.items():
            if role in merged and merged[role] != actor:
                return None
            merged[role] = actor
    return merged

def match_parallel_block_single_segment(
    feats,
    calls: List[Dict[str, Any]],
    duration_s: float,
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    max_results_per_call: int = 5000,
) -> List[Dict[str, Any]]:

    # 1) Force block duration on each call
    forced_calls = [_force_block_duration(c, duration_s) for c in calls]

    # 2) Compile & match each call independently
    compiled: List[Tuple[Dict[str, Any], BlockQuery, List[Dict[str, Any]]]] = []
    for c in forced_calls:
        Q, _pairs = build_block_query(c, fps=fps, cfg=cfg or {})
        hits = match_block(feats, Q, fps=fps, pairs=None, max_results=max_results_per_call)
        compiled.append((c, Q, hits))

    if not compiled:
        return []

    # 3) Index hits per call by window (t0,t1)
    def _key(h): return (int(h["t_start"]), int(h["t_end"]))

    window_keys_sets = []
    indexed = []
    for c, Q, hits in compiled:
        idx = {}
        for h in hits:
            idx.setdefault(_key(h), []).append(h)
        indexed.append((c, Q, idx))
        window_keys_sets.append(set(idx.keys()))

    # Windows that appear in ALL calls (parallel semantics)
    common_windows = set.intersection(*window_keys_sets) if window_keys_sets else set()

    # 4) For each common window, try to merge role bindings
    results: List[Dict[str, Any]] = []
    for (t0, t1) in sorted(common_windows):
        # backtracking across calls to ensure consistent role->actor mapping
        def _search(i: int, acc_roles: List[Dict[str, str]], acc_hits: List[Dict[str, Any]]) -> None:
            if i == len(indexed):
                merged = _join_role_maps(acc_roles)
                if merged is not None:
                    results.append({
                        "t_start": t0,
                        "t_end": t1,
                        "roles": merged,   # {role_name -> actor_id}
                        "hits": acc_hits,  # per-call hits (optional for debugging)
                    })
                return
            c, Q, idx = indexed[i]
            for h in idx[(t0, t1)]:
                role_map = _role_map_from_hit(c, Q, h)
                merged = _join_role_maps(acc_roles + [role_map])
                if merged is not None:
                    _search(i + 1, acc_roles + [role_map], acc_hits + [h])

        _search(0, [], [])

    return results

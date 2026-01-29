# scenario_matching/matching/role_planning.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Generator, Tuple, Callable, Set
from copy import deepcopy
import numpy as np

from scenario_matching.matching.spec import build_block_query
from scenario_matching.matching.match_block import match_block

__all__ = [
    "make_type_to_candidates",
    "role_domains_from_segment",
    "prefilter_domains",
    "build_overlap_matrix",
    "enumerate_bindings",
    "remap_call",
    "roles_used_by_call",
    "match_for_binding",
]

# --------------------------------------------------------------------------------------
# Helpers: boolean runs + "valid presence" mask
# --------------------------------------------------------------------------------------

def _has_consecutive_true(x: np.ndarray, L: int) -> bool:
    """True iff x contains at least one contiguous run of True of length >= L."""
    if L <= 1:
        return bool(np.any(x))
    x = np.asarray(x, dtype=bool)
    if x.size < L:
        return False

    idx = np.flatnonzero(x)
    if idx.size < L:
        return False

    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, idx.size - 1]
    run_lengths = idx[ends] - idx[starts] + 1
    return bool(np.any(run_lengths >= L))


def actor_valid_mask(
    feats,
    actor_id: str,
    *,
    osc_id_abs_max: Optional[float] = None,
) -> np.ndarray:
    """
    Define a "valid presence" mask for an actor.

    Base: feats.present[actor_id] > 0.5
    Optional refinement: also require |feats.osc_id[actor_id]| <= osc_id_abs_max

    Notes:
      - If feats.osc_id does not exist (or missing for actor), we fall back to present-only.
      - Lengths are aligned by min(T_present, T_osc_id) if osc_id is used.
      - NaNs in osc_id are treated as invalid.
    """
    pres_map = getattr(feats, "present", {}) or {}
    p = np.asarray(pres_map.get(actor_id, []), dtype=float) > 0.5
    p = p.astype(bool)

    if osc_id_abs_max is None:
        return p

    osc_map = getattr(feats, "lane_idx", None)
    if not isinstance(osc_map, dict) or actor_id not in osc_map:
        return p

    osc = np.asarray(osc_map.get(actor_id, []), dtype=float)
    if osc.size == 0 or p.size == 0:
        return np.zeros(0, dtype=bool)

    n = min(p.size, osc.size)
    osc_n = osc[:n]
    ok = np.isfinite(osc_n) & (np.abs(osc_n) <= float(osc_id_abs_max))
    return p[:n] & ok


# --------------------------------------------------------------------------------------
# Convenience: compile+match single binding (kept for compatibility)
# --------------------------------------------------------------------------------------

def match_for_binding(feats, call, binding, fps, max_results, cfg):
    call_bound = remap_call(call, binding)
    Q, pairs_hint = build_block_query(call_bound, fps=fps, cfg=cfg)
    return match_block(
        feats=feats,
        Q=Q,
        fps=fps,
        pairs=pairs_hint,
        max_results=max_results,
    )

# --------------------------------------------------------------------------------------
# Candidate resolution
# --------------------------------------------------------------------------------------

def make_type_to_candidates(feats):
    # At least present once (cheap). More expensive minF filtering happens in prefilter_domains().
    present_ok = {
        aid for aid, pres in (getattr(feats, "present", {}) or {}).items()
        if getattr(pres, "sum", lambda: sum(1 for x in pres if x > 0.5))() > 0
    }

    def canon(t: str) -> str:
        t = (t or "").lower()
        if t in ("person", "pedestrian"): return "person"
        if t in ("vehicle", "car"):       return "vehicle"
        return t

    def infer_from_id(aid: str) -> str:
        if aid.startswith("vehicle_"):    return "vehicle"
        if aid.startswith("pedestrian_"): return "person"
        if aid.startswith("cyclist_"):    return "cyclist"
        return "unknown"

    def resolver(type_name: str):
        want = canon(type_name)
        return [aid for aid in present_ok if infer_from_id(aid) == want]

    return resolver

# --------------------------------------------------------------------------------------
# Role discovery helpers
# --------------------------------------------------------------------------------------

_REF_KEYS = ("reference", "same_as", "ahead_of", "behind", "side_of", "right_of", "left_of")

def roles_used_by_call(call) -> Set[str]:
    roles = {call["actor"]}
    aargs = call.get("action_args") or {}

    ref = aargs.get("reference")
    if isinstance(ref, str):
        roles.add(ref)

    for m in call.get("modifiers") or []:
        args = m.get("args") or {}
        for key in ("same_as", "ahead_of", "behind", "side_of", "reference"):
            v = args.get(key)
            if isinstance(v, str):
                roles.add(v)
    return roles

def _declared_role_type(scn_or_constraints: Any, role_name: str) -> str:
    actors_obj = getattr(scn_or_constraints, "actors", None)
    if isinstance(actors_obj, dict):
        inst = actors_obj.get(role_name)
        if inst is not None:
            if hasattr(inst, "type"):
                return str(inst.type).lower()
            if isinstance(inst, dict) and "type" in inst:
                return str(inst["type"]).lower()
        return ""

    if isinstance(scn_or_constraints, dict):
        actors = scn_or_constraints.get("actors", {}) or {}
        info = actors.get(role_name, {}) or {}
        t = info.get("type") or info.get("kind") or ""
        return str(t).lower()

    return ""

def _present_actor_ids(feats) -> List[str]:
    pres = (getattr(feats, "present", {}) or {})
    out = []
    for aid, p in pres.items():
        n_present = int(np.sum(np.asarray(p, dtype=float) > 0.5))
        if n_present > 0:
            out.append(aid)
    return out

def _all_roles_from_scn(scn_or_constraints: Any) -> List[str]:
    actors_obj = getattr(scn_or_constraints, "actors", None)
    if isinstance(actors_obj, dict):
        return list(actors_obj.keys())
    if isinstance(scn_or_constraints, dict):
        return list((scn_or_constraints.get("actors", {}) or {}).keys())
    return []

# --------------------------------------------------------------------------------------
# Domain construction
# --------------------------------------------------------------------------------------

def role_domains_from_segment(
    scn_constraints,
    feats,
    roles: Optional[List[str]] = None,
    *,
    type_to_candidates: Optional[Callable[[str], List[str]] | Dict[str, List[str]]] = None,
    strict_types: bool = True,
    fallback_to_all_present: bool = False,
) -> Dict[str, List[str]]:
    if roles is None:
        roles = _all_roles_from_scn(scn_constraints)

    if type_to_candidates is None:
        resolver = make_type_to_candidates(feats)
    elif callable(type_to_candidates):
        resolver = type_to_candidates
    else:
        mapping = {str(k).lower(): list(v) for k, v in (type_to_candidates or {}).items()}
        resolver = lambda t: list(mapping.get(str(t).lower(), []))

    all_present = _present_actor_ids(feats)
    out: Dict[str, List[str]] = {}

    for r in roles:
        want_type = _declared_role_type(scn_constraints, r)
        if want_type and resolver:
            cands = list(resolver(want_type))
            if cands:
                out[r] = cands
            else:
                if strict_types:
                    out[r] = []
                else:
                    out[r] = list(all_present) if fallback_to_all_present else []
        else:
            out[r] = list(all_present)
    return out

# --------------------------------------------------------------------------------------
# Domain shrinkers and overlap (using the SAME validity mask definition)
# --------------------------------------------------------------------------------------

def prefilter_domains(
    feats,
    domains: Dict[str, List[str]],
    *,
    min_present_frames: int = 1,
    require_speed: bool = True,
    require_consecutive: bool = True,
    osc_id_abs_max: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Cheap 1-actor filters to shrink domains.

    NEW (optional):
      - use actor_valid_mask(feats, actor_id, osc_id_abs_max=...) instead of raw present
      - if require_consecutive=True: require a contiguous run of length >= min_present_frames
        (this matches your "minF consecutive" grounding)
      - else: require mask.sum() >= min_present_frames (legacy behavior)
    """
    spd = getattr(feats, "speed", {}) or {}
    def ok(a: str) -> bool:
        m = actor_valid_mask(feats, a, osc_id_abs_max=osc_id_abs_max)
        if m.size == 0:
            return False

        if require_consecutive:
            if not _has_consecutive_true(m, int(min_present_frames)):
                return False
        else:
            if int(np.sum(m)) < int(min_present_frames):
                return False

        if require_speed and (a not in spd or spd[a] is None):
            return False

        return True

    return {r: [a for a in A if ok(a)] for r, A in domains.items()}


def build_overlap_matrix(
    feats,
    actors: List[str],
    *,
    min_overlap_frames: int = 1,
    osc_id_abs_max: Optional[float] = None,
    require_consecutive: bool = True,
):
    ok = {}
    masks = {a: actor_valid_mask(feats, a, osc_id_abs_max=osc_id_abs_max).astype(bool) for a in actors}

    for a in actors:
        ma = masks.get(a)
        for b in actors:
            if a == b:
                ok[(a, b)] = False
                continue
            mb = masks.get(b)
            if ma is None or mb is None or ma.size == 0 or mb.size == 0:
                ok[(a, b)] = False
                continue

            inter = ma & mb
            if require_consecutive:
                ok[(a, b)] = _has_consecutive_true(inter, int(min_overlap_frames))
            else:
                ok[(a, b)] = int(inter.sum()) >= int(min_overlap_frames)

    return ok


def enumerate_bindings(
    domains: Dict[str, List[str]],
    *,
    distinct: bool = True,
    overlap_ok: Optional[Dict[Tuple[str, str], bool]] = None,
    require_overlap_pairs: Optional[List[Tuple[str, str]]] = None,
    limit: Optional[int] = None,
) -> Generator[Dict[str, str], None, None]:
    roles = sorted(domains, key=lambda r: len(domains[r]))
    used: set[str] = set()
    bind: Dict[str, str] = {}
    out_count = 0

    need_overlap = set(tuple(p) for p in (require_overlap_pairs or []))

    def viable_with_current(r_new: str, a_new: str) -> bool:
        if distinct and a_new in used:
            return False
        if overlap_ok and need_overlap:
            for (r1, r2) in need_overlap:
                if r_new == r1 and r2 in bind:
                    if not overlap_ok.get((a_new, bind[r2]), False):
                        return False
                if r_new == r2 and r1 in bind:
                    if not overlap_ok.get((bind[r1], a_new), False):
                        return False
        return True

    def dfs(i: int):
        nonlocal out_count
        if limit is not None and out_count >= limit:
            return
        if i == len(roles):
            out_count += 1
            yield dict(bind)
            return
        r = roles[i]
        for a in domains[r]:
            if not viable_with_current(r, a):
                continue
            bind[r] = a
            used.add(a)
            yield from dfs(i + 1)
            used.remove(a)
            bind.pop(r, None)

    yield from dfs(0)

# --------------------------------------------------------------------------------------
# Call remapping
# --------------------------------------------------------------------------------------

def remap_call(call: Dict[str, Any], binding: Dict[str, str]) -> Dict[str, Any]:
    c = deepcopy(call)
    if "actor" in c:
        c["actor"] = binding.get(c["actor"], c["actor"])

    aa = c.get("action_args") or {}
    for k in _REF_KEYS:
        if k in aa and isinstance(aa[k], str):
            aa[k] = binding.get(aa[k], aa[k])

    for m in c.get("modifiers", []):
        args = m.get("args", {}) or {}
        for k in _REF_KEYS:
            if k in args and isinstance(args[k], str):
                args[k] = binding.get(args[k], args[k])

    return c
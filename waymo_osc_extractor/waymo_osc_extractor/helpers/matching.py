# matching.py
# Robust, window-aware OpenSCENARIO block matcher with gap tolerance.
# Includes Option A: allow the end condition earlier than duration_s (cfg["allow_shorter_end"]=True).

from typing import Any, Dict, List, Tuple, Optional

# ========= Context =========

class EvalCtx:
    """
    Bundles everything a check needs:
      - stitched: the stitched scenario dict
      - segment:  the current segment dict from stitched["segments"][seg_id]
      - ego/npc:  actor ids (strings)
      - t0..t1:   inclusive window
      - cfg:      optional config dict with tolerance knobs (see below)
    """
    def __init__(
        self,
        stitched: Dict[str, Any],
        segment: Dict[str, Any],
        ego: str,
        npc: str,
        t0: int,
        t1: int,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.stitched = stitched
        self.segment = segment
        self.ego = ego
        self.npc = npc
        self.t0 = t0
        self.t1 = t1
        self.cfg = cfg or {}  # e.g., {"distance_tol": 2.0, "presence_min_coverage": 0.9, ...}


# ========= Shared primitives & tolerant helpers =========

POSITION_NAME = {
    -1: "unknown",
     0: "not related",
     1: "front",
     2: "left",
     3: "right",
     4: "back",
}

LATERAL = {"right_of", "left_of", "same_lane"}
LONGITUDINAL = {"ahead_of", "behind"}


def _is_missing(x) -> bool:
    """Treat None, NaN, or non-finite numeric as missing."""
    try:
        from math import isnan, isfinite
        if x is None:
            return True
        if isinstance(x, (int, float)):
            try:
                if not isfinite(x):
                    return True
                return bool(isinstance(x, float) and isnan(x))
            except Exception:
                return False
        return False
    except Exception:
        return x is None


def _enough_covered(valid_count: int, total: int, cfg_key_allow: str, cfg_key_cov: str, cfg: Dict[str, Any]) -> bool:
    """
    Decide if a window has enough valid samples.
      - If cfg[cfg_key_allow] is an int, allow up to that many missing frames.
      - Else fall back to minimum coverage ratio cfg[cfg_key_cov] (default 1.0).
    """
    allow_missing = cfg.get(cfg_key_allow)
    if isinstance(allow_missing, int):
        return (total - valid_count) <= allow_missing
    min_cov = float(cfg.get(cfg_key_cov, 1.0))
    return (valid_count / max(1, total)) >= min_cov


def lane_index_at(segment: Dict[str, Any], actor_id: str, t: int) -> Optional[int]:
    """Return lane index (1..num_lanes) for actor at time t, or None."""
    seq = segment.get("lane_index", {}).get(actor_id, [])
    return seq[t] if 0 <= t < len(seq) else None


def lane_index_at_snap(segment: Dict[str, Any], actor_id: str, t: int, radius: int) -> Optional[int]:
    """Search within +/- radius around t for the first non-missing lane index."""
    seq = segment.get("lane_index", {}).get(actor_id, [])
    if not isinstance(seq, list) or t < 0 or t >= len(seq):
        return None
    if seq[t] is not None:
        return seq[t]
    for d in range(1, radius + 1):
        lt = t - d
        rt = t + d
        if 0 <= lt < len(seq) and seq[lt] is not None:
            return seq[lt]
        if 0 <= rt < len(seq) and seq[rt] is not None:
            return seq[rt]
    return None


def present_window(segment: Dict[str, Any], actor_id: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    """
    Tolerant presence check:
      - Count frames in [t0..t1] with present==1.
      - Pass if allowed-missing (absolute) or min-coverage (ratio) satisfied.
    """
    seq = segment.get("present", {}).get(actor_id, [])
    if t1 >= len(seq):
        return False
    total = t1 - t0 + 1
    valid = 0
    for t in range(t0, t1 + 1):
        try:
            if seq[t] == 1:
                valid += 1
        except Exception:
            pass
    return _enough_covered(valid, total, "presence_allow_missing", "presence_min_coverage", cfg)


def same_lane_window(segment: Dict[str, Any], actor_id: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    """
    Tolerant 'stay in lane':
      - All known (non-None) lane indices in [t0..t1] must be identical.
      - Missing values allowed per presence coverage knobs.
    """
    seq = segment.get("lane_index", {}).get(actor_id, [])
    if t1 >= len(seq):
        return False
    win = seq[t0 : t1 + 1]
    known = [v for v in win if v is not None]
    if not known:
        return False
    if len(set(known)) != 1:
        return False
    valid = len(known)
    total = len(win)
    return _enough_covered(valid, total, "presence_allow_missing", "presence_min_coverage", cfg)


def speed_window_ok(
    stitched: Dict[str, Any],
    actor_id: str,
    t0: int,
    t1: int,
    lo: float,
    hi: float,
    cfg: Dict[str, Any],
) -> bool:
    """
    Tolerant speed range check:
      - Values in [t0..t1] must fall in [lo - tol, hi + tol] for enough frames.
      - Missing values allowed per 'speed_allow_missing' or 'speed_min_coverage'.
    """
    tol = float(cfg.get("speed_value_tol", 0.0))
    lo2, hi2 = lo - tol, hi + tol

    v = (
        stitched.get("activities", {})
        .get("vehicle", {})
        .get(f"{actor_id}_activity", {})
        .get("long_v")
    )
    if not v or t1 >= len(v):
        return False

    total = t1 - t0 + 1
    valid = 0
    for x in v[t0 : t1 + 1]:
        if _is_missing(x):
            continue
        xv = float(x)
        if lo2 <= xv <= hi2:
            valid += 1

    return _enough_covered(valid, total, "speed_allow_missing", "speed_min_coverage", cfg)


def rel_item_at_snap(stitched: Dict[str, Any], ego: str, npc: str, key: str, t: int, radius: int) -> Optional[float]:
    """
    Tolerant relation getter:
      - relations[ego][npc][key] is a list (e.g., 'position', 'distance')
      - Looks at t; if missing, searches +/- radius for first valid.
    """
    e = stitched.get("relations", {}).get(ego, {})
    arr = e.get(npc, {}).get(key)
    if not isinstance(arr, list) or t < 0 or t >= len(arr):
        return None
    val = arr[t]
    if not _is_missing(val):
        return val
    for d in range(1, radius + 1):
        lt = t - d
        rt = t + d
        if 0 <= lt < len(arr) and not _is_missing(arr[lt]):
            return arr[lt]
        if 0 <= rt < len(arr) and not _is_missing(arr[rt]):
            return arr[rt]
    return None


def pos_is(code_val: float, desired_name: str) -> bool:
    """Compare numeric 'position' code to 'front'/'back'/..."""
    try:
        return POSITION_NAME.get(int(code_val), "unknown") == desired_name
    except Exception:
        return False


def lateral_relation_at_tolerant(segment: Dict[str, Any], ego: str, npc: str, t: int, cfg: Dict[str, Any]) -> str:
    """
    Lateral relation using lane indices with optional snap radius:
      - 'right_of' if npc_lane > ego_lane
      - 'left_of'  if npc_lane < ego_lane
      - 'same_lane' if equal
      - 'unknown' if missing
    """
    rad = int(cfg.get("relation_snap_radius", 0))
    allow_missing = bool(cfg.get("lateral_allow_missing", True))
    if rad > 0 and allow_missing:
        e = lane_index_at_snap(segment, ego, t, rad)
        n = lane_index_at_snap(segment, npc, t, rad)
    else:
        e = lane_index_at(segment, ego, t)
        n = lane_index_at(segment, npc, t)
    if e is None or n is None:
        return "unknown"
    if n > e:
        return "right_of"
    if n < e:
        return "left_of"
    return "same_lane"


def in_range(x: float, lo: Optional[float] = None, hi: Optional[float] = None) -> bool:
    """Inclusive numeric range check."""
    x = float(x)
    if lo is not None and x < float(lo):
        return False
    if hi is not None and x > float(hi):
        return False
    return True


def distance_match(d_obs: Optional[float], cond: Dict[str, Any], tol: float) -> bool:
    """
    Check distance constraint:
      - 'distance' (exact within ±tol)
      - 'distance_range' = [lo, hi]
      - 'distance_min', 'distance_max'
      If no distance constraint provided, returns True.
    """
    if d_obs is None:
        return False

    if "distance" in cond and cond["distance"] is not None:
        try:
            return abs(float(d_obs) - float(cond["distance"])) <= float(tol)
        except Exception:
            return False

    if "distance_range" in cond and cond["distance_range"] is not None:
        try:
            lo, hi = cond["distance_range"]
            return in_range(d_obs, lo, hi)
        except Exception:
            return False

    lo = cond.get("distance_min")
    hi = cond.get("distance_max")
    if lo is not None or hi is not None:
        return in_range(d_obs, lo, hi)

    return True  # no distance constraint


# ========= Modular checks (each returns (ok, reason)) =========

CheckResult = Tuple[bool, str]


def check_map_min_lanes(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """
    Require exact match on lane count if 'map' present.
    Change '==' to '>=' if you later want "at least N lanes".
    """
    if "map" not in block:
        return (True, "no map constraint")
    want = int(block["map"].get("min_lanes", 1))
    have = int(ctx.segment.get("meta", {}).get("num_lanes", 0))
    ok = (have == want)
    return (ok, "num_lanes={} != {}".format(have, want))


def check_presence_window(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """Both ego and npc must be present sufficiently in [t0..t1]."""
    if not present_window(ctx.segment, ctx.ego, ctx.t0, ctx.t1, ctx.cfg):
        return (False, "ego not present enough in window")
    if not present_window(ctx.segment, ctx.npc, ctx.t0, ctx.t1, ctx.cfg):
        return (False, "npc not present enough in window")
    return (True, "")


def check_ego_lane_at_start(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """If specified, ego must start window at given lane index."""
    lane_str = block.get("ego", {}).get("lane_at_start")
    if lane_str is None:
        return (True, "")
    try:
        exp = int(lane_str)
    except Exception:
        return (False, "lane_at_start not integer: {}".format(lane_str))
    l0 = lane_index_at(ctx.segment, ctx.ego, ctx.t0)
    return (l0 == exp, "ego lane_at_start {} != {}".format(l0, exp))


def check_ego_stay_in_lane(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """If 'stay_in_lane' is True, enforce constant lane over window with tolerance."""
    if not block.get("ego", {}).get("stay_in_lane", False):
        return (True, "")
    ok = same_lane_window(ctx.segment, ctx.ego, ctx.t0, ctx.t1, ctx.cfg)
    return (ok, "ego not in the same lane over window")


def check_ego_speed_range(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """
    If 'speed_range' provided, enforce speed within range (with tolerance & coverage).
    Accepts scalar (exact) or (lo, hi).
    """
    sr = block.get("ego", {}).get("speed_range")
    if sr is None:
        return (True, "")
    if isinstance(sr, (int, float)):
        lo = hi = float(sr)
    else:
        lo = float(sr[0])
        hi = float(sr[1])
    ok = speed_window_ok(ctx.stitched, ctx.ego, ctx.t0, ctx.t1, lo, hi, ctx.cfg)
    return (ok, "ego speed outside range/coverage")


def check_ego_change_speed(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """
    If 'change_speed' present, require delta v ~ target over window.
    Uses cfg['change_speed_tol'] as absolute tolerance (default 0.3 m/s).
    """
    if "ego" not in block or "change_speed" not in block["ego"]:
        return (True, "")
    target = float(block["ego"]["change_speed"])
    tol = float(ctx.cfg.get("change_speed_tol", 0.3))
    v = (
        ctx.stitched.get("activities", {})
        .get("vehicle", {})
        .get(f"{ctx.ego}_activity", {})
        .get("long_v")
    )
    if not v or ctx.t1 >= len(v):
        return (False, "no speed series")
    try:
        dv = float(v[ctx.t1]) - float(v[ctx.t0])
    except Exception:
        return (False, "invalid speed samples")
    ok = abs(dv - target) <= tol
    return (ok, "Δv={:.3f} vs target={:.3f} (tol={:.3f})".format(dv, target, tol))


def _check_relatives_at(ctx: EvalCtx, rel_list: List[Dict[str, Any]], t: int) -> CheckResult:
    """
    Evaluate a list of relative conditions at (approximately) time t:
      - Lateral (right_of/left_of/same_lane) via lane indices, tolerant to gaps.
      - Longitudinal (ahead_of/behind) via relations.position code, tolerant to gaps.
      - Optional distance checks via relations.distance, tolerant to gaps.
    """
    if not rel_list:
        return (True, "")
    tol = float(ctx.cfg.get("distance_tol", 2.0))
    rad = int(ctx.cfg.get("relation_snap_radius", 0))

    for cond in rel_list:
        rel = cond.get("relation")

        if rel in LATERAL:
            lat = lateral_relation_at_tolerant(ctx.segment, ctx.ego, ctx.npc, t, ctx.cfg)
            if lat != rel:
                return (False, "lateral near t={} is {} != {}".format(t, lat, rel))
            if any(k in cond for k in ("distance", "distance_range", "distance_min", "distance_max")):
                d = rel_item_at_snap(ctx.stitched, ctx.ego, ctx.npc, "distance", t, rad)
                if not distance_match(d, cond, tol):
                    return (False, "distance mismatch near t={}: {} vs {}".format(t, d, cond))

        elif rel in LONGITUDINAL:
            need = "front" if rel == "ahead_of" else "back"
            pos = rel_item_at_snap(ctx.stitched, ctx.ego, ctx.npc, "position", t, rad)
            if pos is None or not pos_is(pos, need):
                return (False, "longitudinal near t={} is {} != {}".format(t, pos, need))
            if any(k in cond for k in ("distance", "distance_range", "distance_min", "distance_max")):
                d = rel_item_at_snap(ctx.stitched, ctx.ego, ctx.npc, "distance", t, rad)
                if not distance_match(d, cond, tol):
                    return (False, "distance mismatch near t={}: {} vs {}".format(t, d, cond))

        else:
            return (False, "unknown relation: {}".format(rel))

    return (True, "")


def check_npc_rel_start(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """Validate all 'relative_at_start' conditions at (t0, tolerant)."""
    rels = block.get("npc", {}).get("relative_at_start", [])
    return _check_relatives_at(ctx, rels, ctx.t0)


def check_npc_rel_end(ctx: EvalCtx, block: Dict[str, Any]) -> CheckResult:
    """Validate all 'relative_at_end' conditions at (t1, tolerant)."""
    rels = block.get("npc", {}).get("relative_at_end", [])
    return _check_relatives_at(ctx, rels, ctx.t1)


# ========= Registry & evaluator =========

BLOCK_CHECKS = [
    check_map_min_lanes,    # cheap prune: segment-level
    check_presence_window,  # cheap prune: both actors present enough
    check_ego_lane_at_start,
    check_ego_stay_in_lane,
    check_ego_speed_range,
    check_ego_change_speed, # only applies if 'change_speed' provided
    check_npc_rel_start,
    check_npc_rel_end,
]


def check_block(ctx: EvalCtx, block: Dict[str, Any]) -> Tuple[bool, str]:
    """Run all registered checks in order; return first failure or success."""
    for fn in BLOCK_CHECKS:
        ok, reason = fn(ctx, block)
        if not ok:
            return (False, "{}: {}".format(fn.__name__, reason))
    return (True, "")


# ========= Search driver (window-aware, with Option A) =========

def find_block_matches(
    stitched: Dict[str, Any],
    block: Dict[str, Any],
    fps: int = 1,
    max_results: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Slide a time window across all segments and actor pairs.
    With cfg["allow_shorter_end"]=True, the end condition can be satisfied earlier than duration_s.
    Returns:
      [{ "segment": seg_id, "ego": ego_id, "npc": npc_id, "t_start": t0, "t_end": t1 }, ...]
    """
    results: List[Dict[str, Any]] = []
    cfg = cfg or {}

    segments = stitched.get("segments", {})
    dur_frames = max(1, int(block.get("duration_s", 0) * fps))

    # Option A knob
    allow_shorter = bool(cfg.get("allow_shorter_end", False))
    # minimal frames required if allowing shorter ends (you can make this a cfg knob too)
    min_frames = 1 if allow_shorter else dur_frames

    want_lanes = None
    if "map" in block and "min_lanes" in block["map"]:
        try:
            want_lanes = int(block["map"]["min_lanes"])
        except Exception:
            want_lanes = None

    for seg_id, segment in segments.items():
        # Exact lane count filter (if specified)
        if want_lanes is not None:
            have = int(segment.get("meta", {}).get("num_lanes", 0))
            if have != want_lanes:
                continue

        T = int(segment.get("meta", {}).get("T", 91))
        actors = segment.get("actors", [])

        last_t0 = max(0, T - min_frames)
        for i in range(len(actors)):
            for j in range(len(actors)):
                if i == j:
                    continue
                ego = actors[i]
                npc = actors[j]

                for t0 in range(0, last_t0 + 1):
                    # scan t1 within [t0 + min_frames - 1 .. t0 + dur_frames - 1]
                    t1_min = min(T - 1, t0 + min_frames - 1)
                    t1_max = min(T - 1, t0 + dur_frames - 1)

                    found_t1 = None
                    for t1 in range(t1_min, t1_max + 1):
                        ctx = EvalCtx(stitched, segment, ego, npc, t0, t1, cfg)
                        ok, _ = check_block(ctx, block)
                        if ok:
                            found_t1 = t1
                            break

                    if found_t1 is not None:
                        results.append(
                            {
                                "segment": seg_id,
                                "ego": ego,
                                "npc": npc,
                                "t_start": t0,
                                "t_end": found_t1,  # may be earlier than t0+dur_frames-1
                            }
                        )
                        if len(results) >= max_results:
                            return results

    return results


# ========= Optional: sequential chaining across blocks (unchanged) =========

def find_sequence_matches(
    stitched: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    fps: int = 1,
    cfg: Optional[Dict[str, Any]] = None,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Find sequences of windows matching a list of blocks in order.
    Keeps same (segment, ego, npc); searches forward in time.
    """
    cfg = cfg or {}
    sequences: List[Dict[str, Any]] = []

    # seed with block-0 matches
    seeds = find_block_matches(stitched, blocks[0], fps=fps, cfg=cfg, max_results=10_000)

    for seed in seeds:
        seg_id = seed["segment"]
        ego = seed["ego"]
        npc = seed["npc"]
        windows = [{"t_start": seed["t_start"], "t_end": seed["t_end"]}]
        ok_chain = True
        last_end = seed["t_end"]

        seg = stitched.get("segments", {}).get(seg_id)
        if not seg:
            continue
        T = int(seg.get("meta", {}).get("T", 91))

        for k in range(1, len(blocks)):
            block = blocks[k]
            dur_frames = max(1, int(block.get("duration_s", 0) * fps))
            allow_shorter = bool(cfg.get("allow_shorter_end", False))
            min_frames = 1 if allow_shorter else dur_frames

            found_next = None
            # start strictly after the previous window
            for t0 in range(last_end + 1, max(0, T - min_frames) + 1):
                t1_min = min(T - 1, t0 + min_frames - 1)
                t1_max = min(T - 1, t0 + dur_frames - 1)

                for t1 in range(t1_min, t1_max + 1):
                    ctx = EvalCtx(stitched, seg, ego, npc, t0, t1, cfg)
                    ok, _ = check_block(ctx, block)
                    if ok:
                        found_next = {"t_start": t0, "t_end": t1}
                        break
                if found_next:
                    break

            if found_next is None:
                ok_chain = False
                break

            windows.append(found_next)
            last_end = found_next["t_end"]

        if ok_chain:
            sequences.append({
                "segment": seg_id,
                "ego": ego,
                "npc": npc,
                "windows": windows
            })
            if len(sequences) >= max_results:
                break

    return sequences
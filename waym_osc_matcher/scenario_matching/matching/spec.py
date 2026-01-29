# osc_parser/matching/spec.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import numpy as np

# ======================================================================================
# BlockQuery schema
# ======================================================================================
@dataclass
class BlockQuery:
    """
    What the matcher needs for a single-block/window query.

    Backwards-compatible:
      - `checks` is the legacy list of window predicates.
    New (OSC2-näher, optional):
      - `start_checks`, `end_checks`, `during_frame_checks` allow S/E/D decomposition
        (evaluate at t0, at t1, and per-frame respectively) to build per-call intervals.
      - `window_checks` contains predicates that need the full window (e.g. change_lane).
    """
    ego: str
    npc_candidates: List[str]                 # if empty, driver may try all others
    duration_frames: int                      # legacy: typical window length in frames (inclusive end)
    checks: List[Callable[..., bool]]         # legacy: fn(feats, ego, npc, t0, t1, cfg) -> bool
    cfg: Dict[str, Any] = field(default_factory=dict)

    # --- S/E/D decomposition (optional; used by newer matcher) ---
    start_checks: List[Callable[..., bool]] = field(default_factory=list)          # evaluate at t0
    end_checks:   List[Callable[..., bool]] = field(default_factory=list)          # evaluate at t1
    during_frame_checks: List[Callable[..., bool]] = field(default_factory=list)   # evaluate per-frame (t,t)
    window_checks: List[Callable[..., bool]] = field(default_factory=list)         # evaluate on (t0,t1)

    # optional metadata (set by build_block_query)
    arity: int = 1
    roles_used: List[str] = field(default_factory=list)

# ---- local role scanner to avoid circular import with role_planning ----
_REF_KEYS_LOCAL = ("reference", "same_as", "ahead_of", "behind", "side_of")

def _label(label, fn):
    def wrapped(F, E, N, t0, t1, C):
        ok = fn(F, E, N, t0, t1, C)
        if C.get("debug_checks"):
            print(f"[check] {label} t=[{t0},{t1}) -> {ok}")
        return ok
    wrapped._label = label
    return wrapped

def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-π, π]."""
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _lane_sign_from_id(L: np.ndarray) -> np.ndarray:
    """+1 for positive lane ids, -1 for negative, 0 otherwise (vectorized)."""
    L = np.asarray(L)
    s = np.zeros_like(L, dtype=int)
    s[L > 0] = +1
    s[L < 0] = -1
    return s

def _yaw_diff_to_parallel(yaw_delta: np.ndarray, lane_ids: np.ndarray) -> np.ndarray:
    """
    |Δ| to the lane's forward direction:
      - desired = 0 rad for negative lanes
      - desired = π rad for positive lanes
    Returns absolute difference in [0, π].
    """
    yaw = np.asarray(yaw_delta, dtype=float)
    L   = np.asarray(lane_ids, dtype=float)
    desired = np.full_like(yaw, np.nan, dtype=float)
    desired[L <  0] = 0.0
    desired[L >  0] = np.pi
    return np.abs(_wrap_pi(yaw - desired))

def _yaw_dist_to_perp(yaw_delta: np.ndarray, lane_ids: np.ndarray) -> np.ndarray:
    """
    Distance to perpendicular (±π/2) relative to the lane direction.
    Uses the parallel diff then measures | |Δ| - π/2 |.
    """
    dpar = _yaw_diff_to_parallel(yaw_delta, lane_ids)  # [0, π]
    return np.abs(np.abs(dpar) - (np.pi / 2.0))

def relpos_from_s(feats, ego, npc, t):
    se = feats.s.get(ego)
    sn = feats.s.get(npc)
    if se is None or sn is None or not (np.isfinite(se[t]) and np.isfinite(sn[t])):
        return "unknown"
    d = float(sn[t] - se[t])
    return "front" if d > 0 else "back"

def _roles_used_by_call_local(call: Dict[str, Any]) -> List[str]:
    roles = set()
    actor = call.get("actor")
    if isinstance(actor, str):
        roles.add(actor)
    aargs = call.get("action_args") or {}
    for k in _REF_KEYS_LOCAL:
        v = aargs.get(k)
        if isinstance(v, str):
            roles.add(v)
    for m in call.get("modifiers") or []:
        args = m.get("args") or {}
        for k in _REF_KEYS_LOCAL:
            v = args.get(k)
            if isinstance(v, str):
                roles.add(v)
    return sorted(roles)

def _lane_dir_sign_odr(side: Optional[str]) -> int:
    if not isinstance(side, str): return 0
    s = side.strip().lower()
    if s == "left":  return +1   # left = increase in index
    if s == "right": return -1   # right = decrease in index
    return 0

def _step_skip_zero(L0: int, steps: int, sgn: int) -> int:
    if steps <= 0: return L0
    L, moved = L0, 0
    while moved < steps:
        L += sgn
        if L == 0:
            L += sgn
        moved += 1
    return L

def _keep_eval(ok_series: np.ndarray,
               eval_mask: np.ndarray,
               L: int,
               cfg: Dict[str, Any],
               coverage_key: str,
               default_need: float) -> bool:
    """
    ok_series: length-L bool array: True where the predicate passes (False elsewhere)
    eval_mask: length-L bool array: True where the signal was evaluable (present & finite)
    L        : window length (t1 - t0 + 1)
    coverage_key/default_need: min required coverage (e.g., 'speed_min_coverage', 0.9)
    """
    mode = str(cfg.get("during_mode", _DEFAULT_CFG.get("during_mode", "all"))).lower()

    if mode == "coverage":
        good = int(np.sum(ok_series & eval_mask))
        den  = int(L)  # FULL WINDOW denominator
        if den <= 0:
            return False
        need = float(cfg.get(coverage_key, default_need))
        return (good / den) >= need

    # "all" mode: strict → any unknown OR failure → False
    # i.e., must have eval for ALL frames AND all must pass
    all_evaluable = (int(np.sum(eval_mask)) == L)
    all_ok = (int(np.sum(ok_series & eval_mask)) == L)
    return all_evaluable and all_ok
    
# ======================================================================================
# Units, normalization, small helpers
# ======================================================================================
_SPEED_UNITS = {
    "m/s": 1.0,
    "meter_per_second": 1.0,
    "meters_per_second": 1.0,
    "kilometer_per_hour": 1.0 / 3.6,
    "kilometers_per_hour": 1.0 / 3.6,
    "kph": 1.0 / 3.6,
    "km/h": 1.0 / 3.6,
    "mph": 0.44704,
}

_ANGLE_UNITS = {
    "rad": 1.0, "radian": 1.0, "radians": 1.0,
    "deg": math.pi / 180.0, "degree": math.pi / 180.0, "degrees": math.pi / 180.0,
}

_DISTANCE_UNITS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "millimeter": 0.001, "mm": 0.001, "feet": 0.3048, "ft": 0.3048,
}

_TIME_UNITS = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
}

_ACCEL_UNITS = {
    "m/s^2": 1.0, "mps2": 1.0,
    "km/h/s": 1000.0/3600.0, "kph/s": 1000.0/3600.0,
}

_JERK_UNITS = {
    "m/s^3": 1.0, "mps3": 1.0,
}

_DEFAULT_CFG = {

    # presence & coverage
    "presence_min_coverage": 0.9,
    "presence_allow_missing": 0,  # number of frames allowed missing (None to use coverage ratio instead)
    "speed_min_coverage": 0.9,    # also used for distance coverage unless you add a dedicated key

    # tolerances
    "speed_value_tol": 0.10,      # m/s
    "distance_tol": 2.0,          # m
    "change_speed_tol": 0.30,     # m/s

    # lateral / relation
    "relation_snap_radius": 1,    # (placeholder)
    "lateral_allow_missing": True,

    # duration handling
    "allow_shorter_end": True,
    "default_window_s": 5.0,

    # during semantics (window checks)
    "during_mode": "all",
    "during_max_false": 0,         # only used if during_mode == "all"
    "st_reach_tol_s": 1.0,
    "st_reach_tol_t": 0.5,

    # assign_orientation tolerance (radians)
    "yaw_reach_tol": 0.05,  # ≈ 2.9°
    "accel_value_tol": 0.2,  # m/s^2
    "stationary_max_false": 0,      # allow this many violations inside the window (0 = strict)
    "keep_speed_tol": 0.20,   # m/s deviation allowed from sampled speed
    "jerk_value_tol": 0.2,     # m/s^3 tolerance when checking rate_peak
    "accel_min_coverage": 0.9,     # coverage when during_mode == "coverage"

    # --- lane following knobs ---
    "lane_follow_mode": "all",
    "lane_follow_allow_false": 0,
    "lane_min_coverage": 0.95,

    # --- time headway knobs ---
    "time_headway_tol": 0.30,     # seconds
    "headway_min_coverage": 0.85,
    "min_speed_for_headway": 0.30,  # m/s

    # --- lane ordering ---
    "lane_id_convention": "opendrive_rht",
    #--- return block duration to default ---
    "legacy_duration_windows": False,
}

_DEFAULT_CFG.update({
    "speed_series_unit": "m/s",  # set to "km/h" or "mph" if your series is stored that way
})

def _label(title, fn):
    def wrapped(F, E, N, t0, t1, C):
        # If your original _label does more, comment it out for now.
        return fn(F, E, N, t0, t1, C)
    wrapped.__name__ = f"check:{title}"
    return wrapped

def _lat_sign_from_side(side: Optional[str]) -> int:
    s = (side or "").lower()
    if s == "left":  return +1   # OpenDRIVE: +t is left
    if s == "right": return -1
    return 0

def _series_tdot_or_from_t(feats, ego: str, fps: float) -> np.ndarray:
    td = getattr(feats, "t_dot", {}).get(ego)
    if td is not None:
        return np.asarray(td, dtype=float)
    # fallback: finite-diff t
    t = _get(feats.t, ego)
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.array([], dtype=float)
    out = np.full_like(t, np.nan, dtype=float)
    if t.size >= 2:
        out[1:] = np.diff(t) * float(fps)
    return out

def _lane_delta_sign(side: Optional[str], cfg: Dict[str, Any], base_lane: int) -> int:
    # Align with features & modifiers: larger lane index = right
    s = (side or "").lower()
    if s == "left":  return -1
    if s == "right": return +1
    return 0

def _norm_physical(d: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """
    Normalize {"value":..,"unit":..} / {"range":[lo,hi],"unit":..} to SI.
    kind ∈ {"speed","angle","distance","acceleration","jerk"}.
    """
    if not d:
        return {}
    if "value" in d:
        v = float(d["value"]); u = str(d.get("unit", "")).lower()
        if kind == "speed":
            return {"value": v * _SPEED_UNITS.get(u, 1.0), "unit": "m/s"}
        if kind == "angle":
            return {"value": v * _ANGLE_UNITS.get(u, 1.0), "unit": "rad"}
        if kind == "distance":
            return {"value": v * _DISTANCE_UNITS.get(u, 1.0), "unit": "m"}
        if kind == "acceleration":
            return {"value": v * _ACCEL_UNITS.get(u, 1.0), "unit": "m/s^2"}
        if kind == "jerk":
            return {"value": v * _JERK_UNITS.get(u, 1.0), "unit": "m/s^3"}
    if "range" in d:
        lo, hi = d["range"]; u = str(d.get("unit", "")).lower()
        if kind == "speed":
            s = _SPEED_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s"}
        if kind == "angle":
            s = _ANGLE_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "rad"}
        if kind == "distance":
            s = _DISTANCE_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m"}
        if kind == "acceleration":
            s = _ACCEL_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s^2"}
        if kind == "jerk":
            s = _JERK_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s^3"}
    return d

def _anchor_index(t0: int, t1: int, at: Optional[str]) -> int:
    return t1 if (str(at).lower() == "end") else t0

def _val_or_range_to_bounds(spec: Dict[str, Any], tol: float = 0.0) -> Tuple[float, float]:
    if not spec:
        return (-np.inf, np.inf)
    if "value" in spec:
        v = float(spec["value"])
        return v - tol, v + tol
    if "range" in spec:
        lo, hi = spec["range"]
        return float(lo), float(hi)
    return (-np.inf, np.inf)

def _coverage_ratio(x: np.ndarray, lo: float, hi: float) -> float:
    m = np.isfinite(x)
    if not np.any(m): return 0.0
    ok = (x >= lo) & (x <= hi) & m
    return float(np.sum(ok) / np.sum(m))

def _presence_coverage(pres: np.ndarray) -> float:
    if pres is None or len(pres) == 0: return 0.0
    m = np.isfinite(pres)
    if not np.any(m): return 0.0
    return float(np.mean(pres[m] > 0.5))

def _all_during_in_range(x: np.ndarray, lo: float, hi: float, pres: Optional[np.ndarray], max_false: int) -> bool:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if pres is not None:
        m &= (np.asarray(pres) > 0.5)
    if not np.any(m):
        return False
    ok = (x >= lo) & (x <= hi)
    violations = int(np.sum(m & ~ok))
    return violations <= int(max_false)

def _get(arr_map: Dict[str, np.ndarray], key: str) -> np.ndarray:
    a = arr_map.get(key)
    return np.asarray(a) if a is not None else np.array([], dtype=float)

def _to_seconds_unit(u: Optional[str]) -> float:
    if not u:
        return 1.0
    return _TIME_UNITS.get(str(u).lower(), 1.0)

def _time_val_or_range_to_bounds(spec: Dict[str, Any], tol: float) -> Tuple[float, float]:
    if "value" in spec:
        k = _to_seconds_unit(spec.get("unit"))
        v = float(spec["value"]) * k
        return (v - float(tol), v + float(tol))
    if "range" in spec:
        k = _to_seconds_unit(spec.get("unit"))
        lo, hi = spec["range"]
        return (float(lo) * k, float(hi) * k)
    return (-tol, +tol)

def _signed_longitudinal_gap(feats, ego: str, npc: str, t: int) -> Optional[float]:
    s_e = getattr(feats, "s", {}).get(ego)
    s_n = getattr(feats, "s", {}).get(npc)
    if s_e is None or s_n is None:
        return None
    if t >= len(s_e) or t >= len(s_n):
        return None
    se, sn = float(s_e[t]), float(s_n[t])
    if not (np.isfinite(se) and np.isfinite(sn)):
        return None
    return se - sn

def _ego_longitudinal_speed(feats, ego: str, t: int) -> Optional[float]:
    sdot = getattr(feats, "s_dot", {}).get(ego)
    if sdot is not None and t < len(sdot):
        val = float(sdot[t])
        return val if np.isfinite(val) else None
    v = _get(feats.speed, ego)
    if t >= v.size:
        return None
    val = float(v[t])
    if not np.isfinite(val):
        return None
    return abs(val)

def _headway_time_mag(feats, ego: str, npc: str, t: int, min_speed: float) -> Optional[float]:
    ds = _signed_longitudinal_gap(feats, ego, npc, t)
    if ds is None:
        return None
    v_long = _ego_longitudinal_speed(feats, ego, t)
    if v_long is None:
        return None
    v_abs = abs(v_long)
    if not np.isfinite(v_abs) or v_abs < float(min_speed):
        return None
    return abs(ds) / v_abs


# ======================================================================================
# Checks (each returns bool for a given window [t0..t1], inclusive)
# ======================================================================================

def _check_change_lane_action(
    feats,
    ego: str,
    npc: Optional[str],          # unused; kept for signature consistency
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_lane: Optional[int],
    num_of_lanes: Optional[int],
    side: Optional[str],
    reference: Optional[str],
) -> bool:
    """
    Accept if the actor starts in a valid lane at t0, and there exists a time j in [t0, t1]
    such that from j through t1 the lane is CONSTANT and equal to the intended target lane.
    The change can happen anywhere in the window, but once changed, it must remain until t1.
    Unknown / not-present → False. OpenDRIVE indexing. Lane 0 is invalid.
    """
    le = _get(feats.lane_idx, ego)
    if le.size == 0 or t0 >= le.size or t1 >= le.size:
        return False

    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres = np.asarray(pres, dtype=float) > 0.5
    if not (pres[t0] and pres[t1]):
        return False

    # lane at t0 must be known & non-zero
    L0f = float(le[t0])
    if not np.isfinite(L0f):
        return False
    L0 = int(np.rint(L0f))
    if L0 == 0:
        return False

    # Determine target lane
    Ltarget: Optional[int] = None
    if target_lane is not None:
        Ltarget = int(target_lane)
    else:
        steps = int(num_of_lanes) if (num_of_lanes is not None) else 1
        if steps <= 0:
            return False
        sgn = _lane_dir_sign_odr(side)
        if sgn == 0:
            return False

        # Base lane for relative targeting:
        #   - If no reference or reference==ego: base at t0 (your new semantics).
        #   - Else (kept behavior): base from reference at t0 if available; require presence/finite.
        if not reference or reference == ego:
            base_lane = L0
        else:
            lr = _get(feats.lane_idx, reference)
            if lr.size == 0 or t0 >= lr.size:
                return False
            if not pres[t0]:
                return False
            if not np.isfinite(lr[t0]):
                return False
            base_lane = int(np.rint(lr[t0]))
            if base_lane == 0:
                return False

        Ltarget = _step_skip_zero(base_lane, steps, sgn)

    # Must actually change: starting lane must differ from target
    if Ltarget is None or Ltarget == L0:
        return False

    # Optional minimum lateral displacement over the whole window
    min_lat_m = float(cfg.get("change_lane_min_lat_disp_m", 0.0) or 0.0)
    if min_lat_m > 0.0:
        t_ser = getattr(feats, "t", {}).get(ego)
        if t_ser is None or t1 >= len(t_ser):
            return False
        tw = np.asarray(t_ser[t0:t1+1], dtype=float)
        pw = pres[t0:t1+1]
        m = pw & np.isfinite(tw)
        if not np.any(m):
            return False
        if float(np.nanmax(tw[m]) - np.nanmin(tw[m])) < min_lat_m:
            return False

    # Require existence of a suffix [j..t1] with:
    #   - presence True at all frames
    #   - le[k] finite & non-zero
    #   - lane == Ltarget for all k in [j..t1]
    #   - (and since Ltarget != L0, we guarantee a change occurred in the window)
    for j in range(t0, t1 + 1):
        if not pres[j]:
            continue
        # all frames j..t1 must be valid/present
        pw = pres[j:t1+1]
        if not np.all(pw):
            continue
        seg = np.asarray(le[j:t1+1], dtype=float)
        if not np.all(np.isfinite(seg)):
            continue
        Lj = np.rint(seg).astype(int)
        if np.any(Lj == 0):
            continue
        if np.all(Lj == Ltarget):
            # Also ensure we weren’t already at Ltarget at t0 (we checked above),
            # so a true change has happened somewhere in [t0..j].
            return True

    return False

def _check_cross_lane(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    side: Optional[str],
    speed_arg: Optional[Dict[str, Any]],
    lane: Optional[int],
) -> bool:
    fps = float(cfg.get("fps", 10.0))
    td_all = _series_tdot_or_from_t(feats, ego, fps)  # your existing helper
    if td_all.size == 0 or t0 >= td_all.size:
        return False

    le = _get(feats.lane_idx, ego)
    yd = _get(feats.yaw_delta, ego)
    if any(arr is None or np.asarray(arr).size == 0 for arr in (le, yd)):
        return False
    T = min(len(td_all), len(le), len(yd))
    t1 = min(t1, T - 1)

    sl = slice(t0, t1 + 1)
    td = np.asarray(td_all[sl], dtype=float)
    lew = np.asarray(le[sl], dtype=float)
    ydw = np.asarray(yd[sl], dtype=float)

    pres = feats.present.get(ego)
    if pres is None: return False
    pw = (np.asarray(pres, dtype=float) > 0.5)[sl]

    finite = np.isfinite(td) & np.isfinite(lew) & np.isfinite(ydw)
    eval_mask = pw & finite
    if not np.any(eval_mask): return False

    # Orientation by side (right → negative tdot, left → positive)
    sgn = _lat_sign_from_side(side)

    oriented = np.zeros_like(td, dtype=float)
    if sgn == 0:
        # keine Seite vorgegeben -> beide Richtungen ok
        oriented[eval_mask] = np.abs(td[eval_mask])
    else:
        # Seite vorgegeben -> Vorzeichen erzwingen
        oriented[eval_mask] = sgn * td[eval_mask]  # >= 0 heißt "zur gewünschten Seite"

    oriented = np.zeros_like(td, dtype=float)
    oriented[eval_mask] = sgn * td[eval_mask]   # >= 0 if moving to requested side

    # yaw: close to perpendicular to lane direction
    cross_tol = float(cfg.get("cross_yaw_tol_rad", np.deg2rad(25.0)))  # ±25°
    yaw_perp_dist = _yaw_dist_to_perp(ydw, lew)  # distance to π/2
    #print("ypd: ", yaw_perp_dist)
    yaw_ok = np.zeros_like(eval_mask, dtype=bool)
    yaw_ok[eval_mask] = (yaw_perp_dist[eval_mask] <= cross_tol)

    # Target lateral speed (non-negative after orientation)
    spec = _norm_physical(speed_arg or {}, "speed")
    tol  = float(cfg.get("speed_value_tol", _DEFAULT_CFG["speed_value_tol"]))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol) if spec is not None else (None, None)

    speed_ok = np.ones_like(eval_mask, dtype=bool)
    if spec is not None:
        lo = max(0.0, float(lo))
        hi = float(hi)
        speed_ok = np.zeros_like(eval_mask, dtype=bool)
        speed_ok[eval_mask] = (oriented[eval_mask] >= lo) & (oriented[eval_mask] <= hi)

    # Optional lane guard (at start)
    if lane is not None:
        if t0 >= len(le) or not np.isfinite(le[t0]): return False
        if int(round(float(le[t0]))) != int(lane):   return False

    ok_series = yaw_ok & speed_ok

    mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if mode == "coverage":
        den = int(np.sum(eval_mask))
        if den == 0: return False
        cov = float(np.sum(ok_series & eval_mask)) / float(den)
        need = float(cfg.get("speed_min_coverage", _DEFAULT_CFG["speed_min_coverage"]))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", _DEFAULT_CFG["during_max_false"]))
        violations = int(np.sum(eval_mask & ~ok_series))
        return violations <= max_false

def _check_walk(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    side: Optional[str],
    speed_arg: Optional[Dict[str, Any]],
    lane: Optional[int],
) -> bool:
    # series
    le = _get(feats.lane_idx, ego)
    yd = _get(feats.yaw_delta, ego)  # TagFeatures.yaw_delta
    sd = _get(feats.s_dot, ego)      # longitudinal speed in lane-frame

    if any(arr is None or np.asarray(arr).size == 0 for arr in (le, yd, sd)):
        return False

    T = min(len(le), len(yd), len(sd))
    if t0 >= T: return False
    t1 = min(t1, T - 1)

    pres = feats.present.get(ego)
    if pres is None: return False
    pres = np.asarray(pres, dtype=float) > 0.5

    # optional lane guard (at start)
    if lane is not None:
        if not (t0 < len(le) and np.isfinite(le[t0])): return False
        if int(round(float(le[t0]))) != int(lane):     return False

    sl = slice(t0, t1 + 1)
    lew = np.asarray(le[sl], dtype=float)
    ydw = np.asarray(yd[sl], dtype=float)
    sdw = np.asarray(sd[sl], dtype=float)
    pw  = pres[sl]

    finite = np.isfinite(lew) & np.isfinite(ydw) & np.isfinite(sdw)
    eval_mask = pw & finite
    if not np.any(eval_mask): return False

    # yaw constraint: close to parallel
    yaw_tol = float(cfg.get("walk_yaw_tol_rad", np.deg2rad(20.0)))
    yaw_par_diff = _yaw_diff_to_parallel(ydw, lew)
    yaw_ok = np.zeros_like(eval_mask, dtype=bool)
    yaw_ok[eval_mask] = (yaw_par_diff[eval_mask] <= yaw_tol)

    # speed along lane direction (non-negative if moving *with* the lane direction)
    sgn_lane = _lane_sign_from_id(lew)
    s_along = np.zeros_like(sdw, dtype=float)
    s_along[eval_mask] = sgn_lane[eval_mask] * sdw[eval_mask]

    # speed spec (optional)
    spec = _norm_physical(speed_arg or {}, "speed")
    tol  = float(cfg.get("speed_value_tol", _DEFAULT_CFG["speed_value_tol"]))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol) if spec is not None else (None, None)

    speed_ok = np.ones_like(eval_mask, dtype=bool)
    if spec is not None:
        lo = max(0.0, float(lo))
        hi = float(hi)
        speed_ok = np.zeros_like(eval_mask, dtype=bool)
        speed_ok[eval_mask] = (s_along[eval_mask] >= lo) & (s_along[eval_mask] <= hi)

    ok_series = yaw_ok & speed_ok

    mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if mode == "coverage":
        den = int(np.sum(eval_mask))
        if den == 0: return False
        cov = float(np.sum(ok_series & eval_mask)) / float(den)
        need = float(cfg.get("speed_min_coverage", _DEFAULT_CFG["speed_min_coverage"]))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", _DEFAULT_CFG["during_max_false"]))
        violations = int(np.sum(eval_mask & ~ok_series))
        return violations <= max_false
        
def _check_presence(feats, ego, npc, t0, t1, cfg) -> bool:
    win = slice(t0, t1 + 1)
    pres_e = _get(feats.present, ego)[win]
    cov_e = _presence_coverage(pres_e)
    need = float(cfg.get("presence_min_coverage", _DEFAULT_CFG["presence_min_coverage"]))
    if cov_e < need:
        allow = cfg.get("presence_allow_missing", None)
        if isinstance(allow, int):
            if len(pres_e) - int(np.sum(pres_e > 0.5)) > allow:
                return False
        else:
            return False
    if npc is not None:
        pres_n = _get(feats.present, npc)[win]
        cov_n = _presence_coverage(pres_n)
        if cov_n < need:
            allow = cfg.get("presence_allow_missing", None)
            if isinstance(allow, int):
                if len(pres_n) - int(np.sum(pres_n > 0.5)) > allow:
                    return False
            else:
                return False
    return True

def _check_speed(feats, ego, npc, t0, t1, cfg, arg_speed: Dict[str, Any], at: Optional[str]) -> bool:
    spec = _norm_physical(arg_speed, "speed")
    tol = float(cfg.get("speed_value_tol", _DEFAULT_CFG["speed_value_tol"]))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol)
    v = _get(feats.speed, ego)
    if at:
        ti = _anchor_index(t0, t1, at)
        if ti >= v.size or not np.isfinite(v[ti]): return False
        return (v[ti] >= lo) and (v[ti] <= hi)
    sl = slice(t0, t1 + 1)
    mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if mode == "coverage":
        cov = _coverage_ratio(v[sl], lo, hi)
        need = float(cfg.get("speed_min_coverage", _DEFAULT_CFG["speed_min_coverage"]))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", _DEFAULT_CFG["during_max_false"]))
        pres = _get(feats.present, ego)[sl]
        return _all_during_in_range(v[sl], lo, hi, pres=pres, max_false=max_false)

def _check_position(
    feats,
    ego: str,
    npc: str,
    t0: int,
    t1: int,
    cfg,
    where: str,                         # "ahead_of" or "behind"
    dist_arg,                           # optional {"value":..} or {"range":[lo,hi]}
    at: str,                            # "start" | "end" | "mid" etc., per your _anchor_index
) -> bool:
    """
    s-only semantics:
      - ahead_of  :=  s_ego - s_npc > s_tol
      - behind    :=  s_ego - s_npc < -s_tol
    Optionally check |s_ego - s_npc| against distance bounds if distance is specified.
    """
    if not npc:
        return False

    ti = _anchor_index(t0, t1, at)

    se = feats.s.get(ego)
    sn = feats.s.get(npc)
    if se is None or sn is None:
        return False
    if ti >= len(se) or ti >= len(sn):
        return False

    se_t = float(se[ti])
    sn_t = float(sn[ti])

    if not (np.isfinite(se_t) and np.isfinite(sn_t)):
        return False

    xodr_id = _get(feats.lane_idx, ego)
    xodr_id = int(xodr_id[ti])

    if xodr_id is None:
        return False

    s_tol = float(cfg.get("position_s_tol", 0.5))

    delta_raw = se_t - sn_t
    direction = -1.0 if xodr_id > 0 else 1.0
    direction = 1.0
    delta = direction * delta_raw  # >0 heißt jetzt immer: ego ahead (lane-korrigiert)

    if where == "ahead_of":
        ok = (delta > s_tol)
    elif where == "behind":
        ok = (delta < -s_tol)
    else:
        return False

    if not ok:
        return False

    if dist_arg:
        spec = _norm_physical(dist_arg, "distance")
        lo, hi = _val_or_range_to_bounds(
            spec,
            tol=float(cfg.get("distance_tol", _DEFAULT_CFG["distance_tol"]))
        )
        sdist = abs(delta_raw)
        return (sdist >= lo) and (sdist <= hi)

    return True

def _check_lateral(feats, ego, npc, t0, t1, cfg, side: str, dist_arg: Optional[Dict[str, Any]], at: str) -> bool:
    if npc is None: return False
    ti = _anchor_index(t0, t1, at)
    lat = feats.lat_rel.get((ego, npc))
    if lat is None or ti >= len(lat): return False
    if str(lat[ti]).lower() != str(side).lower():
        return False
    if dist_arg:
        t_e = _get(feats.t, ego)
        t_n = _get(feats.t, npc)
        if t_e.size == 0 or t_n.size == 0 or ti >= t_e.size or ti >= t_n.size:
            return bool(cfg.get("lateral_allow_missing", True))
        if not (np.isfinite(t_e[ti]) and np.isfinite(t_n[ti])):
            return bool(cfg.get("lateral_allow_missing", True))
        gap = abs(float(t_e[ti] - t_n[ti]))
        spec = _norm_physical(dist_arg, "distance")
        lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("distance_tol", _DEFAULT_CFG["distance_tol"])))
        return (gap >= lo) and (gap <= hi)
    return True

def _check_lane_number(feats, ego, npc, t0, t1, cfg, lane: int, at: str) -> bool:
    ti = _anchor_index(t0, t1, at)
    ls = _get(feats.lane_idx, ego)
    if ti >= ls.size or not np.isfinite(ls[ti]): return False
    return int(round(ls[ti])) == int(lane)

def _check_lane_same_as(feats, ego, npc, t0, t1, cfg, at: str) -> bool:
    if npc is None: return False
    ti = _anchor_index(t0, t1, at)
    le = _get(feats.lane_idx, ego)
    ln = _get(feats.lane_idx, npc)
    if ti >= le.size or ti >= ln.size: return False
    if not (np.isfinite(le[ti]) and np.isfinite(ln[ti])): return False
    return int(round(le[ti])) == int(round(ln[ti]))

def _check_lane_side_of(feats, ego, npc, t0, t1, cfg, lane: Optional[int], side: str, at: str) -> bool:
    if not _check_lateral(feats, ego, npc, t0, t1, cfg, side=side, dist_arg=None, at=at):
        return False
    if lane is None:
        return True
    return _check_lane_number(feats, ego, npc, t0, t1, cfg, lane, at=at)

def _check_change_lane(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    delta_lane: Dict[str, Any],
    side: Optional[str],
) -> bool:
    """
    Accept iff:
      • lane at t0 is finite and non-zero (OpenDRIVE indexing, 0 invalid)
      • there's a *suffix* [j..t1] with presence==True and lane==Lend (finite, non-zero, constant)
      • Lend != L0 (an actual change)
      • if delta_lane/side is specified, Lend matches the allowed target(s)
      • (optional) min lateral displacement passes
    """
    le = _get(feats.lane_idx, ego)
    if le.size == 0 or t0 >= le.size or t1 >= le.size:
        return False

    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres = (np.asarray(pres, dtype=float) > 0.5)

    # must be present at the window ends
    if not (pres[t0] and pres[t1]):
        return False

    def _norm(x) -> Optional[int]:
        # finite -> nearest int; return None for NaN/inf/None or lane 0
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(xf):
            return None
        xi = int(np.rint(xf))
        return xi if xi != 0 else None

    # start-lane must be valid
    L0 = _norm(le[t0])
    if L0 is None:
        return False

    # find the constant end-lane suffix [j..t1]
    Lend = _norm(le[t1])
    if Lend is None:
        return False

    j = t1
    while j > t0 and pres[j-1] and _norm(le[j-1]) == Lend:
        j -= 1

    # Validate suffix is fully present and constant (defensive check)
    if not np.all(pres[j:t1+1]):
        return False
    seg = np.asarray(le[j:t1+1], dtype=float)
    if not np.all(np.isfinite(seg)):
        return False
    Lj = np.rint(seg).astype(int)
    if np.any(Lj == 0) or not np.all(Lj == Lend):
        return False

    # must have actually changed lanes
    if Lend == L0:
        # print(f"  -> MISS: no suffix j..t1 that is all at a target lane; start={L0}  candidates=[{Lend}]")
        return False

    # Optional: minimum lateral displacement gate
    min_lat_m = float(cfg.get("change_lane_min_lat_disp_m", 0.0) or 0.0)
    if min_lat_m > 0.0:
        t_ser = getattr(feats, "t", {}).get(ego)
        if t_ser is None or t1 >= len(t_ser):
            return False
        tw = np.asarray(t_ser[t0:t1+1], dtype=float)
        m = pres[t0:t1+1] & np.isfinite(tw)
        if not np.any(m):
            return False
        if float(np.nanmax(tw[m]) - np.nanmin(tw[m])) < min_lat_m:
            return False

    # delta/side constraints → compute allowed targets from L0
    candidates: List[int] = []
    if isinstance(delta_lane, dict):
        if "value" in delta_lane:
            steps = int(round(float(delta_lane["value"])))
            if steps == 0:
                return False
            sgn = _lane_dir_sign_odr(side)
            if sgn == 0:
                candidates = [_step_skip_zero(L0, steps, +1), _step_skip_zero(L0, steps, -1)]
            else:
                candidates = [_step_skip_zero(L0, steps, sgn)]
        elif "range" in delta_lane:
            lo, hi = delta_lane["range"]
            lo = int(math.floor(float(lo)))
            hi = int(math.ceil(float(hi)))
            if lo == 0 and hi == 0:
                return False
            sgn = _lane_dir_sign_odr(side)
            for steps in range(max(1, abs(lo)), max(1, abs(hi)) + 1):
                if sgn == 0:
                    candidates.append(_step_skip_zero(L0, steps, +1))
                    candidates.append(_step_skip_zero(L0, steps, -1))
                else:
                    candidates.append(_step_skip_zero(L0, steps, sgn))
            candidates = sorted(set(candidates))

    # If delta not specified, accept any different end-lane; else require match.
    if candidates and Lend not in candidates:
        # print(f"  -> MISS: start={L0}  not in candidates={candidates}  end={Lend}")
        return False

    # Success — suffix is clean and matches constraints
    # print(f"  -> MATCH: start={L0}  target={Lend}  j={j}")
    return True

def _check_change_speed(feats, ego, npc, t0, t1, cfg, delta_speed: Dict[str, Any]) -> bool:
    v = _get(feats.speed, ego)
    if v.size == 0 or t0 >= v.size or t1 >= v.size: return False
    dv = float(v[t1] - v[t0])
    spec = _norm_physical(delta_speed, "speed")
    lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("change_speed_tol", _DEFAULT_CFG["change_speed_tol"])))
    return (dv >= lo) and (dv <= hi)

def _check_acceleration(feats, ego, npc, t0, t1, cfg, accel_arg: Dict[str, Any], at: Optional[str]) -> bool:
    a = _get(feats.s_ddot, ego)
    if a.size == 0:
        return False

    spec = _norm_physical(accel_arg or {}, "acceleration")
    lo, hi = _val_or_range_to_bounds(spec, tol=0.0)

    if at:
        ti = _anchor_index(t0, t1, at)  # in deiner spec.py: end -> t1, sonst t0
        if ti >= a.size or not np.isfinite(a[ti]):
            return False

        ok = (a[ti] >= lo) and (a[ti] <= hi)
        return ok

    # during (mean over window)
    aa = a[t0:t1 + 1]
    if not np.any(np.isfinite(aa)):
        return False
    mean_a = float(np.nanmean(aa))
    ok = (mean_a >= lo) and (mean_a <= hi)

    if cfg.get("debug_accel_end", False) or cfg.get("debug_checks", False):
        print(
            f"[ACCEL@MEAN] actor={ego!r} npc_arg={npc!r} "
            f"t0={t0} t1={t1} mean={mean_a} bounds=[{lo},{hi}] ok={ok}",
            flush=True,
        )

    return ok

def _check_yaw(feats, ego, npc, t0, t1, cfg, angle_arg: Dict[str, Any], at: str) -> bool:
    yaw = _get(feats.yaw, ego)
    spec = _norm_physical(angle_arg, "angle")
    ti = _anchor_index(t0, t1, at)
    if ti >= yaw.size or not np.isfinite(yaw[ti]): return False
    lo, hi = _val_or_range_to_bounds(spec, tol=0.0)
    return (yaw[ti] >= lo) and (yaw[ti] <= hi)

def _check_yaw_delta(feats, ego, npc, t0, t1, cfg, angle_arg: Dict[str, Any], at: str) -> bool:
    ydel = _get(feats.yaw_delta, ego)
    spec = _norm_physical(angle_arg, "angle")
    ti = _anchor_index(t0, t1, at)
    if ti >= ydel.size or not np.isfinite(ydel[ti]): return False
    lo, hi = _val_or_range_to_bounds(spec, tol=0.0)
    return (ydel[ti] >= lo) and (ydel[ti] <= hi)

def _check_distance_traveled(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    dist_arg: Dict[str, Any],
) -> bool:
    spec = _norm_physical(dist_arg or {}, "distance")
    lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("distance_tol", 2.0)))

    s = getattr(feats, "s", {}).get(ego)
    pres = _get(feats.present, ego)
    if s is not None and t0 < len(s):
        te = t1 + 1 if (t1 + 1) < len(s) else t1
        if te < 0:
            return False
        if pres.size and (t0 < pres.size and te < pres.size):
            if not (pres[t0] > 0.5 and pres[te] > 0.5):
                pass
            else:
                s0, se = float(s[t0]), float(s[te])
                if np.isfinite(s0) and np.isfinite(se):
                    d = abs(se - s0)
                    return (d >= lo) and (d <= hi)
        else:
            s0, se = float(s[t0]), float(s[te])
            if np.isfinite(s0) and np.isfinite(se):
                d = abs(se - s0)
                return (d >= lo) and (d <= hi)

    x = _get(feats.x, ego); y = _get(feats.y, ego)
    if x.size and y.size:
        stop = min(max(x.size, y.size), (t1 + 1) + 1)
        if stop <= t0 + 1:
            return False
        sl = slice(t0, stop)
        xi = np.asarray(x[sl], dtype=float) if x.size >= stop else np.array([], dtype=float)
        yi = np.asarray(y[sl], dtype=float) if y.size >= stop else np.array([], dtype=float)
        if xi.size == 0 or yi.size == 0:
            return False
        pres_win = _get(feats.present, ego)[sl] > 0.5
        finite = np.isfinite(xi) & np.isfinite(yi)
        idx = np.where(pres_win & finite)[0]
        if idx.size >= 2:
            dsum = 0.0
            for k in range(idx.size - 1):
                i, j = idx[k], idx[k + 1]
                dx = xi[j] - xi[i]
                dy = yi[j] - yi[i]
                dsum += float(np.hypot(dx, dy))
            return (dsum >= lo) and (dsum <= hi)

    return False

def _check_speed_same_as(feats, ego, npc, t0, t1, cfg, at: Optional[str]) -> bool:
    if npc is None: return False
    v_e = _get(feats.speed, ego)
    v_n = _get(feats.speed, npc)
    tol = float(cfg.get("speed_value_tol", _DEFAULT_CFG["speed_value_tol"]))
    if at:
        ti = _anchor_index(t0, t1, at)
        if ti >= v_e.size or ti >= v_n.size: return False
        if not (np.isfinite(v_e[ti]) and np.isfinite(v_n[ti])): return False
        return abs(float(v_e[ti] - v_n[ti])) <= tol
    ve = v_e[t0:t1+1]; vn = v_n[t0:t1+1]
    m = np.isfinite(ve) & np.isfinite(vn)
    if not np.any(m): return False
    diff = abs(float(np.nanmean(ve[m]) - np.nanmean(vn[m])))
    return diff <= tol

def _check_change_space_gap(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_arg: Dict[str, Any],
    direction: str,
) -> bool:
    if npc is None:
        return False

    dir_l = str(direction or "").lower()
    spec = _norm_physical(target_arg or {}, "distance")
    if "value" in spec:
        tgt = float(spec["value"])
        lo, hi = tgt, tgt
    elif "range" in spec:
        lo, hi = map(float, spec["range"])
    else:
        return False

    tol = float(cfg.get("space_gap_tol", cfg.get("distance_tol", 2.0)))
    ti = t1

    def _fin(v): return (v is not None) and np.isfinite(v)

    if dir_l in ("ahead", "behind"):
        pos = feats.rel_position.get((ego, npc))
        if pos is None or ti >= len(pos):
            return False
        need = "front" if dir_l == "ahead" else "back"
        if str(pos[ti]) != need:
            return False

        s_e = feats.s.get(ego); s_n = feats.s.get(npc)
        if s_e is None or s_n is None or ti >= len(s_e) or ti >= len(s_n):
            return False
        se = float(s_e[ti]); sn = float(s_n[ti])
        if not (_fin(se) and _fin(sn)):
            return False

        ds = se - sn
        val = ds if dir_l == "ahead" else -ds
        return (val >= (lo - tol)) and (val <= (hi + tol))

    if dir_l in ("left", "right"):
        lat = feats.lat_rel.get((ego, npc))
        if lat is None or ti >= len(lat):
            return False
        need = dir_l
        if str(lat[ti]).lower() != need:
            return False

        t_e = feats.t.get(ego); t_n = feats.t.get(npc)
        if t_e is None or t_n is None or ti >= len(t_e) or ti >= len(t_n):
            return bool(cfg.get("lateral_allow_missing", True))
        te = float(t_e[ti]); tn = float(t_n[ti])
        if not (_fin(te) and _fin(tn)):
            return bool(cfg.get("lateral_allow_missing", True))

        dt = te - tn
        val = abs(dt)
        return (val >= (lo - tol)) and (val <= (hi + tol))

    if bool(cfg.get("debug_match_block", False)):
        print(f"[change_space_gap] direction '{direction}' not supported (need inside/outside logic)")
    return False

def _check_keep_space_gap(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    direction: str,
) -> bool:
    """
    Keep space gap (longitudinal or lateral) relative to the gap at t0.
    Unknowns count as false (via _keep_eval); coverage mode uses full L as denominator.
    """
    if npc is None:
        return False

    dir_l = (direction or "").lower()
    sl = slice(t0, t1 + 1)
    tol = float(cfg.get("space_gap_tol", cfg.get("distance_tol", 2.0)))

    def _ok_sign(a: float, b: float, eps: float) -> bool:
        # sign-consistency tolerant to near-zero values
        if not (np.isfinite(a) and np.isfinite(b)):
            return False
        if abs(a) <= eps or abs(b) <= eps:
            return True
        return (a >= 0) == (b >= 0)

    L = int(t1 - t0 + 1)
    if L <= 0:
        return False

    if dir_l in ("longitudinal", "long", "s"):
        s_e = feats.s.get(ego)
        s_n = feats.s.get(npc)
        if s_e is None or s_n is None:
            return False
        if t0 >= len(s_e) or t0 >= len(s_n):
            return False
        if not (np.isfinite(s_e[t0]) and np.isfinite(s_n[t0])):
            return False

        gap0 = float(s_e[t0] - s_n[t0])
        se = np.asarray(s_e[sl], dtype=float)
        sn = np.asarray(s_n[sl], dtype=float)

        m = np.isfinite(se) & np.isfinite(sn)  # evaluable frames
        if not np.any(m):
            return False

        gaps = se - sn
        mag_ok = np.zeros(L, dtype=bool)
        sign_ok = np.zeros(L, dtype=bool)
        mag_ok[m] = np.abs(np.abs(gaps[m]) - abs(gap0)) <= tol
        sign_ok[m] = np.array([_ok_sign(g, gap0, tol) for g in gaps[m]])

        ok = mag_ok & sign_ok

        # Optional categorical orientation (front/back)
        pos = feats.rel_position.get((ego, npc))
        if pos is not None:
            pos_arr = np.asarray(pos[sl], dtype=object)
            need = "front" if gap0 >= 0 else "back"
            known = (pos_arr == "front") | (pos_arr == "back")
            cat_ok = (~known) | (pos_arr == need)
            ok = ok & cat_ok

        eval_mask = m
        return _keep_eval(
            ok, eval_mask, L, cfg,
            coverage_key="speed_min_coverage",
            default_need=_DEFAULT_CFG.get("speed_min_coverage", 0.9),
        )

    elif dir_l in ("lateral", "lat", "t"):
        t_e = feats.t.get(ego)
        t_n = feats.t.get(npc)
        if t_e is None or t_n is None:
            return False
        if t0 >= len(t_e) or t0 >= len(t_n):
            return False
        if not (np.isfinite(t_e[t0]) and np.isfinite(t_n[t0])):
            return False

        gap0 = float(t_e[t0] - t_n[t0])
        te = np.asarray(t_e[sl], dtype=float)
        tn = np.asarray(t_n[sl], dtype=float)

        m = np.isfinite(te) & np.isfinite(tn)  # evaluable frames
        if not np.any(m):
            return False

        gaps = te - tn
        mag_ok = np.zeros(L, dtype=bool)
        sign_ok = np.zeros(L, dtype=bool)
        mag_ok[m] = np.abs(np.abs(gaps[m]) - abs(gap0)) <= tol
        sign_ok[m] = np.array([_ok_sign(g, gap0, tol) for g in gaps[m]])

        ok = mag_ok & sign_ok

        # Optional categorical orientation (left/right)
        lat = feats.lat_rel.get((ego, npc))
        if lat is not None:
            lat_arr = np.asarray(lat[sl], dtype=object)
            need = "left" if gap0 >= 0 else "right"
            known = (lat_arr == "left") | (lat_arr == "right")
            cat_ok = (~known) | (lat_arr == need)
            ok = ok & cat_ok

        eval_mask = m
        return _keep_eval(
            ok, eval_mask, L, cfg,
            coverage_key="speed_min_coverage",
            default_need=_DEFAULT_CFG.get("speed_min_coverage", 0.9),
        )

    else:
        return False
    
def _parse_unit_scale(d: Dict[str, Any]) -> float:
    u = str(d.get("unit", "m")).lower()
    return _DISTANCE_UNITS.get(u, 1.0)

def _check_assign_position_xy(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    x_target: float,
    y_target: float,
) -> bool:
    x = _get(feats.x, ego)
    y = _get(feats.y, ego)
    if t1 >= x.size or t1 >= y.size:
        return False
    if not (np.isfinite(x[t1]) and np.isfinite(y[t1])):
        return False
    tol = float(cfg.get("position_reach_tol", 1.0))
    dx = float(x[t1] - x_target)
    dy = float(y[t1] - y_target)
    return (dx*dx + dy*dy) <= (tol * tol)

def _check_assign_position_st(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    s_target: float,
    t_target: float,
) -> bool:
    s = feats.s.get(ego)
    t = feats.t.get(ego)
    if s is None or t is None:
        return False
    if t1 >= len(s) or t1 >= len(t):
        return False
    if not (np.isfinite(s[t1]) and np.isfinite(t[t1])):
        return False
    tol_s = float(cfg.get("st_reach_tol_s", 1.0))
    tol_t = float(cfg.get("st_reach_tol_t", 0.5))
    return (abs(float(s[t1] - s_target)) <= tol_s) and (abs(float(t[t1] - t_target)) <= tol_t)

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _angdiff(a: float, b: float) -> float:
    return _wrap_pi(a - b)

def _check_assign_orientation_yaw(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    yaw_arg: Dict[str, Any],
) -> bool:
    if isinstance(yaw_arg, (int, float)):
        spec = {"value": float(yaw_arg), "unit": "rad"}
    elif isinstance(yaw_arg, dict):
        spec = _norm_physical(yaw_arg, "angle")
    else:
        return False
    yaw_target = float(spec["value"])
    tol = float(cfg.get("yaw_reach_tol", 0.05))
    yaw = _get(feats.yaw, ego)
    if t1 >= yaw.size or not np.isfinite(yaw[t1]):
        return False
    return abs(_angdiff(float(yaw[t1]), yaw_target)) <= tol
    
def _check_assign_speed(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    speed_arg: Dict[str, Any],
) -> bool:
    if not isinstance(speed_arg, dict):
        return False
    spec = _norm_physical(speed_arg, "speed")
    tol = float(cfg.get("speed_value_tol", 0.10))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol)

    v = _get(feats.speed, ego)
    if t1 >= v.size or not np.isfinite(v[t1]):
        return False
    val = float(v[t1])
    return (val >= lo) and (val <= hi)

def _check_assign_acceleration(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    accel_arg: Dict[str, Any],
) -> bool:
    if not isinstance(accel_arg, dict):
        return False
    spec = _norm_physical(accel_arg, "acceleration")
    tol = float(cfg.get("accel_value_tol", 0.2))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol)

    a = _get(feats.accel, ego)
    if t1 >= a.size or not np.isfinite(a[t1]):
        return False
    val = float(a[t1])
    return (val >= lo) and (val <= hi)

def _check_remain_stationary(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    sl = slice(t0, t1 + 1)
    tol_v  = float(cfg.get("stationary_speed_tol", 0.15))
    tol_ax = float(cfg.get("stationary_axis_tol", 0.15))
    max_false = int(cfg.get("stationary_max_false", 0))

    pres = feats.present.get(ego)
    if pres is None:
        return False
    m_pres = np.asarray(pres[sl], dtype=float) > 0.5
    if not np.any(m_pres):
        return False

    v = _get(feats.speed, ego)
    v_win = np.asarray(v[sl], dtype=float)
    v_ok = (~np.isfinite(v_win)) | (np.abs(v_win) <= tol_v)

    sdot_ok = None
    tdot_ok = None

    sdot = getattr(feats, "s_dot", {}).get(ego) if hasattr(feats, "s_dot") else None
    if sdot is not None:
        s_win = np.asarray(sdot[sl], dtype=float)
        sdot_ok = (~np.isfinite(s_win)) | (np.abs(s_win) <= tol_ax)

    tdot = getattr(feats, "t_dot", {}).get(ego) if hasattr(feats, "t_dot") else None
    if tdot is not None:
        t_win = np.asarray(tdot[sl], dtype=float)
        tdot_ok = (~np.isfinite(t_win)) | (np.abs(t_win) <= tol_ax)

    ok = v_ok
    if sdot_ok is not None:
        ok = ok & sdot_ok
    if tdot_ok is not None:
        ok = ok & tdot_ok

    eval_mask = m_pres
    violations = int(np.sum(eval_mask & ~ok))
    return violations <= max_false

def _check_keep_speed(feats, ego, npc, t0, t1, cfg) -> bool:
    v = _get(feats.speed, ego)
    if t0 >= v.size or not np.isfinite(v[t0]):
        return False
    v0 = float(v[t0])

    sl = slice(t0, t1 + 1)
    v_win = np.asarray(v[sl], dtype=float)

    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5

    finite = np.isfinite(v_win)
    L = int(t1 - t0 + 1)
    if L <= 0:
        return False

    eval_mask = pres_win & finite

    tol = float(cfg.get("keep_speed_tol", cfg.get("speed_value_tol", 0.1)))
    ok_series = np.zeros(L, dtype=bool)
    ok_series[eval_mask] = (np.abs(v_win[eval_mask] - v0) <= tol)

    return _keep_eval(
        ok_series, eval_mask, L, cfg,
        coverage_key="speed_min_coverage", default_need=_DEFAULT_CFG.get("speed_min_coverage", 0.9)
    )

def _check_change_acceleration(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_accel_arg: Dict[str, Any],
    rate_profile: Optional[str] = None,
    rate_peak_arg: Optional[Dict[str, Any]] = None,
) -> bool:
    spec = _norm_physical(target_accel_arg, "acceleration")
    a_tol = float(cfg.get("accel_value_tol", 0.2))
    lo, hi = _val_or_range_to_bounds(spec, tol=a_tol)

    a = _get(feats.accel, ego)
    if t1 >= a.size or not np.isfinite(a[t1]):
        return False
    a_end = float(a[t1])
    if not (lo <= a_end <= hi):
        return False

    if rate_peak_arg is not None:
        j_tol = float(cfg.get("jerk_value_tol", 0.2))
        j_spec = _norm_physical(rate_peak_arg, "jerk")
        if "value" in j_spec:
            j_req_lo = float(j_spec["value"]) - j_tol
        elif "range" in j_spec:
            j_req_lo = float(min(j_spec["range"])) - j_tol
        else:
            return False

        fps = float(cfg.get("fps", 10.0))
        sl = slice(max(0, t0), t1 + 1)

        a_win = np.asarray(a[sl], dtype=float)
        pres = feats.present.get(ego)
        if pres is None:
            return False
        pres_win = np.asarray(pres[sl], dtype=float) > 0.5

        if a_win.size < 2:
            return False
        a_prev = a_win[:-1]
        a_curr = a_win[1:]
        pres_prev = pres_win[:-1]
        pres_curr = pres_win[1:]
        valid = np.isfinite(a_prev) & np.isfinite(a_curr) & pres_prev & pres_curr
        if not np.any(valid):
            return False

        jerk = (a_curr - a_prev) * fps
        peak_abs = float(np.nanmax(np.abs(jerk[valid])))
        if not (peak_abs >= j_req_lo):
            return False

    return True

def _check_keep_acceleration(feats, ego, npc, t0, t1, cfg) -> bool:
    a = _get(feats.accel, ego)
    if t0 >= a.size or not np.isfinite(a[t0]):
        return False
    a0 = float(a[t0])

    sl = slice(t0, t1 + 1)
    a_win = np.asarray(a[sl], dtype=float)

    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5

    finite = np.isfinite(a_win)
    L = int(t1 - t0 + 1)
    if L <= 0:
        return False

    eval_mask = pres_win & finite

    tol = float(cfg.get("keep_accel_tol", cfg.get("accel_value_tol", 0.2)))
    ok_series = np.zeros(L, dtype=bool)
    ok_series[eval_mask] = (np.abs(a_win[eval_mask] - a0) <= tol)

    need_key = "accel_min_coverage" if "accel_min_coverage" in _DEFAULT_CFG else "speed_min_coverage"
    return _keep_eval(
        ok_series, eval_mask, L, cfg,
        coverage_key=need_key, default_need=_DEFAULT_CFG.get(need_key, 0.9)
    )

def _check_follow_lane(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_lane: Optional[int],
) -> bool:
    """
    Keep-/Follow-lane where *all frames in [t0..t1]* are evaluated.
    Unknown lane or not-present counts as a violation (False).
    """
    L = max(0, int(t1 - t0 + 1))
    if L <= 0:
        return False

    # Safe window extractors (pad with NaN / 0 when missing)
    def _win_lane(a):
        if a is None or t0 >= len(a):
            return np.full((L,), np.nan, dtype=float)
        end = min(t1 + 1, len(a))
        w = np.asarray(a[t0:end], dtype=float)
        if w.size < L:
            w = np.concatenate([w, np.full((L - w.size,), np.nan, dtype=float)])
        return w

    def _win_pres(a):
        if a is None or t0 >= len(a):
            return np.zeros((L,), dtype=float)
        end = min(t1 + 1, len(a))
        w = np.asarray(a[t0:end], dtype=float)
        if w.size < L:
            w = np.concatenate([w, np.zeros((L - w.size,), dtype=float)])
        return w

    lane_series = feats.lane_idx.get(ego)
    pres_series = feats.present.get(ego)

    lane_win  = _win_lane(lane_series)
    pres_win  = _win_pres(pres_series) > 0.5
    lane_known = np.isfinite(lane_win)

    # Target lane:
    if target_lane is not None:
        lane_req = int(target_lane)
    else:
        # keep-lane → stick to rounded lane at t0; if unknown at t0 → fail
        if lane_series is None or t0 >= len(lane_series) or not np.isfinite(lane_series[t0]):
            return False
        lane_req = int(np.rint(lane_series[t0]))

    # Per-frame success: must be present, lane known, and equal to lane_req
    lane_eq = np.zeros((L,), dtype=bool)
    lane_eq[pres_win & lane_known] = (np.rint(lane_win[pres_win & lane_known]).astype(int) == lane_req)

    # Frames with !present or !known are now implicitly False (violations)
    mode = str(cfg.get("lane_follow_mode", "all")).lower()
    if mode == "coverage":
        need = float(cfg.get("lane_min_coverage", 0.95))
        cov = float(np.sum(lane_eq)) / float(L)   # denominator is the *whole* window
        return cov >= need
    else:
        allow = int(cfg.get("lane_follow_allow_false", 0))
        violations = int(L - np.sum(lane_eq))     # missing/unknown/present==0 all count as false
        return violations <= allow

def _check_change_time_headway(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_time_arg: Dict[str, Any],
    direction: str,
) -> bool:
    if npc is None:
        return False
    dir_l = (direction or "").lower()
    if dir_l not in ("ahead", "behind"):
        return False

    pos = feats.rel_position.get((ego, npc))
    if pos is None or t1 >= len(pos):
        return False
    need = "front" if dir_l == "ahead" else "back"
    if str(pos[t1]) != need:
        return False

    spec = _norm_physical(target_time_arg, "time") if isinstance(target_time_arg, dict) else {"value": target_time_arg}
    tol = float(cfg.get("time_headway_tol", 0.30))
    lo, hi = _time_val_or_range_to_bounds(spec, tol=tol)

    min_speed = float(cfg.get("min_speed_for_headway", 0.30))
    h_mag = _headway_time_mag(feats, ego, npc, t1, min_speed=min_speed)
    if h_mag is None:
        return False
    return (h_mag >= lo) and (h_mag <= hi)

def _check_keep_time_headway(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    """
    Keep time headway: compare the headway magnitude to its value at t0 and
    (optionally) keep the same front/back orientation. Unknowns count as false
    (via _keep_eval); coverage mode uses full L as denominator.
    """
    if npc is None:
        return False

    # Orientation at t0 (front/back)
    sign0 = None
    pos0 = feats.rel_position.get((ego, npc))
    if pos0 is not None and t0 < len(pos0):
        if str(pos0[t0]) == "front":
            sign0 = +1.0
        elif str(pos0[t0]) == "back":
            sign0 = -1.0

    ds0 = _signed_longitudinal_gap(feats, ego, npc, t0)
    if sign0 is None and ds0 is not None:
        sign0 = 1.0 if ds0 >= 0.0 else -1.0

    min_speed = float(cfg.get("min_speed_for_headway", 0.30))
    h0_mag = _headway_time_mag(feats, ego, npc, t0, min_speed=min_speed)
    if h0_mag is None:
        return False

    sl = slice(t0, t1 + 1)
    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5

    # Optional orientation consistency (front/back) over the window
    pos_series = feats.rel_position.get((ego, npc))
    orient_ok = None
    if pos_series is not None and sign0 is not None:
        pos_win = np.asarray(pos_series[sl], dtype=object)
        need_str = "front" if sign0 >= 0 else "back"
        known = (pos_win == "front") | (pos_win == "back")
        orient_ok = (~known) | (pos_win == need_str)

    # Build eval mask and OK series
    L = int(t1 - t0 + 1)
    if L <= 0:
        return False

    eval_mask = np.zeros(L, dtype=bool)
    h_arr = np.full(L, np.nan, dtype=float)
    for k in range(L):
        if not pres_win[k]:
            continue
        ti = t0 + k
        h = _headway_time_mag(feats, ego, npc, ti, min_speed=min_speed)
        if h is None:
            continue
        h_arr[k] = h
        eval_mask[k] = True

    if not np.any(eval_mask):
        return False

    tol = float(cfg.get("time_headway_tol", 0.30))
    ok_series = np.zeros(L, dtype=bool)
    ok_series[eval_mask] = np.abs(h_arr[eval_mask] - h0_mag) <= tol
    if orient_ok is not None:
        ok_series = ok_series & orient_ok

    return _keep_eval(
        ok_series, eval_mask, L, cfg,
        coverage_key="headway_min_coverage",
        default_need=_DEFAULT_CFG.get("headway_min_coverage", 0.85),
    )

def _check_keep_lane(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    return _check_follow_lane(feats, ego, npc, t0, t1, cfg, target_lane=None)
    
def _check_rise_ttc_less_than(
    feats,
    ego: str,
    npc: str,                  # the concrete actor id (or the bound role id)
    t0: int,
    t1: int,
    cfg: dict,
    threshold_time: dict,
) -> bool:
    dbg_checks = False
    if dbg_checks:
        print("inside _check_rise_ttc_less_than")
        print("[until.debug] feats type:", type(feats))
        print("[until.debug] has ttc:", hasattr(feats, "ttc"))
        if hasattr(feats, "__dict__"):
            print("[until.debug] attrs:", sorted(feats.__dict__.keys()))
        else:
            print("[until.debug] no __dict__ on feats (might be a namedtuple/dataclass with slots)")
    if npc is None:
        print(f"[until] SKIP npc=None E={ego} t=[{t0},{t1}] thr={threshold_time}")
        return False

    # parse threshold seconds
    spec = {"value": float(threshold_time.get("value", 0.0)),
            "unit": threshold_time.get("unit", "s")} if "value" in threshold_time else threshold_time
    _lo, thr = _time_val_or_range_to_bounds(spec, tol=0.0)

    # TTC series for the ordered pair (ego, npc)
    arr = feats.ttc.get((ego, npc))
    if arr is None:
        peers = [n for (e, n) in feats.ttc.keys() if e == ego]
        print(f"[until] MISSING TTC for {ego}→{npc}; ego has TTC peers={peers[:8]}")
        return False

    pe = feats.present.get(ego)
    pn = feats.present.get(npc)
    def present_ok(t):
        ok_e = True if pe is None or t >= len(pe) else (pe[t] > 0.5)
        ok_n = True if pn is None or t >= len(pn) else (pn[t] > 0.5)
        return ok_e and ok_n

    first_true = None
    best = float("inf")
    samples = []
    T = len(arr)

    for t in range(int(t0), int(t1) + 1):
        if t >= T or not present_ok(t):
            samples.append("∅"); continue
        v = float(arr[t])
        samples.append("∞" if not np.isfinite(v) else f"{v:.2f}")
        if np.isfinite(v):
            if v < best: best = v
            if first_true is None and v < thr:
                first_true = t

    any_eval = any(s != "∅" for s in samples)
    best_txt = "∞" if not np.isfinite(best) else f"{best:.2f}"
    if dbg_checks:
        print(f"[until] TTC {ego}→{npc} t=[{t0},{t1}] thr={thr:.2f}s "
            f"first_true={first_true} any_eval={any_eval} best={best_txt} "
            f"samples={samples[:24]}")
    return first_true is not None

# ======================================================================================
# Compiler: build_block_query(call, fps, cfg) → (BlockQuery, candidate_pairs_or_None)
# ======================================================================================
def build_block_query(call: Dict[str, Any], fps: int, cfg: Optional[Dict[str, Any]] = None):
    cfg_eff = _DEFAULT_CFG.copy()
    if cfg:
        cfg_eff.update({k: v for k, v in cfg.items() if v is not None})
    cfg_eff.setdefault("fps", float(fps))

    ego = call.get("actor")
    if not ego:
        raise ValueError("build_block_query: missing 'actor' in call")

    # --- duration → frames (legacy + block/action value or range) ---
    fps_f = float(cfg_eff.get("fps", float(fps)))

    # Legacy fast-path: ignore explicit durations and use default window,
    # allowing early end (previous behavior).
    if bool(cfg_eff.get("legacy_duration_windows", False)):
        duration_frames = max(1, int(round(float(cfg_eff.get("default_window_s", 5.0)) * fps_f)))
        cfg_eff["duration_scope"] = "block"
        cfg_eff["allow_shorter_end"] = True
        cfg_eff["duration_min_frames"] = 1
        cfg_eff["duration_max_frames"] = duration_frames
    else:
        # NEW: block-level duration (call["duration"]) and action-level duration
        # (call["action_args"]["duration"]) both support {"value":..} or {"range":[lo,hi]}
        dur_block  = (call.get("duration") or {})
        dur_action = (call.get("action_args") or {}).get("duration") or {}

        def _frames_value(d: Dict[str, Any]) -> Optional[int]:
            if not d or ("value" not in d):
                return None
            val  = float(d["value"])
            unit = str(d.get("unit", "second")).lower()
            if unit in ("frame", "frames"):
                return max(1, int(round(val)))
            k = _to_seconds_unit(unit)  # supports s, ms, min, h
            return max(1, int(round(val * k * fps_f)))

        def _frames_range(d: Dict[str, Any]) -> Optional[Tuple[int, int]]:
            if not d or ("range" not in d):
                return None
            lo, hi = d["range"]
            unit = str(d.get("unit", "second")).lower()
            if unit in ("frame", "frames"):
                lo_f = int(round(float(lo)))
                hi_f = int(round(float(hi)))
            else:
                k = _to_seconds_unit(unit)
                lo_f = int(round(float(lo) * k * fps_f))
                hi_f = int(round(float(hi) * k * fps_f))
            lo_f = max(1, lo_f)
            hi_f = max(lo_f, hi_f)
            return (lo_f, hi_f)

        # Parse block bounds (used to cap action if both are present)
        block_val = _frames_value(dur_block)
        block_rng = _frames_range(dur_block)
        block_lo = block_hi = None
        if block_val is not None:
            block_lo = block_hi = block_val
        elif block_rng is not None:
            block_lo, block_hi = block_rng

        # Parse action bounds
        action_val = _frames_value(dur_action)
        action_rng = _frames_range(dur_action)

        # Prefer ACTION if provided; cap by BLOCK upper bound when both exist.
        if action_val is not None or action_rng is not None:
            if block_hi is not None:
                if action_val is not None and action_val > block_hi:
                    action_val = block_hi
                if action_rng is not None:
                    lo, hi = action_rng
                    hi = min(hi, block_hi)
                    lo = min(lo, hi)  # keep non-empty after clamping
                    action_rng = (lo, hi)

            if action_val is not None:
                duration_frames = action_val
                cfg_eff["duration_scope"] = "action"
                cfg_eff["allow_shorter_end"] = False
                cfg_eff["duration_min_frames"] = action_val
                cfg_eff["duration_max_frames"] = action_val
            else:
                duration_frames = action_rng[1]
                cfg_eff["duration_scope"] = "action"
                cfg_eff["allow_shorter_end"] = True
                cfg_eff["duration_min_frames"] = action_rng[0]
                cfg_eff["duration_max_frames"] = action_rng[1]

        # Otherwise, use BLOCK if present (value or range) …
        elif block_val is not None or block_rng is not None:
            if block_val is not None:
                duration_frames = block_val
                cfg_eff["duration_scope"] = "block"
                cfg_eff["allow_shorter_end"] = False
                cfg_eff["duration_min_frames"] = block_val
                cfg_eff["duration_max_frames"] = block_val
            else:
                duration_frames = block_rng[1]
                cfg_eff["duration_scope"] = "block"
                cfg_eff["allow_shorter_end"] = True
                cfg_eff["duration_min_frames"] = block_rng[0]
                cfg_eff["duration_max_frames"] = block_rng[1]

        # … or fall back to the default window
        else:
            duration_frames = max(1, int(round(float(cfg_eff.get("default_window_s", 5.0)) * fps_f)))
            cfg_eff["duration_scope"] = "block"
            cfg_eff["allow_shorter_end"] = True
            cfg_eff["duration_min_frames"] = 1
            cfg_eff["duration_max_frames"] = duration_frames

    # ----------------------------------------------------------------------------------
    # Checks (legacy + S/E/D decomposition)
    # ----------------------------------------------------------------------------------
    checks: List[Callable[..., bool]] = []
    start_checks: List[Callable[..., bool]] = []
    end_checks: List[Callable[..., bool]] = []
    during_frame_checks: List[Callable[..., bool]] = []
    window_checks: List[Callable[..., bool]] = []

    referenced: List[str] = []

    def _ref(actor_name: Optional[str]):
        if actor_name and actor_name != ego and actor_name not in referenced:
            referenced.append(actor_name)

    def _at_kind(at: Optional[str]) -> Optional[str]:
        if not isinstance(at, str):
            return None
        a = at.strip().lower()
        if a in ("start", "begin"):
            return "start"
        if a in ("end", "finish"):
            return "end"
        # everything else treated as "during"
        return None

    def add_check(fn, *, at=None, window=False, label=None):
        # attach debug metadata
        try:
            fn._osc_label = label or "check"
            fn._osc_at = at
            fn._osc_window = bool(window)
        except Exception:
            pass

        if window:
            window_checks.append(fn)
        else:
            if at == "start":
                start_checks.append(fn)
            elif at == "end":
                end_checks.append(fn)
            else:
                checks.append(fn)
                during_frame_checks.append(fn)

    # presence gate (window-level)
    add_check(lambda F, E, N, t0, t1, C: _check_presence(F, E, N, t0, t1, C), window=True)

    action_name = str(call.get("action", "")).lower()
    aargs = (call.get("action_args") or {})

    if cfg_eff.get("debug_units"):
        print(f"[build_block_query] action={action_name} modifiers={[(m.get('name'), m.get('args')) for m in (call.get('modifiers') or [])]}")

    # ----------------------------------------------------------------------------------
    # Action checks (all window-level)
    # ----------------------------------------------------------------------------------
    if action_name == "change_lane":
        a_num  = aargs.get("num_of_lanes") or aargs.get("num_lanes") or aargs.get("count")
        a_side = aargs.get("side")
        a_ref  = aargs.get("reference") or ego

        target_lane = aargs.get("target")
        if isinstance(target_lane, dict) and "lane" in target_lane:
            target_lane = target_lane.get("lane")

        if a_ref and a_ref != ego:
            referenced.append(a_ref)

        add_check(
            (lambda target_lane=target_lane, a_num=a_num, a_side=a_side, a_ref=a_ref:
                (lambda F, E, N, t0, t1, C:
                    _check_change_lane_action(F, E, N, t0, t1, C, target_lane, a_num, a_side, a_ref)))(),
            window=True
        )

    elif action_name == "assign_position":
        pos_arg  = aargs.get("position")
        rp_arg   = aargs.get("route_point")
        odr_arg  = aargs.get("odr_point")

        provided = [x for x in (pos_arg, rp_arg, odr_arg) if x is not None]
        if len(provided) != 1:
            raise ValueError("assign_position requires exactly one of {position, route_point, odr_point}")

        target = provided[0]
        if not isinstance(target, dict):
            raise ValueError("assign_position target must be a dict containing either (x,y) or (s,t) or an ODR dict")

        if ("x" in target) and ("y" in target):
            scale = _parse_unit_scale(target)
            xt = float(target["x"]) * scale
            yt = float(target["y"]) * scale
            add_check(
                (lambda xt=xt, yt=yt:
                    (lambda F, E, N, t0, t1, C: _check_assign_position_xy(F, E, N, t0, t1, C, xt, yt)))(),
                window=True
            )

        elif ("s" in target) and ("t" in target):
            scale = _parse_unit_scale(target)
            st = float(target["s"]) * scale
            tt = float(target["t"]) * scale
            add_check(
                (lambda st=st, tt=tt:
                    (lambda F, E, N, t0, t1, C: _check_assign_position_st(F, E, N, t0, t1, C, st, tt)))(),
                window=True
            )

        elif {"road", "lane"} <= {k.lower() for k in target.keys()}:
            s_val = target.get("s"); t_val = target.get("t")
            if (s_val is not None) and (t_val is not None):
                st = float(s_val); tt = float(t_val)
                add_check(
                    (lambda st=st, tt=tt:
                        (lambda F, E, N, t0, t1, C: _check_assign_position_st(F, E, N, t0, t1, C, st, tt)))(),
                    window=True
                )
        else:
            raise ValueError("assign_position target must include (x,y) or (s,t), or be an ODR dict with road/lane/(s,t)")

    elif action_name == "change_space_gap":
        target = aargs.get("target")
        direction = aargs.get("direction")
        reference = aargs.get("reference")
        if not (target and direction and reference):
            raise ValueError("change_space_gap requires 'target', 'direction', and 'reference'")
        _ref(reference)
        add_check(
            (lambda target=target, direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_change_space_gap(F, E, reference, t0, t1, C, target, direction)))(),
            window=True
        )

    elif action_name == "keep_space_gap":
        direction = aargs.get("direction")
        reference = aargs.get("reference")
        if not reference:
            raise ValueError("keep_space_gap requires 'reference'")
        _ref(reference)
        add_check(
            (lambda direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_keep_space_gap(F, E, reference, t0, t1, C, direction)))(),
            window=True
        )

    elif action_name == "change_time_headway":
        target = aargs.get("target")
        direction = aargs.get("direction")
        reference = aargs.get("reference")
        if not (target and direction and reference):
            raise ValueError("change_time_headway requires 'target', 'direction', and 'reference'")
        _ref(reference)
        add_check(
            (lambda target=target, direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_change_time_headway(F, E, reference, t0, t1, C, target, direction)))(),
            window=True
        )

    elif action_name == "keep_time_headway":
        reference = aargs.get("reference")
        if not reference:
            raise ValueError("keep_time_headway requires 'reference'")
        _ref(reference)
        add_check(
            (lambda reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_keep_time_headway(F, E, reference, t0, t1, C)))(),
            window=True
        )

    elif action_name == "follow_lane":
        target_lane = aargs.get("target")
        if isinstance(target_lane, dict) and "lane" in target_lane:
            target_lane = target_lane.get("lane")
        if target_lane is not None:
            target_lane = int(target_lane)
        add_check(
            (lambda target_lane=target_lane:
                (lambda F, E, N, t0, t1, C:
                    _check_follow_lane(F, E, N, t0, t1, C, target_lane)))(),
            window=True
        )

    elif action_name == "remain_stationary":
        add_check((lambda: (lambda F, E, N, t0, t1, C: _check_remain_stationary(F, E, N, t0, t1, C)))(), window=True)

    elif action_name == "keep_speed":
        add_check((lambda: (lambda F, E, N, t0, t1, C: _check_keep_speed(F, E, N, t0, t1, C)))(), window=True)

    elif action_name == "change_acceleration":
        target = aargs.get("target") or aargs.get("acceleration")
        rp     = aargs.get("rate_profile")
        peak   = aargs.get("rate_peak")
        if target is None:
            raise ValueError("change_acceleration requires 'target' acceleration")
        add_check(
            (lambda target=target, rp=rp, peak=peak:
                (lambda F, E, N, t0, t1, C:
                    _check_change_acceleration(F, E, N, t0, t1, C, target, rp, peak)))(),
            window=True
        )

    elif action_name == "keep_acceleration":
        add_check((lambda: (lambda F, E, N, t0, t1, C: _check_keep_acceleration(F, E, N, t0, t1, C)))(), window=True)

    elif action_name == "assign_speed":
        speed_arg = aargs.get("target") or aargs.get("speed")
        if speed_arg is None:
            raise ValueError("assign_speed requires 'target' (or legacy 'speed')")
        add_check(
            (lambda speed_arg=speed_arg:
                (lambda F, E, N, t0, t1, C:
                    _check_assign_speed(F, E, N, t0, t1, C, speed_arg)))(),
            window=True
        )

    elif action_name == "assign_acceleration":
        accel_arg = aargs.get("target") or aargs.get("acceleration") or aargs.get("accel")
        if accel_arg is None:
            raise ValueError("assign_acceleration requires 'target' (or legacy 'acceleration')")
        add_check(
            (lambda accel_arg=accel_arg:
                (lambda F, E, N, t0, t1, C:
                    _check_assign_acceleration(F, E, N, t0, t1, C, accel_arg)))(),
            window=True
        )

    elif action_name == "assign_orientation":
        tgt = aargs.get("target") or aargs.get("orientation") or aargs.get("orientation_3d")
        yaw_arg = None
        if isinstance(tgt, dict):
            yaw_arg = tgt.get("yaw") or tgt.get("angle")
        else:
            yaw_arg = tgt
        if yaw_arg is None:
            raise ValueError("assign_orientation requires 'target' with a yaw angle")
        add_check(
            (lambda yaw_arg=yaw_arg:
                (lambda F, E, N, t0, t1, C:
                    _check_assign_orientation_yaw(F, E, N, t0, t1, C, yaw_arg)))(),
            window=True
        )

    elif action_name == "cross_lane":
        side  = aargs.get("side")
        speed = aargs.get("speed")
        lane  = aargs.get("lane")
        add_check(
            (lambda side=side, speed=speed, lane=lane:
                (lambda F, E, N, t0, t1, C:
                    _check_cross_lane(F, E, N, t0, t1, C, side, speed, lane)))(),
            window=True
        )
    elif action_name == "cross":
        side  = aargs.get("side")
        speed = aargs.get("speed")
        lane  = aargs.get("lane")
        add_check(
            (lambda side=side, speed=speed, lane=lane:
                (lambda F, E, N, t0, t1, C:
                    _check_cross_lane(F, E, N, t0, t1, C, side, speed, lane)))(),
            window=True
        )

    elif action_name == "walk":
        side  = aargs.get("side")
        speed = aargs.get("speed")
        lane  = aargs.get("lane")
        add_check(
            (lambda side=side, speed=speed, lane=lane:
                (lambda F, E, N, t0, t1, C:
                    _check_walk(F, E, N, t0, t1, C, side, speed, lane)))(),
            window=True
        )

    # ----------------------------------------------------------------------------------
    # translate modifiers
    # ----------------------------------------------------------------------------------
    for m in (call.get("modifiers") or []):
        name = str(m.get("name", "")).lower()
        args = m.get("args", {}) or {}
        if cfg_eff.get("debug_units"):
            print(f"[build_block_query] visiting modifier: {name}")

        if name == "speed":
            at_raw = args.get("at")
            if "same_as" in args:
                other = args.get("same_as"); _ref(other)
                add_check(
                    (lambda other=other, at=at_raw:
                        (lambda F, E, N, t0, t1, C: _check_speed_same_as(F, E, other, t0, t1, C, at)) )(),
                    at=at_raw
                )
            else:
                sp_raw = args.get("speed") or {}
                sp = _norm_physical(sp_raw, "speed")  # convert to m/s now
                lo, hi = _val_or_range_to_bounds(sp, tol=float(cfg_eff.get("speed_value_tol", 0.10)))
                if cfg_eff.get("debug_units"):
                    print(f"[build_block_query] speed modifier → [{lo:.2f}, {hi:.2f}] m/s (at={at_raw})")
                add_check(
                    _label(f"speed in [{lo:.2f},{hi:.2f}] m/s at={at_raw}",
                        (lambda sp=sp, at=at_raw:
                            (lambda F, E, N, t0, t1, C: _check_speed(F, E, N, t0, t1, C, sp, at)))()),
                    at=at_raw
                )

        elif name == "position":
            # position is always anchored (default start)
            at_eff = args.get("at", "start")
            if "ahead_of" in args:
                other = args.get("ahead_of"); _ref(other)
                dist  = args.get("distance")
                title = f"position ahead_of {other} at={at_eff}" + ("" if dist is None else f" & dist")
                add_check(
                    _label(title,
                        (lambda other=other, dist=dist, at=at_eff:
                            (lambda F, E, N, t0, t1, C:
                                _check_position(F, E, other, t0, t1, C, "ahead_of", dist, at)))()),
                    at=at_eff
                )
            elif "behind" in args:
                other = args.get("behind"); _ref(other)
                dist  = args.get("distance")
                title = f"position behind {other} at={at_eff}" + ("" if dist is None else f" & dist")
                add_check(
                    _label(title,
                        (lambda other=other, dist=dist, at=at_eff:
                            (lambda F, E, N, t0, t1, C:
                                _check_position(F, E, other, t0, t1, C, "behind", dist, at)))()),
                    at=at_eff
                )

        elif name == "lateral":
            other = args.get("side_of"); _ref(other)
            side = args.get("side")
            at_eff = args.get("at", "start")
            dist = args.get("distance")
            add_check(
                (lambda other=other, side=side, dist=dist, at=at_eff:
                    (lambda F, E, N, t0, t1, C: _check_lateral(F, E, other, t0, t1, C, side, dist, at)))(),
                at=at_eff
            )

        elif name == "lane":
            at_raw = args.get("at")  # None => during (per-frame)
            at_eff = at_raw or "start"
            at_label = at_raw if at_raw is not None else "all"

            if "same_as" in args:
                other = args.get("same_as"); _ref(other)
                add_check(
                    _label(f"lane same_as {other} at={at_label}",
                        (lambda other=other, at=at_eff:
                            (lambda F, E, N, t0, t1, C: _check_lane_same_as(F, E, other, t0, t1, C, at)))()),
                    at=at_raw
                )

            elif "side_of" in args and "side" in args:
                other = args.get("side_of"); _ref(other)
                side  = args.get("side")
                lane  = args.get("lane")  # optional filter
                title = f"lane side_of {other} side={side} at={at_label}"
                if lane is not None:
                    title += f" & lane=={lane}"
                add_check(
                    _label(title,
                        (lambda other=other, side=side, lane=lane, at=at_eff:
                            (lambda F, E, N, t0, t1, C: _check_lane_side_of(F, E, other, t0, t1, C, lane, side, at)))()),
                    at=at_raw
                )

            elif "lane" in args:
                lane = int(args.get("lane"))
                add_check(
                    _label(f"lane == {lane} at={at_label}",
                        (lambda lane=lane, at=at_eff:
                            (lambda F, E, N, t0, t1, C: _check_lane_number(F, E, N, t0, t1, C, lane, at)))()),
                    at=at_raw
                )

        elif name == "change_lane":
            # window-level
            dl = args.get("lane")
            if isinstance(dl, (int, float)):
                delta_lane = {"value": float(dl)}
            else:
                delta_lane = dl or None
            side = args.get("side") or args.get("from")
            if delta_lane is None and side is not None:
                delta_lane = {"value": 1}
            if delta_lane is not None:
                add_check(
                    (lambda d=delta_lane, s=side:
                        (lambda F, E, N, t0, t1, C: _check_change_lane(F, E, N, t0, t1, C, d, s)))(),
                    window=True
                )

        elif name == "change_speed":
            dv = args.get("speed") or {}
            add_check((lambda dv=dv:
                      (lambda F, E, N, t0, t1, C: _check_change_speed(F, E, N, t0, t1, C, dv)))(),
                      window=True)

        elif name == "acceleration":
            at_raw = args.get("at")  # None | "start" | "end"
            acc = args.get("acceleration") or {}
            is_window = (at_raw is None)

            chk = (lambda acc=acc, at=at_raw:
                (lambda F, E, N, t0, t1, C:
                    _check_acceleration(F, E, N, t0, t1, C, acc, at)))()

            add_check(
                chk,
                at=at_raw,
                window=is_window,
                label=f"acceleration@{at_raw or 'window'}",
            )

        elif name == "yaw":
            at_raw = args.get("at")
            at_eff = at_raw or "start"
            at_label = at_raw if at_raw is not None else "all"
            ang = args.get("angle") or {}
            add_check((lambda ang=ang, at=at_eff:
                      (lambda F, E, N, t0, t1, C: _check_yaw(F, E, N, t0, t1, C, ang, at)))(),
                      at=at_raw)

        elif name == "yaw_delta":
            at_raw = args.get("at")
            at_eff = at_raw or "start"
            at_label = at_raw if at_raw is not None else "all"
            ang = args.get("angle") or {}
            add_check((lambda ang=ang, at=at_eff:
                      (lambda F, E, N, t0, t1, C: _check_yaw_delta(F, E, N, t0, t1, C, ang, at)))(),
                      at=at_raw)

        elif name == "distance":
            dist = args.get("distance") or {}
            add_check(
                (lambda dist=dist:
                    (lambda F, E, N, t0, t1, C:
                        _check_distance_traveled(F, E, N, t0, t1, C, dist)))(),
                window=True
            )

        elif name == "keep_speed":
            add_check(
                (lambda: (lambda F, E, N, t0, t1, C: _check_keep_speed(F, E, N, t0, t1, C)))(),
                window=True
            )

        elif name == "keep_lane":
            add_check(
                (lambda: (lambda F, E, N, t0, t1, C: _check_keep_lane(F, E, N, t0, t1, C)))(),
                window=True
            )

        elif name == "until":
            # window-level
            ref = args.get("reference")
            time_arg = args.get("time")
            title = f"until rise(TTC<{time_arg.get('value')}s) ref={ref}"
            until_fn = (lambda ref=ref, time_arg=time_arg:
                            (lambda F, E, N, t0, t1, C:
                                _check_rise_ttc_less_than(F, E, ref, t0, t1, C, time_arg)
                            ))()
            add_check(_label(title, until_fn), window=True)


    def _names(L):
        out = []
        for f in L:
            out.append(getattr(f, "_osc_label", getattr(f, "__name__", type(f).__name__)))
        return out

    if cfg.get("debug_checks"):
        print("[Q-END-LABELS]", [getattr(f, "_osc_label", None) for f in end_checks], flush=True)
        print("[Q-WIN-LABELS]", [getattr(f, "_osc_label", None) for f in window_checks], flush=True)
        print(
            f"[Q-CHECKS] actor={ego!r} "
            f"n_start={len(start_checks)} n_end={len(end_checks)} "
            f"n_during={len(during_frame_checks)} n_window={len(window_checks)}",
            flush=True,
        )
    Q = BlockQuery(
        ego=str(ego),
        npc_candidates=referenced[:],
        duration_frames=int(duration_frames),
        checks=checks,
        cfg=cfg_eff,
        start_checks=start_checks,
        end_checks=end_checks,
        during_frame_checks=during_frame_checks,
        window_checks=window_checks,
    )

    pairs = [(ego, r) for r in referenced] if referenced else None

    if Q.cfg.get("debug_checks"):
        print(f"[build_block_query] ego={ego}  duration_frames={duration_frames}  fps={fps}")
        if referenced:
            print(f"[build_block_query] referenced NPCs: {referenced}")
        print(f"[build_block_query] compiled checks: {len(checks)}")
        print(f"[build_block_query] start_checks={len(start_checks)} end_checks={len(end_checks)} "
              f"during_frame_checks={len(during_frame_checks)} window_checks={len(window_checks)}")

    roles = _roles_used_by_call_local(call)
    Q.arity = 1 if len(roles) == 1 else 2
    Q.roles_used = roles
    return Q, pairs
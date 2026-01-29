# scenario_matching/analysis_stats/stats_extractors_change_lane.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import numpy as np


# -----------------------------------------------------------------------------
# Binning helpers
# -----------------------------------------------------------------------------
def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _q(x: Optional[float], step: float) -> Optional[float]:
    """Quantize to reduce cardinality in stats_shard.json."""
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if not math.isfinite(xf):
        return None
    if step <= 0:
        return xf
    return round(xf / step) * step


def _get_actor_id(roles: Dict[str, str], *keys: str) -> Optional[str]:
    for k in keys:
        v = roles.get(k)
        if v:
            return str(v)
    return None


def _present_mask(feats: Any, actor: str) -> np.ndarray:
    pres = getattr(feats, "present", None) or {}
    m = np.asarray(pres.get(actor, []), dtype=float)
    if m.size == 0:
        # fall back: treat "no present array" as all-present for the segment length if we can infer T
        T = int(getattr(feats, "T", 0) or 0)
        return np.ones((T,), dtype=bool) if T > 0 else np.zeros((0,), dtype=bool)
    return m > 0.5


def _lane_series(feats: Any, actor: str) -> np.ndarray:
    lane = getattr(feats, "lane_idx", None) or {}
    return np.asarray(lane.get(actor, []), dtype=float)


def _speed_series(feats: Any, actor: str) -> np.ndarray:
    spd = getattr(feats, "speed", None) or {}
    return np.asarray(spd.get(actor, []), dtype=float)


def _pair_series(d: Dict[Any, Any], a: str, b: str) -> Optional[np.ndarray]:
    """Try (a,b) and (b,a)."""
    if not isinstance(d, dict):
        return None
    v = d.get((a, b))
    if v is None:
        v = d.get((b, a))
    if v is None:
        return None
    arr = np.asarray(v, dtype=float)
    return arr if arr.size else None


def _slice01(t0: int, t1: int, T: int) -> Tuple[int, int]:
    t0 = max(0, int(t0))
    t1 = max(0, int(t1))
    if T > 0:
        t0 = min(t0, T - 1)
        t1 = min(t1, T - 1)
    if t1 < t0:
        t0, t1 = t1, t0
    return t0, t1


# -----------------------------------------------------------------------------
# Param specs (used by report scripts / for documentation; collector bins via _q)
# -----------------------------------------------------------------------------
PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    # Keep steps coarse to keep JSON small & histograms readable
    "ego_speed_mean_kph": {"unit": "kph", "step": 1.0},
    "npc_speed_mean_kph": {"unit": "kph", "step": 1.0},
    "dist_start_m": {"unit": "m", "step": 1.0},
    "dist_end_m": {"unit": "m", "step": 1.0},
    "window_dur_s": {"unit": "s", "step": 0.5},
    # NEW: min TTC after lane change completion (both in lane -1)
    "ttc_min_same_lane_s": {"unit": "s", "step": 0.5},

    # NEW: absolute longitudinal positions (meters) at window endpoints
    "ego_s_start_m": {"unit": "m", "step": 1.0},
    "ego_s_end_m": {"unit": "m", "step": 1.0},
    "npc_s_start_m": {"unit": "m", "step": 1.0},
    "npc_s_end_m": {"unit": "m", "step": 1.0},
    # signed delta s = ego_s - npc_s (meters)
    "delta_s_start_m": {"unit": "m", "step": 1.0},
    "delta_s_end_m": {"unit": "m", "step": 1.0},
}

def _at(arr: np.ndarray, ti: int) -> Optional[float]:
            if arr is None or arr.size == 0:
                return None
            if ti < 0 or ti >= arr.size:
                return None
            v = float(arr[ti])
            return v if np.isfinite(v) else None

def _s_series(feats: Any, actor_id: str) -> np.ndarray:
    """Longitudinal station (s) series for an actor (meters)."""
    s = getattr(feats, "s", {}) or {}
    arr = s.get(actor_id)
    if arr is None:
        return np.asarray([], dtype=float)
    return np.asarray(arr, dtype=float)

def _sddot_series(feats: Any, actor_id: str) -> np.ndarray:
    """Longitudinal station (s) series for an actor (meters)."""
    s = getattr(feats, "s_ddot", {}) or {}
    arr = s.get(actor_id)
    if arr is None:
        return np.asarray([], dtype=float)
    return np.asarray(arr, dtype=float)

def extract_params_for_window(
    feats: Any,
    roles: Dict[str, str],
    *,
    t0: int,
    t1: int,
    fps: int,
    left_is_decreasing: bool = True,  # kept for API compatibility; not needed here
) -> Dict[str, Optional[float]]:
    """
    Change-lane params extracted on a SINGLE representative window [t0..t1] (inclusive).

    Returned values are already quantized (see PARAM_SPECS steps) so StatsCollector can
    build compact histograms without exploding the number of unique keys.
    """
    
    out: Dict[str, Optional[float]] = {}
    #print(f"t0:  {t0} t1: {t1}")
    ego = _get_actor_id(roles, "ego_vehicle", "ego")
    npc = _get_actor_id(roles, "npc", "actor", "other", "target")  # be permissive
    T = int(getattr(feats, "T", 0) or 0)
    if fps and fps > 0:
        out["window_dur_s"] = _q((t1 - t0 + 1) / float(fps), PARAM_SPECS["window_dur_s"]["step"])

    # --- ego speed mean ---
    if ego:
        spd = _speed_series(feats, ego)
        spdt0 = _at(spd, int(t0))*3.6
        out["ego_speed_t0"] = spdt0
        spdt1 = _at(spd, int(t1))*3.6
        out["ego_speed_t1"] = spdt1
        pm = _present_mask(feats, ego)
        if spd.size and pm.size:
            sl = slice(t0, t1 + 1)
            m = pm[sl]
            v = spd[sl]
            if v.size and m.size and np.any(m):
                mean_mps = float(np.mean(v[m]))
                out["ego_speed_mean_kph"] = _q(mean_mps * 3.6, PARAM_SPECS["ego_speed_mean_kph"]["step"])
            else:
                out["ego_speed_mean_kph"] = None
    if npc:
        spd = _speed_series(feats, npc)
        s_ = getattr(feats, "s", None)
        spdt0 = _at(spd, int(t0)) *3.6
        out["npc_speed_t0"] = spdt0
        spdt1 = _at(spd, int(t1))*3.6
        out["npc_speed_t1"] = spdt1
        pm = _present_mask(feats, npc)
        if spd.size and pm.size:
            sl = slice(t0, t1 + 1)
            m = pm[sl]
            v = spd[sl]
            if v.size and m.size and np.any(m):
                mean_mps = float(np.mean(v[m]))
                out["npc_speed_mean_kph"] = _q(mean_mps * 3.6, PARAM_SPECS["ego_speed_mean_kph"]["step"])
            else:
                out["npc_speed_mean_kph"] = None

    # --- pair distance at t0 / t1 ---
    if ego and npc:
        rel_d = _pair_series(getattr(feats, "rel_distance", None) or {}, ego, npc)
        if rel_d is not None and rel_d.size:
            out["dist_start_m"] = _q(float(rel_d[t0]), PARAM_SPECS["dist_start_m"]["step"])
            out["dist_end_m"] = _q(float(rel_d[t1]), PARAM_SPECS["dist_end_m"]["step"])
        else:
            out["dist_start_m"] = None
            out["dist_end_m"] = None

        # --- NEW: absolute longitudinal positions at start/end ---
        se = _s_series(feats, ego)
        sn = _s_series(feats, npc)
        an = _sddot_series(feats, npc)        

        ego_s0 = _at(se, int(t0))
        ego_s1 = _at(se, int(t1))
        npc_s0 = _at(sn, int(t0))
        npc_s1 = _at(sn, int(t1))
        npc_sdd1 = _at(an, int(t1))

        out["ego_s_start_m"] = ego_s0
        out["ego_s_end_m"] = ego_s1
        out["npc_s_start_m"] = npc_s0
        out["npc_s_end_m"] = npc_s1

        out["npc_sddot_end_m"] = npc_sdd1

        if ego_s0 is not None and npc_s0 is not None:
            out["delta_s_start_m"] = _q(ego_s0 - npc_s0, PARAM_SPECS["delta_s_start_m"]["step"])
        else:
            out["delta_s_start_m"] = None

        if ego_s1 is not None and npc_s1 is not None:
            out["delta_s_end_m"] = _q(ego_s1 - npc_s1, PARAM_SPECS["delta_s_end_m"]["step"])
        else:
            out["delta_s_end_m"] = None

        ttc = _pair_series(getattr(feats, "ttc", None) or {}, ego, npc)
        la  = _lane_series(feats, ego)
        lb  = _lane_series(feats, npc)

        if ttc is not None and la.size and lb.size:
            pm_a = _present_mask(feats, ego)
            pm_b = _present_mask(feats, npc)
            sl = slice(t0, t1 + 1)

            # beide da
            pres_ok = pm_a[sl] & pm_b[sl]

            # beide lane-id gültig und gleich
            la_sl = la[sl]
            lb_sl = lb[sl]
            same_lane = np.isfinite(la_sl) & np.isfinite(lb_sl) & np.isclose(la_sl, lb_sl)

            mask = pres_ok & same_lane

            if np.any(mask):
                vals = np.asarray(ttc[sl], dtype=float)[mask]
                vals = vals[np.isfinite(vals) & (vals > 0)]
                out["ttc_min_same_lane_s"] = (
                    _q(float(np.min(vals)), PARAM_SPECS["ttc_min_same_lane_s"]["step"])
                    if vals.size else None
                )
            else:
                out["ttc_min_same_lane_s"] = None
        else:
            out["ttc_min_same_lane_s"] = None
    return out


TTC_PARAM_SPECS = PARAM_SPECS

def extract_ttc_features(feats, roles, t0, t1, *, fps=10, left_is_decreasing=True):
    return extract_params_for_window(
        feats=feats, 
        roles=roles, 
        t0=t0, 
        t1=t1,
        fps=fps,
        left_is_decreasing=left_is_decreasing,
    )
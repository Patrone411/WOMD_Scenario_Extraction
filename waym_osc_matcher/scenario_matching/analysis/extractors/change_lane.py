"""scenario_matching.analysis.extractors.change_lane

Extractors + parameter specs for the OSC block 'change_lane' (as posted in chat).

Lane-change direction:
- left_is_decreasing=True assumes lane_idx decreases when moving left.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "ego_speed_mean_kph": {"kind": "hist", "bins": np.linspace(0, 200, 41).tolist()},
    "ego_speed_start_kph": {"kind": "hist", "bins": np.linspace(0, 200, 41).tolist()},
    "ego_speed_end_kph":   {"kind": "hist", "bins": np.linspace(0, 200, 41).tolist()},
    "dist_start_m": {"kind": "hist", "bins": np.linspace(0, 300, 61).tolist()},
    "dist_end_m":   {"kind": "hist", "bins": np.linspace(0, 300, 61).tolist()},

    "ego_lane_start": {"kind": "cat"},
    "ego_lane_end":   {"kind": "cat"},
    "npc_lane_start": {"kind": "cat"},
    "npc_lane_end":   {"kind": "cat"},
    "same_lane_end":  {"kind": "cat"},
    "relpos_start":   {"kind": "cat"},
    "relpos_end":     {"kind": "cat"},

    "npc_lane_change_any": {"kind": "cat"},
    "npc_lane_change_left_any": {"kind": "cat"},
}


def ego_speed_mean_kph(feats, ego: str, t0: int, t1: int) -> Optional[float]:
    v = feats.speed.get(ego)
    if v is None:
        return None
    seg = v[t0:t1+1]
    if not np.isfinite(seg).any():
        return None
    return float(np.nanmean(seg) * 3.6)


def ego_speed_start_kph(feats, ego: str, t0: int) -> Optional[float]:
    v = feats.speed.get(ego)
    if v is None:
        return None
    x = v[t0]
    return float(x * 3.6) if np.isfinite(x) else None


def ego_speed_end_kph(feats, ego: str, t1: int) -> Optional[float]:
    v = feats.speed.get(ego)
    if v is None:
        return None
    x = v[t1]
    return float(x * 3.6) if np.isfinite(x) else None


def lane_at(feats, actor: str, t: int) -> Optional[int]:
    la = feats.lane_idx.get(actor)
    if la is None:
        return None
    x = la[t]
    return int(x) if np.isfinite(x) else None


def dist_at(feats, a: str, b: str, t: int) -> Optional[float]:
    d = feats.rel_distance.get((a, b))
    if d is None:
        return None
    x = d[t]
    return float(x) if np.isfinite(x) else None


def relpos_at(feats, a: str, b: str, t: int) -> Optional[str]:
    rp = feats.rel_position.get((a, b))
    if rp is None:
        return None
    return rp[t]


def same_lane(feats, a: str, b: str, t: int) -> Optional[int]:
    la = lane_at(feats, a, t)
    lb = lane_at(feats, b, t)
    if la is None or lb is None:
        return None
    return int(la == lb)


def npc_lane_change_any(feats, npc: str, t0: int, t1: int) -> Optional[bool]:
    la = feats.lane_idx.get(npc)
    pres = feats.present.get(npc)
    if la is None or pres is None:
        return None
    seg = la[t0:t1+1]
    valid = pres[t0:t1+1].astype(bool) & np.isfinite(seg)
    if int(valid.sum()) < 2:
        return None
    segv = seg[valid].astype(int)
    return bool(np.any(np.diff(segv) != 0))


def npc_lane_change_left_any(feats, npc: str, t0: int, t1: int, *, left_is_decreasing: bool) -> Optional[bool]:
    la = feats.lane_idx.get(npc)
    pres = feats.present.get(npc)
    if la is None or pres is None:
        return None
    seg = la[t0:t1+1]
    valid = pres[t0:t1+1].astype(bool) & np.isfinite(seg)
    if int(valid.sum()) < 2:
        return None
    segv = seg[valid].astype(int)
    d = np.diff(segv)
    return bool(np.any(d < 0)) if left_is_decreasing else bool(np.any(d > 0))


def extract_change_lane_features(
    feats,
    roles: Dict[str, str],
    t0: int,
    t1: int,
    *,
    left_is_decreasing: bool = True,
) -> Optional[Dict[str, Any]]:
    ego = roles.get("ego_vehicle")
    npc = roles.get("npc")
    if not ego or not npc:
        return None

    return {
        "ego_speed_mean_kph": ego_speed_mean_kph(feats, ego, t0, t1),
        "ego_speed_start_kph": ego_speed_start_kph(feats, ego, t0),
        "ego_speed_end_kph": ego_speed_end_kph(feats, ego, t1),

        "ego_lane_start": lane_at(feats, ego, t0),
        "ego_lane_end": lane_at(feats, ego, t1),
        "npc_lane_start": lane_at(feats, npc, t0),
        "npc_lane_end": lane_at(feats, npc, t1),

        "dist_start_m": dist_at(feats, npc, ego, t0),
        "dist_end_m": dist_at(feats, npc, ego, t1),
        "relpos_start": relpos_at(feats, npc, ego, t0),
        "relpos_end": relpos_at(feats, npc, ego, t1),
        "same_lane_end": same_lane(feats, npc, ego, t1),

        "npc_lane_change_any": npc_lane_change_any(feats, npc, t0, t1),
        "npc_lane_change_left_any": npc_lane_change_left_any(
            feats, npc, t0, t1, left_is_decreasing=left_is_decreasing
        ),
    }

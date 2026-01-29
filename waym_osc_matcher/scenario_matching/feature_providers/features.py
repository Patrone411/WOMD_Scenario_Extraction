# osc_parser/matching/features.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import json
import numpy as np

# Lateral labels
LATERAL_LEFT    = "left"
LATERAL_RIGHT   = "right"
LATERAL_SAME    = "same"
LATERAL_UNKNOWN = "unknown"

REL_POS_ALLOWED = {"front", "back", "unknown"}

# Finite-difference settings
DEFAULT_FPS = 10  # Hz
DEFAULT_DT  = 1.0 / DEFAULT_FPS

# --- NEW: robust defaults / helpers ------------------------------------------

DEFAULT_T = 91  # your dataset’s fixed frame count

class SkipSegment(Exception):
    """Raised to indicate a segment should be skipped (e.g., no actor data)."""

def _collect_actor_ids(seg_actor_block: Dict[str, Any]) -> List[str]:
    """
    Return actor IDs that have any per-segment payload.
    We consider keys present under segment_actor_data[seg_id].
    """
    return sorted(list(seg_actor_block.keys())) if isinstance(seg_actor_block, dict) else []

def _infer_T_from_seg_dict(seg_actor_block: Dict[str, Any], ga_block: Dict[str, Any]) -> int:
    """
    Try to infer T from per-actor per-segment arrays first (lane ids, 's'),
    then fall back to global long_v length for any actor.
    """
    max_len = 0
    if isinstance(seg_actor_block, dict):
        for _actor, payload in seg_actor_block.items():
            if not isinstance(payload, dict):
                continue
            for key in ("osc_lane_id", "s"):
                arr = payload.get(key)
                if isinstance(arr, list):
                    max_len = max(max_len, len(arr))
    if max_len > 0:
        return max_len

    if isinstance(ga_block, dict):
        for _actor, blk in ga_block.items():
            if isinstance(blk, dict) and isinstance(blk.get("long_v"), list):
                max_len = max(max_len, len(blk["long_v"]))
                if max_len > 0:
                    break
    return max_len  # may be 0 → caller can fall back to DEFAULT_T

# --- helpers -----------------------------------------------------------------

def _presence_from_s(s_list, T: int) -> Optional[np.ndarray]:
    """
    Presence = 1 when s[t] is not null (or not the literal "null"), else 0.
    Pads/crops to length T. Returns None if s_list is missing/empty.
    """
    if s_list is None:
        return None
    arr = np.asarray(s_list, dtype=object)
    if arr.size == 0:
        return None
    arr = _pad_or_crop(arr, T, fill=None)
    pres = np.ones((T,), dtype=float)
    # treat None or the string "null" as absent
    mask = (arr == None) | (arr == "null")  # noqa: E711
    pres[mask] = 0.0
    return pres

def _to_np(a, dtype=float):
    if a is None:
        return None
    try:
        arr = np.asarray(a, dtype=dtype)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None

def _pad_or_crop(arr: Optional[np.ndarray], T: int, fill=np.nan) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim != 1:
        arr = arr.ravel()
    n = arr.shape[0]
    if n == T:
        return arr
    if n > T:
        return arr[:T].copy()
    out = np.full((T,), fill, dtype=arr.dtype)
    out[:n] = arr
    return out

def _accel_from_speed(v: Optional[np.ndarray], dt: float) -> np.ndarray:
    """
    Compute acceleration from speed using finite differences.
    - Central difference for interior: a[t] = (v[t+1]-v[t-1])/(2*dt)
    - Forward/backward difference at edges
    - If neighbors are not finite, result is NaN at that index
    """
    if v is None:
        return np.array([], dtype=float)
    v = np.asarray(v, dtype=float)
    T = v.shape[0]
    a = np.full((T,), np.nan, dtype=float)
    if T == 0:
        return a
    # interior central diff
    if T >= 3:
        v_prev = np.roll(v, 1)
        v_next = np.roll(v, -1)
        ok_c = np.isfinite(v_prev) & np.isfinite(v_next)
        ok_c[0] = False
        ok_c[-1] = False
        a[ok_c] = (v_next[ok_c] - v_prev[ok_c]) / (2.0 * dt)
    # forward at 0
    if T >= 2 and np.isfinite(v[0]) and np.isfinite(v[1]):
        a[0] = (v[1] - v[0]) / dt
    # backward at T-1
    if T >= 2 and np.isfinite(v[-1]) and np.isfinite(v[-2]):
        a[-1] = (v[-1] - v[-2]) / dt
    return a

def segment_ids(stitched: Dict[str, Any]) -> List[str]:
    """
    Return all segment ids. Your data stores them under 'road_segments'.
    """
    return list((stitched.get("road_segments") or {}).keys())

def segment_num_lanes(stitched: Dict[str, Any], seg_id: str) -> Optional[int]:
    """
    num_lanes is stored per segment in 'road_segments'[seg_id]['num_lanes'].
    """
    seg = (stitched.get("road_segments") or {}).get(seg_id) or {}
    val = seg.get("num_lanes")
    try:
        return int(val) if val is not None else None
    except Exception:
        return None

def actor_ids_in_segment(stitched: Dict[str, Any], seg_id: str) -> List[str]:
    """
    Actor ids present in a given segment come from 'segment_actor_data'[seg_id].
    """
    return list((stitched.get("segment_actor_data") or {}).get(seg_id, {}).keys())

def segment_length(stitched: Dict[str, Any], seg_id: str) -> int:
    """
    Determine the number of frames for a segment.
    Prefer per-segment actor arrays (e.g., 's' or 'osc_lane_id').
    Fallback to global long_v series if needed.
    """
    seg_acts = (stitched.get("segment_actor_data") or {}).get(seg_id, {}) or {}
    max_len = 0
    for _actor, payload in seg_acts.items():
        if not isinstance(payload, dict):
            continue
        for key in ("s", "osc_lane_id"):
            arr = payload.get(key)
            if isinstance(arr, list):
                max_len = max(max_len, len(arr))
    if max_len > 0:
        return max_len

    # fallback: use any global long_v length
    ga = (stitched.get("general_actor_data") or {}).get("actor_activities") or {}
    for _actor, blk in ga.items():
        if isinstance(blk, dict) and isinstance(blk.get("long_v"), list):
            max_len = max(max_len, len(blk["long_v"]))
    return max_len  # 0 means nothing found

# --- schema-specific helpers -------------------------------------------------

def _ga_block(data: Dict[str, Any]) -> Dict[str, Any]:
    """general_actor_data.actor_activities dict (actor_id -> per-actor arrays)."""
    return (data.get("general_actor_data") or {}).get("actor_activities") or {}

def _per_seg_actor_block(data: Dict[str, Any], seg_id: str) -> Dict[str, Any]:
    """segment_actor_data[seg_id] dict (actor_id -> per-segment arrays)."""
    return (data.get("segment_actor_data") or {}).get(seg_id, {}) or {}

def _pairs_root(data: Dict[str, Any]) -> Dict[str, Any]:
    """inter_actor_activities dict at the ROOT (ego_id -> npc_id -> pair arrays)."""
    return data.get("inter_actor_activities") or {}

# -----------------------------------------------------------------------------


@dataclass
class TagFeatures:
    """Per-segment time series bundle used by the matcher."""
    segment_id: str
    num_lanes: int
    length_m: float
    actors: List[str]
    T: int

    # per-actor series (length T)
    speed: Dict[str, np.ndarray] = field(default_factory=dict)       # m/s
    yaw:   Dict[str, np.ndarray] = field(default_factory=dict)       # rad (if available; else NaN)
    x:     Dict[str, np.ndarray] = field(default_factory=dict)       # meters (map frame)
    y:     Dict[str, np.ndarray] = field(default_factory=dict)
    lane_idx: Dict[str, np.ndarray] = field(default_factory=dict)    # 1..N (0/NaN unknown)
    present:  Dict[str, np.ndarray] = field(default_factory=dict)    # 0/1 per frame
    accel:    Dict[str, np.ndarray] = field(default_factory=dict)    # m/s^2 (finite diff of speed)

    # lane-frame longitudinal/lateral kinematics
    s:        Dict[str, np.ndarray] = field(default_factory=dict)    # m
    t:        Dict[str, np.ndarray] = field(default_factory=dict)    # m
    s_dot:    Dict[str, np.ndarray] = field(default_factory=dict)    # m/s
    t_dot:    Dict[str, np.ndarray] = field(default_factory=dict)    # m/s
    s_ddot:   Dict[str, np.ndarray] = field(default_factory=dict)    # m/s^2
    t_ddot:   Dict[str, np.ndarray] = field(default_factory=dict)    # m/s^2
    yaw_delta:Dict[str, np.ndarray] = field(default_factory=dict)    # rad

    # per-pair series (length T)
    rel_position: Dict[Tuple[str,str], np.ndarray] = field(default_factory=dict)   # "front"/"back"/"unknown"
    lat_rel: Dict[Tuple[str,str], np.ndarray] = field(default_factory=dict)        # "left"/"right"/"same"/"unknown"
    rel_distance: Dict[Tuple[str,str], np.ndarray] = field(default_factory=dict)   # meters
    ttc: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    # --------- factory ---------
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def from_tag_json(cls, data: Dict[str, Any], seg_id: str) -> "TagFeatures":
        # lanes per segment
        seg_meta = (data.get("road_segments") or {}).get(seg_id, {}) or {}
        num_lanes = int(seg_meta.get("num_lanes", 0))
        num_segments = int(seg_meta.get("num_segments", 0)) if seg_meta.get("num_segments") is not None else 0
        length_m = float(num_segments) * 5.0

        # blocks from the current schema
        seg_acts = _per_seg_actor_block(data, seg_id)                  # segment_actor_data[seg_id]
        ga       = _ga_block(data)                                     # general_actor_data.actor_activities
        pairs    = _pairs_root(data)                                   # inter_actor_activities (root)

        # actors listed for the segment (keys in seg_acts)
        actors = _collect_actor_ids(seg_acts)

        # --- NEW: if no actor series at all → skip this segment cleanly -----
        if not actors:
            raise SkipSegment(f"no actor series present in {seg_id}")

        # choose T from the longest osc_lane_id/s present, else global long_v; else default 91
        T = _infer_T_from_seg_dict(seg_acts, ga)
        if T <= 0:
            T = int(DEFAULT_T)

        # --- per-actor series -------------------------------------------------
        speed: Dict[str, np.ndarray] = {}
        yaw:   Dict[str, np.ndarray] = {}
        x:     Dict[str, np.ndarray] = {}
        y:     Dict[str, np.ndarray] = {}
        lane_idx: Dict[str, np.ndarray] = {}
        present:  Dict[str, np.ndarray] = {}
        accel:    Dict[str, np.ndarray] = {}

        s:        Dict[str, np.ndarray] = {}
        t:        Dict[str, np.ndarray] = {}
        s_dot:    Dict[str, np.ndarray] = {}
        t_dot:    Dict[str, np.ndarray] = {}
        s_ddot:   Dict[str, np.ndarray] = {}
        t_ddot:   Dict[str, np.ndarray] = {}
        yaw_delta:Dict[str, np.ndarray] = {}

        for a in actors:
            a_seg = seg_acts.get(a, {}) or {}
            a_glb = ga.get(a, {}) or {}

            # lanes
            l = _pad_or_crop(_to_np(a_seg.get("osc_lane_id"), dtype=float), T, fill=np.nan)
            if l is None:
                l = np.full((T,), np.nan)
            lane_idx[a] = l

            # speed (long_v) – already m/s in current schema
            v = _pad_or_crop(_to_np(a_glb.get("long_v"), dtype=float), T)
            if v is None:
                v = np.full((T,), np.nan, dtype=float)
            speed[a] = v

            # yaw/x/y
            yy  = _pad_or_crop(_to_np(a_glb.get("yaw"), dtype=float), T)
            xx  = _pad_or_crop(_to_np(a_glb.get("x"),   dtype=float), T)
            yy2 = _pad_or_crop(_to_np(a_glb.get("y"),   dtype=float), T)
            yaw[a] = yy  if yy  is not None else np.full((T,), np.nan)
            x[a]   = xx  if xx  is not None else np.full((T,), np.nan)
            y[a]   = yy2 if yy2 is not None else np.full((T,), np.nan)

            # presence: prefer 's' from segment data; else based on finite x/y
            pres = _presence_from_s(a_seg.get("s"), T)
            if pres is None:
                pres = np.where(np.isfinite(x[a]) & np.isfinite(y[a]), 1.0, 0.0)
            present[a] = pres

            # acceleration from speed via finite differences
            accel[a] = _accel_from_speed(speed[a], DEFAULT_DT)

            # lane-frame signals (per-segment block)
            s_arr      = _pad_or_crop(_to_np(a_seg.get("s"),        dtype=float), T)
            t_arr      = _pad_or_crop(_to_np(a_seg.get("t"),        dtype=float), T)
            s_dot_arr  = _pad_or_crop(_to_np(a_seg.get("s_dot"),    dtype=float), T)
            t_dot_arr  = _pad_or_crop(_to_np(a_seg.get("t_dot"),    dtype=float), T)
            s_ddot_arr = _pad_or_crop(_to_np(a_seg.get("s_ddot"),   dtype=float), T)
            t_ddot_arr = _pad_or_crop(_to_np(a_seg.get("t_ddot"),   dtype=float), T)
            yd_arr     = _pad_or_crop(_to_np(a_seg.get("yaw_delta"),dtype=float), T)

            s[a]        = s_arr      if s_arr      is not None else np.full((T,), np.nan)
            t[a]        = t_arr      if t_arr      is not None else np.full((T,), np.nan)
            s_dot[a]    = s_dot_arr  if s_dot_arr  is not None else np.full((T,), np.nan)
            t_dot[a]    = t_dot_arr  if t_dot_arr  is not None else np.full((T,), np.nan)
            s_ddot[a]   = s_ddot_arr if s_ddot_arr is not None else np.full((T,), np.nan)
            t_ddot[a]   = t_ddot_arr if t_ddot_arr is not None else np.full((T,), np.nan)
            yaw_delta[a]= yd_arr     if yd_arr     is not None else np.full((T,), np.nan)

        # --- per-pair: lateral (from lanes) ----------------------------------
        lat_rel: Dict[Tuple[str,str], np.ndarray] = {}
        for i in range(len(actors)):
            for j in range(len(actors)):
                if i == j:
                    continue
                e = actors[i]; n = actors[j]
                l_e = np.asarray(lane_idx[e], dtype=float)
                l_n = np.asarray(lane_idx[n], dtype=float)
                lbl = np.full((T,), LATERAL_UNKNOWN, dtype=object)
                known = np.isfinite(l_e) & np.isfinite(l_n)
                gt = np.zeros_like(known, dtype=bool)
                lt = np.zeros_like(known, dtype=bool)
                np.greater(l_n, l_e, where=known, out=gt)
                np.less(l_n,  l_e, where=known, out=lt)
                eq = (l_n == l_e) & known
                lbl[gt] = LATERAL_RIGHT
                lbl[lt] = LATERAL_LEFT
                lbl[eq] = LATERAL_SAME
                lat_rel[(e, n)] = lbl

        # --- per-pair rel_position + rel_distance (ROOT: inter_actor_activities) ---
        inter_map = pairs  # ego -> npc -> pair_block
        rel_position: Dict[Tuple[str,str], np.ndarray] = {}
        rel_distance: Dict[Tuple[str,str], np.ndarray] = {}
        ttc_map: Dict[Tuple[str,str], np.ndarray] = {}


        for e in actors:
            e_block = inter_map.get(e) or {}
            for n in actors:
                if n == e:
                    continue
                pair_block = e_block.get(n)
                if not isinstance(pair_block, dict):
                    # Skip silently if that pair isn't provided in the file
                    continue

                # position
                pos_seq = pair_block.get("position")
                if isinstance(pos_seq, list):
                    pos_arr = _pad_or_crop(np.asarray(pos_seq, dtype=object), T, fill="unknown")
                    if pos_arr is None:
                        pos_arr = np.full((T,), "unknown", dtype=object)
                    mask_bad = ~np.isin(pos_arr, list(REL_POS_ALLOWED))
                    if np.any(mask_bad):
                        pos_arr = pos_arr.copy()
                        pos_arr[mask_bad] = "unknown"
                    rel_position[(e, n)] = pos_arr

                # euclidean distance
                dist_seq = pair_block.get("eucl_distance")
                if not isinstance(dist_seq, list):
                    # Allow missing distance too; skip if absent
                    continue
                dist_arr = _pad_or_crop(_to_np(dist_seq, dtype=float), T, fill=np.nan)
                if dist_arr is None:
                    continue
                dist_arr = dist_arr.copy()
                dist_arr[dist_arr < 0] = 0.0
                rel_distance[(e, n)] = dist_arr

                ttc_seq = (
                    pair_block.get("ttc")
                )
                if isinstance(ttc_seq, list):
                    arr = _pad_or_crop(_to_np(ttc_seq, dtype=float), T, fill=np.nan)
                    if arr is not None:
                        # sanitize: negative → 0, extremely large sentinel → +inf
                        arr = arr.copy()
                        arr[arr < 0] = 0.0
                        # treat 1e9-ish (or None) as "no collision"
                        huge = np.isfinite(arr) & (arr > 1e8)
                        arr[huge] = np.inf
                        ttc_map[(e, n)] = arr

        return cls(
            segment_id=seg_id,
            num_lanes=num_lanes,
            length_m=length_m,
            actors=actors,
            T=T,
            speed=speed,
            yaw=yaw,
            x=x,
            y=y,
            lane_idx=lane_idx,
            present=present,
            accel=accel,
            s=s, t=t, s_dot=s_dot, t_dot=t_dot, s_ddot=s_ddot, t_ddot=t_ddot, yaw_delta=yaw_delta,
            rel_position=rel_position,
            lat_rel=lat_rel,
            rel_distance=rel_distance,
            ttc= ttc_map
        )

    # convenience: build features for all segments (optionally filtered by min lanes)
    @classmethod
    def load_all_segments(cls, data: Dict[str, Any], min_lanes: Optional[int] = None) -> Dict[str, "TagFeatures"]:
        feats_by_seg: Dict[str, TagFeatures] = {}
        road = data.get("road_segments") or {}
        for seg_id, meta in road.items():
            nlanes = int(meta.get("num_lanes", 0))
            if min_lanes is not None and nlanes < int(min_lanes):
                continue
            try:
                feats_by_seg[seg_id] = cls.from_tag_json(data, seg_id)
            except SkipSegment:
                # silently ignore segments with zero actor data
                continue
            except ValueError as ex:
                # keep behavior quiet; log if you prefer:
                # print(f"[features] skip {seg_id}: {ex}")
                continue
        return feats_by_seg

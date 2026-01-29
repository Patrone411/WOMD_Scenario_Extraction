PATH = "./osc2_parser/data/4680e1fb10c57daa_tags.json"  # adjust if needed
# osc2_parser/sanity_check_tagfeatures.py
import json, math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from osc2_parser.matching.features import TagFeatures

# ----------------------------- config ---------------------------------------
DEFAULT_FPS = 10.0
DEFAULT_DT  = 1.0 / DEFAULT_FPS

# Tolerances for *exporter-stencil* equality
FD1_ABS_TOL   = 1e-6   # s_dot ~= fd1(s), t_dot ~= fd1(t)
FD2_ABS_TOL   = 1e-6   # s_ddot ~= fd2(s), t_ddot ~= fd2(t)

# New decomposition tolerances
VEL_PROJ_ABS_TOL      = 0.50   # |s_dot - v*cos(δ)|, |t_dot - v*sin(δ)|
SPEED_RECON_ABS_TOL   = 0.50   # |v - hypot(s_dot, t_dot)|
DERIV_PROJ_ABS_TOL    = 1.00   # |fd1(s_dot) - (a*cosδ − v*sinδ*δ_dot)| etc.

# General plausibility checks
MAX_SPEED_MS    = 100.0
MIN_SPEED_MS    = -0.5
MAX_ACCEL_MS2   = 20.0
MIN_ACCEL_MS2   = -20.0
MAX_YAW_ABS_RAD = math.tau + 0.2
MAX_YAWDELTA_ABS_RAD = math.pi + 0.2
MAX_LAT_ABS_M   = 20.0
MAX_LONG_RATE   = 100.0
MAX_LAT_RATE    = 30.0

# ----------------------------- helpers --------------------------------------
def _series_len(arr) -> int:
    if isinstance(arr, (list, tuple)): return len(arr)
    if isinstance(arr, np.ndarray): return int(arr.shape[0])
    return 0

def _ensure_1d_float(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.ravel()

def mae(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m): return float('nan')
    return float(np.mean(np.abs(a[m] - b[m])))

# --- exact clones of your exporter stencils ---
def fd1_like_exporter(arr: np.ndarray, dt: float) -> np.ndarray:
    """Your 1st derivative: forward/backward at edges; central inside; NaN if neighbors missing."""
    x = np.asarray(arr, dtype=float)
    T = x.shape[0]
    out = np.full(T, np.nan, dtype=float)
    if T < 2: return out
    if np.isfinite(x[0]) and np.isfinite(x[1]):
        out[0] = (x[1] - x[0]) / dt
    if np.isfinite(x[-1]) and np.isfinite(x[-2]):
        out[-1] = (x[-1] - x[-2]) / dt
    if T >= 3:
        a = x[:-2]; c = x[2:]
        ok = np.isfinite(a) & np.isfinite(c)
        out[1:-1][ok] = (c[ok] - a[ok]) / (2.0 * dt)
    return out

def fd2_like_exporter(arr: np.ndarray, dt: float) -> np.ndarray:
    """Your 2nd derivative: central; edges NaN; requires three finite neighbors."""
    x = np.asarray(arr, dtype=float)
    T = x.shape[0]
    out = np.full(T, np.nan, dtype=float)
    if T < 3: return out
    a = x[:-2]; b = x[1:-1]; c = x[2:]
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    out[1:-1][ok] = (c[ok] - 2.0*b[ok] + a[ok]) / (dt*dt)
    return out

# ----------------------------- main checks -----------------------------------
def print_segment_overview(feats_by_seg: Dict[str, TagFeatures], limit: int = 10):
    print(f"Loaded {len(feats_by_seg)} segments")
    for i, (sid, feats) in enumerate(feats_by_seg.items()):
        if i >= limit: break
        print(f"- seg={sid}  lanes={feats.num_lanes}  length_m={getattr(feats,'length_m','?')}  T={feats.T}  actors={list(feats.actors)}")

def validate_tagfeatures(
    feats_by_seg: Dict[str, TagFeatures],
    dt: float = DEFAULT_DT,
    min_lanes_expected: Optional[int] = None,
    distance_tol_m: float = 0.5
) -> List[Tuple[str, str]]:
    issues: List[Tuple[str, str]] = []

    for seg_id, feats in feats_by_seg.items():
        # meta
        if min_lanes_expected is not None and feats.num_lanes < min_lanes_expected:
            issues.append((seg_id, f"num_lanes={feats.num_lanes} but filtered with min_lanes={min_lanes_expected}"))

        T = feats.T
        actors = list(feats.actors)

        # per-actor fields
        required = ["speed","yaw","x","y","lane_idx","present","accel",
                    "s","t","s_dot","t_dot","s_ddot","t_ddot","yaw_delta"]
        for a in actors:
            for k in required:
                d = getattr(feats, k, {})
                if d.get(a) is None:
                    issues.append((seg_id, f"{k}[{a}] missing"))
                elif _series_len(d[a]) != T:
                    issues.append((seg_id, f"{k}[{a}] len={_series_len(d[a])} != T={T}"))

            # simple plausibility
            v = _ensure_1d_float(feats.speed.get(a, np.array([])))
            if v.size:
                vf = v[np.isfinite(v)]
                if vf.size:
                    if vf.min() < MIN_SPEED_MS: issues.append((seg_id, f"speed[{a}] min={vf.min():.2f} < {MIN_SPEED_MS}"))
                    if vf.max() > MAX_SPEED_MS: issues.append((seg_id, f"speed[{a}] max={vf.max():.2f} > {MAX_SPEED_MS}"))

            y = _ensure_1d_float(feats.yaw.get(a, np.array([])))
            if y.size:
                yf = y[np.isfinite(y)]
                if yf.size and np.max(np.abs(yf)) > MAX_YAW_ABS_RAD:
                    issues.append((seg_id, f"yaw[{a}] looks non-radian (max abs={np.max(np.abs(yf)):.2f})"))

            ydel = _ensure_1d_float(feats.yaw_delta.get(a, np.array([])))
            if ydel.size:
                ydf = ydel[np.isfinite(ydel)]
                if ydf.size and np.max(np.abs(ydf)) > MAX_YAWDELTA_ABS_RAD:
                    issues.append((seg_id, f"yaw_delta[{a}] out of typical bounds (max abs={np.max(np.abs(ydf)):.2f})"))

            lanes = _ensure_1d_float(feats.lane_idx.get(a, np.array([])))
            if lanes.size:
                fm = np.isfinite(lanes)
                nonint = np.where(fm & (np.floor(lanes)!=lanes))[0]
                if nonint.size:
                    issues.append((seg_id, f"lane_idx[{a}] non-integer at idx {nonint[:5].tolist()} ..."))

            # ---------------- exporter-stencil consistency ----------------
            s     = _ensure_1d_float(feats.s.get(a, np.array([])))
            t     = _ensure_1d_float(feats.t.get(a, np.array([])))
            sdot  = _ensure_1d_float(feats.s_dot.get(a, np.array([])))
            tdot  = _ensure_1d_float(feats.t_dot.get(a, np.array([])))
            sddot = _ensure_1d_float(feats.s_ddot.get(a, np.array([])))
            tddot = _ensure_1d_float(feats.t_ddot.get(a, np.array([])))
            pres  = feats.present.get(a, np.ones_like(s))

            # rebuild via same stencils
            sdot_ref  = fd1_like_exporter(s, dt)
            tdot_ref  = fd1_like_exporter(t, dt)
            sddot_ref = fd2_like_exporter(s, dt)
            tddot_ref = fd2_like_exporter(t, dt)

            m = mae(sdot, sdot_ref)
            if m == m and m > FD1_ABS_TOL:
                issues.append((seg_id, f"s_dot[{a}] != fd1(s): MAE={m:.6f} > {FD1_ABS_TOL}"))

            m = mae(tdot, tdot_ref)
            if m == m and m > FD1_ABS_TOL:
                issues.append((seg_id, f"t_dot[{a}] != fd1(t): MAE={m:.6f} > {FD1_ABS_TOL}"))

            m = mae(sddot, sddot_ref)
            if m == m and m > FD2_ABS_TOL:
                issues.append((seg_id, f"s_ddot[{a}] != fd2(s): MAE={m:.6f} > {FD2_ABS_TOL}"))

            m = mae(tddot, tddot_ref)
            if m == m and m > FD2_ABS_TOL:
                issues.append((seg_id, f"t_ddot[{a}] != fd2(t): MAE={m:.6f} > {FD2_ABS_TOL}"))

            # ---------------- velocity decomposition checks ----------------
            v     = _ensure_1d_float(feats.speed.get(a, np.array([])))
            delta = _ensure_1d_float(feats.yaw_delta.get(a, np.array([])))

            v_s = v * np.cos(delta)   # expected s_dot
            v_t = v * np.sin(delta)   # expected t_dot

            m = mae(sdot, v_s)
            if m == m and m > VEL_PROJ_ABS_TOL:
                issues.append((seg_id, f"s_dot[{a}] vs v*cos(delta): MAE={m:.3f} > {VEL_PROJ_ABS_TOL}"))

            m = mae(tdot, v_t)
            if m == m and m > VEL_PROJ_ABS_TOL:
                issues.append((seg_id, f"t_dot[{a}] vs v*sin(delta): MAE={m:.3f} > {VEL_PROJ_ABS_TOL}"))

            # speed reconstruction from (s_dot, t_dot)
            v_rec = np.hypot(sdot, tdot)
            m = mae(v, v_rec)
            if m == m and m > SPEED_RECON_ABS_TOL:
                issues.append((seg_id, f"speed[{a}] vs hypot(s_dot,t_dot): MAE={m:.3f} > {SPEED_RECON_ABS_TOL}"))

            # ---------------- acceleration decomposition checks ------------
            # From s_dot = v cos δ, t_dot = v sin δ:
            #   d(s_dot)/dt = a cos δ − v sin δ * δ_dot
            #   d(t_dot)/dt = a sin δ + v cos δ * δ_dot
            a_long   = _ensure_1d_float(feats.accel.get(a, np.array([])))  # from dv/dt
            delta_dt = fd1_like_exporter(delta, dt)
            dsdt_ref = fd1_like_exporter(sdot, dt)
            dtdt_ref = fd1_like_exporter(tdot, dt)

            dsdt_from_v = a_long * np.cos(delta) - v * np.sin(delta) * delta_dt
            dtdt_from_v = a_long * np.sin(delta) + v * np.cos(delta) * delta_dt

            m = mae(dsdt_ref, dsdt_from_v)
            if m == m and m > DERIV_PROJ_ABS_TOL:
                issues.append((seg_id, f"d(s_dot)/dt[{a}] from v,δ: MAE={m:.3f} > {DERIV_PROJ_ABS_TOL}"))

            m = mae(dtdt_ref, dtdt_from_v)
            if m == m and m > DERIV_PROJ_ABS_TOL:
                issues.append((seg_id, f"d(t_dot)/dt[{a}] from v,δ: MAE={m:.3f} > {DERIV_PROJ_ABS_TOL}"))

        # (per-pair quick validations can remain as you had them)
    return issues

def quick_peek(feats_by_seg: Dict[str, TagFeatures], seg_id: Optional[str] = None, n: int = 5):
    if seg_id is None:
        seg_id = next(iter(feats_by_seg))
    feats = feats_by_seg[seg_id]
    actors = list(feats.actors)
    print(f"SEG {seg_id}: lanes={feats.num_lanes}, length_m={getattr(feats,'length_m','?')}, T={feats.T}, actors={actors}")
    for a in actors:
        print(f"  {a}:")
        print(f"    s[0:{n}]      = {feats.s[a][:n].tolist()}")
        print(f"    s_dot[0:{n}]  = {feats.s_dot[a][:n].tolist()}")
        print(f"    s_ddot[0:{n}] = {feats.s_ddot[a][:n].tolist()}")
        print(f"    t[0:{n}]      = {feats.t[a][:n].tolist()}")
        print(f"    t_dot[0:{n}]  = {feats.t_dot[a][:n].tolist()}")
        print(f"    t_ddot[0:{n}] = {feats.t_ddot[a][:n].tolist()}")

# ----------------------------- run ------------------------------------------
if __name__ == "__main__":
    with open(PATH, "r") as f:
        data = json.load(f)

    feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=2)

    print_segment_overview(feats_by_seg, limit=10)
    issues = validate_tagfeatures(feats_by_seg, dt=DEFAULT_DT, min_lanes_expected=2)
    if issues:
        print("\n❗ Found potential issues:")
        for seg_id, msg in issues[:200]:
            print(f"  [{seg_id}] {msg}")
        if len(issues) > 200:
            print(f"  ... and {len(issues)-200} more")
    else:
        print("\n✅ TagFeatures fields look consistent with the exporter and Frenet projections.")

    quick_peek(feats_by_seg)

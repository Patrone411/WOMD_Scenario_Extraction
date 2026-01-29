import numpy as np
from shapely.geometry import Point, LineString

def _outside_extent(ls, P, s, tol=1e-6):
    # returns True if P is beyond the start or end (not merely near)
    coords = np.asarray(ls.coords, dtype=float)
    p0, pL = coords[0], coords[-1]
    L = ls.length
    # forward tangents at ends
    t0 = coords[1] - coords[0] if len(coords) > 1 else np.array([1.0, 0.0])
    tL = coords[-1] - coords[-2] if len(coords) > 1 else np.array([1.0, 0.0])
    t0 /= np.linalg.norm(t0) or 1.0
    tL /= np.linalg.norm(tL) or 1.0
    v0 = np.array(P) - p0
    vL = np.array(P) - pL
    if s <= tol and v0 @ t0 < 0:        # before start
        return True
    if s >= L - tol and vL @ tL > 0:    # after end
        return True
    return False

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0])

def _wrap_pi(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _yaw_from_xy(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Estimate yaw from positions via finite differences."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    T = len(xs)
    yaw = np.zeros(T, dtype=float)
    if T >= 2:
        yaw[0] = np.arctan2(ys[1] - ys[0], xs[1] - xs[0])
        yaw[-1] = np.arctan2(ys[-1] - ys[-2], xs[-1] - xs[-2])
    if T >= 3:
        dy = ys[2:] - ys[:-2]
        dx = xs[2:] - xs[:-2]
        yaw[1:-1] = np.arctan2(dy, dx)
    return yaw

def _tangent_and_point(ls: LineString, s: float, ds: float = 0.5):
    """
    Returns (t_hat, pcl) where:
      - t_hat is the unit tangent of the LineString at arc-length s
      - pcl is the point on the LineString at arc-length s
    """
    L = ls.length
    s = float(np.clip(s, 0.0, L))
    s1 = max(0.0, s - ds)
    s2 = min(L,   s + ds)
    p1 = np.array(ls.interpolate(s1).coords[0])
    p2 = np.array(ls.interpolate(s2).coords[0])
    t = _unit(p2 - p1)
    if np.allclose(t, 0.0):
        # fallback: widen the stencil
        eps = min(max(ds*2, 0.05*L), max(1e-3, ds*4))
        s1 = max(0.0, s - eps)
        s2 = min(L,   s + eps)
        p1 = np.array(ls.interpolate(s1).coords[0])
        p2 = np.array(ls.interpolate(s2).coords[0])
        t = _unit(p2 - p1)
        if np.allclose(t, 0.0) and len(ls.coords) >= 2:
            # final fallback: use nearest segment direction
            i = 0 if s < L/2 else -2
            p1 = np.array(ls.coords[i])
            p2 = np.array(ls.coords[i+1])
            t = _unit(p2 - p1)
    pcl = np.array(ls.interpolate(s).coords[0])
    return t, pcl

def frenet_st_series(centerline: LineString, xs: np.ndarray, ys: np.ndarray,
                     valid=None, ds: float = 0.5, mask_outside_valid: bool = True):
    """
    Compute OpenSCENARIO (s,t) for a trajectory w.r.t. a reference LineString.

    s: arc-length along the reference line from its start [m]
    t: signed lateral offset (left of the ref line is +, right is -) [m]
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    T = len(xs)
    if centerline is None: L = 0
    else: L = centerline.length

    s_out  = [None] * T
    t_out  = [None] * T

    v0, v1 = (0, T-1) if valid is None else (int(valid[0]), int(valid[1]))
    v0 = max(0, v0); v1 = min(T-1, v1)

    # iterate only valid window to save work
    for t in range(v0, v1 + 1):
        x, y = xs[t], ys[t]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        P = Point(float(x), float(y))
        s = centerline.project(P)              # arc length (closest point)
        if _outside_extent(centerline, [x, y], s):
            s_out[t] = None
            t_out[t] = None
            continue
        t_hat, pcl = _tangent_and_point(centerline, s, ds=ds)
        n_hat = np.array([-t_hat[1], t_hat[0]])   # left normal
        r = np.array([x, y]) - pcl
        s_out[t] = float(s)
        t_out[t] = float(r @ n_hat)

    if not mask_outside_valid:
        # optional: fill outside valid
        for t in range(0, v0):
            s_out[t] = float(0.0); t_out[t] = float(0.0)
        for t in range(v1+1, T):
            s_out[t] = float(L);  t_out[t] = float(0.0)

    return s_out, t_out, (v0, v1)



# ===================== velocities & accelerations in s/t (+ yaw_delta) =====================

def _finite_diff_first(arr: np.ndarray, dt: float) -> np.ndarray:
    """1st derivative with fwd/backward at ends, central inside; respects NaNs."""
    T = len(arr)
    out = np.full(T, np.nan, dtype=float)
    if T < 2:
        return out
    if np.isfinite(arr[0]) and np.isfinite(arr[1]):
        out[0] = (arr[1] - arr[0]) / dt
    if np.isfinite(arr[-1]) and np.isfinite(arr[-2]):
        out[-1] = (arr[-1] - arr[-2]) / dt
    if T >= 3:
        a = arr[:-2]; c = arr[2:]
        ok = np.isfinite(a) & np.isfinite(c)
        out[1:-1][ok] = (c[ok] - a[ok]) / (2.0 * dt)
    return out

def _finite_diff_second(arr: np.ndarray, dt: float) -> np.ndarray:
    """2nd derivative with central stencil; NaNs where neighbors missing; edges stay NaN."""
    T = len(arr)
    out = np.full(T, np.nan, dtype=float)
    if T < 3:
        return out
    a = arr[:-2]; b = arr[1:-1]; c = arr[2:]
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    out[1:-1][ok] = (c[ok] - 2.0*b[ok] + a[ok]) / (dt*dt)
    return out

def _nan_to_none_list(a: np.ndarray) -> list:
    return [None if (x is None or not np.isfinite(x)) else float(x) for x in a]


def process_actor_data_per_segment(condensed, segment, dt: float, ds: float = 0.5):
    """
    Returns:
       {
        's','t','valid', 'yaw_delta',
        's_dot','t_dot','s_ddot','t_ddot',
        'osc_lane_id'  # per-timestep OSC lane id mapped from lane_id via processed_segs[seg_id]["oscid_by_lane"]
      }
    """
    per_segment_ids = condensed["per_segment_ids"]
    all_payloads    = condensed["actor_activities"]

    ref_line = segment["reference_line"]
    osc_map  = segment.get("oscid_by_lane", {})  # lane_id -> osc id
    per_segment_kin = {}
    #print(f"processing seg id (kinematics): {seg_id}")
    for actor_key in per_segment_ids:
        rec = all_payloads.get(actor_key)
        if rec is None:
            continue

        # pull kinematics
        x = rec.get('x', rec.get('kinematics', {}).get('x', None))
        y = rec.get('y', rec.get('kinematics', {}).get('y', None))
        yaw = rec.get('yaw',
                rec.get('bbox_yaw',
                rec.get('kinematics', {}).get('yaw',
                rec.get('kinematics', {}).get('bbox_yaw', None))))
        valid = rec.get('valid', rec.get('meta', {}).get('valid', None))
        lane_series = rec.get("lane_id", []) or []
        if x is None or y is None:
            continue

        s_list, t_list, (v0, v1) = frenet_st_series(ref_line, x, y, valid=valid, ds=ds, mask_outside_valid=True)

        # arrays with NaN outside valid or where s/t is None
        T = len(s_list)
        s_arr = np.full(T, np.nan, dtype=float)
        t_arr = np.full(T, np.nan, dtype=float)
        for ti in range(v0, v1 + 1):
            s_val = s_list[ti]; t_val = t_list[ti]
            if s_val is not None and t_val is not None:
                s_arr[ti] = s_val
                t_arr[ti] = t_val

        # s/t derivatives
        s_dot  = _finite_diff_first(s_arr, dt)
        t_dot  = _finite_diff_first(t_arr, dt)
        s_ddot = _finite_diff_second(s_arr, dt)
        t_ddot = _finite_diff_second(t_arr, dt)

        # yaw delta (actor yaw - ref tangent)
        if yaw is None:
            yaw_arr = _yaw_from_xy(x, y)
        else:
            yaw_arr = np.asarray(yaw, dtype=float)

        yaw_delta = [None] * T
        for ti in range(v0, v1 + 1):
            if not np.isfinite(yaw_arr[ti]):
                continue
            s = s_list[ti]
            if s is None:
                continue
            t_hat, _ = _tangent_and_point(ref_line, s, ds=ds)
            psi_ref = np.arctan2(t_hat[1], t_hat[0])
            yaw_delta[ti] = float(_wrap_pi(yaw_arr[ti] - psi_ref))

        # NEW: per-timestep OSC lane id mapped from lane_id via osc_map
        osc_lane_id = [None] * T
        Ls = len(lane_series)
        for ti in range(v0, v1 + 1):
            if ti < Ls:
                lid = lane_series[ti]
                
                if lid is not None:
                    osc_lane_id[ti] = osc_map.get(int(lid))  # None if not found

        per_segment_kin[actor_key] = {
            "s": s_list,
            "t": t_list,
            "valid": (int(v0), int(v1)),
            "yaw_delta": yaw_delta,
            "s_dot":  _nan_to_none_list(s_dot),
            "t_dot":  _nan_to_none_list(t_dot),
            "s_ddot": _nan_to_none_list(s_ddot),
            "t_ddot": _nan_to_none_list(t_ddot),
            "osc_lane_id": osc_lane_id,
        }

    return per_segment_kin

def compute_segment_openscenario_coords(condensed, processed_segs, dt: float, ds: float = 0.5):
    """
    Returns:
      per_segment_kin[seg_id][actor_key] = {
        's','t','valid', 'yaw_delta',
        's_dot','t_dot','s_ddot','t_ddot',
        'osc_lane_id'  # per-timestep OSC lane id mapped from lane_id via processed_segs[seg_id]["oscid_by_lane"]
      }
    """
    per_segment_ids = condensed["per_segment_ids"]
    all_payloads    = condensed["actor_activities"]

    per_segment_kin = {}
    for seg_id in sorted(per_segment_ids.keys()):
        ref_line = processed_segs[seg_id]["reference_line"]
        osc_map  = processed_segs[seg_id].get("oscid_by_lane", {})  # lane_id -> osc id
        per_segment_kin[seg_id] = {}
        print(f"processing seg id (kinematics): {seg_id}")
        for actor_key in per_segment_ids[seg_id]:
            rec = all_payloads.get(actor_key)
            if rec is None:
                continue

            # pull kinematics
            x = rec.get('x', rec.get('kinematics', {}).get('x', None))
            y = rec.get('y', rec.get('kinematics', {}).get('y', None))
            yaw = rec.get('yaw',
                    rec.get('bbox_yaw',
                    rec.get('kinematics', {}).get('yaw',
                    rec.get('kinematics', {}).get('bbox_yaw', None))))
            valid = rec.get('valid', rec.get('meta', {}).get('valid', None))
            lane_series = rec.get("lane_id", []) or []
            if x is None or y is None:
                continue

            s_list, t_list, (v0, v1) = frenet_st_series(ref_line, x, y, valid=valid, ds=ds, mask_outside_valid=True)

            # arrays with NaN outside valid or where s/t is None
            T = len(s_list)
            s_arr = np.full(T, np.nan, dtype=float)
            t_arr = np.full(T, np.nan, dtype=float)
            for ti in range(v0, v1 + 1):
                s_val = s_list[ti]; t_val = t_list[ti]
                if s_val is not None and t_val is not None:
                    s_arr[ti] = s_val
                    t_arr[ti] = t_val

            # s/t derivatives
            s_dot  = _finite_diff_first(s_arr, dt)
            t_dot  = _finite_diff_first(t_arr, dt)
            s_ddot = _finite_diff_second(s_arr, dt)
            t_ddot = _finite_diff_second(t_arr, dt)

            # yaw delta (actor yaw - ref tangent)
            if yaw is None:
                yaw_arr = _yaw_from_xy(x, y)
            else:
                yaw_arr = np.asarray(yaw, dtype=float)

            yaw_delta = [None] * T
            for ti in range(v0, v1 + 1):
                if not np.isfinite(yaw_arr[ti]):
                    continue
                s = s_list[ti]
                if s is None:
                    continue
                t_hat, _ = _tangent_and_point(ref_line, s, ds=ds)
                psi_ref = np.arctan2(t_hat[1], t_hat[0])
                yaw_delta[ti] = float(_wrap_pi(yaw_arr[ti] - psi_ref))

            # NEW: per-timestep OSC lane id mapped from lane_id via osc_map
            osc_lane_id = [None] * T
            Ls = len(lane_series)
            for ti in range(v0, v1 + 1):
                if ti < Ls:
                    lid = lane_series[ti]
                    
                    if lid is not None:
                        osc_lane_id[ti] = osc_map.get(int(lid))  # None if not found

            per_segment_kin[seg_id][actor_key] = {
                "s": s_list,
                "t": t_list,
                "valid": (int(v0), int(v1)),
                "yaw_delta": yaw_delta,
                "s_dot":  _nan_to_none_list(s_dot),
                "t_dot":  _nan_to_none_list(t_dot),
                "s_ddot": _nan_to_none_list(s_ddot),
                "t_ddot": _nan_to_none_list(t_ddot),
                "osc_lane_id": osc_lane_id,
            }
    return per_segment_kin


import numpy as np

# ---------- utilities ----------

def _central_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros_like(arr, dtype=float)
    if len(arr) >= 2:
        v[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
        v[0]    = (arr[1]  - arr[0])   / dt
        v[-1]   = (arr[-1] - arr[-2])  / dt
    return v

def _yaw_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    yaw = np.zeros_like(x, dtype=float)
    if len(x) >= 2:
        yaw[1:-1] = np.arctan2(y[2:] - y[:-2], x[2:] - x[:-2])
        yaw[0]    = np.arctan2(y[1]  - y[0],   x[1]  - x[0])
        yaw[-1]   = np.arctan2(y[-1] - y[-2],  x[-1] - x[-2])
    return yaw

def _ttc_circle(p1, v1, r1, p2, v2, r2) -> float:
    r = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=float)
    v = np.array([v2[0]-v1[0], v2[1]-v1[1]], dtype=float)
    R = float(r1 + r2)
    c = r @ r - R*R
    if c <= 0.0:
        return 0.0
    a = v @ v
    if a <= 1e-9:
        return np.inf
    b = 2.0 * (r @ v)
    disc = b*b - 4*a*c
    if disc < 0.0:
        return np.inf
    t1 = (-b - np.sqrt(disc)) / (2*a)
    t2 = (-b + np.sqrt(disc)) / (2*a)
    if t1 >= 0.0:
        return float(t1)
    if t2 >= 0.0:
        return float(t2)
    return np.inf

def _front_or_back_str(x1, y1, yaw1, x2, y2) -> str:
    dx, dy = (x2 - x1), (y2 - y1)
    if dx == 0.0 and dy == 0.0:
        return "unknown"
    hx, hy = np.cos(yaw1), np.sin(yaw1)
    dot = hx*dx + hy*dy
    cross = hx*dy - hy*dx
    phi = np.arctan2(cross, dot)  # (-pi, pi]
    return "front" if (-0.5*np.pi < phi <= 0.5*np.pi) else "back"

def _first_present(d, candidates):
    """
    candidates may include strings (top-level keys) or tuples for nested lookups, e.g. ('kinematics','x').
    Returns the first present value, else raises KeyError.
    """
    for key in candidates:
        if isinstance(key, tuple):
            cur = d
            ok = True
            for k in key:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok:
                return cur
        else:
            if isinstance(d, dict) and key in d:
                return d[key]
    raise KeyError(f"None of the keys found: {candidates}. Available top-level keys: {list(d.keys())}")

def _radius_for_actor(actor_key: str, type_radii: dict) -> float:
    a_type = actor_key.split('_', 1)[0]  # 'vehicle_123' -> 'vehicle'
    return float(type_radii.get(a_type, 1.0))

# ---------- main ----------

# Try your project's timestep
try:
    from parameters.tag_parameters import t_s as _DEFAULT_DT
except Exception:
    _DEFAULT_DT = None

_DEFAULT_RADII = {'vehicle': 2.3, 'pedestrian': 0.5, 'cyclist': 0.8}

def build_inter_actor_position_and_ttc(per_actor: dict, dt: float = None, type_radii: dict = None) -> dict:
    """
    per_actor: output of per_actor_minimal(...) OR a similar dict.
               Accepts flat keys ('x','y','yaw','valid') or nested under 'kinematics'.
               Yaw can be 'yaw' or 'bbox_yaw'. If yaw missing, it is derived from xy.
    """
    if dt is None:
        if _DEFAULT_DT is None:
            raise ValueError("dt not provided and parameters.tag_parameters.t_s not found.")
        dt = float(_DEFAULT_DT)
    if type_radii is None:
        type_radii = dict(_DEFAULT_RADII)

    # normalize each actor record into x,y,yaw,valid
    kin = {}
    for key, d in per_actor.items():
        # x/y
        try:
            x = np.asarray(_first_present(d, ['x', ('kinematics','x')]), dtype=float)
            y = np.asarray(_first_present(d, ['y', ('kinematics','y')]), dtype=float)
        except KeyError as e:
            raise KeyError(f"{key}: missing x/y arrays. {e}")
        # yaw (several possible field names; else compute from xy)
        try:
            yaw = np.asarray(_first_present(d, ['yaw', 'bbox_yaw', ('kinematics','yaw'), ('kinematics','bbox_yaw')]), dtype=float)
        except KeyError:
            yaw = _yaw_from_xy(x, y)
        # valid
        try:
            v0, v1 = _first_present(d, ['valid', 'validity', ('meta','valid')])
            v0, v1 = int(v0), int(v1)
        except Exception:
            v0, v1 = 0, len(x) - 1

        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        yaw = np.nan_to_num(yaw, nan=0.0)

        vx = _central_diff(x, dt)
        vy = _central_diff(y, dt)

        kin[key] = dict(x=x, y=y, yaw=yaw, vx=vx, vy=vy, valid=(v0, v1), radius=_radius_for_actor(key, type_radii))

    keys = list(kin.keys())
    inter_actor_relation = {}

    for i, ki in enumerate(keys):
        Ai = kin[ki]
        Ti = len(Ai['x'])
        inter_actor_relation[ki] = {}

        for j, kj in enumerate(keys):
            if kj == ki:
                continue

            Aj = kin[kj]

            pos = np.full(Ti, "unknown", dtype=object)
            ttc = np.full(Ti, np.inf, dtype=float)
            distance = np.full(Ti, np.nan, dtype=float)  # <-- NEW

            # overlap of valid ranges (and array bounds)
            vi0, vi1 = Ai['valid']; vj0, vj1 = Aj['valid']
            t0 = max(vi0, vj0)
            t1 = min(vi1, vj1, Ti - 1, len(Aj['x']) - 1)
            if t1 < t0:
                inter_actor_relation[ki][kj] = {
                    'position': pos.tolist(),
                    'ttc': ttc.tolist(),
                    'eucl_distance': distance.tolist(),  # <-- NEW
                }
                continue

            # vectorized distance for the valid overlap
            dx = Ai['x'][t0:t1+1] - Aj['x'][t0:t1+1]
            dy = Ai['y'][t0:t1+1] - Aj['y'][t0:t1+1]
            distance[t0:t1+1] = np.hypot(dx, dy)  # <-- NEW

            # per-timestep fields that depend on yaw/velocities
            for t in range(t0, t1 + 1):
                pos[t] = _front_or_back_str(Ai['x'][t], Ai['y'][t], Ai['yaw'][t], Aj['x'][t], Aj['y'][t])
                ttc[t] = _ttc_circle(
                    (Ai['x'][t], Ai['y'][t]), (Ai['vx'][t], Ai['vy'][t]), Ai['radius'],
                    (Aj['x'][t], Aj['y'][t]), (Aj['vx'][t], Aj['vy'][t]), Aj['radius']
                )

            inter_actor_relation[ki][kj] = {
                'position': pos.tolist(),
                'ttc': ttc.tolist(),
                'eucl_distance': distance.tolist(),  # <-- NEW
            }

    return inter_actor_relation

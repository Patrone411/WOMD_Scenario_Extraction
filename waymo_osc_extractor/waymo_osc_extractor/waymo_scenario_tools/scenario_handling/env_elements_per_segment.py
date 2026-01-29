from typing import List, Tuple
from shapely.geometry import LineString, Polygon, Point, MultiPoint, MultiLineString
import math
from shapely.geometry import LineString, Point

try:
    # Shapely 2.x
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

def _clean_poly(g: Polygon) -> Polygon:
    """Return a valid single Polygon or None."""
    if g is None or g.is_empty:
        return None
    try:
        if not g.is_valid:
            g = _make_valid(g) if _make_valid else g.buffer(0)
    except Exception:
        return None
    # collapse MultiPolygon to largest part
    if getattr(g, "geom_type", "") == "MultiPolygon":
        parts = [p for p in g.geoms if not p.is_empty and p.is_valid]
        if not parts:
            return None
        g = max(parts, key=lambda p: p.area)
    return g if (not g.is_empty and g.is_valid) else None

def signed_lateral_offset(centerline: LineString, pt: Point, ds: float = 0.5):
    """
    Returns (s, foot_point_on_centerline, t_signed).
      s: arc length from start of centerline [m]
      t_signed: left-of-centerline is +, right is - (same as your frenet_st_series)
    """
    # arc-length position and foot point
    s = centerline.project(pt)
    foot = centerline.interpolate(s)

    # local tangent via finite difference
    L = centerline.length
    s0 = max(0.0, s - 0.5*ds)
    s1 = min(L,   s + 0.5*ds)
    p0 = centerline.interpolate(s0)
    p1 = centerline.interpolate(s1)

    vx, vy = (p1.x - p0.x), (p1.y - p0.y)
    nrm = math.hypot(vx, vy)
    if nrm == 0.0:                       # rare: degenerate or very short segment; widen stencil
        eps = max(1e-3, min(ds, 0.01*L))
        s0 = max(0.0, s - eps); p0 = centerline.interpolate(s0)
        s1 = min(L,   s + eps); p1 = centerline.interpolate(s1)
        vx, vy = (p1.x - p0.x), (p1.y - p0.y)
        nrm = math.hypot(vx, vy) or 1.0

    tx, ty = vx/nrm, vy/nrm                  # unit tangent (start->end direction)
    nx, ny = -ty, tx                         # unit left-normal
    rx, ry = (pt.x - foot.x), (pt.y - foot.y)
    t_signed = rx*nx + ry*ny
    return float(s), foot, float(t_signed)


def is_point_on_lane(pt: Point, lane_poly: Polygon, tol: float = 0.5) -> bool:
    """Returns True if the point lies within the lane polygon (with tolerance in meters)."""
    return lane_poly.buffer(tol).contains(pt)

def crosswalk_on_lane(crosswalk: Polygon, lane_poly: Polygon,
                      tol: float = 0.0, min_overlap_ratio: float = 0.1) -> bool:
    """Robust overlap test: (optionally) buffer the CROSSWALK, not the lane."""
    cw = _clean_poly(crosswalk)
    lp = _clean_poly(lane_poly)
    if cw is None or lp is None or cw.area == 0:
        return False
    try:
        test_cw = cw.buffer(tol) if tol > 0 else cw
        inter_area = test_cw.intersection(lp).area
        if tol == 0.0:
            return inter_area > 0.0
        return (inter_area / cw.area) >= min_overlap_ratio
    except Exception:
        return False

def s_along_centerline_point(centerline: LineString, pt: Point,
                             normalized: bool = False) -> Tuple[float, Point, float]:
    """
    Curvilinear position s of a point along the centerline.
    Returns (s, foot_point, lateral_offset).
    s is in meters unless normalized=True (then in [0,1]).
    """
    s = centerline.project(pt, normalized=normalized)
    foot = centerline.interpolate(s, normalized=normalized)
    lat = pt.distance(foot)
    return float(s), foot, float(lat)

def s_intervals_for_polygon(centerline: LineString, poly: Polygon) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Where does the centerline pass through the polygon?
    Returns:
      - intervals: list of (s_start, s_end) along the centerline for each crossing segment
      - points_s: list of s positions for point-only intersections (tangential touch)
    If there is no intersection, returns [] and [s_centroid] as a fallback.
    """
    inter = centerline.intersection(poly)
    intervals: List[Tuple[float, float]] = []
    points_s: List[float] = []

    def _proj(p: Point) -> float:
        return float(centerline.project(p, normalized=False))

    if inter.is_empty:
        # Fallback: project centroid (nearest approach)
        c = poly.centroid
        return [], [ _proj(c) ]

    if isinstance(inter, Point):
        points_s.append(_proj(inter))
    elif isinstance(inter, MultiPoint):
        for p in inter.geoms:
            points_s.append(_proj(p))
    elif isinstance(inter, LineString):
        a = Point(inter.coords[0]); b = Point(inter.coords[-1])
        s0, s1 = _proj(a), _proj(b)
        if s1 < s0: s0, s1 = s1, s0
        intervals.append((s0, s1))
    elif isinstance(inter, MultiLineString):
        for seg in inter.geoms:
            a = Point(seg.coords[0]); b = Point(seg.coords[-1])
            s0, s1 = _proj(a), _proj(b)
            if s1 < s0: s0, s1 = s1, s0
            intervals.append((s0, s1))
    else:
        # GeometryCollection: recurse basic cases
        for g in getattr(inter, "geoms", []):
            if isinstance(g, Point):
                points_s.append(_proj(g))
            elif isinstance(g, LineString):
                a = Point(g.coords[0]); b = Point(g.coords[-1])
                s0, s1 = _proj(a), _proj(b)
                if s1 < s0: s0, s1 = s1, s0
                intervals.append((s0, s1))

    points_s.sort()
    intervals.sort()
    return intervals, points_s


def process_env_elements_for_segment(segment, scenario_env_elements):
    
    on_lane_tol = 0.5            # meters tolerance for point-in-lane
    cw_overlap_tol = 0.2         # meters buffer for overlap test
    cw_min_overlap_ratio = 0.10  # require at least 10% of crosswalk area overlapping the lane

    ref_line = segment["reference_line"]
    
    crosswalks = scenario_env_elements.get_other_object('cross_walk')          # list[Polygon]
    tl_points = scenario_env_elements.traffic_lights['points']                 # list[Point]
    tl_state = scenario_env_elements.traffic_lights['traffic_lights_state']    # np.ndarray [T, N]
    
    tl_hit_ids = []
    cw_hit_ids = []
    tl_results = []
    cw_results = []

    for cid, rec in segment["chains"].items():
        poly = rec["polygon"]        # None if polygons were skipped; otherwise check poly.is_empty
        oscid  = rec["oscid"]
        cl   = rec["centerline"]

        for i, pt in enumerate(tl_points):
            if not is_point_on_lane(pt, poly, tol=on_lane_tol):
                continue
            tl_hit_ids.append(i)
            s, foot, t = signed_lateral_offset(ref_line, pt, ds=0.5)            
            tl_results.append({
                "idx": i,
                "point": pt,
                "s": s,                                # meters from start of centerline
                "t": t,                 # meters from centerline
            })
        # Sort them along the lane

        for i, cw in enumerate(crosswalks):
            if not crosswalk_on_lane(cw, poly, tol=cw_overlap_tol, min_overlap_ratio=cw_min_overlap_ratio):
                continue
            cw_hit_ids.append(i)
            intervals, point_hits = s_intervals_for_polygon(cl, cw)
            # representative s value(s) to label the crosswalk
            reps = [0.5*(a+b) for a,b in intervals] if intervals else point_hits
            cw_results.append({
                "idx": i,
                "intervals": intervals,                              # list of (s_start, s_end) in meters
                "s": point_hits,                        # s positions if centerline only touches
                "repr_s": reps,                                      # convenient labeling positions
            })

    tl_results.sort(key=lambda r: r["s"])
    cw_results.sort(key=lambda r: (r["repr_s"][0] if r["repr_s"] else float("inf")))

    per_segment_envs = {
        "tl_ids": set(tl_hit_ids),
        "cw_ids": set(cw_hit_ids),
        "tl_results": tl_results,
        "cw_results": cw_results,
    }
    return per_segment_envs


def process_env_elements_segment_wise(all_segments, scenario_env_elements):

    per_segment_env = {}
    for seg_key, res in all_segments.items():
        env_res = process_env_elements_for_segment(res,scenario_env_elements)
        per_segment_env[seg_key] = env_res

    return per_segment_env
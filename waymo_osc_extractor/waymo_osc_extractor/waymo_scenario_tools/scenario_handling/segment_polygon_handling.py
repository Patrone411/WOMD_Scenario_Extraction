from typing import Dict, List, Tuple, Optional
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, linemerge

from typing import Optional, Union
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon

try:
    # Shapely >= 2.0
    from shapely.validation import make_valid
    HAVE_MAKE_VALID = True
except ImportError:
    from shapely.ops import unary_union
    HAVE_MAKE_VALID = False

def compare_lane_geoms(original_poly_raw, sanitized_poly_clean):
    o = normalize_area_geom(original_poly_raw)
    c = normalize_area_geom(sanitized_poly_clean)

    # handle empties
    if o.is_empty and c.is_empty:
        return {
            "both_empty": True,
            "iou": 1.0,
            "area_ratio": 1.0,
            "rel_diff": 0.0,
            "hausdorff": 0.0,
            "suspicious": False,
        }
    if o.is_empty != c.is_empty:
        # one vanished or one just appeared
        return {
            "both_empty": False,
            "iou": 0.0,
            "area_ratio": 0.0,
            "rel_diff": 1.0,
            "hausdorff": float("inf"),
            "suspicious": True,
        }

    inter = o.intersection(c)
    uni   = o.union(c)
    symd  = o.symmetric_difference(c)

    # compute metrics
    A = o.area
    B = c.area
    area_sim = (min(A, B) / max(A, B)) if max(A, B) > 1e-9 else 1.0
    iou = (inter.area / uni.area) if not uni.is_empty else 1.0
    rel_diff = symd.area / (A + 1e-9)
    haus = o.hausdorff_distance(c)

    # pick your own thresholds
    suspicious = (
        iou < 0.9 or
        area_sim < 0.95 or
        rel_diff > 0.2 or
        haus > 1.0   # lane moved >1 meter
    )
    if suspicious:
        result = {
        "both_empty": False,
        "iou": iou,
        "area_ratio": area_sim,
        "rel_diff": rel_diff,
        "hausdorff": haus,
        "suspicious": suspicious,
        }
        print(result)
    else:
        print('fine')
    return {
        "both_empty": False,
        "iou": iou,
        "area_ratio": area_sim,
        "rel_diff": rel_diff,
        "hausdorff": haus,
        "suspicious": suspicious,
    }

def area_ratio(a, b, eps=1e-9):
    A = a.area
    B = b.area
    if A < eps and B < eps:
        return 1.0  # both basically zero
    return min(A, B) / max(A, B)


def normalize_area_geom(g):
    """
    Return a Polygon or MultiPolygon representing the areal part of g.
    Returns empty Polygon() if unusable.
    """
    if g is None or g.is_empty:
        return Polygon()

    if isinstance(g, (Polygon, MultiPolygon)):
        return g

    # geometry collection etc -> pull only polygons
    parts = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    return unary_union(parts)

def sanitize_lane_polygon(
    geom: BaseGeometry,
    max_abs_coord: float = 1e6,
    simplify_tol: float = 0.05,
) -> Optional[Union[Polygon, MultiPolygon]]:
    """
    Take a raw lane polygon and return a geometry that is:
    - finite (no NaNs, no absurd coordinates)
    - non-empty
    - valid (no self-intersections)
    - slightly simplified to reduce razor-thin spikes

    Returns None if we can't get something trustworthy.
    """

    # 0. reject obviously broken inputs
    if geom is None:
        return None
    if geom.is_empty:
        return None

    # 1. coordinate sanity check
    #    (if bounds are astronomically large or NaN, this is garbage map data)
    try:
        minx, miny, maxx, maxy = geom.bounds
    except Exception:
        return None

    for v in (minx, miny, maxx, maxy):
        # NaN check: NaN != NaN
        if v != v:
            return None
        if abs(v) > max_abs_coord:
            return None

    g = geom

    # 2. optional light simplify to shave off micro-spikes / self overlaps
    #    we keep preserve_topology=True so we don't turn a lane into nonsense
    try:
        if getattr(g, "length", 0.0) > 0.0:
            g = g.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        # if simplify explodes, ignore and keep original
        pass

    # 3. if geometry is already valid after simplify, we're done
    if g.is_valid:
        return g

    # 4. try to repair invalid polygons (self-intersections etc.)
    try:
        if HAVE_MAKE_VALID:
            g_fixed = make_valid(g)
        else:
            # Shapely <2 trick:
            #   - buffer(0) often fixes self-crossing rings
            #   - unary_union can dissolve weird overlaps
            g_fixed = g.buffer(0)
            if g_fixed.is_empty:
                g_fixed = unary_union([g])
    except Exception:
        return None

    # 5. after repair, final checks
    if g_fixed is None or g_fixed.is_empty:
        return None
    if not g_fixed.is_valid:
        return None

    # we only want surface-like geometry back (Polygon or MultiPolygon).
    if not isinstance(g_fixed, (Polygon, MultiPolygon)):
        # e.g. make_valid() can sometimes spit back GeometryCollection
        # if it's not an area, it's not useful as a lane footprint.
        polys = [part for part in getattr(g_fixed, "geoms", []) if isinstance(part, (Polygon, MultiPolygon))]
        if not polys:
            return None
        # merge those area parts
        if len(polys) == 1:
            g_fixed = polys[0]
        else:
            # union them into one MultiPolygon
            try:
                g_fixed = unary_union(polys) if not HAVE_MAKE_VALID else make_valid(unary_union(polys))
            except Exception:
                return None

    return g_fixed

# -----------------------------
# Tiny geometry + stitching
# -----------------------------
def _as2d(ls: LineString) -> LineString:
    return LineString([(x, y) for (x, y, *_) in ls.coords])

def _flip(ls: LineString) -> LineString:
    return LineString(list(reversed(ls.coords)))

def _dist2(p, q) -> float:
    dx = p[0]-q[0]; dy = p[1]-q[1]
    return dx*dx + dy*dy

def _stitch_greedy(pieces: List[LineString]) -> LineString:
    """
    Greedy continuity stitch (respect input order). Flips each piece
    if that better continues the previous piece's end. Then concatenates.
    """
    out: List[LineString] = []
    for g in pieces:
        if g.is_empty or len(g.coords) < 2:
            continue
        g2 = _as2d(g)
        if not out:
            out.append(g2)
            continue
        pe = out[-1].coords[-1]  # previous end
        s0 = g2.coords[0]; s1 = g2.coords[-1]
        if _dist2(pe, s1) < _dist2(pe, s0):
            g2 = _flip(g2)
        out.append(g2)

    coords: List[Tuple[float, float]] = []
    for g in out:
        pts = list(g.coords)
        if not coords:
            coords.extend(pts)
        else:
            coords.extend(pts[1:] if coords[-1] == pts[0] else pts)
    return LineString(coords) if len(coords) >= 2 else LineString()

# -----------------------------
# Chain slicing by endpoints
# -----------------------------
def _slice_chain_segments_with_laneids(
    lane_graph,
    lane_ids: List[int],
    start: Optional[Tuple[int,int]],
    end:   Optional[Tuple[int,int]],
) -> Dict[int, List[LineString]]:
    """
    Return {lane_id: [segments]} limited to (start_lane, start_idx) .. (end_lane, end_idx) inclusive.
    If start/end are None -> take the whole chain in the given lane_ids order.
    """
    def segs(lid): return lane_graph.lane_segments.get(lid, [])
    selected: Dict[int, List[LineString]] = {lid: [] for lid in lane_ids}

    if not lane_ids:
        return selected

    began = False
    s_lane = start[0] if start else lane_ids[0]
    s_idx  = start[1] if start else 0
    e_lane = end[0]   if end   else lane_ids[-1]
    e_idx  = end[1]   if end   else max(0, len(segs(lane_ids[-1])) - 1)

    for lid in lane_ids:
        pieces = segs(lid)
        if not pieces:
            continue
        if not began:
            if lid == s_lane:
                if lid == e_lane:
                    sel = pieces[s_idx:e_idx+1]
                    began = True
                else:
                    sel = pieces[s_idx:]
                    began = True
            else:
                continue
        else:
            if lid == e_lane:
                sel = pieces[:e_idx+1]
                selected[lid].extend([_as2d(g) for g in sel if (not g.is_empty and len(g.coords) >= 2)])
                break
            else:
                sel = pieces

        selected[lid].extend([_as2d(g) for g in sel if (not g.is_empty and len(g.coords) >= 2)])

    return selected

# =====================================================
# Boundaries from centerline + per-lane halfwidths
# =====================================================
def _build_chain_boundaries(
    lane_graph,
    lane_ids: List[int],
    slice_by: Optional[Tuple[Tuple[int,int], Tuple[int,int]]] = None,
    *,
    cap_style: int = 2,
    join_style: int = 2,
    mitre_limit: float = 5.0,
    bridge_tol: float = 0.75,
) -> Tuple[LineString, LineString, LineString]:
    """
    Build:
      - chain_centerline (stitched across lane_ids)
      - left_boundary    (offset-by-halfwidth, oriented to chain direction)
      - right_boundary   (offset-by-halfwidth, oriented to chain direction)
    using ONLY the centerline pieces and each lane_id's own halfwidth (no inflation).

    Returns (chain_centerline, left_boundary, right_boundary).
    """
    def hw(lid): return float(lane_graph.lane_graph[lid]["buffer_size"])
    start, end = slice_by if slice_by else (None, None)
    sel = _slice_chain_segments_with_laneids(lane_graph, lane_ids, start, end)

    # stitch per-lane centerlines
    stitched_by_lane: List[Tuple[int, LineString]] = []
    for lid in lane_ids:
        pieces = sel.get(lid, [])
        if not pieces:
            continue
        cl = _stitch_greedy(pieces)
        if not cl.is_empty and len(cl.coords) >= 2:
            stitched_by_lane.append((lid, cl))

    # chain centerline = concatenate stitched_by_lane
    chain_coords: List[Tuple[float, float]] = []
    for _, cl in stitched_by_lane:
        pts = list(cl.coords)
        if not chain_coords:
            chain_coords.extend(pts)
        else:
            chain_coords.extend(pts[1:] if chain_coords[-1] == pts[0] else pts)
    chain_centerline = LineString(chain_coords) if len(chain_coords) >= 2 else LineString()

    if chain_centerline.is_empty or len(stitched_by_lane) == 0:
        return chain_centerline, LineString(), LineString()

    chain_start = chain_centerline.coords[0]

    def _orient_to_chain_direction(ls: LineString) -> LineString:
        if ls.is_empty or len(ls.coords) < 2:
            return ls
        s = ls.coords[0]; e = ls.coords[-1]
        d_s = _dist2((s[0], s[1]), (chain_start[0], chain_start[1]))
        d_e = _dist2((e[0], e[1]), (chain_start[0], chain_start[1]))
        return ls if d_s <= d_e else _flip(ls)

    # offset each stitched lane centerline by its own hw
    left_parts: List[LineString] = []
    right_parts: List[LineString] = []

    for lid, cl in stitched_by_lane:
        r = hw(lid)
        try:
            l = cl.parallel_offset(r, 'left',  join_style=join_style, mitre_limit=mitre_limit)
            rgt = cl.parallel_offset(r, 'right', join_style=join_style, mitre_limit=mitre_limit)
        except Exception:
            continue

        # normalize to LineString (in case MultiLineString comes back)
        def _as_line(g):
            if g.is_empty:
                return LineString()
            if isinstance(g, LineString):
                return g
            if isinstance(g, MultiLineString):
                best = max(list(g.geoms), key=lambda s: s.length, default=None)
                return best if best is not None else LineString()
            m = linemerge(g)
            if isinstance(m, LineString):
                return m
            if isinstance(m, MultiLineString):
                best = max(list(m.geoms), key=lambda s: s.length, default=None)
                return best if best is not None else LineString()
            return LineString()

        l = _as_line(l)
        rgt = _as_line(rgt)
        if not l.is_empty:
            left_parts.append(_orient_to_chain_direction(l))
        if not rgt.is_empty:
            right_parts.append(_orient_to_chain_direction(rgt))

    # concatenate parts in lane order, bridging tiny gaps
    def _concat_parts(parts: List[LineString]) -> LineString:
        coords: List[Tuple[float, float]] = []
        for seg in parts:
            pts = list(seg.coords)
            if not pts:
                continue
            if not coords:
                coords.extend(pts)
                continue
            end_prev = coords[-1]
            start_next = pts[0]
            gap = ((end_prev[0]-start_next[0])**2 + (end_prev[1]-start_next[1])**2) ** 0.5
            if gap > 1e-12 and gap <= bridge_tol:
                coords.append(start_next)
            coords.extend(pts[1:] if coords and coords[-1] == pts[0] else pts)
        return LineString(coords) if len(coords) >= 2 else LineString()

    left_boundary  = _concat_parts(left_parts)
    right_boundary = _concat_parts(right_parts)
    return chain_centerline, left_boundary, right_boundary

# -----------------------------
# Build a buffered polygon per chain (inflate each lane_id)
# -----------------------------
def _buffer_chain_polygon(
    lane_graph,
    lane_ids: List[int],
    slice_by: Optional[Tuple[Tuple[int,int], Tuple[int,int]]] = None,
    inflate_pct: float = 0.10,  # +10% by default
    cap_style: int = 2,
    join_style: int = 2,
    mitre_limit: float = 5.0,
    resolution: int = 8,
) -> Tuple[Polygon, LineString]:
    """
    - Slice segments per lane_id (respecting 'slice_by' if provided).
    - Greedy-stitch all segments across lane_ids (to one chain centerline).
    - Buffer each lane_id centerline by (halfwidth * (1+inflate_pct)).
    - Union all lane-polys to one chain polygon (may be MultiPolygon).

    Returns:
      (chain_polygon (Polygon or MultiPolygon), stitched_chain_centerline)
    """
    def hw(lid): return float(lane_graph.lane_graph[lid]["buffer_size"])

    start, end = slice_by if slice_by else (None, None)
    sel = _slice_chain_segments_with_laneids(lane_graph, lane_ids, start, end)

    # 1) stitched chain centerline (lane-id concatenation with greedy stitch)
    stitched_by_lane: List[LineString] = []
    for lid in lane_ids:
        pieces = sel.get(lid, [])
        if not pieces:
            continue
        stitched_by_lane.append(_stitch_greedy(pieces))

    chain_cl_coords: List[Tuple[float, float]] = []
    for cl in stitched_by_lane:
        if cl.is_empty or len(cl.coords) < 2:
            continue
        pts = list(cl.coords)
        if not chain_cl_coords:
            chain_cl_coords.extend(pts)
        else:
            chain_cl_coords.extend(pts[1:] if chain_cl_coords[-1] == pts[0] else pts)
    chain_centerline = LineString(chain_cl_coords) if len(chain_cl_coords) >= 2 else LineString()

    # 2) buffer per lane_id, then union
    polys: List[Polygon] = []
    for lid in lane_ids:
        cl = _stitch_greedy(sel.get(lid, []))
        if cl.is_empty or cl.length == 0:
            continue
        r = hw(lid) * (1.0 + float(inflate_pct))
        p = cl.buffer(r, cap_style=cap_style, join_style=join_style, mitre_limit=mitre_limit, resolution=resolution)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            polys.append(p)

    if not polys:
        return Polygon(), chain_centerline

    chain_poly = unary_union(polys)  # may be Polygon or MultiPolygon
    return chain_poly, chain_centerline

# -----------------------------
# Utility: difference that tolerates any geometry combos
# -----------------------------
def _geom_difference(a, b):
    if a.is_empty:
        return a
    try:
        return a.difference(b)
    except Exception:
        return unary_union([a]).difference(unary_union([b]))

# -----------------------------
# Colormap helper (new API with fallback)
# -----------------------------
def _get_cmap(name: str):
    try:
        import matplotlib
        return matplotlib.colormaps.get_cmap(name)  # modern API
    except Exception:
        import matplotlib.cm as cm
        return cm.get_cmap(name)

# -----------------------------
# MAIN (inflate + force target boundary) + REF + OSCIDs
# -----------------------------
def build_polys_target_first_inflate(
    lane_graph,
    Lanesegments_block,
    *,
    compute_polygons: bool = True,  # <-- NEW: skip polygon work if False
    inflate_pct: float = 0.10,      # +10% (0.10) by default
    bridge_tol: float = 0.75,       # used only for boundary assembly
    show_plot: bool = True,
    plot_boundaries: bool = True,   # boundaries still plotted if desired
    arrows_per_line: int = 10,      # number of arrows on centerlines/boundaries
):
    """
    1) Identify roles from endpoints_by_id: 'target' (always present), optional 'left', 'right'.
    2) (Optional) Build buffered polygons for each present chain with +inflate_pct on each lane_id width.
    3) If polygons are computed, force inner boundaries of left/right to be the target boundary via difference.
    4) Compute directed left/right boundary lines per chain using ONLY centerlines + per-lane halfwidths.
    5) Choose a single reference line (using endpoints direction relative to target).
    6) Also compute OSC IDs per chain based on the same rule table; access via oscid_by_chain[chain_id].
    7) Plot (polygons if computed) + centerlines + boundaries + reference line.
    """
    chains = Lanesegments_block['chains']
    endpoints_by_id = Lanesegments_block.get('endpoints_by_id', {})
    endpoints = Lanesegments_block.get('endpoints', {})

    # roles → chain ids
    target_chain_id: Optional[int] = None
    left_chain_id: Optional[int] = None
    right_chain_id: Optional[int] = None
    for cid, info in endpoints_by_id.items():
        role = info.get('role')
        if role == 'target':
            target_chain_id = cid
        elif role == 'left':
            left_chain_id = cid
        elif role == 'right':
            right_chain_id = cid
    if target_chain_id is None:
        target_chain_id = chains[0]['id']

    # helper: find lane_ids & slice by endpoints
    def _lane_ids(cid): return next(c['lane_ids'] for c in chains if c['id'] == cid)
    def _slice_for_chain(cid):
        info = endpoints_by_id.get(cid)
        if info and 'start' in info and 'end' in info:
            return (info['start'], info['end'])
        return None

    # --- (Optional) Build polygons (inflated) ---
    if compute_polygons:
        # --- target lane ---
        t_lane_ids = _lane_ids(target_chain_id)
        t_slice = _slice_for_chain(target_chain_id)

        target_poly_raw, t_centerline_poly = _buffer_chain_polygon(
            lane_graph, t_lane_ids, slice_by=t_slice, inflate_pct=inflate_pct
        )

        target_poly_clean = sanitize_lane_polygon(target_poly_raw)
        if target_poly_clean is None:
            target_poly_clean = Polygon()  # safe empty fallback
        
        

        # --- init neighbors ---
        left_centerline_poly = LineString()
        right_centerline_poly = LineString()
        left_poly_clean = Polygon()
        right_poly_clean = Polygon()

        # --- left lane (if it exists) ---
        if left_chain_id is not None:
            ln_lane_ids = _lane_ids(left_chain_id)
            ln_slice = _slice_for_chain(left_chain_id)

            left_chain_poly_raw, left_centerline_poly = _buffer_chain_polygon(
                lane_graph, ln_lane_ids, slice_by=ln_slice, inflate_pct=inflate_pct
            )

            left_poly_clean = sanitize_lane_polygon(left_chain_poly_raw)
            if left_poly_clean is None:
                left_poly_clean = Polygon()
            

        # --- right lane (if it exists) ---
        if right_chain_id is not None:
            rn_lane_ids = _lane_ids(right_chain_id)
            rn_slice = _slice_for_chain(right_chain_id)

            right_chain_poly_raw, right_centerline_poly = _buffer_chain_polygon(
                lane_graph, rn_lane_ids, slice_by=rn_slice, inflate_pct=inflate_pct
            )

            right_poly_clean = sanitize_lane_polygon(right_chain_poly_raw)
            if right_poly_clean is None:
                right_poly_clean = Polygon()
            

        # --- difference step ---
        # remove overlap between neighbor lanes and target lane
        # IMPORTANT: use sanitized polys here, NOT the raw ones
        if not left_poly_clean.is_empty and not target_poly_clean.is_empty:
            left_final = _geom_difference(left_poly_clean, target_poly_clean)
        else:
            left_final = Polygon()

        if not right_poly_clean.is_empty and not target_poly_clean.is_empty:
            right_final = _geom_difference(right_poly_clean, target_poly_clean)
        else:
            right_final = Polygon()
    else:
        # placeholders when skipped
        target_poly_clean = Polygon()
        left_final = Polygon()
        right_final = Polygon()
        # also still need the lane_id lists and slices for boundaries below
        t_lane_ids = _lane_ids(target_chain_id)
        t_slice = _slice_for_chain(target_chain_id)
        ln_lane_ids = _lane_ids(left_chain_id) if left_chain_id is not None else None
        ln_slice    = _slice_for_chain(left_chain_id) if left_chain_id is not None else None
        rn_lane_ids = _lane_ids(right_chain_id) if right_chain_id is not None else None
        rn_slice    = _slice_for_chain(right_chain_id) if right_chain_id is not None else None

    # --- Build (non-inflated) directed boundaries for each present chain ---
    left_boundary_by_chain: Dict[int, LineString] = {}
    right_boundary_by_chain: Dict[int, LineString] = {}
    centerline_by_chain: Dict[int, LineString] = {}

    # target boundaries
    t_centerline_b, t_left_b, t_right_b = _build_chain_boundaries(
        lane_graph, t_lane_ids, slice_by=t_slice, bridge_tol=bridge_tol
    )
    centerline_by_chain[target_chain_id] = t_centerline_b
    left_boundary_by_chain[target_chain_id] = t_left_b
    right_boundary_by_chain[target_chain_id] = t_right_b

    # left
    if left_chain_id is not None:
        if not compute_polygons:
            ln_lane_ids = _lane_ids(left_chain_id)
            ln_slice    = _slice_for_chain(left_chain_id)
        ln_centerline_b, ln_left_b, ln_right_b = _build_chain_boundaries(
            lane_graph, ln_lane_ids, slice_by=ln_slice, bridge_tol=bridge_tol
        )
        centerline_by_chain[left_chain_id] = ln_centerline_b
        left_boundary_by_chain[left_chain_id] = ln_left_b
        right_boundary_by_chain[left_chain_id] = ln_right_b

    # right
    if right_chain_id is not None:
        if not compute_polygons:
            rn_lane_ids = _lane_ids(right_chain_id)
            rn_slice    = _slice_for_chain(right_chain_id)
        rn_centerline_b, rn_left_b, rn_right_b = _build_chain_boundaries(
            lane_graph, rn_lane_ids, slice_by=rn_slice, bridge_tol=bridge_tol
        )
        centerline_by_chain[right_chain_id] = rn_centerline_b
        left_boundary_by_chain[right_chain_id] = rn_left_b
        right_boundary_by_chain[right_chain_id] = rn_right_b

    # ---------------------------------------------------------
    # Select a single reference line using directions from `endpoints`
    # AND compute oscid_by_chain based on the same rule table
    # ---------------------------------------------------------
    has_left  = (left_chain_id  is not None
                 and left_chain_id  in centerline_by_chain
                 and not centerline_by_chain[left_chain_id].is_empty)
    has_right = (right_chain_id is not None
                 and right_chain_id in centerline_by_chain
                 and not centerline_by_chain[right_chain_id].is_empty)

    # "same" | "opposite" relative to target
    left_rel  = endpoints.get('left',  {}).get('direction')  if has_left  else None
    right_rel = endpoints.get('right', {}).get('direction')  if has_right else None

    def _get_left_boundary(cid):
        ls = left_boundary_by_chain.get(cid)
        return ls if (ls is not None and not ls.is_empty) else None

    reference_line = None
    reference_line_source = "none"

    # OSC IDs per chain (default None)
    oscid_by_chain: Dict[int, Optional[int]] = {
        target_chain_id: None
    }
    if left_chain_id is not None:
        oscid_by_chain[left_chain_id] = None
    if right_chain_id is not None:
        oscid_by_chain[right_chain_id] = None

    if has_left and has_right:
        # If right is opposite -> all None (overrides)
        if right_rel == "opposite":
            reference_line = None
            reference_line_source = "both: right opposite -> None"
            # all oscids None
            oscid_by_chain[target_chain_id] = None
            if left_chain_id is not None:  oscid_by_chain[left_chain_id]  = None
            if right_chain_id is not None: oscid_by_chain[right_chain_id] = None
        else:
            if left_rel == "same" and right_rel == "same":
                reference_line = _get_left_boundary(left_chain_id)
                reference_line_source = "both: all same -> left.left"
                # left=1, target=2, right=3
                if left_chain_id is not None:  oscid_by_chain[left_chain_id]  = -1
                oscid_by_chain[target_chain_id] = -2
                if right_chain_id is not None: oscid_by_chain[right_chain_id] = -3
            elif left_rel == "opposite" and right_rel == "same":
                reference_line = _get_left_boundary(target_chain_id)
                reference_line_source = "both: left opposite -> target.left"
                # left=-1, target=1, right=2
                if left_chain_id is not None:  oscid_by_chain[left_chain_id]  = 1
                oscid_by_chain[target_chain_id] = -1
                if right_chain_id is not None: oscid_by_chain[right_chain_id] = -2
            else:
                # fallback: left.left or target.left; keep OSCIDs None if ambiguous
                reference_line = _get_left_boundary(left_chain_id) or _get_left_boundary(target_chain_id)
                reference_line_source = "both: fallback -> left.left or target.left"
                # leave oscids as None
    elif has_left and not has_right:
        if left_rel == "opposite":
            reference_line = _get_left_boundary(target_chain_id)
            reference_line_source = "only left: opposite -> target.left"
            # target=1, left=-1
            if left_chain_id is not None:  oscid_by_chain[left_chain_id]  = 1
            oscid_by_chain[target_chain_id] = -1
        else:
            reference_line = _get_left_boundary(left_chain_id)
            reference_line_source = "only left: same -> left.left"
            # left=1, target=2
            if left_chain_id is not None:  oscid_by_chain[left_chain_id]  = -1
            oscid_by_chain[target_chain_id] = -2
    elif has_right and not has_left:
        if right_rel == "opposite":
            reference_line = None
            reference_line_source = "only right: opposite -> None"
            # both None
            oscid_by_chain[target_chain_id] = None
            if right_chain_id is not None: oscid_by_chain[right_chain_id] = None
        else:
            reference_line = _get_left_boundary(target_chain_id)
            reference_line_source = "only right: same -> target.left"
            # target=1, right=2
            oscid_by_chain[target_chain_id] = -1
            if right_chain_id is not None: oscid_by_chain[right_chain_id] = -2
    else:
        reference_line = None
        reference_line_source = "no neighbors -> None"
        # all remain None

    if reference_line is not None and reference_line.is_empty:
        reference_line = None
        reference_line_source += " (empty -> None)"

    # ---------------- Plot (optional) ----------------
    if show_plot:
        import matplotlib.pyplot as plt

        def _plot_line(ax, line, lw=1.5, ls='-', label=None, color=None):
            if line.is_empty: return
            x, y = line.xy
            ax.plot(x, y, linestyle=ls, linewidth=lw, label=label, color=color)

        def _plot_line_with_arrows(ax, line: LineString, *, color=None, lw=1.5, ls='-', n_arrows=8):
            if line.is_empty or line.length <= 0:
                return
            x, y = line.xy
            ax.plot(x, y, linestyle=ls, linewidth=lw, color=color)
            n = max(1, n_arrows)
            for i in range(1, n+1):
                t1 = i/(n+1)
                t0 = max(0.0, t1 - 0.02)
                p0 = line.interpolate(t0, normalized=True)
                p1 = line.interpolate(t1, normalized=True)
                ax.annotate("", xy=(p1.x, p1.y), xytext=(p0.x, p0.y),
                            arrowprops=dict(arrowstyle="->", color=color, lw=max(1.0, lw-0.2)))

        def _plot_poly(ax, poly: Polygon, face_alpha=0.25, edge_lw=1.0, label=None, color=None):
            if poly.is_empty: return
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=face_alpha, label=label, color=color)
            ax.plot(x, y, linewidth=edge_lw, color=color)
            for ring in poly.interiors:
                xi, yi = ring.xy
                ax.plot(xi, yi, linewidth=edge_lw, color=color)

        def _plot_poly_any(ax, geom, face_alpha=0.25, edge_lw=1.0, label=None, color=None):
            if geom.is_empty: return
            if isinstance(geom, Polygon):
                _plot_poly(ax, geom, face_alpha, edge_lw, label, color)
            elif isinstance(geom, MultiPolygon):
                for i, g in enumerate(geom.geoms):
                    lab = (label if i == 0 else None)
                    _plot_poly(ax, g, face_alpha, edge_lw, lab, color)
            elif isinstance(geom, GeometryCollection):
                for g in geom.geoms:
                    _plot_poly_any(ax, g, face_alpha, edge_lw, label, color)

        cmap = _get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(10, 7))

        # polygons (only if computed)
        if compute_polygons:
            _plot_poly_any(ax, target_poly_clean, face_alpha=0.30, edge_lw=1.5,
                           label=f"target chain {target_chain_id}", color=cmap(0))
            _plot_poly_any(ax, left_final,  face_alpha=0.25, edge_lw=1.2,
                           label="left (final)", color=cmap(1))
            _plot_poly_any(ax, right_final, face_alpha=0.25, edge_lw=1.2,
                           label="right (final)", color=cmap(2))

        # centerlines (dotted) with arrows
        for i, (cid, cl) in enumerate(centerline_by_chain.items()):
            _plot_line_with_arrows(ax, cl, color=cmap(3+i), lw=1.5, ls=':', n_arrows=arrows_per_line)
            # annotate OSCID if available
            if cid in oscid_by_chain and oscid_by_chain[cid] is not None:
                p = cl.interpolate(0.5, normalized=True)
                ax.text(p.x, p.y, f"OSC {oscid_by_chain[cid]}", fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # boundaries (optional)
        if plot_boundaries:
            for i, cid in enumerate(centerline_by_chain.keys()):
                lb = left_boundary_by_chain.get(cid, LineString())
                rb = right_boundary_by_chain.get(cid, LineString())
                col = cmap(6 + i)
                _plot_line_with_arrows(ax, lb, color=col, lw=2.0, ls='-',  n_arrows=arrows_per_line)
                _plot_line_with_arrows(ax, rb, color=col, lw=2.0, ls='--', n_arrows=arrows_per_line)

        # reference line (if any) highlighted
        if reference_line is not None:
            _plot_line_with_arrows(ax, reference_line, color='black', lw=3.0, ls='-', n_arrows=max(3, arrows_per_line//2))
            ax.text(*reference_line.interpolate(0.5, normalized=True).coords[0],
                    f"REF ({reference_line_source})", fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linewidth=0.4)
        ax.margins(0.05)
        ax.legend(loc="best", fontsize=8)
        ttl = f"Centerlines + boundaries (arrows)"
        if compute_polygons:
            ttl = f"Polygons + " + ttl + f". Inflate={int(inflate_pct*100)}%"
        ax.set_title(ttl)
        plt.show()
    
    lane2chain = {}
    for c in chains:
        for lid in c["lane_ids"]:
            lane2chain[lid] = c["id"]

    oscid_by_lane = {lid: oscid_by_chain.get(cid) for lid, cid in lane2chain.items()}


    # --- Minimal, easy-to-loop output ---

    # Collect polygons per chain (None when polygons were skipped)
    polygon_by_chain = {
        target_chain_id: (target_poly_clean if compute_polygons else None),
    }
    if left_chain_id is not None:
        polygon_by_chain[left_chain_id] = (left_final if compute_polygons else None)
    if right_chain_id is not None:
        polygon_by_chain[right_chain_id] = (right_final if compute_polygons else None)

    # Assemble compact per-chain records (target first, then left/right if present)
    chains_min = {}
    for cid in [target_chain_id] + [c for c in (left_chain_id, right_chain_id) if c is not None]:
        chains_min[cid] = {
            "id": cid,
            "polygon": polygon_by_chain.get(cid),                # Polygon | MultiPolygon | None
            "oscid": oscid_by_chain.get(cid),                    # int | None
            "centerline": centerline_by_chain.get(cid, LineString()),  # LineString (may be empty)
        }
    valid = True
    if reference_line is None:
        valid = False
    
    boundaries_by_chain = {
        cid: {"left": left_boundary_by_chain.get(cid, LineString()),
            "right": right_boundary_by_chain.get(cid, LineString())}
        for cid in centerline_by_chain.keys()
    }

    return {
        "target_chain_id": target_chain_id,
        "target_polygon": target_poly_clean if compute_polygons else None,
        "left_polygon": left_final if compute_polygons else None,
        "right_polygon": right_final if compute_polygons else None,
        "centerline_by_chain": centerline_by_chain,
        "boundaries_by_chain": boundaries_by_chain,
        "reference_line": reference_line,
        "reference_line_source": reference_line_source,  # helpful for debugging
        "oscid_by_chain": oscid_by_chain,               # <- NEW: {chain_id: int|None}
        "oscid_by_lane": oscid_by_lane,
        "chains": chains_min, 
        "valid": valid,
    }


def run_for_all_segments(lane_graph, segments_by_key, *,
                         compute_polygons=True,
                         inflate_pct=0.10,
                         bridge_tol=0.75,
                         show_plot=True,           # usually turn off when batching
                         plot_boundaries=False,
                         arrows_per_line=10):
    """
    segments_by_key: dict like {'seg_0': {...}, 'seg_1': {...}, ...}
    Returns: {'seg_0': {'segment': <orig>, 'results': <build_* output>}, ...}
    """
    out = {}
    for seg_key, seg_block in segments_by_key.items():
        res = build_polys_target_first_inflate(
            lane_graph,
            seg_block,
            compute_polygons=compute_polygons,
            inflate_pct=inflate_pct,
            bridge_tol=bridge_tol,
            show_plot=show_plot,
            plot_boundaries=plot_boundaries,
            arrows_per_line=arrows_per_line,
        )
        out[seg_key] = res
        
        """{
            #"segments_raw": seg_block,   # original data you passed in
            #"segments_processed": res,         # output from build_polys_target_first_inflate
        }"""
    return out

def compute_segment_result(
    lane_graph,
    segments_by_key,
    segment_key: str,
    *,
    compute_polygons: bool = True,
    inflate_pct: float = 0.10,
    bridge_tol: float = 0.75,
    plot_boundaries: bool = True,
    arrows_per_line: int = 10,
):
    """
    Compute the same geometric result as build_polys_target_first_inflate,
    but only for a single segment identified by `segment_key`.

    This DOES NOT plot anything.
    """
    seg_block = segments_by_key[segment_key]

    result = build_polys_target_first_inflate(
        lane_graph,
        seg_block,
        compute_polygons=compute_polygons,
        inflate_pct=inflate_pct,
        bridge_tol=bridge_tol,
        show_plot=False,          # <- no plotting here
        plot_boundaries=plot_boundaries,
        arrows_per_line=arrows_per_line,
    )
    return result

def plot_single_segment(
    lane_graph,
    segments_by_key,
    segment_key: str,
    *,
    compute_polygons: bool = True,
    inflate_pct: float = 0.10,
    bridge_tol: float = 0.75,
    plot_boundaries: bool = True,
    arrows_per_line: int = 10,
    show_reference_line: bool = True,
    show_centerlines: bool = True,
    show_polygons: bool = True,
    figsize=(10, 7),
):
    """
    Compute and plot ONLY the requested segment (e.g. 'seg_0').

    Returns (fig, ax).

    All the knobs here are plotting/visualization oriented and can be freely
    tuned without touching build_polys_target_first_inflate.
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection

    # --- 1) compute geoms only for this segment ---
    res = compute_segment_result(
        lane_graph,
        segments_by_key,
        segment_key,
        compute_polygons=compute_polygons,
        inflate_pct=inflate_pct,
        bridge_tol=bridge_tol,
        plot_boundaries=plot_boundaries,
        arrows_per_line=arrows_per_line,
    )

    target_poly      = res["target_polygon"]
    left_poly        = res["left_polygon"]
    right_poly       = res["right_polygon"]
    centerlines      = res["centerline_by_chain"]
    boundaries       = res["boundaries_by_chain"]
    reference_line   = res["reference_line"]
    ref_line_source  = res["reference_line_source"]
    target_chain_id  = res["target_chain_id"]

    # --- 2) small local plotting helpers (safe to customize) ---

    def _plot_line_with_arrows(ax, line: LineString, *, color=None,
                               lw=1.5, ls='-', n_arrows=8):
        if line.is_empty or line.length <= 0:
            return
        x, y = line.xy
        ax.plot(x, y, linestyle=ls, linewidth=lw, color=color)
        n = max(1, n_arrows)
        for i in range(1, n + 1):
            t1 = i / (n + 1)
            t0 = max(0.0, t1 - 0.02)
            p0 = line.interpolate(t0, normalized=True)
            p1 = line.interpolate(t1, normalized=True)
            ax.annotate(
                "",
                xy=(p1.x, p1.y),
                xytext=(p0.x, p0.y),
                arrowprops=dict(arrowstyle="->", color=color, lw=max(1.0, lw - 0.2)),
            )

    def _plot_poly(ax, poly: Polygon, face_alpha=0.25, edge_lw=1.0, label=None, color=None):
        if poly.is_empty:
            return
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=face_alpha, label=label, color=color)
        ax.plot(x, y, linewidth=edge_lw, color=color)
        for ring in poly.interiors:
            xi, yi = ring.xy
            ax.plot(xi, yi, linewidth=edge_lw, color=color)

    def _plot_poly_any(ax, geom, face_alpha=0.25, edge_lw=1.0, label=None, color=None):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, Polygon):
            _plot_poly(ax, geom, face_alpha, edge_lw, label, color)
        elif isinstance(geom, MultiPolygon):
            for i, g in enumerate(geom.geoms):
                lab = label if i == 0 else None
                _plot_poly(ax, g, face_alpha, edge_lw, lab, color)
        elif isinstance(geom, GeometryCollection):
            for g in geom.geoms:
                _plot_poly_any(ax, g, face_alpha, edge_lw, label, color)

    # --- 3) actual plotting ---

    cmap = _get_cmap("tab10")  # uses your existing helper
    fig, ax = plt.subplots(figsize=figsize)

    # polygons
    if show_polygons and compute_polygons:
        _plot_poly_any(
            ax,
            target_poly,
            face_alpha=0.30,
            edge_lw=1.5,
            label=f"target chain {target_chain_id}",
            color=cmap(0),
        )
        _plot_poly_any(
            ax,
            left_poly,
            face_alpha=0.25,
            edge_lw=1.2,
            label="left (final)",
            color=cmap(1),
        )
        _plot_poly_any(
            ax,
            right_poly,
            face_alpha=0.25,
            edge_lw=1.2,
            label="right (final)",
            color=cmap(2),
        )

    # centerlines
    if show_centerlines:
        for i, (cid, cl) in enumerate(centerlines.items()):
            _plot_line_with_arrows(
                ax,
                cl,
                color=cmap(3 + i),
                lw=1.5,
                ls=":",
                n_arrows=arrows_per_line,
            )

    # boundaries
    if plot_boundaries:
        for i, cid in enumerate(centerlines.keys()):
            b = boundaries.get(cid, {})
            lb = b.get("left")
            rb = b.get("right")
            col = cmap(6 + i)

            if isinstance(lb, LineString):
                _plot_line_with_arrows(
                    ax,
                    lb,
                    color=col,
                    lw=2.0,
                    ls="-",
                    n_arrows=arrows_per_line,
                )
            if isinstance(rb, LineString):
                _plot_line_with_arrows(
                    ax,
                    rb,
                    color=col,
                    lw=2.0,
                    ls="--",
                    n_arrows=arrows_per_line,
                )

    # reference line
    if show_reference_line and reference_line is not None and not reference_line.is_empty:
        _plot_line_with_arrows(
            ax,
            reference_line,
            color="black",
            lw=3.0,
            ls="-",
            n_arrows=max(3, arrows_per_line // 2),
        )
        p = reference_line.interpolate(0.5, normalized=True)
        """ax.text(
            p.x,
            p.y,
            f"REF ({ref_line_source})",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )"""

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.4)
    ax.margins(0.05)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Segment {segment_key} (inflate={int(inflate_pct * 100)}%)")

    # NOTE: we do NOT call plt.show() here.
    return fig, ax

def plot_single_segment_ma(
    lane_graph,
    segments_by_key,
    segment_key: str,
    *,
    compute_polygons: bool = True,
    inflate_pct: float = 0.10,
    bridge_tol: float = 0.75,
    plot_boundaries: bool = True,
    arrows_per_line: int = 10,
    show_reference_line: bool = True,
    show_centerlines: bool = True,
    show_polygons: bool = True,
    figsize=(10, 7),
):
    """
    Compute and plot ONLY the requested segment (e.g. 'seg_0').

    Returns (fig, ax).

    All the knobs here are plotting/visualization oriented and can be freely
    tuned without touching build_polys_target_first_inflate.
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection

    # --- 1) compute geoms only for this segment ---
    res = compute_segment_result(
        lane_graph,
        segments_by_key,
        segment_key,
        compute_polygons=compute_polygons,
        inflate_pct=inflate_pct,
        bridge_tol=bridge_tol,
        plot_boundaries=plot_boundaries,
        arrows_per_line=arrows_per_line,
    )

    target_poly      = res["target_polygon"]
    left_poly        = res["left_polygon"]
    right_poly       = res["right_polygon"]
    centerlines      = res["centerline_by_chain"]
    boundaries       = res["boundaries_by_chain"]
    reference_line   = res["reference_line"]
    ref_line_source  = res["reference_line_source"]
    target_chain_id  = res["target_chain_id"]

    # --- 2) small local plotting helpers (safe to customize) ---

    def _plot_line_with_arrows(ax, line: LineString, *, color=None,
                               lw=1.5, ls='-', n_arrows=8, ref_line=False):
        if line.is_empty or line.length <= 0:
            return
        x, y = line.xy
        if ref_line:
            ax.plot(x, y, linestyle=ls, linewidth=lw, color=color, label="Reference Line")
        else:
            ax.plot(x, y, linestyle=ls, linewidth=lw, color=color)
        n = max(0, n_arrows)
        for i in range(1, n + 1):
            t1 = i / (n + 1)
            t0 = max(0.0, t1 - 0.02)
            p0 = line.interpolate(t0, normalized=True)
            p1 = line.interpolate(t1, normalized=True)
            ax.annotate(
                "",
                xy=(p1.x, p1.y),
                xytext=(p0.x, p0.y),
                arrowprops=dict(arrowstyle="->", color=color, lw=max(1.0, lw - 0.2)),
            )

    def _plot_poly(ax, poly: Polygon, face_alpha=0.25, edge_lw=1.0, label=None, color=None):
        if poly.is_empty:
            return
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=face_alpha, label=label, color=color)
        ax.plot(x, y, linewidth=edge_lw, color=color)
        for ring in poly.interiors:
            xi, yi = ring.xy
            ax.plot(xi, yi, linewidth=edge_lw, color=color)

    def _plot_poly_any(ax, geom, face_alpha=0.25, edge_lw=1.0, label=None, color=None):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, Polygon):
            _plot_poly(ax, geom, face_alpha, edge_lw, label, color)
        elif isinstance(geom, MultiPolygon):
            for i, g in enumerate(geom.geoms):
                lab = label if i == 0 else None
                _plot_poly(ax, g, face_alpha, edge_lw, lab, color)
        elif isinstance(geom, GeometryCollection):
            for g in geom.geoms:
                _plot_poly_any(ax, g, face_alpha, edge_lw, label, color)

    # --- 3) actual plotting ---

    cmap = _get_cmap("tab10")  # uses your existing helper
    fig, ax = plt.subplots(figsize=figsize)
    road_color = (191/255, 191/255, 191/255)
    line_color = (255/255, 255/255, 255/255)
    # polygons
    if show_polygons and compute_polygons:
        _plot_poly_any(
            ax,
            target_poly,
            face_alpha=0.7,
            edge_lw=1.5,
            #label=f"target chain {target_chain_id}",
            color=road_color,
        )
        _plot_poly_any(
            ax,
            left_poly,
            face_alpha=0.7,
            edge_lw=1.2,
            #label="left (final)",
            color=road_color,
        )
        _plot_poly_any(
            ax,
            right_poly,
            face_alpha=0.7,
            edge_lw=1.2,
            #label="right (final)",
            color=road_color,
        )

    # centerlines
    show_centerlines = False
    if show_centerlines:
        for i, (cid, cl) in enumerate(centerlines.items()):
            _plot_line_with_arrows(
                ax,
                cl,
                color=cmap(3 + i),
                lw=1.5,
                ls=":",
                n_arrows=arrows_per_line,
                label="Reference Line",
            )

    # boundaries
    if plot_boundaries:
        b = boundaries.get(target_chain_id, {})
        lb = b.get("left")
        rb = b.get("right")

        # determine which neighbor polygon exists
        neighbor_poly = None
        if left_poly is not None and not left_poly.is_empty:
            neighbor_poly = left_poly
        elif right_poly is not None and not right_poly.is_empty:
            neighbor_poly = right_poly

        # pick the target-side boundary that actually touches the neighbor polygon
        shared = None
        if neighbor_poly is not None:
            eps = 0.05  # tolerance in meters-ish; adjust if needed
            if isinstance(lb, LineString) and lb.buffer(eps).intersects(neighbor_poly):
                shared = lb
            elif isinstance(rb, LineString) and rb.buffer(eps).intersects(neighbor_poly):
                shared = rb
            else:
                # fallback: choose the closer one
                cand = []
                if isinstance(lb, LineString): cand.append(lb)
                if isinstance(rb, LineString): cand.append(rb)
                if cand:
                    shared = min(cand, key=lambda ln: ln.distance(neighbor_poly))

        if isinstance(shared, LineString):
            _plot_line_with_arrows(
                ax,
                shared,
                color=(1.0, 1.0, 1.0),  # white
                lw=2.0,
                ls="--",                # dashed
                n_arrows=0,
            )

    # reference line
    if show_reference_line and reference_line is not None and not reference_line.is_empty:
        _plot_line_with_arrows(
            ax,
            reference_line,
            color="black",
            lw=3.0,
            ls="-",
            n_arrows=max(3, arrows_per_line // 2),
            ref_line=True
        )
        p = reference_line.interpolate(0.5, normalized=True)
        """ax.text(
            p.x,
            p.y,
            f"REF ({ref_line_source})",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )"""

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.4)
    ax.margins(0.05)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Segment {segment_key} (inflate={int(inflate_pct * 100)}%)")

    # NOTE: we do NOT call plt.show() here.
    return fig, ax


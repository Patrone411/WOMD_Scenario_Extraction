# export_to_opendrive.py
import math
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from typing import Tuple

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# ---------------------------------
# constants / numeric helpers
# ---------------------------------
EPS = 1e-3  # generous for meters

def _clamp(v, lo, hi):  # hi treated as exclusive for laneSection end
    if v < lo: return lo
    if v > hi - EPS: return hi - EPS
    return v

def _dedupe_cuts(cuts, total_len, eps=EPS):
    cuts = sorted(cuts)
    out = []
    last = None
    for s in cuts:
        # snap near 0 or near total_len
        if abs(s) < eps:
            s = 0.0
        elif abs(s - total_len) < eps:
            s = total_len
        if last is None or abs(s - last) > eps:
            out.append(s)
            last = s
    # never keep a trailing duplicate
    if len(out) >= 2 and abs(out[-1] - out[-2]) < eps:
        out.pop()
    return out

# ---------------------------------
# geometry helpers
# ---------------------------------
def _ref_project_s(ref: LineString, pt: Point) -> float:
    return float(ref.project(pt))

def _segment_s_range_on_ref(ref: LineString, edge: LineString) -> Tuple[float, float]:
    if edge.is_empty or len(edge.coords) < 2:
        return (math.inf, -math.inf)
    s_vals = [_ref_project_s(ref, Point(x, y)) for (x, y) in edge.coords]
    return (min(s_vals), max(s_vals))

def _ref_tangent(ref: LineString, s: float) -> Tuple[float, float]:
    # approximate tangent via small forward step
    s0 = max(0.0, min(s, ref.length))
    ds = min(0.25, max(0.05, 0.001 * ref.length))
    p0 = ref.interpolate(s0)
    p1 = ref.interpolate(min(ref.length, s0 + ds))
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    L = math.hypot(dx, dy) or 1e-9
    return dx / L, dy / L

def _signed_offset_to_edge(ref: LineString, edge: LineString, s: float) -> float:
    """Signed lateral offset from ref(s) to the nearest point on 'edge'.
       Positive = to the left of the ref direction."""
    p_ref = ref.interpolate(s)
    p_e = nearest_points(p_ref, edge)[1]
    vx, vy = (p_e.x - p_ref.x), (p_e.y - p_ref.y)
    tx, ty = _ref_tangent(ref, s)
    # left normal = rotate tangent by +90°
    nx, ny = -ty, tx
    return (vx * nx + vy * ny)

def _lane_presence_on_section(s0, s1, rng, eps=1e-6):
    a, b = rng
    return (a < s1 - eps) and (b > s0 + eps)

SIMPLIFY_TOL = 0.05    # 5 cm – safe for removing micro-zigzags
MIN_SEG_LEN  = 0.01    # 1 cm – ignore micro planView pieces

def _clean_line(ls: LineString, tol: float = SIMPLIFY_TOL) -> LineString:
    if ls.is_empty or len(ls.coords) < 3:
        return ls
    ls2 = ls.simplify(tol, preserve_topology=False)
    # make sure we still have at least 2 points
    if ls2.is_empty or len(ls2.coords) < 2:
        return ls
    return ls2

# ---------------------------------
# xml helpers
# ---------------------------------
def _pretty_xml(elem: Element) -> str:
    rough = tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

def _linestring_to_planview(ls: LineString):
    """Return list of geometries [{s,x,y,hdg,length}, ...] and total length."""
    coords = list(ls.coords)
    geoms = []
    s_acc = 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        dx, dy = (x1 - x0), (y1 - y0)
        L = math.hypot(dx, dy)
        if L <= MIN_SEG_LEN:
            continue
        hdg = math.atan2(dy, dx)
        geoms.append({"s": s_acc, "x": x0, "y": y0, "hdg": hdg, "length": L})
        s_acc += L
    return geoms, s_acc

# ---------------------------------
# main
# ---------------------------------
def export_segment_to_xodr(seg_key: str,
                           results: dict,
                           *,
                           road_id: int = 1,
                           road_name: str = None,
                           author: str = "auto",
                           rev_major: int = 1,
                           rev_minor: int = 5) -> str:
    """
    Build a minimal OpenDRIVE from a single segment using:
      • reference_line (a boundary) for planView
      • per-lane edges to derive laneSections & widths
      • stable lane IDs from oscid_by_chain (sign = side)
    """
    road_name = road_name or f"segment_{seg_key}"

    # --- 1) Plan view from the reference boundary ---
    ref: LineString = results.get("reference_line")
    if ref is None or ref.is_empty or len(ref.coords) < 2:
        # fallback: target centerline if reference_line missing
        target_cid = results["target_chain_id"]
        ref = results["centerline_by_chain"].get(target_cid)
        if ref is None or ref.is_empty or len(ref.coords) < 2:
            raise ValueError(f"{seg_key}: no usable reference line")
    ref = _clean_line(ref)

    geoms, total_len = _linestring_to_planview(ref)
    if not geoms or total_len <= 1e-6:
        raise ValueError(f"{seg_key}: degenerate planView")

    # --- 2) Gather lanes to export from edges + oscid ---
    oscid_by_chain = results.get("oscid_by_chain", {}) or {}
    boundaries_by_chain = results.get("boundaries_by_chain", {}) or {}

    lanes_export = []  # list of dicts per chain/lane with stable id = oscid
    for cid, _rec in results["chains"].items():
        oscid = oscid_by_chain.get(cid)
        if oscid is None:
            continue
        bnd = boundaries_by_chain.get(cid) or {}
        left_edge  = _clean_line(bnd.get("left",  LineString()))
        right_edge = _clean_line(bnd.get("right", LineString()))
        if left_edge.is_empty or right_edge.is_empty:
            # skip lanes with missing edges (fallback to lane_graph if you really want)
            continue

        sL = _segment_s_range_on_ref(ref, left_edge)
        sR = _segment_s_range_on_ref(ref, right_edge)
        s_min = _clamp(min(sL[0], sR[0]), 0.0, total_len)
        s_max = _clamp(max(sL[1], sR[1]), 0.0, total_len)
        if s_max <= s_min + EPS:
            continue

        lanes_export.append({
            "cid": cid,
            "oscid": int(oscid),  # stable lane id (sign = side)
            "left_edge": left_edge,
            "right_edge": right_edge,
            "s_range": (s_min, s_max),
        })

    if not lanes_export:
        raise ValueError(f"{seg_key}: no lanes with edges to export")

    # --- 3) Build laneSection cuts from all edge coverage ---
    cuts = {0.0, total_len}
    for L in lanes_export:
        a, b = L["s_range"]
        cuts.add(_clamp(a, 0.0, total_len))
        cuts.add(_clamp(b, 0.0, total_len))
    cuts = _dedupe_cuts(cuts, total_len)

    # --- 4) OpenDRIVE DOM scaffold ---
    odr = Element("OpenDRIVE")
    SubElement(odr, "header", {
        "revMajor": str(rev_major),
        "revMinor": str(rev_minor),
        "name": road_name,
        "version": "1.00",
        "date": "2025-01-01T00:00:00",
        "north": "0", "south": "0", "east": "0", "west": "0",
        "vendor": author,
    })
    road = SubElement(odr, "road", {
        "name": road_name,
        "length": f"{total_len:.3f}",
        "id": str(road_id),
        "junction": "-1",
    })

    plan = SubElement(road, "planView")
    for g in geoms:
        geo = SubElement(plan, "geometry", {
            "s": f"{g['s']:.6f}",
            "x": f"{g['x']:.6f}",
            "y": f"{g['y']:.6f}",
            "hdg": f"{g['hdg']:.9f}",
            "length": f"{g['length']:.6f}",
        })
        SubElement(geo, "line")

    SubElement(road, "elevationProfile")

    lanes = SubElement(road, "lanes")
    SubElement(lanes, "laneOffset", {"s": "0", "a": "0", "b": "0", "c": "0", "d": "0"})

    def _lane_width_at(lane_rec, s_mid):
        le = lane_rec["left_edge"]; re = lane_rec["right_edge"]
        dL = _signed_offset_to_edge(ref, le, s_mid)
        dR = _signed_offset_to_edge(ref, re, s_mid)
        return abs(dL - dR), min(dL, dR), max(dL, dR)

    def _ensure_link_child(lane_elem):
        link = lane_elem.find("link")
        if link is None:
            link = SubElement(lane_elem, "link")
        return link

    # for lane linking across sections
    prev_lane_elems = {}  # lane_id -> xml element (from previous section)

    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        # prevent zero-length or "start at end" sections
        if s0 >= total_len - EPS:
            continue
        if s1 - s0 <= EPS:
            continue

        lsec = SubElement(lanes, "laneSection", {"s": f"{s0:.6f}"})

        # center lane
        center = SubElement(lsec, "center")
        c0 = SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})
        SubElement(c0, "roadMark", {
            "sOffset": "0", "type": "solid", "weight": "standard", "color": "standard", "width": "0.12"
        })

        # which lanes exist in this section?
        active = [L for L in lanes_export if _lane_presence_on_section(s0, s1, L["s_range"])]
        if not active:
            prev_lane_elems = {}
            continue

        s_mid = 0.5 * (s0 + s1)
        left_group, right_group = [], []
        for L in active:
            w, inner, outer = _lane_width_at(L, s_mid)
            lid = L["oscid"]
            rec = {"id": lid, "width": w, "inner": inner, "outer": outer}
            if lid > 0:
                left_group.append(rec)
            else:
                right_group.append(rec)

        # ordering is cosmetic; IDs are stable in the XML
        left_group.sort(key=lambda d: abs(d["id"]))
        right_group.sort(key=lambda d: abs(d["id"]))

        current_lane_elems = {}

        if left_group:
            left_xml = SubElement(lsec, "left")
            for rec in left_group:
                lid = abs(int(rec["id"]))          # left ids are positive
                lane_xml = SubElement(left_xml, "lane", {"id": str(lid), "type": "driving", "level": "false"})
                SubElement(lane_xml, "width", {"sOffset": "0", "a": f"{rec['width']:.3f}", "b": "0", "c": "0", "d": "0"})
                SubElement(lane_xml, "roadMark", {
                    "sOffset": "0", "type": "broken", "weight": "standard", "color": "standard", "width": "0.12"
                })
                current_lane_elems[lid] = lane_xml

        if right_group:
            right_xml = SubElement(lsec, "right")
            for rec in right_group:
                lid = -abs(int(rec["id"]))         # right ids are negative
                lane_xml = SubElement(right_xml, "lane", {"id": str(lid), "type": "driving", "level": "false"})
                SubElement(lane_xml, "width", {"sOffset": "0", "a": f"{rec['width']:.3f}", "b": "0", "c": "0", "d": "0"})
                SubElement(lane_xml, "roadMark", {
                    "sOffset": "0", "type": "broken", "weight": "standard", "color": "standard", "width": "0.12"
                })
                current_lane_elems[lid] = lane_xml

        # link lanes with same ID to previous section (predecessor/successor)
        for lid, curr in current_lane_elems.items():
            if lid in prev_lane_elems:
                prev = prev_lane_elems[lid]
                _ensure_link_child(curr)
                SubElement(curr.find("link"), "predecessor", {"id": str(lid)})
                _ensure_link_child(prev)
                SubElement(prev.find("link"), "successor", {"id": str(lid)})
        prev_lane_elems = current_lane_elems

    return _pretty_xml(odr)

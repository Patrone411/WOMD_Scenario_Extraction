import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import TextBox, RadioButtons
import matplotlib.cm as cm
from shapely.strtree import STRtree
from shapely.ops import linemerge
import math
from collections import Counter



"""since there is no lane connectivity information given in the waymo data,
this module constructs a graph from lane centerlines using spatial and directional information,
then infers connections between them (successors and neighbors)"""

def _derive_left_right_ids_2lane(seg):
    """
    Returns (left_ids, right_ids) for a 2-lane segment.
    Prefers endpoints (ground truth), falls back to adjacency.
    """
    chain_a_ids = seg["chains"][0]["lane_ids"]  # target (by construction)
    chain_b_ids = seg["chains"][1]["lane_ids"]  # neighbor

    # 1) Source of truth: endpoints
    ep = seg.get("endpoints", {}) or {}
    if "left" in ep and "right" not in ep:
        # neighbor is on the left of target
        return chain_b_ids, chain_a_ids
    if "right" in ep and "left" not in ep:
        # neighbor is on the right of target
        return chain_a_ids, chain_b_ids

    # 2) Fallback: adjacency
    adj = seg.get("adjacency", {}) or {}
    a_adj = adj.get("chain_a", {}) or {}
    if a_adj.get("left_of"):
        # target is left of neighbor
        return chain_a_ids, chain_b_ids
    if a_adj.get("right_of"):
        # target is right of neighbor
        return chain_b_ids, chain_a_ids

    # 3) Last resort: keep order (won't be strictly guaranteed without side info)
    return chain_a_ids, chain_b_ids

def _attach_endpoints_and_target(seg, chains, nlanes, opendrive_style=True):
    """
    Return (endpoints_with_ids, endpoints_by_id, target_chain_id)
    - endpoints_with_ids: same roles ('target','left','right') but each has 'chain_id'
    - endpoints_by_id: {chain_id: {role: 'target'|'left'|'right', ...endpoint fields...}}
    - target_chain_id: int or None
    """
    endpoints = seg.get("endpoints") or {}

    # Map every lane_id to its final chain id (OpenDRIVE ids already assigned in `chains`)
    lane_to_chain = {}
    for ch in chains:
        cid = ch["id"]
        for lid in ch["lane_ids"]:
            lane_to_chain[lid] = cid

    ep_with_ids = {}
    ep_by_id = {}
    for role, info in endpoints.items():
        s_lid = info["start"][0]
        e_lid = info["end"][0]
        cid = lane_to_chain.get(s_lid) or lane_to_chain.get(e_lid)
        info2 = dict(info)
        info2["chain_id"] = cid
        ep_with_ids[role] = info2
        if cid is not None:
            ep_by_id[cid] = {"role": role, **info}

    # Resolve target_chain_id robustly
    target_info = endpoints.get("target")
    target_cid = None
    if target_info:
        s_lid = target_info["start"][0]
        e_lid = target_info["end"][0]
        target_cid = lane_to_chain.get(s_lid) or lane_to_chain.get(e_lid)

    # Sensible fallback for 3-lane OpenDRIVE (target is middle)
    if target_cid is None and opendrive_style and nlanes == 3:
        target_cid = 2

    return ep_with_ids, ep_by_id, target_cid

#helper functions used in lane graph class
def lateral_buffer(line: LineString, width: float) -> Polygon:
    '''buffers a line string element in lateral direction, creating
    a rectangle. used to create lane segment polygons by passing lane centerline linestrings
    and a width corresponding to the lane type. using shapely linestring.buffer buffers in 
    longitudinal direction as well as lateral, which can be circumvented using this method instead. '''
    left = line.parallel_offset(width, 'left', join_style=2)
    right = line.parallel_offset(width, 'right', join_style=2)

    try:
        coords = list(left.coords) + list(reversed(right.coords))
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)  # Clean it up
        return poly
    except Exception as e:
        print(f"Failed to buffer line: {e}")
        return None

def compute_heading(line: LineString):
    '''get approx heading of linestring using first and last vertex'''
    x1, y1 = line.coords[0][:2]
    x2, y2 = line.coords[-1][:2]
    dx, dy = x2 - x1, y2 - y1
    return math.atan2(dy, dx)  # in radians

def side_of(candidate_line, ref_line, ref_heading_rad):
    '''computes relative side of a lane given a reference lane'''
    # Reference segment: start and direction
    x0, y0 = ref_line.coords[0][:2]
    dx = math.cos(ref_heading_rad)
    dy = math.sin(ref_heading_rad)

    # Vector from start of ref to centroid of candidate
    cx, cy = candidate_line.centroid.xy
    vec_x = cx[0] - x0
    vec_y = cy[0] - y0

    # Compute cross product (ref_dir × vec_to_candidate)
    cross = dx * vec_y - dy * vec_x
    if cross > 0:
        return "left"
    elif cross < 0:
        return "right"
    else:
        return "colinear"

def segment_lane(points, segment_length=5.0):
    """
    Given a list of points of a lane (sampled at 0.5m in waymo), return resampled points list.
    """
    segments = []
    acc = [points[0]]
    total_len = 0.0

    for i in range(1, len(points)):
        acc.append(points[i])
        total_len += np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))

        if total_len >= segment_length:
            segments.append(LineString(acc))
            acc = [points[i]]
            total_len = 0.0

    # Add remaining if long enough
    if len(acc) > 1:
        segments.append(LineString(acc))

    return segments



class LaneGraph:
    '''main lanegraph class that takes in lane information provided in waymo tfrecords
    and builds structured information such as lane successor/predecessor lists and lane neighbor information'''
    def __init__(self, do_setup: bool, xyz, direction, type_, id_, valid, lane_types=None,
                 max_dist=0.001):
        assert isinstance(do_setup,bool), "do_setup must be a boolean"
        self.xyz = xyz
        self.dir = direction
        self.type = type_.flatten()
        self.id = id_.flatten()
        self.valid = valid.flatten()
        self.lane_types = lane_types
        self.max_dist = max_dist
        self.lane_graph = {}
        self.connectivity = defaultdict(list)
        self.neighbors = defaultdict(list)
        self.lane_polygon_neighbors = {}
        self.segment_neighbors = defaultdict(list)
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.sequences = []
        self.lane_to_sequence = {}
        self.lane_polygons = {}
        self.lane_segments = {}  # <-- will exist even in empty scenes
        self.neighbor_info = defaultdict(lambda: defaultdict(list))
        self.lane_line_strings = {}

        self.preferred_types = (1, 2, 3)
        has_centerline = np.any((self.valid == 1) & np.isin(self.type, self.preferred_types))
        self.is_empty = not has_centerline

        # Initialize a few geometry helpers
        self.start_points = np.empty((0, 3), dtype=float)
        self.start_ids = []
        self.kdtree = None

        self.lane_widhts = {  # (typo kept as-is since you reference it elsewhere)
            'freeway': 3.5,
            'surface_street': 3.5,
            'bike_lane': 1.5,
            'brokenSingleWhite': 0.2,
            'brokenSingleYellow': 0.2,
            'brokenDoubleYellow': 0.2
        }

        # Normalize lane_types
        if lane_types is None:
            self.lane_types = None
        elif isinstance(lane_types, int):
            self.lane_types = [lane_types]
        else:
            self.lane_types = list(lane_types)

        if do_setup:
            self.setup()
    
    def setup(self):
        # --- Return early only if it's empty ---
        if self.is_empty:
            return 
        self._build_lane_graph()
        self._build_connectivity()
        self._compute_lane_neighbors_per_lane_segment()
        self.compute_lane_neighbors()
        self._compute_polygon_neighbors()
        self._compute_neighbors_from_neighbor_info()
        self._divide_root_and_branch_sequences()
        self.root_seqs = [s['lane_ids'] for s in self.sequences if not s["is_branch_root"]]
        # Build a mapping from lane_id -> chain index
        self.lane_to_chain = {}
        for chain_idx, chain in enumerate(self.root_seqs):
            for lane_id in chain:
                self.lane_to_chain[lane_id] = chain_idx

    def _build_lane_graph(self):
        '''Constructs lanegraph from waymo data.
        builds per lane id dicts containing the respective
        lane_line_strings, lane_polygons and resampled lane_segments.
        lane_graph dict stores per id start points, end points,
        an approx directional vector and the buffer size (half lane width)
        corresponding to its lane type.
        '''
        # Filter valid lane centerline points
        if self.lane_types is None:
            lane_mask = (self.valid == 1)
        else:
            lane_mask = (self.valid == 1) & np.isin(self.type, self.lane_types)
        xyz_lanes = self.xyz[lane_mask]
        dir_lanes = self.dir[lane_mask]
        id_lanes = self.id[lane_mask]

        # Group by lane ID
        lane_points = defaultdict(list)
        lane_dirs = defaultdict(list)

        for i in range(len(id_lanes)):
            lane_points[id_lanes[i]].append(xyz_lanes[i])
            lane_dirs[id_lanes[i]].append(dir_lanes[i])
        self.start_points = []
        self.start_ids = []

        for lane_id, points in lane_points.items():
            if len(points) < 2:
                continue
            points_np = np.array(points)
            
            start = points_np[0]
            end = points_np[-1]
            
            dir_vec = end - start
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                final_dir = np.array([0.0, 0.0, 0.0])
            else:
                final_dir = dir_vec / norm

            type_mask = (self.id == lane_id)
            lane_type_ids = self.type[type_mask]
            lane_type_id = int(lane_type_ids[0]) if len(lane_type_ids) > 0 else -1

            buffer_size = self._get_buffer_distance(lane_type_id)

            self.lane_graph[lane_id] = {
                "start": start,
                "end": end,
                "dir": final_dir,
                "buffer_size": buffer_size,
                
            }

            self.start_points.append(start)
            self.start_ids.append(lane_id)

        # Before: self.kdtree = cKDTree(self.start_points)
        if len(self.start_points) == 0:
            self.kdtree = None
            return
        self.kdtree = cKDTree(np.asarray(self.start_points, dtype=float))

        # Compute shapely polygons for each lane using LineString buffer
        for lane_id, points in lane_points.items():
            if len(points) < 2:
                print(f"Skipping lane {lane_id} due to insufficient points: {len(points)}")
                continue
            coords_2d = [(pt[0], pt[1]) for pt in points]  # drop z
            line = LineString(coords_2d)

            # Get lane type for this lane ID (assuming all points share the same type)
            type_mask = (self.id == lane_id)
            lane_type_ids = self.type[type_mask]
            lane_type_id = int(lane_type_ids[0]) if len(lane_type_ids) > 0 else -1

            buffer_size = self._get_buffer_distance(lane_type_id)
            #polygon = line.buffer(buffer_size)
            polygon = lateral_buffer(line, buffer_size)
            self.lane_line_strings[lane_id] = line
            self.lane_polygons[lane_id] = polygon
            #self.lane_type[lane_id] = lane_type_id
            self.lane_segments[lane_id] = segment_lane(points=points, segment_length=5)

    def _build_connectivity(self):
        """Determines which lanes connect to which,
        based on spatial proximity. builds successor and predecesor lsits"""
        if self.kdtree is None:
            return
        
        for lane_id, props in self.lane_graph.items():
            end = props["end"]

            neighbors = self.kdtree.query_ball_point(end, self.max_dist)

            for idx in neighbors:
                candidate_id = self.start_ids[idx]
                if candidate_id == lane_id:
                    continue
                
                self.connectivity[lane_id].append(candidate_id)
                self.successors[lane_id].append(candidate_id)
                self.predecessors[candidate_id].append(lane_id)

    def _target_pair_from_window(self, target_chain, g):
        """
        Return (lane_id, seg_idx) in target_chain at global index g,
        or None if g is out of range.
        """
        g2lane, _, _ = self._build_target_global_index(target_chain)
        if g < 0 or g >= len(g2lane):
            return None
        return g2lane[g]


    def _neighbor_endpoints_from_window(self, target_chain, gstart, gend, side, neighbor_ids=None):
        """
        Scan the neighbor_info along the target window [gstart, gend) and
        return:
        (first_pair, last_pair, direction_summary)

        first_pair / last_pair are (neighbor_lane_id, neighbor_seg_idx) or None
        direction_summary ∈ {'same', 'opposite', 'mixed', 'unknown'}.
        If neighbor_ids is provided, only consider those lane_ids.
        """
        g2lane, _, _ = self._build_target_global_index(target_chain)
        want = set(neighbor_ids) if neighbor_ids is not None else None

        first = None
        last = None
        dirs = []

        for g in range(gstart, gend):
            if g < 0 or g >= len(g2lane):
                continue
            t_lane, t_seg = g2lane[g]
            for (n_side, n_id, n_seg, n_dir) in self.neighbor_info.get(t_lane, {}).get(t_seg, []):
                if n_side != side:
                    continue
                if (want is not None) and (n_id not in want):
                    continue
                if first is None:
                    first = (n_id, n_seg)
                last = (n_id, n_seg)
                if n_dir in ("same", "opposite"):
                    dirs.append(n_dir)
                # take the first matching neighbor at this step; move on
                break

        if not dirs:
            dsum = "unknown"
        else:
            u = set(dirs)
            dsum = next(iter(u)) if len(u) == 1 else "mixed"

        return first, last, dsum
    def compute_lane_neighbors(    
        self, 
        lateral_dist_threshold=5.0, 
        direction_threshold=0.7, 
        num_samples=20, 
        min_longitudinal_overlap=0.5,  # percent of lane length
        aligned =True
    ):
        '''use only in documentation, using start end end points and check for proximity and alignment,
        good for a rough first estimate of lane neighbor relationship'''
        collected_neighbor_info = defaultdict(list)

        sampled_points = {}
        directions = {}

        for lane_id, props in self.lane_graph.items():
            start = props["start"]
            end = props["end"]
            dir_vec = props["dir"]

            samples = np.linspace(start, end, num_samples)
            sampled_points[lane_id] = samples
            directions[lane_id] = dir_vec

        for lane_id, ref_samples in sampled_points.items():
            ref_dir = directions[lane_id]
            ref_dir_norm = ref_dir / (np.linalg.norm(ref_dir) + 1e-8)

            successors = set(self.successors.get(lane_id, []))
            predecessors = set(self.predecessors.get(lane_id, []))
            non_neighbors = successors | predecessors | {lane_id}
            seen = set()

            for other_id, other_samples in sampled_points.items():
                if other_id in non_neighbors or other_id in seen:
                    continue

                # Directional alignment
                other_dir = directions[other_id]
                dot = np.dot(ref_dir_norm, other_dir)
                if abs(dot) < direction_threshold:
                    continue

                # Lateral distance between points
                dists = np.linalg.norm(ref_samples[:, None, :2] - other_samples[None, :, :2], axis=2)
                min_dists = np.min(dists, axis=1)

                close_mask = min_dists < lateral_dist_threshold
                overlap_ratio = np.sum(close_mask) / num_samples

                if overlap_ratio < min_longitudinal_overlap:
                    continue

                relation = (
                    "parallel" if dot > 0.7 else
                    "opposite" if dot < -0.7 else
                    "unaligned"
                )
                if aligned == True and relation == "unaligned":
                    continue
                collected_neighbor_info[lane_id].append({
                    "neighbor_id": other_id,
                    "direction_relation": relation,
                    "alignment_score": float(dot),
                    "overlap_ratio": float(overlap_ratio)
                })
                seen.add(other_id)
        self.neighbors = {
            lane_id: {entry["neighbor_id"] for entry in neighbors}
            for lane_id, neighbors in collected_neighbor_info.items()
        }
           
    def _compute_polygon_neighbors(self, min_overlap_ratio=0.00):
        """use only in documentation. Compute neighbors using only geometrical overlap of lane polygons,
        excluding direct successors/predecessors."""
        self.lane_polygon_neighbors = defaultdict(list)

        # Get valid polygons
        lane_ids = []
        polygons = []

        #clean up öolygon list
        for lane_id, poly in self.lane_polygons.items():
            if poly is not None and poly.is_valid:
                lane_ids.append(lane_id)
                polygons.append(poly)

        # Exhaustive pairwise check
        for i in range(len(polygons)):
            id_a = lane_ids[i]
            poly_a = polygons[i]

            for j in range(i + 1, len(polygons)):
                id_b = lane_ids[j]
                poly_b = polygons[j]

                # Skip if they are direct successors or predecessors
                if id_b in self.get_successors(id_a) or id_b in self.get_predecessors(id_a):
                    continue
                if id_a in self.get_successors(id_b) or id_a in self.get_predecessors(id_b):
                    continue

                if poly_a.intersects(poly_b):
                    intersection = poly_a.intersection(poly_b)
                    if not intersection.is_empty and intersection.area > 0:
                        # Compute overlap ratio w.r.t. poly_a (or use union for IoU)
                        overlap_ratio = intersection.area / poly_a.area
                        if overlap_ratio >= min_overlap_ratio:
                            self.lane_polygon_neighbors[id_a].append(id_b)
                            self.lane_polygon_neighbors[id_b].append(id_a)
    
    def _compute_lane_neighbors_per_lane_segment(self, buffer_width_percentage = 1.2, cut_ends=True):
        """
        final version of neighbor calculations.
        use downsampled lane segments to lower computational cost.
        use c implemented shapely str tree for spacial indexing to reduce search space from O(n) to roughly O(log n) 
        average complexity while finding nearby segments of other lanes.
        buffer lane segment line to polygon using lane widht. check segments for intersections. calculate metadata
        (heading, side, direction) of neighbor segments. append neighbor information on a per lane per segment basis.
        use buffer_width_percentage to increase lane widths to increase overlap chance. use max_andgle_diff
        to decide what angle diffrence between lane segments count as neighbors
        information can be querried in multiple ways, see 'longest_neighbor_chain_one_side()' methods for example.
        """
        all_segments = []
        segment_to_lane_and_index = {}

        for lane_id, segments in self.lane_segments.items():
            for seg_idx, seg in enumerate(segments):
                all_segments.append(seg)
                segment_to_lane_and_index[id(seg)] = (lane_id, seg_idx)

        if not all_segments:
            # nothing to index, keep neighbor_info empty and return
            self.neighbor_info = defaultdict(lambda: defaultdict(list))
            return
        
        tree = STRtree(all_segments)

        # Result: for each lane_id and segment index → list of (side, neighbor_lane_id)
        neighbor_info = defaultdict(lambda: defaultdict(list))

        for lane_id, segments in self.lane_segments.items():
            buffer_radius = self.lane_graph[lane_id]["buffer_size"] * buffer_width_percentage
            # Get predecessors and successors
            predecessors = set(self.get_predecessors(lane_id))
            successors = set(self.get_successors(lane_id))
            for idx, segment in enumerate(segments):
                heading = compute_heading(segment)
                buf = lateral_buffer(segment, buffer_radius)

                search_area = lateral_buffer(segment, buffer_radius*2)
                hits = tree.query(search_area)

                for candidate_idx in hits:
                    candidate = all_segments[candidate_idx]  # get actual LineString
                    if candidate == segment:
                        continue

                    neighbor_info_entry = segment_to_lane_and_index.get(id(candidate))
                    if neighbor_info_entry is None:
                        continue
                    neighbor_id, neighbor_seg_idx = neighbor_info_entry
                    if neighbor_id == lane_id:
                        continue
                    if neighbor_id in predecessors or neighbor_id in successors:
                        continue

                    if cut_ends:
                        if idx == 0 and neighbor_seg_idx == len(self.lane_segments[neighbor_id]) - 1:
                            continue
                        if idx == len(segments) - 1 and neighbor_seg_idx == 0:
                            continue
                    
                    neighbor_buffer_size = self.lane_graph[neighbor_id]["buffer_size"] * buffer_width_percentage
                    neighbor_seg_buffer = candidate.buffer(neighbor_buffer_size)

                    neighbor_seg_buffer = lateral_buffer(candidate,neighbor_buffer_size)
                    if not buf.intersects(neighbor_seg_buffer):
                        continue

                    candidate_heading = compute_heading(candidate)
                    delta_heading = abs((heading - candidate_heading + np.pi) % (2 * np.pi) - np.pi)
                    if delta_heading < np.pi / 4:
                        direction = "same"
                    elif delta_heading > 3 * np.pi / 4:
                        direction = "opposite"
                    else:
                        direction = "angled"

                    side = side_of(candidate, segment, heading)
                    if side in ("left", "right"):
                        if (side, neighbor_id, direction) not in neighbor_info[lane_id][idx]:                            
                            if direction == "angled":
                                continue
                            neighbor_info[lane_id][idx].append((side, neighbor_id, neighbor_seg_idx, direction))
        #print(neighbor_info[115])
        self.neighbor_info = neighbor_info

    def _compute_neighbors_from_neighbor_info(self):
        '''sums up lane neighbor info of _compute_lane_neighbors_per_lane_segment,
        disregarding segment information to use it for comparison against other neighbor
        calculation methods'''

        for lane_id, segments_info in self.neighbor_info.items():
            neighbor_ids = set()
            for segment_neighbors in segments_info.values():
                for entry in segment_neighbors:
                    _, neighbor_id, _, _ = entry
                    neighbor_ids.add(neighbor_id)
            self.segment_neighbors[lane_id] = neighbor_ids

    def longest_neighbor_chain_one_side(self, lane_chain, side, enforce_continuity= False, min_overlap=0, neighbor_direction = 'any'):
        """
        Track the longest valid chain of neighbors (left/right) across the lane_chain that.
        only allows neighbor chains along the target lane chain that are either of a constant lane,
        or are immediate successor/predecessor lanes.
        leaving enforced contionuity false allows neighbor chains with gaps,
        meaning they are neighbors in the beginning and the end of a target lane, but not for all segments.
        direction of searched neighbors can be set to 'same' or 'oppoisite'
        """
        def longest_consecutive_subchain(chain, lane_chain):
            lane_to_idx = {lane_id: i for i, lane_id in enumerate(lane_chain)}

            longest_subchain = []
            current_subchain = []

            prev_lane_i = None
            prev_seg_idx = None

            for item in chain:
                lane_id, seg_idx, _, _ = item
                lane_i = lane_to_idx[lane_id]

                if prev_lane_i is None:
                    # Start first element of a new subchain
                    current_subchain = [item]

                else:
                    # Check lane continuity
                    lane_ok = (lane_i == prev_lane_i) or (lane_i == prev_lane_i + 1)

                    if not lane_ok:
                        # Lane jump too big, break chain
                        if len(current_subchain) > len(longest_subchain):
                            longest_subchain = current_subchain
                        current_subchain = [item]

                    else:
                        if lane_i == prev_lane_i:
                            # Same lane: segment must be same or +1
                            seg_ok = (seg_idx == prev_seg_idx) or (seg_idx == prev_seg_idx + 1)

                            if seg_ok:
                                current_subchain.append(item)
                            else:
                                # Segment gap in same lane breaks chain
                                if len(current_subchain) > len(longest_subchain):
                                    longest_subchain = current_subchain
                                current_subchain = [item]

                        else:
                            # lane_i == prev_lane_i + 1 (next lane)
                            prev_lane_id = lane_chain[prev_lane_i]
                            prev_segments = self.lane_segments.get(prev_lane_id, [])
                            last_seg_idx = len(prev_segments) - 1

                            # New lane segment must start at 0 and previous lane must be fully traversed
                            if prev_seg_idx == last_seg_idx and seg_idx == 0:
                                current_subchain.append(item)
                            else:
                                if len(current_subchain) > len(longest_subchain):
                                    longest_subchain = current_subchain
                                current_subchain = [item]

                prev_lane_i = lane_i
                prev_seg_idx = seg_idx

            # Check last subchain at end of loop
            if len(current_subchain) > len(longest_subchain):
                longest_subchain = current_subchain

            return longest_subchain

        active_chains = {}  # keys: neighbor_id, values: chain list
        best_chain = []
        for lane_id in lane_chain:
            segments = self.lane_segments.get(lane_id, [])

            for seg_idx in range(len(segments)):
                neighbors = self.neighbor_info.get(lane_id, {}).get(seg_idx, [])
                side_neighbors = [n for n in neighbors if n[0] == side]
                if neighbor_direction != 'any':
                    side_neighbors = [n for n in side_neighbors if n[3] == neighbor_direction]
                new_active_chains = {}

                for _, neighbor_id, neighbor_seg_idx, _ in side_neighbors:
                    extended = False
                    for prev_neighbor, chain in active_chains.items():
                        if neighbor_id == prev_neighbor or neighbor_id in self.get_successors(prev_neighbor):
                            candidate_chain = chain + [(lane_id, seg_idx, neighbor_id, neighbor_seg_idx)]
                            if neighbor_id not in new_active_chains or len(candidate_chain) > len(new_active_chains[neighbor_id]):
                                new_active_chains[neighbor_id] = candidate_chain
                            extended = True

                    if not extended:
                        if neighbor_id not in new_active_chains:
                            new_active_chains[neighbor_id] = [(lane_id, seg_idx, neighbor_id, neighbor_seg_idx)]

                for prev_neighbor, chain in active_chains.items():
                    if prev_neighbor not in new_active_chains:
                        new_active_chains[prev_neighbor] = chain

                active_chains = new_active_chains

            for chain in active_chains.values():
                if enforce_continuity:
                    chain = longest_consecutive_subchain(chain, lane_chain)
                if len(chain) > len(best_chain):
                    best_chain = chain

        #print(f"longest chain length for chain {lane_chain} on side:{side}: {len(best_chain)}")
        if len(best_chain) > min_overlap:
            return best_chain
        else:
            return []

    def longest_neighbor_chain_left_and_right(self, lane_chain, min_overlap, enforce_cont = False, neighbor_direction= 'any'):
        left_best_chain = self.longest_neighbor_chain_one_side(lane_chain=lane_chain, side='left', enforce_continuity= enforce_cont, neighbor_direction=neighbor_direction)
        right_best_chain = self.longest_neighbor_chain_one_side(lane_chain=lane_chain, side='right', enforce_continuity= enforce_cont, neighbor_direction=neighbor_direction)
        results = self.intersect_chains(left_best_chain,right_best_chain, min_overlap)
        #print('left', left_best_chain, ' right: ', right_best_chain)
        return results
    
    def intersect_chains(self, longest_left, longest_right, min_overlap):
        # Map from (lane_id, seg_idx) to neighbor_id
        left_map = {(lane_id, seg_idx): left_nid for (lane_id, seg_idx, left_nid, _) in longest_left}
        right_map = {(lane_id, seg_idx): right_nid for (lane_id, seg_idx, right_nid, _) in longest_right}

        # Intersect segment positions
        common_keys = sorted(set(left_map.keys()) & set(right_map.keys()))
        if len(common_keys) < min_overlap:
            return False  # or None, depending on what you want
        # Build the combined chain
        combined = [(lane_id, seg_idx, left_map[(lane_id, seg_idx)], right_map[(lane_id, seg_idx)])
                    for (lane_id, seg_idx) in common_keys]

        return combined
        
    def _get_buffer_distance(self, lane_type_id):
        """helper function to compute lane segment overlaps using polygons.
        Maps lane type to a buffer (half the width of a lane) in meters."""
        type_map = {
            1: 1.75,  # e.g., freeway
            2: 1.75,  # e.g., surface street
            3: 0.75,  # e.g., bike lane
            6: 0.1,  # e.g., brokenSingleWhite
            9: 0.1,  # e.g., brokenSingleYellow
            10: 0.1,  # e.g., brokenDoubleYellow

        }
        return type_map.get(lane_type_id, 1.5)  # default buffer if unknown = 1.5m
    
    def neighbor_chains_one_side(self, lane_chain, side, enforce_continuity=False, min_overlap=0, neighbor_direction='any'):
        """
        Track all valid neighbor chains (left/right) across the lane_chain that are longer than min_overlap.
        Returns all chains, sorted by length (descending).
        Each chain entry is (lane_id, seg_idx, neighbor_id, neighbor_seg_idx, direction).
        """
        def longest_consecutive_subchain(chain, lane_chain):
            lane_to_idx = {lane_id: i for i, lane_id in enumerate(lane_chain)}

            longest_subchain = []
            current_subchain = []

            prev_lane_i = None
            prev_seg_idx = None

            for item in chain:
                lane_id, seg_idx, _, _, _ = item
                lane_i = lane_to_idx[lane_id]

                if prev_lane_i is None:
                    current_subchain = [item]
                else:
                    lane_ok = (lane_i == prev_lane_i) or (lane_i == prev_lane_i + 1)

                    if not lane_ok:
                        if len(current_subchain) > len(longest_subchain):
                            longest_subchain = current_subchain
                        current_subchain = [item]
                    else:
                        if lane_i == prev_lane_i:
                            seg_ok = (seg_idx == prev_seg_idx) or (seg_idx == prev_seg_idx + 1)

                            if seg_ok:
                                current_subchain.append(item)
                            else:
                                if len(current_subchain) > len(longest_subchain):
                                    longest_subchain = current_subchain
                                current_subchain = [item]
                        else:
                            prev_lane_id = lane_chain[prev_lane_i]
                            prev_segments = self.lane_segments.get(prev_lane_id, [])
                            last_seg_idx = len(prev_segments) - 1

                            if prev_seg_idx == last_seg_idx and seg_idx == 0:
                                current_subchain.append(item)
                            else:
                                if len(current_subchain) > len(longest_subchain):
                                    longest_subchain = current_subchain
                                current_subchain = [item]

                prev_lane_i = lane_i
                prev_seg_idx = seg_idx

            if len(current_subchain) > len(longest_subchain):
                longest_subchain = current_subchain

            return longest_subchain

        active_chains = {}  # keys: neighbor_id, values: chain list
        valid_chains = []   # collect all chains longer than min_overlap

        for lane_id in lane_chain:
            segments = self.lane_segments.get(lane_id, [])

            for seg_idx in range(len(segments)):
                neighbors = self.neighbor_info.get(lane_id, {}).get(seg_idx, [])
                side_neighbors = [n for n in neighbors if n[0] == side]
                if neighbor_direction != 'any':
                    side_neighbors = [n for n in side_neighbors if n[3] == neighbor_direction]
                new_active_chains = {}

                for _, neighbor_id, neighbor_seg_idx, direction in side_neighbors:
                    extended = False
                    for prev_neighbor, chain in active_chains.items():
                        if neighbor_id == prev_neighbor or neighbor_id in self.get_successors(prev_neighbor):
                            candidate_chain = chain + [(lane_id, seg_idx, neighbor_id, neighbor_seg_idx, direction)]
                            if neighbor_id not in new_active_chains or len(candidate_chain) > len(new_active_chains[neighbor_id]):
                                new_active_chains[neighbor_id] = candidate_chain
                            extended = True

                    if not extended:
                        if neighbor_id not in new_active_chains:
                            new_active_chains[neighbor_id] = [(lane_id, seg_idx, neighbor_id, neighbor_seg_idx, direction)]

                for prev_neighbor, chain in active_chains.items():
                    if prev_neighbor not in new_active_chains:
                        new_active_chains[prev_neighbor] = chain

                active_chains = new_active_chains

            # After processing each lane, collect chains above threshold
            for chain in active_chains.values():
                if enforce_continuity:
                    chain = longest_consecutive_subchain(chain, lane_chain)
                if len(chain) > min_overlap:
                    valid_chains.append(chain)

        # Deduplicate chains
        unique_chains = []
        seen = set()
        for chain in valid_chains:
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                unique_chains.append(chain)

        # Sort by length, descending
        unique_chains.sort(key=len, reverse=True)

        return unique_chains

    def neighbor_chains_both_sides(self, lane_chain, enforce_continuity=False, min_overlap=0, neighbor_direction='any'):
        """
        Run neighbor_chains_one_side() for both left and right sides of a lane_chain.
        Merge contiguous chains with the same neighbors so that each list only contains maximal contiguous chains.

        Returns:
            {
                "left": [...merged chains...],
                "right": [...merged chains...],
                "overlap_chains": [...merged overlap chains...]
            }
        """
        # Step 1: get raw chains
        left_chains = self.neighbor_chains_one_side(
            lane_chain, "left", enforce_continuity=enforce_continuity,
            min_overlap=min_overlap, neighbor_direction=neighbor_direction
        )
        right_chains = self.neighbor_chains_one_side(
            lane_chain, "right", enforce_continuity=enforce_continuity,
            min_overlap=min_overlap, neighbor_direction=neighbor_direction
        )

        # Step 2: merge contiguous chains with the same neighbors
        def merge_chains(chains):
            if not chains:
                return []

            # Sort chains by starting segment index of chain_a
            chains.sort(key=lambda c: (c[0][0], c[0][1]))  # sort by (lane_id, seg_idx)

            merged = []
            current = chains[0]

            for next_chain in chains[1:]:
                # Check if contiguous in chain_a
                current_last_lane, current_last_idx = current[-1][0], current[-1][1]
                next_first_lane, next_first_idx = next_chain[0][0], next_chain[0][1]

                prev_segments = self.lane_segments.get(current_last_lane, [])
                contiguous = (current_last_lane == next_first_lane and next_first_idx == current_last_idx + 1) or \
                            (self.lane_segments.get(current_last_lane) and next_first_idx == 0 and
                            lane_chain.index(next_first_lane) == lane_chain.index(current_last_lane) + 1)

                # Check if neighbors are identical
                same_neighbors = set((item[2], item[4]) for item in current) == set((item[2], item[4]) for item in next_chain)

                if contiguous and same_neighbors:
                    current += next_chain  # extend current
                else:
                    merged.append(current)
                    current = next_chain

            merged.append(current)
            return merged

        left_chains = merge_chains(left_chains)
        right_chains = merge_chains(right_chains)

        # Step 3: compute overlaps as before
        overlap_chains = []
        for left_chain in left_chains:
            left_segments = [(lane_id, seg_idx) for lane_id, seg_idx, _, _, _ in left_chain]

            for right_chain in right_chains:
                right_segments = [(lane_id, seg_idx) for lane_id, seg_idx, _, _, _ in right_chain]

                i = 0
                while i < len(left_segments):
                    try:
                        j = right_segments.index(left_segments[i])
                    except ValueError:
                        i += 1
                        continue

                    overlap_entry = []
                    ii, jj = i, j
                    while ii < len(left_segments) and jj < len(right_segments) and left_segments[ii] == right_segments[jj]:
                        left_info = {"id": left_chain[ii][2], "direction": left_chain[ii][4]}
                        right_info = {"id": right_chain[jj][2], "direction": right_chain[jj][4]}
                        lane_id, seg_idx = left_segments[ii]
                        overlap_entry.append({
                            "lane_id": lane_id,
                            "seg_idx": seg_idx,
                            "left": left_info,
                            "right": right_info
                        })
                        ii += 1
                        jj += 1

                    if len(overlap_entry) > 0:
                        overlap_chains.append(overlap_entry)

                    i = ii

        return {
            "left": left_chains,
            "right": right_chains,
            "overlap_chains": overlap_chains
        }
    
    def build_road_segments(self, lane_chain, enforce_continuity=False, min_overlap=0, neighbor_direction="any"):
        """
        Build road segment structures from neighbor_chains_both_sides().
        - num_lanes = 2: two chains (left/right neighbor relation)
        - num_lanes = 3: three chains (overlap from both sides)
        """
        result = self.neighbor_chains_both_sides(
            lane_chain,
            enforce_continuity=enforce_continuity,
            min_overlap=min_overlap,
            neighbor_direction=neighbor_direction
        )

        left_chains = result["left"]
        right_chains = result["right"]
        overlap_chains = result["overlap_chains"]

        road_segments = []
        seen = set()

        # ---- num_lanes = 2 (left/right neighbor chains) ----
        for side, chains in [("left", left_chains), ("right", right_chains)]:
            for chain in chains:
                lane_ids = list(dict.fromkeys(l for l, _, _, _, _ in chain))  # preserve order, deduplicate
                neighbor_ids = list(dict.fromkeys(n for _, _, n, _, _ in chain))
                num_segments = len(chain)  # count every segment entry

                # Deduplication key
                key = (tuple(lane_ids), tuple(neighbor_ids), num_segments)
                if key in seen:
                    continue
                seen.add(key)

                road_segments.append({
                    "num_lanes": 2,
                    "num_segments": num_segments,
                    "chains": [
                        {"id": "chain_a", "lane_ids": lane_ids},
                        {"id": "chain_b", "lane_ids": neighbor_ids}
                    ],
                    "adjacency": {
                        "chain_a": {
                            "right_of": [{"neighbor": "chain_b"}] if side == "left" else [],
                            "left_of": [{"neighbor": "chain_b"}] if side == "right" else []
                        },
                        "chain_b": {
                            "left_of": [{"neighbor": "chain_a"}] if side == "left" else [],
                            "right_of": [{"neighbor": "chain_a"}] if side == "right" else []
                        }
                    }
                })

        # ---- num_lanes = 3 (overlap chains) ----
        for overlap in overlap_chains:
            if not overlap:
                continue

            num_segments = len(overlap)  # count total segment entries
            lane_ids = list(dict.fromkeys(e["lane_id"] for e in overlap))
            left_ids = list(dict.fromkeys(e["left"]["id"] for e in overlap))
            right_ids = list(dict.fromkeys(e["right"]["id"] for e in overlap))

            key = (tuple(lane_ids), tuple(left_ids), tuple(right_ids), num_segments)
            if key in seen:
                continue
            seen.add(key)

            road_segments.append({
                "num_lanes": 3,
                "num_segments": num_segments,
                "chains": [
                    {"id": "chain_a", "lane_ids": lane_ids},
                    {"id": "chain_left", "lane_ids": left_ids},
                    {"id": "chain_right", "lane_ids": right_ids}
                ],
                "adjacency": {
                    "chain_a": {
                        "left_of": [{"neighbor": "chain_left"}],
                        "right_of": [{"neighbor": "chain_right"}]
                    },
                    "chain_left": {
                        "right_of": [{"neighbor": "chain_a"}]
                    },
                    "chain_right": {
                        "left_of": [{"neighbor": "chain_a"}]
                    }
                }
            })

        return road_segments
    

    def build_lane_lookup(self, road_segments):
        """
        Build a lookup from lane_id -> list of OpenSCENARIO lane mappings.
        Handles overlaps where the same lane appears in multiple segments.
        """
        lane_lookup = {}

        for seg_id, seg_data in road_segments.items():
            chains = seg_data["chains"]

            for osc_index, chain in enumerate(chains, start=1):
                chain_id = chain["id"]
                for lane_id in chain["lane_ids"]:
                    mapping = {
                        "segment": seg_id,
                        "chain_id": chain_id,
                        "osc_index": osc_index
                    }
                    lane_lookup.setdefault(lane_id, []).append(mapping)

        return lane_lookup

    def _build_target_global_index(self, target_chain):
        """
        Returns:
        g2lane: list of (lane_id, seg_idx) for every global segment in target_chain
        lane_start: {lane_id: global_start_index}
        lane_end:   {lane_id: global_end_index_exclusive}
        """
        g2lane = []
        lane_start, lane_end = {}, {}
        g = 0
        for lane_id in target_chain:
            segs = self.lane_segments.get(lane_id, [])
            lane_start[lane_id] = g
            for s in range(len(segs)):
                g2lane.append((lane_id, s))
                g += 1
            lane_end[lane_id] = g
        return g2lane, lane_start, lane_end
    
    def neighbor_chains_one_side_wrap_allchains(
    self, target_chain, all_chains, side,
    enforce_continuity=False, min_overlap=0, neighbor_direction='any'
    ):
        """
        Track neighbor chains on one side using full chain info and a global segment index.

        Returns a list of *maximal* runs. Each run:
        {
            "side": "left" | "right",
            "target_global_start": int,           # inclusive
            "target_global_end": int,             # exclusive
            "num_segments": int,                  # = target_global_end - target_global_start
            "target_lane_ids": [...],             # only target lane IDs that actually appear in this run (ordered, unique)
            "neighbor_lane_ids": [...],           # only neighbor lane IDs that appear in this run (ordered, unique)
            "neighbor_lane_spans": {              # per-neighbor-lane spans in *target global* coords
                neighbor_lane_id: [(gstart, gend), ...]  # non-overlapping, contiguous spans
            }
        }
        """
        # Build lane->chain lookup from all_chains
        lane_to_chain = {lane: i for i, chain in enumerate(all_chains) for lane in chain}

        # Build global index for target_chain
        g2lane, _, _ = self._build_target_global_index(target_chain)
        G = len(g2lane)

        # Active runs keyed by neighbor_chain_id
        active = {}  # neighbor_chain_id -> run dict

        results = []

        def finalize_run(run):
            if run["num_segments"] >= min_overlap:
                results.append(run)

        # Iterate across *global* target segments
        for g in range(G):
            lane_id, seg_idx = g2lane[g]
            # neighbors for this target segment
            neighbors = self.neighbor_info.get(lane_id, {}).get(seg_idx, [])
            side_neighbors = [n for n in neighbors if n[0] == side]
            if neighbor_direction != 'any':
                side_neighbors = [n for n in side_neighbors if n[3] == neighbor_direction]

            # Group present neighbors by their neighbor_chain_id
            present_by_chain = {}
            for _, neighbor_id, neighbor_seg_idx, direction in side_neighbors:
                chain_id = lane_to_chain.get(neighbor_id)
                if chain_id is None:
                    continue
                present_by_chain.setdefault(chain_id, []).append((neighbor_id, neighbor_seg_idx, direction))

            # Step 1: extend the runs that are present this tick
            extended_chain_ids = set()
            for chain_id, neighs in present_by_chain.items():
                if chain_id in active:
                    # extend existing run
                    run = active[chain_id]
                    run["target_global_end"] = g + 1
                    run["num_segments"] += 1
                    # record target lane id if new
                    if not run["target_lane_ids"] or run["target_lane_ids"][-1] != lane_id:
                        if run["target_lane_ids"] and run["target_lane_ids"][-1] == lane_id:
                            pass
                        elif lane_id not in run["target_lane_ids"]:
                            run["target_lane_ids"].append(lane_id)
                    # update neighbor lanes + spans
                    for (nid, _nseg, _dir) in neighs:
                        if not run["neighbor_lane_ids"] or run["neighbor_lane_ids"][-1] != nid:
                            if nid not in run["neighbor_lane_ids"]:
                                run["neighbor_lane_ids"].append(nid)
                        spans = run["neighbor_lane_spans"].setdefault(nid, [])
                        if spans and spans[-1][1] == g:  # contiguous -> extend
                            spans[-1] = (spans[-1][0], g + 1)
                        else:  # start new span at g
                            spans.append((g, g + 1))
                    extended_chain_ids.add(chain_id)
                else:
                    # start a new run
                    run = {
                        "side": side,
                        "target_global_start": g,
                        "target_global_end": g + 1,
                        "num_segments": 1,
                        "target_lane_ids": [lane_id],
                        "neighbor_lane_ids": [],
                        "neighbor_lane_spans": {}
                    }
                    for (nid, _nseg, _dir) in neighs:
                        if nid not in run["neighbor_lane_ids"]:
                            run["neighbor_lane_ids"].append(nid)
                        run["neighbor_lane_spans"].setdefault(nid, []).append((g, g + 1))
                    active[chain_id] = run
                    extended_chain_ids.add(chain_id)

            # Step 2: any active runs that were NOT extended at this tick must be finalized
            to_close = [cid for cid in active.keys() if cid not in extended_chain_ids]
            for cid in to_close:
                finalize_run(active[cid])
                del active[cid]

        # Flush remaining active runs at end
        for run in list(active.values()):
            finalize_run(run)
        active.clear()

        # Optionally, sort results by length desc
        results.sort(key=lambda r: r["num_segments"], reverse=True)
        return results



    def neighbor_chains_both_sides_wrap_allchains(
    self, target_chain, all_chains, enforce_continuity=False, min_overlap=0, neighbor_direction='any'
    ):
        """
        Runs one-side on left and right, then finds both-sides overlaps.
        Uses neighbor_lane_spans + target_global windows to keep only the *involved* lane IDs.
        """
        left_runs = self.neighbor_chains_one_side_wrap_allchains(
            target_chain, all_chains, 'left', enforce_continuity, min_overlap=0, neighbor_direction=neighbor_direction
        )
        right_runs = self.neighbor_chains_one_side_wrap_allchains(
            target_chain, all_chains, 'right', enforce_continuity, min_overlap=0, neighbor_direction=neighbor_direction
        )

        # global index for mapping global windows -> target lane IDs
        g2lane, _, _ = self._build_target_global_index(target_chain)

        def ids_in_global_window(gstart, gend):
            seen = set()
            out = []
            for g in range(gstart, gend):
                lid = g2lane[g][0]
                if lid not in seen:
                    seen.add(lid)
                    out.append(lid)
            return out

        def neighbor_ids_in_window(neighbor_lane_spans, gstart, gend):
            # keep neighbor ids whose *any* span intersects [gstart, gend)
            out = []
            for nid, spans in neighbor_lane_spans.items():
                for (s, e) in spans:
                    if not (e <= gstart or s >= gend):  # overlap
                        out.append(nid)
                        break
            return out

        # Build 2-lane outputs: left-only and right-only (already maximal runs)
        left_only = []
        for r in left_runs:
            if r["num_segments"] >= min_overlap:
                left_only.append({
                    "side": "left",
                    "target_lane_ids": r["target_lane_ids"][:],
                    "neighbor_lane_ids": r["neighbor_lane_ids"][:],
                    "num_segments": r["num_segments"],
                    "target_global_start": r["target_global_start"],
                    "target_global_end": r["target_global_end"],
                    "neighbor_lane_spans": r["neighbor_lane_spans"]  # keep for debugging if needed
                })

        right_only = []
        for r in right_runs:
            if r["num_segments"] >= min_overlap:
                right_only.append({
                    "side": "right",
                    "target_lane_ids": r["target_lane_ids"][:],
                    "neighbor_lane_ids": r["neighbor_lane_ids"][:],
                    "num_segments": r["num_segments"],
                    "target_global_start": r["target_global_start"],
                    "target_global_end": r["target_global_end"],
                    "neighbor_lane_spans": r["neighbor_lane_spans"]
                })

        # Build 3-lane overlaps by intersecting left/right global windows
        both_sides = []
        for L in left_runs:
            for R in right_runs:
                gs = max(L["target_global_start"], R["target_global_start"])
                ge = min(L["target_global_end"],   R["target_global_end"])
                if ge <= gs:
                    continue
                length = ge - gs
                if length < min_overlap:
                    continue

                # Filter lane IDs to *those actually present in the overlap window*
                target_ids = ids_in_global_window(gs, ge)
                left_ids  = neighbor_ids_in_window(L["neighbor_lane_spans"], gs, ge)
                right_ids = neighbor_ids_in_window(R["neighbor_lane_spans"], gs, ge)

                if not left_ids or not right_ids:
                    continue  # must have both sides inside the same window

                both_sides.append({
                    "target_lane_ids": target_ids,
                    "left_lane_ids": left_ids,
                    "right_lane_ids": right_ids,
                    "num_segments": length,
                    "target_global_start": gs,
                    "target_global_end": ge
                })

        # Optional: dedupe (same triple of ID tuples + length)
        def key2(r):
            return (tuple(r["target_lane_ids"]), tuple(r.get("left_lane_ids", [])),
                    tuple(r.get("right_lane_ids", [])), r["num_segments"])
        seen = set()
        left_only = [r for r in left_only if not (key2(r) in seen or seen.add(key2(r)))]
        right_only = [r for r in right_only if not (key2(r) in seen or seen.add(key2(r)))]
        for r in both_sides:
            key = key2(r)
            if key not in seen:
                seen.add(key)

        return {
            "left": left_only,
            "right": right_only,
            "overlap_chains": both_sides
        }
    
    def build_road_segments_wrap_allchains(self, target_chain, all_chains,
                                       enforce_continuity=False, min_overlap=0,
                                       neighbor_direction="any"):
        res = self.neighbor_chains_both_sides_wrap_allchains(
            target_chain, all_chains, enforce_continuity, min_overlap, neighbor_direction
        )

        left_runs  = res["left"]
        right_runs = res["right"]
        overlaps   = res["overlap_chains"]

        road_segments = []
        seen = set()

        # 2-lane: left-only
        for r in left_runs:
            lane_ids     = r["target_lane_ids"]
            neighbor_ids = r["neighbor_lane_ids"]
            nseg         = r["num_segments"]
            gstart       = r["target_global_start"]
            gend         = r["target_global_end"]

            # endpoints
            t_start = self._target_pair_from_window(target_chain, gstart)
            t_end   = self._target_pair_from_window(target_chain, gend - 1)
            l_start, l_end, l_dir = self._neighbor_endpoints_from_window(
                target_chain, gstart, gend, side="left", neighbor_ids=neighbor_ids
            )

            key = ("2L-left", tuple(lane_ids), tuple(neighbor_ids), nseg)
            if key in seen:
                continue
            seen.add(key)

            road_segments.append({
                "num_lanes": 2,
                "num_segments": nseg,
                "chains": [
                    {"id": "chain_a", "lane_ids": lane_ids},
                    {"id": "chain_b", "lane_ids": neighbor_ids}
                ],
                "adjacency": {
                    "chain_a": {"right_of": [{"neighbor": "chain_b"}], "left_of": []},
                    "chain_b": {"left_of":  [{"neighbor": "chain_a"}], "right_of": []}
                },
                # <<< minimal add-on >>>
                "endpoints": {
                    "target": {"start": t_start, "end": t_end},
                    "left":   {"start": l_start, "end": l_end, "direction": l_dir}
                }
            })

        # 2-lane: right-only
        for r in right_runs:
            lane_ids     = r["target_lane_ids"]
            neighbor_ids = r["neighbor_lane_ids"]
            nseg         = r["num_segments"]
            gstart       = r["target_global_start"]
            gend         = r["target_global_end"]

            # endpoints
            t_start = self._target_pair_from_window(target_chain, gstart)
            t_end   = self._target_pair_from_window(target_chain, gend - 1)
            r_start, r_end, r_dir = self._neighbor_endpoints_from_window(
                target_chain, gstart, gend, side="right", neighbor_ids=neighbor_ids
            )

            key = ("2L-right", tuple(lane_ids), tuple(neighbor_ids), nseg)
            if key in seen:
                continue
            seen.add(key)

            road_segments.append({
                "num_lanes": 2,
                "num_segments": nseg,
                "chains": [
                    {"id": "chain_a", "lane_ids": lane_ids},
                    {"id": "chain_b", "lane_ids": neighbor_ids}
                ],
                "adjacency": {
                    "chain_a": {"left_of": [{"neighbor": "chain_b"}], "right_of": []},
                    "chain_b": {"right_of": [{"neighbor": "chain_a"}], "left_of":  []}
                },
                # <<< minimal add-on >>>
                "endpoints": {
                    "target": {"start": t_start, "end": t_end},
                    "right":  {"start": r_start, "end": r_end, "direction": r_dir}
                }
            })

        # 3-lane: both-sides overlaps (already filtered to involved lane IDs)
        for o in overlaps:
            lane_ids  = o["target_lane_ids"]
            left_ids  = o["left_lane_ids"]
            right_ids = o["right_lane_ids"]
            nseg      = o["num_segments"]
            gstart    = o["target_global_start"]
            gend      = o["target_global_end"]

            # endpoints
            t_start = self._target_pair_from_window(target_chain, gstart)
            t_end   = self._target_pair_from_window(target_chain, gend - 1)
            l_start, l_end, l_dir = self._neighbor_endpoints_from_window(
                target_chain, gstart, gend, side="left",  neighbor_ids=left_ids
            )
            r_start, r_end, r_dir = self._neighbor_endpoints_from_window(
                target_chain, gstart, gend, side="right", neighbor_ids=right_ids
            )

            key = ("3L", tuple(lane_ids), tuple(left_ids), tuple(right_ids), nseg)
            if key in seen:
                continue
            seen.add(key)

            road_segments.append({
                "num_lanes": 3,
                "num_segments": nseg,
                "chains": [
                    {"id": "chain_a", "lane_ids": lane_ids},
                    {"id": "chain_left", "lane_ids": left_ids},
                    {"id": "chain_right", "lane_ids": right_ids}
                ],
                "adjacency": {
                    "chain_a":    {"left_of": [{"neighbor": "chain_left"}],
                                "right_of": [{"neighbor": "chain_right"}]},
                    "chain_left": {"right_of": [{"neighbor": "chain_a"}]},
                    "chain_right":{"left_of":  [{"neighbor": "chain_a"}]}
                },
                # <<< minimal add-on >>>
                "endpoints": {
                    "target": {"start": t_start, "end": t_end},
                    "left":   {"start": l_start, "end": l_end, "direction": l_dir},
                    "right":  {"start": r_start, "end": r_end, "direction": r_dir}
                }
            })

        return road_segments

    def build_global_road_segments(self, all_chains,
                               enforce_continuity=False, min_overlap=0,
                               neighbor_direction="any",
                               opendrive_style=True):
        """
        Build a global dictionary of road segments across all chains.
        1) Deduplicates mirrored 2-lane segments.
        2) Deduplicates subset/superset 2-lane segments while preserving lane order.
        3) Does not deduplicate 3-lane segments.
        Assigns unique IDs to each segment.
        
        If opendrive_style=True, outputs OpenDRIVE-style lane numbering:
        - 2 lanes: id=1 (left), id=2 (right)
        - 3 lanes: id=1 (left), id=2 (middle), id=3 (right)
        """

        global_segments = {}
        seg_counter = 0

        # Step 1: initial deduplication of mirrored 2-lane segments
        temp_segments = []
        seen_2lane = []
        seen_3lane = set()

        for chain in all_chains:
            segs = self.build_road_segments_wrap_allchains(
                target_chain=chain,
                all_chains=all_chains,
                enforce_continuity=enforce_continuity,
                min_overlap=min_overlap,
                neighbor_direction=neighbor_direction
            )

            for seg in segs:
                nlanes = seg["num_lanes"]

                if nlanes == 2:
                    chain_a = tuple(seg["chains"][0]["lane_ids"])
                    chain_b = tuple(seg["chains"][1]["lane_ids"])
                    nseg    = seg["num_segments"]

                    key = tuple(sorted([chain_a, chain_b])) + (nseg,)
                    if key in seen_2lane:
                        continue
                    seen_2lane.append(key)

                elif nlanes == 3:
                    target_ids = tuple(seg["chains"][0]["lane_ids"])
                    left_ids   = tuple(seg["chains"][1]["lane_ids"])
                    right_ids  = tuple(seg["chains"][2]["lane_ids"])
                    nseg       = seg["num_segments"]

                    key = (target_ids, left_ids, right_ids, nseg)
                    if key in seen_3lane:
                        continue
                    seen_3lane.add(key)

                temp_segments.append(seg)

        # Step 2: subset/superset deduplication of 2-lane segments
        deduped_segments = []
        skipped_indices = set()
        for i, seg1 in enumerate(temp_segments):
            if i in skipped_indices or seg1["num_lanes"] != 2:
                continue

            A1 = seg1["chains"][0]["lane_ids"]
            B1 = seg1["chains"][1]["lane_ids"]
            keep_seg1 = True

            for j, seg2 in enumerate(temp_segments):
                if j <= i or seg2["num_lanes"] != 2:
                    continue

                A2 = seg2["chains"][0]["lane_ids"]
                B2 = seg2["chains"][1]["lane_ids"]

                # Check full subset/superset conditions
                case1 = set(A1).issubset(B2) and set(B1).issubset(A2)
                case2 = set(A2).issubset(B1) and set(B2).issubset(A1)

                if case1 or case2:
                    len1 = len(A1) + len(B1)
                    len2 = len(A2) + len(B2)

                    if len1 < len2:
                        keep_seg1 = False
                        break
                    elif len1 > len2:
                        skipped_indices.add(j)
                    else:  # same length, discard first
                        keep_seg1 = False
                        break

            if keep_seg1:
                deduped_segments.append(seg1)

        # Step 3: add 3-lane segments untouched
        deduped_segments += [s for s in temp_segments if s["num_lanes"] == 3]

        # Step 4: assign unique IDs and format output
        for seg in deduped_segments:
            nlanes = seg["num_lanes"]
            seg_id = f"seg_{seg_counter}"
            seg_counter += 1

            if opendrive_style:
                if nlanes == 2:
                    left_ids, right_ids = _derive_left_right_ids_2lane(seg)
                    chains = [
                        {"id": 1, "lane_ids": left_ids},   # OpenDRIVE: left
                        {"id": 2, "lane_ids": right_ids},  # OpenDRIVE: right
                    ]
                elif nlanes == 3:
                    target_ids = seg["chains"][0]["lane_ids"]
                    left_ids   = seg["chains"][1]["lane_ids"]
                    right_ids  = seg["chains"][2]["lane_ids"]
                    chains = [
                        {"id": 1, "lane_ids": left_ids},   # left
                        {"id": 2, "lane_ids": target_ids}, # middle
                        {"id": 3, "lane_ids": right_ids},  # right
                    ]
                else:
                    chains = seg["chains"]

                ep_with_ids, ep_by_id, target_cid = _attach_endpoints_and_target(seg, chains, nlanes, opendrive_style)

                global_segments[seg_id] = {
                    "num_lanes": nlanes,
                    "num_segments": seg["num_segments"],
                    "chains": chains,                 # OpenDRIVE ids: 1=left, 2=middle/right, 3=right
                    "endpoints": ep_with_ids,         # roles + chain_id
                    "endpoints_by_id": ep_by_id,      # keyed by final chain id
                    "target_chain_id": target_cid,    # <-- easy access to the target
                }
                
                

            else:
                # Keep original chain naming
                global_segments[seg_id] = seg

        return global_segments

    
    def plot_segment_highlight(self, global_segments, segment_id, ax=None):
        """
        Plot all lane polygons in light gray, and highlight the lanes 
        involved in a selected segment.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        else:
            fig = ax.figure

        patches = []
        colors = []

        # --- Step 1: Plot all lanes faintly ---
        for lane_id, poly in self.lane_polygons.items():
            if poly is None or not poly.is_valid:
                continue
            coords = list(poly.exterior.coords)
            patch = MplPolygon(coords, closed=True)
            patches.append(patch)
            colors.append((0.8, 0.8, 0.8, 0.3))  # light gray with transparency

        base_collection = PatchCollection(
            patches, facecolors=colors, edgecolor="black", linewidth=0.5, alpha=0.5
        )
        ax.add_collection(base_collection)

        # --- Step 2: Highlight lanes from the selected segment ---
        if segment_id not in global_segments:
            raise ValueError(f"Segment ID {segment_id} not found in global_segments")

        seg = global_segments[segment_id]
        highlight_lane_ids = [lid for chain in seg["chains"] for lid in chain["lane_ids"]]

        patches_highlight = []
        colors_highlight = []

        cmap = cm.get_cmap("tab10", len(highlight_lane_ids))
        for idx, lane_id in enumerate(highlight_lane_ids):
            poly = self.lane_polygons.get(lane_id)
            if poly is None or not poly.is_valid:
                continue
            coords = list(poly.exterior.coords)
            patch = MplPolygon(coords, closed=True)
            patches_highlight.append(patch)
            colors_highlight.append(cmap(idx % 10))  # rotate colors

            # Label each highlighted lane
            centroid = poly.centroid
            ax.text(
                centroid.x, centroid.y, str(lane_id),
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="red"
            )

        highlight_collection = PatchCollection(
            patches_highlight, facecolors=colors_highlight, edgecolor="red", linewidth=1.5, alpha=0.7
        )
        ax.add_collection(highlight_collection)

        # --- Final formatting ---
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"All Lane Polygons with Highlighted {segment_id}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        plt.show()   # <-- force plotting here

    def interactive_lane_neighborhood_viewer(self):
        """Launch an interactive viewer to explore lane neighbors by ID,
         using the 3 diffrent approaches to neighbor computation"""

         # Store your neighbor dicts in a mapping
        neighbor_dicts = {
            "spatial_proximity": self.neighbors,
            "lane_polygon_overlap": self.lane_polygon_neighbors,
            "per_segment_overlap": self.segment_neighbors,
        }
        
        
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Space for the text box

        # Start with the first lane ID
        current_dict_name = "spatial_proximity"
        initial_lane_id = next(iter(self.lane_graph))  # Assuming self.id is a dict or list of lane IDs

        self.plot_lane_neighbors(ax, initial_lane_id, neighbor_dict=neighbor_dicts[current_dict_name])

        # Add text box for user input
        axbox = plt.axes([0.25, 0.05, 0.5, 0.075])
        text_box = TextBox(axbox, 'Enter Lane ID: ')

        # --- Radio buttons for neighbor type ---
        axradio = plt.axes([0.05, 0.05, 0.15, 0.15])
        radio = RadioButtons(axradio, list(neighbor_dicts.keys()), active=0)

        def update_plot(lane_id):
            ax.clear()
            self.plot_lane_neighbors(ax, lane_id, neighbor_dicts[current_dict_name])
            fig.canvas.draw_idle()

        def on_submit(text):
            nonlocal initial_lane_id
            try:
                lane_id = int(text)
                if lane_id in self.lane_graph:
                    initial_lane_id = lane_id
                    update_plot(initial_lane_id)
                else:
                    print(f"Lane ID {lane_id} not found.")
            except ValueError:
                print("Invalid input. Please enter an integer lane ID.")

        def on_radio(label):
            nonlocal current_dict_name
            current_dict_name = label
            update_plot(initial_lane_id)

        text_box.on_submit(on_submit)
        radio.on_clicked(on_radio)
        plt.show()

    def interactive_lane_segments_neighborhood_viewer(self):
        """Launch an interactive viewer to explore lane neighbors by ID."""

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Space for the text box

        # Start with the first lane ID
        initial_lane_id = next(iter(self.lane_graph))  # Assuming self.id is a dict or list of lane IDs

        self.plot_lane_neighbors_polygons(ax, initial_lane_id, 'red', 'green')

        # Add text box for user input
        axbox = plt.axes([0.25, 0.05, 0.5, 0.075])
        text_box = TextBox(axbox, 'Enter Lane ID: ')

        def on_submit(text):
            try:
                lane_id = int(text)
                if lane_id in self.id:
                    ax.clear()  # Clear previous drawings
                    self.plot_lane_neighbors_polygons(ax, lane_id, 'red', 'green')
                    fig.canvas.draw_idle()
                else:
                    print(f"Lane ID {lane_id} not found.")
            except ValueError:
                print("Invalid input. Please enter an integer lane ID.")

        text_box.on_submit(on_submit)
        plt.show()

    def _divide_root_and_branch_sequences(self):
        """
        divides lane graph lane items into root and branch sequences:
        if lane has no predecessor, it makrs the start of a root sequence. 
        if a lane segment of a root sequence has multiple successors,
        the successor with the least angular deviation from the last root 
        segment gets added to root sequence, others start a new branch sequence.
        Tracks whether a sequence started from a branch and,
        if so, from which root sequence it originated.

        Returns:
            - sequences: list of dicts with:
                'lane_ids', 'polyline', 'is_branch_root', 'root_sequence_id'
            - lane_to_sequence: dict mapping lane_id → sequence index
        """
        visited = set()
        pending_starts = []

        # Step 1: add true root lanes (no predecessors)
        for lane_id in self.lane_graph:
            if not self.predecessors.get(lane_id):
                pending_starts.append((lane_id, False, None))  # (lane_id, is_branch_root, root_seq_id)

        def get_best_aligned_successor(lane_id):
            successors = self.successors.get(lane_id, [])
            if not successors:
                return None, []

            current_dir = self.lane_graph[lane_id]["dir"]
            scores = [(succ_id, np.dot(current_dir, self.lane_graph[succ_id]["dir"]))
                    for succ_id in successors]
            scores.sort(key=lambda x: -x[1])
            best = scores[0][0]
            others = [s[0] for s in scores[1:]]
            return best, others

        while pending_starts:
            start_id, is_branch_root, root_seq_id = pending_starts.pop(0)
            if start_id in visited:
                continue

            path = []
            current_id = start_id

            while True:
                path.append(current_id)
                visited.add(current_id)

                successors = self.successors.get(current_id, [])
                if not successors:
                    break

                if len(successors) == 1:
                    next_id = successors[0]
                    if next_id in visited:
                        break
                    current_id = next_id
                else:
                    best, others = get_best_aligned_successor(current_id)

                    for other_id in others:
                        if other_id not in visited:
                            pending_starts.append((other_id, True, len(self.sequences)))  # record parent seq ID

                    if best in visited:
                        break
                    current_id = best

            # Build geometry
            lane_geometries = [self.lane_line_strings[lid] for lid in path]
            polyline = linemerge(lane_geometries)   
            
            seq_index = len(self.sequences)
            for lid in path:
                self.lane_to_sequence[lid] = seq_index

            self.sequences.append({
                "lane_ids": path,
                "polyline": polyline,
                "is_branch_root": is_branch_root,
                "root_sequence_id": root_seq_id  # None if it's a true root
            })

        return self.sequences, self.lane_to_sequence
    
    def get_successors(self, lane_id):
        return self.successors.get(lane_id, [])

    def get_predecessors(self, lane_id):
        return self.predecessors.get(lane_id, [])

    def get_neighbors(self, lane_id):
        return self.neighbors.get(lane_id, [])
    
    def get_polygon_neighbors(self, lane_id):
        """Returns list of lane_ids whose polygons intersect with the given lane's polygon."""
        return self.lane_polygon_neighbors.get(lane_id, [])

    def get_polygon(self, lane_id):
        """Returns the shapely Polygon for a given lane ID."""
        return self.lane_polygons.get(lane_id, None)
    
    def plot_all_polygons_by_lane_type(self, ax=None):
        """
        Plot all lane polygons colored by lane type and mark start points.
        Returns: (fig, ax) so caller can handle plt.show().
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        patches = []
        colors = []
        start_points_x = []
        start_points_y = []

        # Assign consistent colors per lane type
        type_ids = list(set(self.type.flatten()))
        type_ids.sort()
        cmap = cm.get_cmap('tab20', len(type_ids))
        type_to_color = {tid: cmap(i) for i, tid in enumerate(type_ids)}

        lane_ids = list(self.lane_polygons.keys())

        for lane_id in lane_ids:
            poly = self.lane_polygons.get(lane_id)
            if poly is None or not poly.is_valid:
                continue

            # Determine lane type
            mask = self.id == lane_id
            lane_type = int(self.type[mask][0]) if np.any(mask) else -1
            color = type_to_color.get(lane_type, (0.5, 0.5, 0.5))  # gray fallback

            coords = list(poly.exterior.coords)
            patch = MplPolygon(coords, closed=True)
            patches.append(patch)
            colors.append(color)

            # Plot lane ID at polygon centroid
            centroid = poly.centroid
            ax.text(
                centroid.x, centroid.y, str(lane_id),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='black'
            )

            # Get and store start point
            start = self.lane_graph.get(lane_id, {}).get("start")
            if start is not None:
                start_points_x.append(start[0])
                start_points_y.append(start[1])

        # Add polygons
        collection = PatchCollection(patches, facecolors=colors, edgecolor="black", alpha=0.6)
        ax.add_collection(collection)

        # Add start points
        ax.scatter(start_points_x, start_points_y, c='red', s=20, label='Start Points', zorder=10)

        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title("Lane Polygons Colored by Lane Type")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend()

        return fig, ax
    
    def plot_all_lane_polygons(self, ax=None):
            """
            Plot all lane polygons with unique colors.
            If an Axes is passed, draw on it. Otherwise, create a new one.
            Returns: (fig, ax) so caller can handle plt.show().
            """
            # Create figure and axes if not provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
            else:
                fig = ax.figure

            patches = []
            colors = []

            lane_ids = list(self.lane_polygons.keys())
            num_lanes = len(lane_ids)
            cmap = cm.get_cmap('tab20', num_lanes)

            for idx, lane_id in enumerate(lane_ids):
                poly = self.lane_polygons[lane_id]
                if poly is None or not poly.is_valid:
                    continue

                coords = list(poly.exterior.coords)
                patch = MplPolygon(coords, closed=True)
                patches.append(patch)
                colors.append(cmap(idx % 20))  # wrap to avoid index error

                # Label each polygon
                centroid = poly.centroid
                ax.text(
                    centroid.x, centroid.y, str(lane_id),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color='black'
                )

            collection = PatchCollection(patches, facecolors=colors, edgecolor="black", alpha=0.6)
            ax.add_collection(collection)

            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_title("All Lane Polygons with Unique Colors")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)

            return fig, ax
            
    def plot_lane_neighbors(self,ax, target_lane_id, neighbor_dict):
        '''plots target id lane in red and neighbor ids of that lane in green.
        since theres multiple methods of computing neighbors,
        this method can be called with any of them as arg'''

        ax.clear()
        # Gather lane categories
        target_lane = {target_lane_id}
        neighbor_ids = neighbor_dict.get(target_lane_id, [])
        # Define colors
        colors = {
            "target": "red",
            "neighbor": "green",
        }

        ax.clear()
        
        def get_color(lid):
            if lid in target_lane:
                return colors["target"]
            elif lid in neighbor_ids:
                return colors["neighbor"]
            else:
                return "gray"

        # Draw lane arrows
        for lid, props in self.lane_graph.items():
            start = props["start"]
            end = props["end"]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            color = get_color(lid)
            if color == "gray":
                lw = 1
            else:
                lw = 2
                
            ax.arrow(start[0], start[1], dx, dy,
                        head_width=0.3, length_includes_head=True,
                        fc=color, ec=color, alpha=0.8, linewidth = lw)

            #ax.text(start[0], start[1], str(lid), fontsize=8, color='black')
            midpoint = (start + end) / 2 # plot lane ids at middle of edge instead of node  
            ax.text(midpoint[0], midpoint[1], str(lid), fontsize=8, color='black')

        ax.set_title(f"Lane ID: {target_lane_id} (Red), Neighbors (Green)")
        ax.axis('equal')
        ax.grid(True)
        plt.draw()

    def plot_segment_buffers_with_neighbors(self, lane_id, buffer_width_percentage=1.2):
        '''debugging visualization of all buffered neighbor
            polygons and segments for a given lane id'''

        segments = self.lane_segments.get(lane_id)
        if not segments:
            print(f"No segments found for lane_id {lane_id}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Buffered Segments and Neighbors for Lane {lane_id}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        buffer_patches = []         # Target lane segment buffers
        neighbor_patches = []       # Neighbor segment buffers
        visited_neighbors = set()
        all_bounds = []

        # Buffer size for the main lane
        buffer_size = self.lane_graph[lane_id]["buffer_size"] * buffer_width_percentage

        # Plot main lane segments with buffer
        for idx, segment in enumerate(segments):
            x, y = segment.xy
            ax.plot(x, y, color='blue', linewidth=2, label='Lane Segment' if idx == 0 else "")
            ax.text(x[0], y[0], f"{lane_id}:{idx}", fontsize=8, color='blue')
            all_bounds.append(segment.bounds)

            # Buffer
            buf = segment.buffer(buffer_size)
            if isinstance(buf, Polygon):
                poly = MplPolygon(list(buf.exterior.coords), closed=True)
                buffer_patches.append(poly)
                all_bounds.append(buf.bounds)

            # Plot neighbors
            neighbors = self.neighbor_info.get(lane_id, {}).get(idx, [])
            for side, neighbor_id, neighbor_seg_idx, direction in neighbors:
                if (neighbor_id, neighbor_seg_idx) in visited_neighbors:
                    continue
                visited_neighbors.add((neighbor_id, neighbor_seg_idx))

                neighbor_segments = self.lane_segments.get(neighbor_id, [])
                if 0 <= neighbor_seg_idx < len(neighbor_segments):
                    neighbor_seg = neighbor_segments[neighbor_seg_idx]
                    nx, ny = neighbor_seg.xy
                    color = 'red' if side == 'left' else 'green'
                    label = 'Left Neighbor' if side == 'left' else 'Right Neighbor'
                    ax.plot(nx, ny, color=color, linewidth=2, linestyle='--', label=label)
                    ax.text(nx[0], ny[0], f"{neighbor_id}:{neighbor_seg_idx}", fontsize=8, color=color)
                    all_bounds.append(neighbor_seg.bounds)

                    # Buffer for neighbor using its own lane's buffer size
                    neighbor_buffer_size = self.lane_graph[neighbor_id]["buffer_size"] * buffer_width_percentage
                    neighbor_buf = neighbor_seg.buffer(neighbor_buffer_size)
                    if isinstance(neighbor_buf, Polygon):
                        poly = MplPolygon(list(neighbor_buf.exterior.coords), closed=True)
                        neighbor_patches.append((poly, color))
                        all_bounds.append(neighbor_buf.bounds)

        # Add target lane segment buffers
        patch_collection = PatchCollection(buffer_patches, facecolor='orange', edgecolor='red', alpha=0.3, zorder=2)
        ax.add_collection(patch_collection)

        # Add neighbor buffers with matching side color
        for poly, color in neighbor_patches:
            patch = PatchCollection([poly], facecolor=color, edgecolor=color, alpha=0.2, zorder=1)
            ax.add_collection(patch)


        ax.grid(True)
        ax.set_aspect('equal')
        plt.show()
    
    def plot_neighbors_for_lane(self,ax, lane_id):
        """
        Visualize a single lane's segments and its left/right neighbors with direction-based color coding.

        - Blue: segments of the specified lane
        - Green: left neighbors with same direction
        - Cyan: left neighbors with opposite direction
        - Red: right neighbors with same direction
        - Orange: right neighbors with opposite direction
        """
        ax.clear()
        print(lane_id)
        if lane_id not in self.neighbor_info:
            print(f"No neighbor information found for lane {lane_id}")
            return

        # Color map for (side, direction) combinations
        color_map = {
            ('left', 'same'): 'green',
            ('left', 'opposite'): 'cyan',
            ('right', 'same'): 'red',
            ('right', 'opposite'): 'orange',
        }
        
        # Step 2: Plot all lanes
        for _lane_id, segments in self.lane_segments.items():
            for seg in segments:
                if seg.is_empty or not seg.is_valid or len(seg.coords) == 0:
                    continue
                xs, ys = seg.xy
                color = 'gray'
                lw = 1

                # Plot the segment
                ax.plot(xs, ys, color=color, linewidth=lw)

            # Arrow for direction (optional, using mid-segment)
            mid_idx = len(segments) // 2
            if len(segments) >= 2 and mid_idx + 1 < len(segments):
                seg_mid = segments[mid_idx]
                seg_next = segments[mid_idx + 1]

                x0, y0 = seg_mid.centroid.x, seg_mid.centroid.y
                x1, y1 = seg_next.centroid.x, seg_next.centroid.y
                dx, dy = x1 - x0, y1 - y0

                ax.arrow(x0, y0, dx, dy,
                        head_width=0.5, head_length=0.7, fc=color, ec=color,
                        length_includes_head=True)



            # Label the lane ID
            if segments:
                mid_seg = segments[len(segments)//2]
                x, y = mid_seg.centroid.x, mid_seg.centroid.y
                ax.text(x, y, str(_lane_id), fontsize=6, color='black')

        for _lane_id, props in self.lane_graph.items():
            start = props["start"]
            end = props["end"]
            ax.plot(start[0],start[1], marker='o', color='blue', markersize=4)
            ax.plot(end[0],end[1], marker='o', color='red', markersize=4)
        # 2. Plot the target lane's segments in blue
        for seg_idx, seg in enumerate(self.lane_segments.get(lane_id, [])):
            x, y = seg.xy
            ax.plot(x, y, color='blue', linewidth=2)

        # 3. Plot the neighbor segments with appropriate color
        segment_neighbors = self.neighbor_info.get(lane_id, {})
        for seg_idx, neighbors in segment_neighbors.items():
            for side, neighbor_id, neighbor_seg_idx, direction in neighbors:
                if direction == 'angled':
                    continue  # Skip angled neighbors

                color = color_map.get((side, direction), 'gray')  # Fallback to gray if unexpected

                neighbor_seg = self.lane_segments[neighbor_id][neighbor_seg_idx]
                x, y = neighbor_seg.xy
                ax.plot(x, y, color=color, linewidth=2)

        ax.set_title(f"Lane ID: {lane_id} (Red), Neighbors (Green)")
        ax.axis('equal')
        ax.grid(True)
        plt.draw()

    def plot_neighbor_chain(self, lane_chain, _enforce_continuity = False, _neighbor_direction = 'any' ):
        # Get the longest chains for both sides
        chain_left = self.longest_neighbor_chain_one_side(
            lane_chain,
            'left',
            enforce_continuity= _enforce_continuity,
            neighbor_direction= _neighbor_direction
            )
        chain_right = self.longest_neighbor_chain_one_side(
            lane_chain, 
            'right', 
            enforce_continuity= _enforce_continuity, 
            neighbor_direction= _neighbor_direction
            )
        
        # Extract sets for comparisons
        segments_left = set((lane_id, seg_idx) for lane_id, seg_idx, _, _ in chain_left)
        segments_right = set((lane_id, seg_idx) for lane_id, seg_idx, _, _ in chain_right)

        # Identify overlapping target segments
        overlapping_segments = segments_left & segments_right
        unique_left = segments_left - overlapping_segments
        unique_right = segments_right - overlapping_segments

        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. Plot all lane segments in light grey
        for lane_id, segments in self.lane_segments.items():
            for seg_idx, linestring in enumerate(segments):
                x, y = linestring.xy
                ax.plot(x, y, color='lightgrey', linewidth=1)

        # 2. Plot target lane_chain in blue
        for lane_id in lane_chain:
            for seg_idx, linestring in enumerate(self.lane_segments.get(lane_id, [])):
                x, y = linestring.xy
                ax.plot(x, y, color='blue', linewidth=2)

        # 3. Plot overlapping segments (left + right) in purple
        for lane_id, seg_idx in overlapping_segments:
            linestring = self.lane_segments[lane_id][seg_idx]
            x, y = linestring.xy
            ax.plot(x, y, color='purple', linewidth=4)

        # 4. Plot left-only segments in orange
        for lane_id, seg_idx in unique_left:
            linestring = self.lane_segments[lane_id][seg_idx]
            x, y = linestring.xy
            ax.plot(x, y, color='orange', linewidth=3)

        # 5. Plot right-only segments in cyan
        for lane_id, seg_idx in unique_right:
            linestring = self.lane_segments[lane_id][seg_idx]
            x, y = linestring.xy
            ax.plot(x, y, color='cyan', linewidth=3)

        # 6. Plot neighbor segments for left chain (green)
        for _, _, neighbor_id, neighbor_seg_idx in chain_left:
            linestring = self.lane_segments[neighbor_id][neighbor_seg_idx]
            x, y = linestring.xy
            ax.plot(x, y, color='green', linewidth=2, linestyle='--')

        # 7. Plot neighbor segments for right chain (blue)
        for _, _, neighbor_id, neighbor_seg_idx in chain_right:
            linestring = self.lane_segments[neighbor_id][neighbor_seg_idx]
            x, y = linestring.xy
            ax.plot(x, y, color='blue', linewidth=2, linestyle='--')

        ax.set_aspect('equal')
        ax.set_title("Longest Neighbor Chains (Left & Right)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_longest_neighbor_chain(self, lane_chain, neighbor_chain, side):
        """
        Plot the lane chain and its longest neighbor chain on the given side.

        Args:
            lane_chain (list): List of lane IDs in the main chain
            neighbor_chain (list): List of tuples (lane_id, segment_index, neighbor_lane_id)
            side (str): 'left' or 'right', used for coloring
        """
        colors = {"left": "green", "right": "red"}
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot all lanes in light gray for context
        for lid, segments in self.lane_segments.items():
            for seg in segments:
                x, y = seg.xy
                ax.plot(x, y, color="lightgray", linewidth=1, zorder=1)

        # Plot the main lane chain in blue
        for lane_id in lane_chain:
            segments = self.lane_segments.get(lane_id, [])
            for seg in segments:
                x, y = seg.xy
                ax.plot(x, y, color="blue", linewidth=2, zorder=3)

        # Plot neighbors in the longest chain in side color
        neighbor_color = colors.get(side, "orange")
        for lane_id, seg_idx, neighbor_id in neighbor_chain:
            # Plot the neighbor segments
            neighbor_segments = self.lane_segments.get(neighbor_id, [])
            for nseg in neighbor_segments:
                x, y = nseg.xy
                ax.plot(x, y, color=neighbor_color, linewidth=2, zorder=4)

            # Optionally, draw lines connecting the centroids of the main segment and neighbor segment
            main_seg = self.lane_segments[lane_id][seg_idx]
            main_x, main_y = main_seg.centroid.xy

            # Find the neighbor segments intersecting buffered main segment to highlight connection
            buf = main_seg.buffer(self.lane_graph[lane_id]["buffer_size"] * 1.5)
            for nseg in neighbor_segments:
                if buf.intersects(nseg):
                    n_x, n_y = nseg.centroid.xy
                    ax.plot([main_x[0], n_x[0]], [main_y[0], n_y[0]], color=neighbor_color, linestyle="--", zorder=5)
                    break  # Just connect one matching segment for clarity

        ax.set_aspect("equal")
        ax.set_title(f"Longest {side.capitalize()} Neighbor Chain for Lane Chain")
        plt.show()

    def plot_neighbor_chain_with_both_sides(self, chain, neighbor_debug = False):
        
        def get_segment_position(lane_id, seg_idx):
            segments = self.lane_segments.get(lane_id, [])
            if 0 <= seg_idx < len(segments):
                return segments[seg_idx]
            else:
                return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
                # Plot all lanes in light gray for context
        for lid, segments in self.lane_segments.items():            
            for seg in segments:
                x, y = seg.xy
                if any(lid == item[0] for item in chain):
                    ax.plot(x, y, color="pink", linewidth=2, zorder=1)
                else:
                    ax.plot(x, y, color="lightgray", linewidth=1, zorder=1)

        # Plot neighboring segments colored by side with labels
        for lane_id, seg_idx, left_n, right_n in chain:
            # Plot the main segment point and label
            pos = get_segment_position(lane_id, seg_idx)
            if pos is None:
                continue
            x, y = pos.interpolate(0.5, normalized=True).xy
            x, y = x[0], y[0]

            # Black circle and label for main segment
            ax.plot(x, y, 'ko')
            ax.text(x, y + 0.5, f"{lane_id}:{seg_idx}", fontsize=8, ha='center', va='bottom')

            # Plot left neighbor segments in red + labels
            left_segments = self.lane_segments.get(left_n, [])
            for i, seg in enumerate(left_segments):
                x_seg, y_seg = seg.xy
                ax.plot(x_seg, y_seg, color='red', linewidth=2, alpha=0.7, zorder=2)
                # Label each segment at midpoint
                if neighbor_debug:
                    mx, my = seg.interpolate(0.5, normalized=True).xy
                    ax.text(mx[0], my[0] + 0.3, f"{left_n}:{i}", color='red', fontsize=7, ha='center', va='bottom')

            # Plot right neighbor segments in green + labels
            right_segments = self.lane_segments.get(right_n, [])
            for i, seg in enumerate(right_segments):
                x_seg, y_seg = seg.xy
                ax.plot(x_seg, y_seg, color='green', linewidth=2, alpha=0.7, zorder=2)
                # Label each segment at midpoint
                if neighbor_debug:
                    mx, my = seg.interpolate(0.5, normalized=True).xy
                    ax.text(mx[0], my[0] + 0.3, f"{right_n}:{i}", color='green', fontsize=7, ha='center', va='bottom')

        ax.set_title("Segments with Both Left and Right Neighbors")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        plt.axis('equal')
        plt.show()

    def plot_lane_with_neighbors(self, target_lane_id):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all lanes in light gray
        for lid, segments in self.lane_segments.items():
            for seg in segments:
                x, y = seg.xy
                ax.plot(x, y, color="lightgray", linewidth=1, zorder=1)

        # Plot the target lane in blue
        for seg in self.lane_segments[target_lane_id]:
            x, y = seg.xy
            ax.plot(x, y, color="blue", linewidth=2, zorder=2)

        # Plot neighbors
        segment_neighbors = self.neighbor_info.get(target_lane_id, {})
        colors = {"left": "green", "right": "red"}
        target_radius = self.lane_graph[target_lane_id]["buffer_size"] * 1.5


        for idx, seg in enumerate(self.lane_segments[target_lane_id]):
            neighbors = segment_neighbors.get(idx, [])
            x0, y0 = seg.centroid.xy

            for side, neighbor_id, direction in neighbors:
                #print(idx)
                #print('target_lane_id: ', target_lane_id)
                #print("side ", side, ' neighbor_id ', neighbor_id)

                if direction == 'angled':
                    continue
                color = colors.get(side, "orange")
                # Draw a line from center of seg to center of neighbor seg(s)
                for nseg in self.lane_segments[neighbor_id]:
                    neighbor_radius = self.lane_graph[neighbor_id]["buffer_size"] * 1.5

                    if seg.buffer(target_radius).intersects(nseg.buffer(neighbor_radius)):
                        xn, yn = nseg.centroid.xy
                        ax.plot([x0[0], xn[0]], [y0[0], yn[0]], color=color, linestyle="--", zorder=3)
                        ax.annotate(
                            f"{side[0].upper()}:{neighbor_id}",
                            xy=(xn[0], yn[0]),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                        )

        ax.set_aspect("equal")
        ax.set_title(f"Lane {target_lane_id} with Left (green) / Right (red) Neighbors")
        return fig, ax
    
    def plot_lane_predecessors_and_successors(self, lane_id):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Define colors
        colors = {
            "target": "red",
            "successor": "green",
            "predecessor": "purple"
        }

        # Gather lane categories
        target_lane = {lane_id}
        successors = set(self.get_successors(lane_id))
        predecessors = set(self.get_successors(lane_id))

        def get_color(lid):
            if lid in target_lane:
                return colors["target"]
            elif lid in successors:
                return colors["successor"]
            elif lid in predecessors:
                return colors["predecessor"]
            else:
                return "gray"

        # Draw lane arrows
        for lid, props in self.lane_graph.items():
            start = props["start"]
            end = props["end"]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            color = get_color(lid)

            ax.arrow(start[0], start[1], dx, dy,
                    head_width=0.3, length_includes_head=True,
                    fc=color, ec=color, alpha=0.8)


            midpoint = (start + end) / 2 # plot lane ids at middle of edge  
            ax.text(midpoint[0], midpoint[1], str(lid), fontsize=8, color='black')

        legend_patches = [
            mpatches.Patch(color=colors["target"], label='Target Lane'),
            mpatches.Patch(color=colors["neighbor"], label='Neighbor Lane'),
            mpatches.Patch(color=colors["successor"], label='Successor Lane'),
            mpatches.Patch(color="gray", label='Unrelated Lane')
        ]
        ax.legend(handles=legend_patches, loc='upper right')

        ax.set_aspect('equal')
        ax.set_title(f"Lane successors/predecessors for Lane ID {lane_id}")
        ax.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_branch_vs_root_sequence_polygons(self):
        """
        Creates a side-by-side plot of root-started vs branch-started sequences.
        Does NOT call plt.show() so multiple plots can be composed.
        """
        root_seqs = [s for s in self.sequences if not s["is_branch_root"]]
        branch_seqs = [s for s in self.sequences if s["is_branch_root"]]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        titles = ["Root-Started Sequences", "Branch-Started Sequences"]

        for ax, group, title in zip(axes, [root_seqs, branch_seqs], titles):
            cmap = cm.get_cmap("tab20", len(group))
            for idx, seq in enumerate(group):
                coords = list(seq["polyline"].coords)
                xs, ys = zip(*coords)
                color = cmap(idx % 20)
                # Plot the polyline
                ax.plot(xs, ys, color=color, linewidth=2)

                # Start point
                ax.plot(xs[0], ys[0], 'o', color=color, markersize=5)
                ax.text(xs[0], ys[0], str(seq["lane_ids"][0]), fontsize=8)

                # Add a direction arrow at the midpoint of the sequence
                if len(xs) >= 2:
                    for i in range(len(xs) - 1):
                        dx = xs[i + 1] - xs[i]
                        dy = ys[i + 1] - ys[i]
                        ax.arrow(xs[i], ys[i], dx, dy,
                                head_width=0.3, head_length=0.6,
                                fc=color, ec=color, length_includes_head=True)
                        break  # remove break to draw multiple arrows along the path

            ax.set_aspect("equal")
            ax.set_title(title)
            ax.grid(True)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        plt.tight_layout()
        return fig, axes
    
    
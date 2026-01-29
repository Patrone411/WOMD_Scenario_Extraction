from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Polygon, Point


class EnvironmentElementsWaymo:
    """
    Minimal module to extract:
      - crosswalks: list[Polygon]
      - controlled_lanes: list[Polygon]
      - traffic_lights:
          * 'points': list[Point]
          * 'traffic_lights_state': np.ndarray [T, N]
          * 'traffic_lights_lane_id': np.ndarray [T, N]

    Expected `parsed` dict (NumPy arrays or tensors with .numpy()):
      roadgraph:
        'roadgraph_samples/xyz'  -> [P, 3]
        'roadgraph_samples/type' -> [P, 1] or [P]
        'roadgraph_samples/id'   -> [P, 1] or [P]
      traffic lights:
        EITHER single-shot:
          'traffic_light_state/id'    -> [T, N]
          'traffic_light_state/valid' -> [T, N]
          'traffic_light_state/state' -> [T, N]
          'traffic_light_state/x'     -> [T, N]
          'traffic_light_state/y'     -> [T, N]
        OR Waymo-style split (will be concatenated along time):
          'traffic_light_state/past/*', '.../current/*', '.../future/*' (same shapes as above)
    """

    # Waymo type codes we actually need here
    CROSSWALK_CODE = 18
    SPEED_BUMP_CODE = 19  # ignored, but filtered out of lane-building
    DEFAULT_LANE_WIDTH = 3.5  # meters

    def __init__(self, parsed: Dict, lane_width: float = DEFAULT_LANE_WIDTH) -> None:
        self.parsed = parsed
        self.lane_width = float(lane_width)

        # outputs
        self._crosswalks: List[Polygon] = []
        self._controlled_lane_polys: List[Polygon] = []
        self.traffic_lights: Dict[str, object] = {
            "points": [],                     # list[Point]
            "traffic_lights_state": None,     # np.ndarray [T, N]
            "traffic_lights_lane_id": None,   # np.ndarray [T, N]
        }

        # internals
        self._rg_xyz: np.ndarray = None
        self._rg_type: np.ndarray = None
        self._rg_id: np.ndarray = None

    # ----------------------- public API -----------------------

    def __call__(self, eval_mode: bool = False):
        """Build all requested elements. `eval_mode=True` treats polyline coords as polygon shells."""
        self._parse_inputs()
        self._build_crosswalks(eval_mode=eval_mode)
        self._build_traffic_lights()
        self._build_controlled_lane_polys(eval_mode=eval_mode)
        return self

    def get_other_object(self, key: str):
        if key != "cross_walk":
            raise KeyError("Only 'cross_walk' is supported in this minimal module.")
        return self._crosswalks

    def get_controlled_lane(self):
        return self._controlled_lane_polys

    # ----------------------- parsing helpers -----------------------

    @staticmethod
    def _to_numpy(arr):
        return arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)

    def _concat_or_plain(self, base: str):
        """
        Return [T, N] array for a traffic-light field `base` ('id', 'valid', 'state', 'x', 'y').
        Supports either single-shot keys or past/current/future triplets.
        """
        p = self.parsed
        split_keys = [
            f"traffic_light_state/past/{base}",
            f"traffic_light_state/current/{base}",
            f"traffic_light_state/future/{base}",
        ]
        single_key = f"traffic_light_state/{base}"

        if all(k in p for k in split_keys):
            parts = [self._to_numpy(p[k]) for k in split_keys]
            return np.concatenate(parts, axis=0)
        elif single_key in p:
            return self._to_numpy(p[single_key])
        else:
            # allow empty, but keep consistent 2-D shape
            return np.empty((1, 0), dtype=float if base in ("x", "y") else int)

    def _parse_inputs(self):
        p = self.parsed
        self._rg_xyz = self._to_numpy(p["roadgraph_samples/xyz"])        # [P, 3]
        self._rg_type = self._to_numpy(p["roadgraph_samples/type"]).reshape(-1)  # [P]
        self._rg_id = self._to_numpy(p["roadgraph_samples/id"]).reshape(-1)      # [P]

    # ----------------------- builders -----------------------

    def _build_crosswalks(self, eval_mode: bool):
        self._crosswalks.clear()
        mask = (self._rg_type == self.CROSSWALK_CODE)
        if not np.any(mask):
            return

        pts = self._rg_xyz[mask, :2]        # [K, 2]
        ids = self._rg_id[mask]             # [K]
        for oid in np.unique(ids):
            coords = pts[ids == oid, :]
            if len(coords) == 0:
                continue
            # Crosswalk samples are usually polygon boundary arcs; build directly.
            try:
                poly = Polygon(coords) if eval_mode else Polygon(coords)
                if not poly.is_empty:
                    self._crosswalks.append(poly)
            except Exception:
                # Fall back to a thin buffer around a LineString if polygon fails
                if len(coords) >= 2:
                    self._crosswalks.append(LineString(coords).buffer(0.2))
                else:
                    self._crosswalks.append(Point(coords[0]).buffer(0.2))

    def _build_traffic_lights(self):
        # Shapes: [T, N]
        tl_id = self._concat_or_plain("id")
        tl_valid = self._concat_or_plain("valid")
        tl_state = self._concat_or_plain("state")
        tl_x = self._concat_or_plain("x")
        tl_y = self._concat_or_plain("y")

        # Store raw arrays
        self.traffic_lights["traffic_lights_lane_id"] = tl_id
        self.traffic_lights["traffic_lights_state"] = tl_state

        # Build one averaged Point per physical light (across time where valid==1)
        self.traffic_lights["points"] = []
        if tl_id.size == 0:
            return

        # transpose to iterate over lights: [N, T]
        x_TN = tl_x.T
        y_TN = tl_y.T
        v_TN = tl_valid.T

        for x_hist, y_hist, v_hist in zip(x_TN, y_TN, v_TN):
            valid_idx = np.where(v_hist == 1)[0]
            if valid_idx.size == 0:
                continue
            px = float(np.mean(x_hist[valid_idx]))
            py = float(np.mean(y_hist[valid_idx]))
            self.traffic_lights["points"].append(Point(px, py))

    def _build_controlled_lane_polys(self, eval_mode: bool):
        """Build polygons only for lane_ids referenced by ANY valid traffic light state."""
        self._controlled_lane_polys.clear()

        tl_id = self.traffic_lights["traffic_lights_lane_id"]
        tl_valid = self._concat_or_plain("valid")  # need valid to filter ids

        if tl_id is None or tl_id.size == 0:
            return

        controlled_ids = np.unique(tl_id[tl_valid == 1])
        if controlled_ids.size == 0:
            return

        # Exclude crosswalk/speed bump samples from lane reconstruction
        lane_mask = (self._rg_type != self.CROSSWALK_CODE) & (self._rg_type != self.SPEED_BUMP_CODE)
        lane_pts = self._rg_xyz[lane_mask, :2]
        lane_ids = self._rg_id[lane_mask]

        for lid in controlled_ids:
            coords = lane_pts[lane_ids == lid, :]
            if coords.shape[0] == 0:
                continue
            if eval_mode:
                # Treat coords as polygon shell if they already define an area
                try:
                    poly = Polygon(coords)
                    if not poly.is_empty:
                        self._controlled_lane_polys.append(poly)
                        continue
                except Exception:
                    pass  # fall back to buffered line below

            # Buffer a polyline (or point) into an area
            if coords.shape[0] >= 2:
                self._controlled_lane_polys.append(LineString(coords).buffer(self.lane_width / 2.0))
            else:
                self._controlled_lane_polys.append(Point(coords[0]).buffer(self.lane_width / 2.0))


# ----------------------- example usage -----------------------
# env = EnvironmentElementsWaymo(parsed_dict)
# env(eval_mode=False)  # or True if your coords are polygon shells
# crosswalks = env.get_other_object('cross_walk')          # list[Polygon]
# controlled = env.get_controlled_lane()                   # list[Polygon]
# tl_points = env.traffic_lights['points']                 # list[Point]
# tl_state = env.traffic_lights['traffic_lights_state']    # np.ndarray [T, N]
# tl_lane_id = env.traffic_lights['traffic_lights_lane_id']# np.ndarray [T, N]

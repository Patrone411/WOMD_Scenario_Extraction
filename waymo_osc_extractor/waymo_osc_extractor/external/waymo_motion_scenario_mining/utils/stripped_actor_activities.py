from typing import Dict, List, Tuple
import numpy as np
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely.geometry import Polygon

from .helpers.create_rect_from_file import get_agent_list, actor_creator
from .environ_elements import EnvironmentElementsWaymo
from .parameters.tag_parameters import actor_dict, lane_key


def build_lane_index(env) -> Tuple[List[Polygon], List[float], List[str], STRtree]:
    """Flatten all lane polygons across lane_key into a single index."""
    polys, areas, ids = [], [], []
    keys = ("freeway", "surface_street", "bike_lane")  # 1,2,3
    for key in keys:
        lane_polys = env.get_lane(key)
        lane_ids   = env.lane_id[key]
        for p, lid in zip(lane_polys, lane_ids):
            polys.append(p)
            areas.append(p.area)
            ids.append(str(lid))
    tree = STRtree(polys)
    return polys, areas, ids, tree


def per_actor_minimal(parsed, eval_mode=False):
    """
    Returns dict: actor_key -> {
        'x': np.ndarray, 'y': np.ndarray, 'yaw': np.ndarray,
        'long_v': np.ndarray, 'lane_id': List[str|None], 'valid': (v0,v1)
    }
    """
    # --- map geometry ---
    env = EnvironmentElementsWaymo(parsed)
    env.create_polygon_set(eval_mode=eval_mode)
    lane_polys, lane_areas, lane_ids, lane_tree = build_lane_index(env)

    # cache prepared geometries for faster intersects & a geom->index map
    prepared = [prep(p) for p in lane_polys]
    geom2idx = {id(g): i for i, g in enumerate(lane_polys)}

    out: Dict[str, dict] = {}

    for actor_type in actor_dict:
        agent_type = actor_dict[actor_type]
        ids = get_agent_list(agent_type, parsed, eval_mode=eval_mode)
        ids = [int(ids.item())] if getattr(ids, "shape", ()) == () else ids.tolist()

        for agent in ids:
            st, _ = actor_creator(agent_type, int(agent), parsed, eval_mode=eval_mode)
            _ = st.data_preprocessing()
            v0, v1 = st.get_validity_range()
            if v0 == v1:
                continue

            # basic kinematics
            x    = np.asarray(st.kinematics['x'])
            y    = np.asarray(st.kinematics['y'])
            yaw  = np.asarray(st.kinematics['bbox_yaw'])
            vx   = np.asarray(st.kinematics['velocity_x'])
            vy   = np.asarray(st.kinematics['velocity_y'])

            # longitudinal velocity in body frame
            long_v = vx * np.cos(yaw) + vy * np.sin(yaw)

            # bbox polygons (per timestep) – used for lane assignment
            bboxes = st.polygon_set()

            # per-timestep lane id by max intersection area
            lane_id_series: List[str] = [None] * len(x)
            for t in range(v0, v1 + 1):
                bb = bboxes[t]

                # shortlist candidates via STRtree
                cands = lane_tree.query(bb)
                # Normalize to a Python list; STRtree may return a NumPy array
                if isinstance(cands, np.ndarray):
                    cands = cands.tolist()
                # empty guard
                if len(cands) == 0:
                    lane_id_series[t] = None
                    continue

                # normalize candidates to integer indices (robust across shapely/pygeos)
                first = cands[0]
                if isinstance(first, (int, np.integer)):
                    idxs = cands  # query returned indices directly
                else:
                    # query returned geometry objects
                    idxs = [geom2idx[id(g)] for g in cands]

                best_area, best_i = 0.0, None
                for i in idxs:
                    # STRtree checks envelopes; confirm real geometry intersection
                    if not prepared[i].intersects(bb):
                        continue
                    inter_area = bb.intersection(lane_polys[i]).area
                    if inter_area > best_area:
                        best_area, best_i = inter_area, i

                lane_id_series[t] = lane_ids[best_i] if best_i is not None else None

            key = f"{actor_type}_{agent}"
            out[key] = dict(
                x=x, y=y, yaw=yaw, long_v=long_v,
                lane_id=lane_id_series, valid=(int(v0), int(v1))
            )

    return out

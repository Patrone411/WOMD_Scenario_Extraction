import json
import os
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List, Dict, Iterator, Any

from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import (
    features_description,
    Scenario,
)
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.segment_polygon_handling import (
    plot_single_segment,
)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scenario_matching.features.providers.s3_pickle_feature_provider import (
    S3PickleFeatureProvider,
)

# ============================================================
# NEW: JSONL reader for the NEW format (type="block_hit")
# ============================================================

@dataclass
class BlockHit:
    source_uri: str
    waymo_tf_file_id: str          # "00000"
    scene_id: str                  # "104b4a3e67b26ce1"
    segment_id: str                # "seg_0"
    actors_to_roles: Dict[str, str]  # {"vehicle_10":"ego_vehicle", ...}
    intervals: Optional[List[Tuple[int, int]]] = None

    # optional fields (not required by your loop)
    block: Optional[str] = None
    osc: Optional[str] = None
    T: Optional[int] = None
    fps: Optional[int] = None
    t0: Optional[int] = None
    t1: Optional[int] = None
    windows_by_t0: Optional[Dict[str, Any]] = None


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Parse s3://bucket/path -> (bucket, path)
    """
    if not uri or "://" not in uri:
        return "", ""
    scheme, rest = uri.split("://", 1)
    if scheme.lower() != "s3":
        return "", ""
    parts = rest.split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    return bucket, path

def _tfid_scene_from_source_uri(source_uri: str) -> Tuple[str, str]:
    """
    Expects .../<tfid>/<scene>.pkl
    Example:
      s3://waymo/results/run_1331520_full/00000/104b4a3e67b26ce1.pkl
    -> tfid="00000", scene_id="104b4a3e67b26ce1"
    """
    _, path = _parse_s3_uri(source_uri)
    if not path:
        return "", ""
    parts = path.split("/")
    if len(parts) < 2:
        return "", ""
    tfid = parts[-2]
    scene_file = parts[-1]
    scene_id = os.path.splitext(scene_file)[0]
    return tfid, scene_id

def _base_prefix_from_source_uri(source_uri: str) -> Tuple[str, str]:
    """
    For provider we want:
      bucket="waymo"
      base_prefix="results/run_1331520_full/00000"
    from source_uri:
      s3://waymo/results/run_1331520_full/00000/<scene>.pkl
    """
    bucket, path = _parse_s3_uri(source_uri)
    if not path:
        return bucket, ""
    # drop the filename
    prefix = path.rsplit("/", 1)[0] if "/" in path else ""
    return bucket, prefix

def _invert_roles_map(roles: Dict[str, str]) -> Dict[str, str]:
    """
    New JSON: {"ego_vehicle": "vehicle_10", "npc": "vehicle_22"}
    Needed by plotter: {"vehicle_10": "ego_vehicle", "vehicle_22": "npc"}
    """
    out: Dict[str, str] = {}
    for role, actor_id in (roles or {}).items():
        if actor_id is None:
            continue
        out[str(actor_id)] = str(role)
    return out

def iter_jsonl_block_hits(log_path: str) -> Iterator[BlockHit]:
    """
    Reads ONLY the new JSONL format:
      {"type":"block_hit", ...}
    """
    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {log_path}:{lineno}: {e}") from e

            if rec.get("type") != "example_window":
                continue

            source_uri = rec.get("source_uri", "")
            tfid, scene_id = _tfid_scene_from_source_uri(source_uri)

            seg = str(rec.get("segment_id", ""))
            actors_to_roles = _invert_roles_map(rec.get("roles", {}))
            intervals_raw = rec.get("intervals")
            t0 = rec.get("t0")
            t1 = rec.get("t1_first")
            if t1-t0< 40:
                t1 = t0 + 40
                #t1 = rec.get("t1_greedy")
            intervals: Optional[List[Tuple[int, int]]] = None
            if isinstance(intervals_raw, list) and intervals_raw and isinstance(intervals_raw[0], list):
                intervals = [tuple(map(int, iv)) for iv in intervals_raw]

            yield BlockHit(
                source_uri=source_uri,
                waymo_tf_file_id=tfid,
                scene_id=scene_id,
                segment_id=seg,
                actors_to_roles=actors_to_roles,
                intervals=intervals,
                block=rec.get("block"),
                osc=rec.get("osc"),
                T=rec.get("T"),
                fps=rec.get("fps"),
                windows_by_t0=rec.get("windows_by_t0"),
                t0=t0,
                t1=t1
            )


# ============================================================
# Your plotting/animation utilities (unchanged)
# ============================================================

# first = red, second = green; others fall back to Matplotlib cycle
_INDEX_COLORS = ["tab:red", "tab:green"]

def _build_active_mask(T: int, intervals: Optional[Iterable[Tuple[int, int]]]) -> np.ndarray:
    """Boolean mask of length T: True where any [t0,t1] interval is active (inclusive)."""
    m = np.zeros(int(T), dtype=bool)
    if not intervals:
        m[:] = True
        return m
    for t0, t1 in intervals:
        a = max(0, int(t0))
        b = max(a, int(t1))
        if a >= T:
            continue
        m[a:min(b + 1, T)] = True
    return m

def _coerce_series(v, dtype=float):
    """Return a 1-D numpy array. Scalars -> length-1 arrays. None -> empty array."""
    if v is None:
        return np.array([], dtype=dtype)
    a = np.asarray(v, dtype=dtype)
    if a.ndim == 0:
        a = a.reshape(1)
    return a

def normalize_actor_data(actor_data: dict, estimate_yaw_if_missing=True):
    """
    Ensures actor_data[aid] has x,y,yaw,present as 1-D arrays (same length after trim).
    If yaw missing/NaN and estimate_yaw_if_missing, yaw is derived from dx/dy.
    """
    norm = {}
    for aid, d in actor_data.items():
        x = _coerce_series(d.get("x"))
        y = _coerce_series(d.get("y"))
        yaw = _coerce_series(d.get("yaw"))
        present = _coerce_series(d.get("present"), dtype=float)
        if present.size == 0:
            present = np.ones_like(x, dtype=float)

        T = max(0, min(x.size, y.size, present.size if present.size else x.size))
        x, y, present = x[:T], y[:T], present[:T]
        if yaw.size == 0:
            yaw = np.full(T, np.nan, dtype=float)
        else:
            yaw = yaw[:T]

        if estimate_yaw_if_missing and not np.isfinite(yaw).any():
            if T >= 2:
                vx = np.gradient(x)
                vy = np.gradient(y)
                yaw = np.arctan2(vy, vx)
            else:
                yaw = np.zeros(T, dtype=float)

        if T == 0:
            continue

        out = {"x": x, "y": y, "yaw": yaw, "present": present}
        if "speed" in d and d["speed"] is not None:
            out["speed"] = _coerce_series(d["speed"])
        if "role" in d:
            out["role"] = d["role"]
        norm[aid] = out
    return norm

def _color_for_index(i: int) -> str:
    if i < len(_INDEX_COLORS):
        return _INDEX_COLORS[i]
    return f"C{(i % 10)}"

def _finite_mask(*arrays):
    m = None
    for a in arrays:
        a = np.asarray(a, dtype=float)
        cur = np.isfinite(a)
        m = cur if m is None else (m & cur)
    return m

def animate_trajectories(
    fig,
    ax,
    actor_data: Dict[str, dict],
    colors: Optional[Dict[str, str]] = None,
    interval_ms: int = 60,
    show_heading: bool = True,
    arrow_len: float = 3.0,
    speed_scale: Optional[float] = None,
    arrow_max: float = 8.0,
    active_intervals: Optional[List[Tuple[int, int]]] = None,
    restrict_to_active: bool = True,
    t0: int = None,
    t1: int = None, 
):
    """Animated trajectories with an arrow at the current tip."""
    actor_data = normalize_actor_data(actor_data)

    max_T = max(len(d["x"]) for d in actor_data.values()) if actor_data else 0
    active_mask = _build_active_mask(max_T, active_intervals)
    frames_arr = np.where(active_mask)[0] if restrict_to_active else np.arange(max_T)
    # Restrict to [t0, t1]
    if t0 is not None:
        frames_arr = frames_arr[frames_arr >= t0] if restrict_to_active else np.arange(max_T)
    if t1 is not None:
        frames_arr = frames_arr[frames_arr <= t1] if restrict_to_active else np.arange(max_T)
    
    print(f"t0: {t0}, t1: {t1}")

    
    color_for_aid = {aid: (colors or {}).get(aid, _color_for_index(i))
                     for i, aid in enumerate(actor_data.keys())}

    artists = {}
    for aid, d in actor_data.items():
        color = color_for_aid[aid]
        ln, = ax.plot([], [], lw=2, color=color, label=f"{d.get('role','actor')} ({aid})")
        pt, = ax.plot([], [], marker="o", ms=6, color=color)
        qv = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1.0, color=color, width=0.003) \
             if (show_heading and d.get("yaw") is not None) else None
        artists[aid] = (ln, pt, qv)

    def init():
        arts = []
        for ln, pt, qv in artists.values():
            ln.set_data([], [])
            pt.set_data([], [])
            if qv is not None:
                qv.set_offsets(np.array([[np.nan, np.nan]]))
                qv.set_UVC([], [])
            arts.extend([ln, pt] + ([qv] if qv is not None else []))
        return arts

    def update(cur_t: int):
        arts = []
        k = np.searchsorted(frames_arr, cur_t, side="right")
        used_frames = frames_arr[:k]

        for aid, d in actor_data.items():
            x = np.asarray(d["x"], float)
            y = np.asarray(d["y"], float)
            yaw = np.asarray(d["yaw"], float) if d.get("yaw") is not None else None
            spd = np.asarray(d["speed"], float) if d.get("speed") is not None else None

            ln, pt, qv = artists[aid]
            sel = used_frames[used_frames < x.size]

            xi = x[sel]
            yi = y[sel]
            ln.set_data(xi, yi)

            if sel.size:
                j = sel[-1]
                if np.isfinite(x[j]) and np.isfinite(y[j]):
                    pt.set_data([x[j]], [y[j]])
                else:
                    pt.set_data([], [])
            else:
                pt.set_data([], [])

            if qv is not None and yaw is not None and sel.size:
                j = sel[-1]
                if j < yaw.size and np.isfinite(x[j]) and np.isfinite(y[j]) and np.isfinite(yaw[j]):
                    L = float(arrow_len)
                    if speed_scale is not None and spd is not None and j < spd.size and np.isfinite(spd[j]):
                        L = float(np.clip(max(0.5, spd[j] * float(speed_scale)), 0.0, float(arrow_max)))
                    u = np.cos(yaw[j]) * L
                    v = np.sin(yaw[j]) * L
                    qv.set_offsets(np.array([[x[j], y[j]]]))
                    qv.set_UVC(u, v)
                else:
                    qv.set_offsets(np.array([[np.nan, np.nan]]))
                    qv.set_UVC([], [])

            arts.extend([ln, pt] + ([qv] if qv is not None else []))
        return arts

    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    anim = FuncAnimation(fig, update, frames=frames_arr.tolist(), init_func=init, blit=True, interval=interval_ms)
    return anim

def plot_trajectories_static(
    fig,
    ax,
    actor_data: Dict[str, dict],
    colors: Optional[Dict[str, str]] = None,
    show_heading: bool = True,
    arrow_len: float = 3.0,
    speed_scale: Optional[float] = None,
    arrow_max: float = 8.0,
    active_intervals: Optional[List[Tuple[int, int]]] = None,
    restrict_to_active: bool = True,
    t0: int = None,
    t1: int = None,
    mark_start_end: bool = True,
    fade_with_time: bool = False,
    lane_width: int = 5,
    marker_size: int = 10,
    end_marker_size: Optional[int] = None,
    quiver_s: float = 0.01,
    zorder_base: int = 100,
    tfid: str = None,
    scid: str = None,
    seg_id: str = None,
    # NEW: timestep labels
    label_every: int = 10,
    label_fontsize: int = 16,
    label_offset: float = 0.4,   # in data units (e.g. meters)
    label_color: str = "white",
    label_box_alpha: float = 0.35,
    label_along_normal: bool = True,
):
    """Static trajectories in one plot (same info as animation, but condensed)."""

    # Default role colors (can be overridden by passing `colors`)
    default_colors = {
        "ego_vehicle": (255 / 255, 97 / 255, 72 / 255),
        "npc": (84 / 255, 160 / 255, 230 / 255),
    }
    if colors is None:
        colors = default_colors

    actor_data = normalize_actor_data(actor_data)

    max_T = max(len(d["x"]) for d in actor_data.values()) if actor_data else 0
    active_mask = _build_active_mask(max_T, active_intervals)

    # frames to use
    frames_arr = np.where(active_mask)[0] if restrict_to_active else np.arange(max_T)
    if t0 is not None:
        frames_arr = frames_arr[frames_arr >= int(t0)]
    if t1 is not None:
        frames_arr = frames_arr[frames_arr <= int(t1)]

    # map actor id -> color (supports colors keyed by actor-id OR by role)
    color_for_aid = {}
    for i, (aid, d) in enumerate(actor_data.items()):
        role = d.get("role", "")
        color_for_aid[aid] = colors.get(aid, colors.get(role, _color_for_index(i)))

    # Plot each actor
    for aid, d in actor_data.items():
        x = np.asarray(d["x"], float)
        y = np.asarray(d["y"], float)
        yaw = np.asarray(d["yaw"], float) if d.get("yaw") is not None else None
        spd = np.asarray(d["speed"], float) if d.get("speed") is not None else None
        present = np.asarray(d.get("present", np.ones_like(x)), float)

        # limit frames to this actor length
        sel = frames_arr[frames_arr < x.size]
        if sel.size == 0:
            continue

        xi = x[sel]
        yi = y[sel]

        # respect "present" and finite values
        pm = present[sel] > 0.5
        fm = _finite_mask(xi, yi)
        keep = pm & fm
        if not np.any(keep):
            continue

        xi = xi[keep]
        yi = yi[keep]
        sel_kept = sel[keep]  # original frame indices for these points

        color = color_for_aid[aid]
        label = f"{d.get('role','actor')} ({aid})"

        # trajectory line(s)
        if fade_with_time and len(xi) >= 2:
            for i_seg in range(1, len(xi)):
                alpha_min = 0.35
                a = alpha_min + (1 - alpha_min) * (i_seg / (len(xi) - 1))
                ax.plot(
                    xi[i_seg - 1 : i_seg + 1],
                    yi[i_seg - 1 : i_seg + 1],
                    lw=lane_width,
                    color=color,
                    alpha=a,
                    zorder=zorder_base,
                )
            # one dummy line so legend works
            ax.plot([], [], lw=lane_width, color=color, label=label, zorder=zorder_base)
        else:
            ax.plot(xi, yi, lw=lane_width, color=color, label=label, zorder=zorder_base)

        # timestep labels every N frames
        if label_every is not None and label_every > 0 and len(xi) > 0:
            # bbox for readability
            bbox = dict(
                boxstyle="round,pad=0.15",
                fc=(0, 0, 0, float(label_box_alpha)),
                ec="none",
            )

            for k, t in enumerate(sel_kept):
                t_int = int(t)
                if (t_int % int(label_every)) != 0:
                    continue

                # choose offset direction
                if label_along_normal and len(xi) >= 2:
                    k0 = max(0, k - 1)
                    k1 = min(len(xi) - 1, k + 1)
                    dx = xi[k1] - xi[k0]
                    dy = yi[k1] - yi[k0]
                    n = (dx * dx + dy * dy) ** 0.5
                    if n > 1e-9:
                        nx, ny = -dy / n, dx / n  # normal direction
                    else:
                        nx, ny = 1.0, 0.0
                    tx = xi[k] + label_offset * nx
                    ty = yi[k] + label_offset * ny
                else:
                    tx = xi[k] + label_offset
                    ty = yi[k] + label_offset

                ax.text(
                    tx,
                    ty,
                    f"t{t_int}",
                    color=label_color,
                    fontsize=label_fontsize,
                    zorder=zorder_base + 5,
                    ha="left",
                    va="bottom",
                    bbox=bbox,
                )

        # mark start/end
        if mark_start_end:
            ms_end = int(end_marker_size) if end_marker_size is not None else int(marker_size * 1.4)
            ax.plot(
                [xi[0]],
                [yi[0]],
                marker="o",
                ms=marker_size,
                color=color,
                alpha=0.6,
                zorder=zorder_base + 2,
            )
            ax.plot(
                [xi[-1]],
                [yi[-1]],
                marker="o",
                ms=ms_end,
                color=color,
                zorder=zorder_base + 3,
            )

        # heading arrow at the last valid point
        if show_heading and yaw is not None and sel_kept.size:
            j = int(sel_kept[-1])
            if j < yaw.size and np.isfinite(yaw[j]) and np.isfinite(x[j]) and np.isfinite(y[j]):
                L = float(arrow_len)
                if speed_scale is not None and spd is not None and j < spd.size and np.isfinite(spd[j]):
                    L = float(np.clip(max(0.5, spd[j] * float(speed_scale)), 0.0, float(arrow_max)))

                u = np.cos(yaw[j]) * L
                v = np.sin(yaw[j]) * L

                head_scale = 4.0 * lane_width  # tune: 3..6 * lane_width

                ax.annotate(
                    "",
                    xy=(x[j] + u, y[j] + v),
                    xytext=(x[j], y[j]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=lane_width,           # same thickness as trajectory
                        color=color,
                        mutation_scale=head_scale,  # head size in points
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=zorder_base + 4,
                )

    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    if tfid is not None and seg_id is not None:
        ax.set_title(f"Actor Trajectories for Tfrecord {tfid} Scene {scid} timestep range [{t0},{t1}]")
    else:
        ax.set_title("Actor Trajectories")
    return ax

# ============================================================
# Driving code (adapted for new JSONL)
# ============================================================

tf_path = "tf_records/training_tfexample.tfrecord-00670-of-01000"
#tf_path = "tf_records/training_tfexample.tfrecord-00670-of-01000"
#log_path = "change_lane_000.jsonl"  # <-- dein neues JSONL
log_path = "cross_215.jsonl"  # <-- dein neues JSONL
log_path = "CCrb_670.jsonl"  # <-- dein neues JSONL

for hit in iter_jsonl_block_hits(log_path):
    tfid = hit.waymo_tf_file_id
    scene = hit.scene_id
    seg = hit.segment_id
    actors_to_roles = hit.actors_to_roles
    t0 = hit.t0
    t1 = hit.t1

    print("seg_id:", seg)
    print("actors_to_roles:", actors_to_roles)
    print("source_uri:", hit.source_uri)

    dataset = tf.data.TFRecordDataset(tf_path)

    active_iv = getattr(hit, "intervals", None)
    if active_iv and isinstance(active_iv[0], list):
        active_iv = [tuple(iv) for iv in active_iv]

    for raw in dataset:
        parsed = tf.io.parse_single_example(raw, features_description)
        parsed_id = parsed["scenario/id"].numpy().item().decode("utf-8")
        if parsed_id != scene:
            continue

        scenario = Scenario(example=parsed)
        scenario.setup()

        sequences = scenario.lane_graph.sequences
        root_seqs = [s["lane_ids"] for s in sequences if not s["is_branch_root"]]
        road_segs = scenario.lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
        scenario.lane_graph.plot_branch_vs_root_sequence_polygons()
        fig, ax = plot_single_segment(
            scenario.lane_graph,
            road_segs,
            segment_key=seg,
            compute_polygons=True,
            inflate_pct=0.20,
            plot_boundaries=True,
            show_reference_line=True,
            show_centerlines=False,
        )
         
        plt.show()

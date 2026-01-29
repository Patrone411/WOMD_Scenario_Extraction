"""
Animate all actors on top of lane polygons, scene by scene,
from a hardcoded Waymo TFRecord path.

For each scenario in the TFRecord:
  - build Scenario
  - plot lane polygons as map background
  - animate all actors' trajectories with their IDs

Close the figure window to move to the next scenario.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 🔧 CHANGE THIS IMPORT to match your project layout:
# from your_scenario_module import Scenario, features_description
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import features_description, Scenario


# 🔧 HARD-CODED PATH & SETTINGS -----------------------------------------

TFRECORD_PATH = "tf_records/training_tfexample.tfrecord-00000-of-01000"
# Set to None to iterate through the whole file
NUM_SCENES_TO_SHOW = 5
FRAME_INTERVAL_MS = 11

# ----------------------------------------------------------------------
# Animation helper
# ----------------------------------------------------------------------

def animate_all_actors_on_polygons(scenario, interval=60, show_ids=True):
    """
    Animate ALL actors for a given Scenario on top of lane polygons.

    - Background: lane polygons (Scenario.lane_graph.plot_all_polygons_by_lane_type)
    - One colored line + moving dot per actor
    - Text label with actor ID next to the dot
    """
    # 1) Background: lane polygons
    fig, ax = scenario.lane_graph.plot_all_polygons_by_lane_type()
    ax.set_title(f"Scenario {scenario.name} - actor trajectories on lane polygons")

    trajectories = []  # list of (x, y)
    actor_ids = []
    lines = []
    dots = []
    labels = []

    # Determine which actors actually move (have any valid positions)
    valid_actor_indices = []
    for idx in range(len(scenario.actor_ids)):
        x = scenario.full_x[idx]
        y = scenario.full_y[idx]
        valid = (x != -1) & (y != -1)
        if np.any(valid):
            valid_actor_indices.append(idx)

    if not valid_actor_indices:
        print("No valid actors with positions in this scenario.")
        return fig, None

    cmap = plt.get_cmap("tab20", max(1, len(valid_actor_indices)))

    # Build one animated track per actor
    for i, idx in enumerate(valid_actor_indices):
        aid = scenario.actor_ids[idx]
        x = scenario.full_x[idx]
        y = scenario.full_y[idx]

        trajectories.append((x, y))
        actor_ids.append(aid)

        color = cmap(i % 20)
        line, = ax.plot([], [], color=color, linewidth=2, label=f"{int(aid)}")
        dot, = ax.plot([], [], "o", color=color)

        lines.append(line)
        dots.append(dot)

        if show_ids:
            txt = ax.text(
                np.nan,
                np.nan,
                str(int(aid)),
                fontsize=10,
                fontweight="bold",
                color="black",
                ha="left",
                va="bottom",
                zorder=10,
            )
        else:
            txt = None
        labels.append(txt)

    # 2) Expand axis limits so actors fit with polygons
    all_valid_x = []
    all_valid_y = []
    for (x, y) in trajectories:
        valid = (x != -1) & (y != -1)
        if np.any(valid):
            all_valid_x.append(x[valid])
            all_valid_y.append(y[valid])

    if all_valid_x:
        all_valid_x = np.concatenate(all_valid_x)
        all_valid_y = np.concatenate(all_valid_y)

        xmin0, xmax0 = ax.get_xlim()
        ymin0, ymax0 = ax.get_ylim()

        xmin = min(xmin0, float(np.min(all_valid_x)))
        xmax = max(xmax0, float(np.max(all_valid_x)))
        ymin = min(ymin0, float(np.min(all_valid_y)))
        ymax = max(ymax0, float(np.max(all_valid_y)))

        pad = 5.0
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # 3) Build animation
    max_len = max((len(x) for x, _ in trajectories), default=0)
    last_valid_positions = [None] * len(trajectories)

    def init():
        for line, dot, txt in zip(lines, dots, labels):
            line.set_data([], [])
            dot.set_data([], [])
            if txt is not None:
                txt.set_position((np.nan, np.nan))
        artists = lines + dots + [t for t in labels if t is not None]
        return artists

    def update(frame):
        for i, ((x, y), line, dot, txt) in enumerate(zip(trajectories, lines, dots, labels)):
            valid = (x[: frame + 1] != -1) & (y[: frame + 1] != -1)
            if np.any(valid):
                last_x = float(x[: frame + 1][valid][-1])
                last_y = float(y[: frame + 1][valid][-1])

                line.set_data(x[: frame + 1][valid], y[: frame + 1][valid])
                dot.set_data(last_x, last_y)
                last_valid_positions[i] = (last_x, last_y)

                if txt is not None:
                    txt.set_position((last_x + 0.5, last_y + 0.5))
            elif last_valid_positions[i] is not None:
                last_x, last_y = last_valid_positions[i]
                dot.set_data(last_x, last_y)
                if txt is not None:
                    txt.set_position((last_x + 0.5, last_y + 0.5))

        artists = lines + dots + [t for t in labels if t is not None]
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_len,
        init_func=init,
        interval=interval,
        blit=False,   # more robust with many artists & text
        repeat=False,
    )

    # Legend with actor IDs
    ax.legend(title="Actor IDs", loc="upper right")

    return fig, ani


# ----------------------------------------------------------------------
# Main loop over the TFRecord
# ----------------------------------------------------------------------

def main():
    print(f"Reading TFRecord: {TFRECORD_PATH}")
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)

    animations = []  # keep references so they don't get GC'd

    for i, raw in enumerate(dataset):
        if NUM_SCENES_TO_SHOW is not None and i >= NUM_SCENES_TO_SHOW:
            print(f"Reached NUM_SCENES_TO_SHOW={NUM_SCENES_TO_SHOW}, stopping.")
            break

        # Parse TF Example into a Scenario
        example = tf.io.parse_single_example(raw, features_description)
        scenario = Scenario(example, do_setup=False)
        scenario.setup()  # builds lane_graph, polygons, etc.

        print(f"\n=== Scenario {i} ===")
        print(f"Scenario ID: {scenario.name}")

        fig, ani = animate_all_actors_on_polygons(
            scenario,
            interval=FRAME_INTERVAL_MS,
            show_ids=True,
        )

        animations.append(ani)  # keep reference alive
        plt.show()              # close this window to proceed to next scenario

    print("Done iterating through scenarios.")


if __name__ == "__main__":
    main()

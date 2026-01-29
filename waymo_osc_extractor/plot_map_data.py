from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import features_description, Scenario
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.segment_polygon_handling import plot_single_segment
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.segment_polygon_handling import run_for_all_segments

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

def plot_side_neighbor_run_schematic(side="left"):
    """
    Schematische Visualisierung eines einseitigen Nachbarschaftslaufs.

    Vertikale Achse: globaler Segmentindex g (Fahrtrichtung nach oben).
    Horizontale Achse: Spalten für Target-Sequenz C und eine Nachbar-Root-Sequenz C^(k).

    - Target-Sequenz C: für jedes g ein Kästchen mit (lane_id, lokalem Index i),
      jede Target-Lane in einem eigenen Blauton.
    - Nachbar-Sequenz C^(k): für jedes g im Laufintervall ein Kästchen mit (m_k, lokalem Index j),
      jede Nachbar-Lane in einem eigenen Rotton.
    """

    # Beispielhafte globale Indizierung der Target-Sequenz C:
    # g : (lane_id, seg_idx = lokaler Index i in dieser Lane)
    g2lane = [
        ("ℓ_A", 0), ("ℓ_A", 1),                 # Lane ℓ_A
        ("ℓ_B", 0), ("ℓ_B", 1), ("ℓ_B", 2),     # Lane ℓ_B
        ("ℓ_C", 0), ("ℓ_C", 1),                 # Lane ℓ_C
        ("ℓ_D", 0)                              # Lane ℓ_D
    ]
    G = len(g2lane)

    # Beispiel: ein Lauf von g=2 bis g=7 (exklusiv)
    g_start, g_end = 2, 7

    # Beispielhafte neighbor_lane_spans innerhalb EINER Root-Sequenz C^(k):
    # m_1 ist von g=2..4 aktiv, m_2 von g=4..7
    neighbor_lane_spans = {
        "m_1": [(2, 4)],
        "m_2": [(4, 7)],
    }

    fig, ax = plt.subplots(figsize=(6, 7))

    # x-Positionen der Spalten:
    x_target = 0.0
    x_neighbor = -2.0 if side == "left" else 2.0   # links/rechts in Zeichenebene = Seite relativ zur Fahrtrichtung

    # ---------------------------------------------------------
    # Farbzuweisung für Target-Lanes: verschiedene Blautöne
    # ---------------------------------------------------------
    target_lane_ids = []
    for lane_id, _ in g2lane:
        if lane_id not in target_lane_ids:
            target_lane_ids.append(lane_id)

    cmap_target = plt.get_cmap("Blues")
    # Werte im Bereich [0.3, 0.9] -> mittlere bis kräftige Blautöne
    if len(target_lane_ids) > 1:
        target_values = list(
            [0.3 + 0.6 * i / (len(target_lane_ids) - 1) for i in range(len(target_lane_ids))]
        )
    else:
        target_values = [0.6]

    target_lane_colors = {
        lid: cmap_target(v) for lid, v in zip(target_lane_ids, target_values)
    }

    # Target-Sequenz C: ein Kästchen pro g (Lane + lokaler Index i)
    for g, (lane_id, seg_idx) in enumerate(g2lane):
        color = target_lane_colors[lane_id]
        rect = Rectangle(
            (x_target - 0.45, g - 0.45),
            0.9, 0.9,
            facecolor=color,
            edgecolor="black",
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(
            x_target, g,
            rf"{lane_id}, $i={seg_idx}$",
            ha="center", va="center",
            fontsize=9, color="white",
        )

    ax.text(
        x_target, G + 0.7,
        r"Target-Sequenz $C$" "\n" r"(lokale Indizes $i$ je Lane)",
        ha="center", va="bottom", fontsize=10,
    )

    # Rahmen für das Laufintervall [g_start, g_end) auf der Target-Seite
    ax.add_patch(Rectangle(
        (x_target - 0.55, g_start - 0.55),
        1.1, (g_end - g_start),
        facecolor="none",
        edgecolor="red",
        linewidth=2.0,
    ))
    ax.text(
        x_target + (0.8 if side == "left" else -0.8),
        (g_start + g_end) / 2,
        rf"Lauf auf Seite ``{side}''",
        rotation=90,
        ha="center", va="center", fontsize=9, color="red",
    )

    # ---------------------------------------------------------
    # Nachbar-Root-Sequenz C^(k): verschiedene Rottöne
    # ---------------------------------------------------------
    neighbor_ids = list(neighbor_lane_spans.keys())

    cmap_neighbor = plt.get_cmap("Reds")
    if len(neighbor_ids) > 1:
        neighbor_values = list(
            [0.3 + 0.6 * i / (len(neighbor_ids) - 1) for i in range(len(neighbor_ids))]
        )
    else:
        neighbor_values = [0.6]

    neighbor_colors = {
        nid: cmap_neighbor(v) for nid, v in zip(neighbor_ids, neighbor_values)
    }

    # lokaler j-Zähler pro Nachbar-Lane (nur für Darstellung)
    neighbor_local_index = {nid: 0 for nid in neighbor_ids}

    for nid in neighbor_ids:
        spans = neighbor_lane_spans[nid]
        col = neighbor_colors[nid]

        for (gs, ge) in spans:
            for g in range(gs, ge):
                j = neighbor_local_index[nid]

                rect = Rectangle(
                    (x_neighbor - 0.45, g - 0.45),
                    0.9, 0.9,
                    facecolor=col,
                    edgecolor="black",
                    alpha=0.9,
                )
                ax.add_patch(rect)
                ax.text(
                    x_neighbor, g,
                    rf"{nid}, $j={j}$",
                    ha="center", va="center",
                    fontsize=9, color="black",
                )

                neighbor_local_index[nid] += 1

    ax.text(
        x_neighbor, G + 0.7,
        r"Nachbar-Root-Sequenz $C^{(k)}$" "\n" r"(lokale Indizes $j$ je Lane Element)",
        ha="center", va="bottom", fontsize=10,
    )

    # Laufintervall auch auf der Nachbarseite markieren
    ax.add_patch(Rectangle(
        (x_neighbor - 0.55, g_start - 0.55),
        1.1, (g_end - g_start),
        facecolor="none",
        edgecolor="red",
        linewidth=1.5,
        linestyle="--",
    ))

    # ---------------------------------------------------------
    # g-Achse / Fahrtrichtung
    # ---------------------------------------------------------
    ax.set_ylim(-0.5, G - 0.5)
    ax.set_xlim(min(x_neighbor, x_target) - 1.2, max(x_neighbor, x_target) + 1.2)

    ax.set_yticks(range(G))
    ax.set_yticklabels([rf"$g={g}$" for g in range(G)])
    ax.set_xticks([x_neighbor, x_target])
    ax.set_xticklabels(["Nachbarseite", "Target"])

    # Fahrtrichtungs-Pfeil (wachsendes g)
    arrow_x = max(x_neighbor, x_target) + 0.9 if side == "left" else min(x_neighbor, x_target) - 0.9
    ax.annotate(
        "", xy=(arrow_x, G - 0.2), xytext=(arrow_x, -0.2),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )
    ax.text(
        arrow_x + (0.3 if side == "left" else -0.3),
        (G - 0.2) / 2,
        "Fahrtrichtung /\n" r"wachsendes $g$",
        rotation=90,
        ha="center", va="center", fontsize=9,
    )

    ax.set_title(
        rf"Einseitiger Nachbarschaftslauf ({side})" "\n"
        r"mit globalem Index $g$ und lokalen Indizes $i$, $j$"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_side_neighbor_run_schematic(side="left")

tf_path = "tf_records/training_tfexample.tfrecord-00000-of-01000"
log_path = "match_run_walk.osc_test.jsonl"
dataset = tf.data.TFRecordDataset(tf_path)
for raw in dataset:
        parsed = tf.io.parse_single_example(raw, features_description)
        parsed_id = parsed['scenario/id'].numpy().item().decode("utf-8")
        if parsed_id == "35910825e3844c7a":
            scenario = Scenario(example=parsed)
            scenario.setup()
            lg = scenario.lane_graph
            lg.plot_neighbor_candidates_searchbox(lane_id=292, seg_idx=21, annotate=True)
            lg.plot_neighbor_overlap_polygons(lane_id=292, seg_idx=21,)

            quit()
            #lg.debug_strtree_candidate(292, 17, 257, 6)
            #lg.debug_segment_pair(292, 17, 257, 6)

            i = 15
            while i < 30:

                lg.plot_neighbor_info_step(lane_id=292, seg_idx=i, buffer_width_percentage=1.2)
                
                i = i + 1
            quit()


            for i, x in enumerate(lg.root_seqs):
                target_chain = lg.root_seqs[i]
                all_chains   = lg.root_seqs

                lg.debug_plot_side_run(target_chain, all_chains, side="left", min_overlap=5, run_idx=0)

            fig, axes = lg.plot_branch_vs_root_sequence_polygons(
                lane_ids=None,        # or a subset if you want to focus on some lanes
                draw_polygons=False,  # or True if you want lane rectangles
                draw_directions=True,
                show=True,
            )
            """fig, ax = lg.plot_centerlines_with_boundaries_by_type(
                show=True,
                draw_polygons=True,   # polygons off
            )


            plt.show()
            fig, ax = lg.plot_polygon_overlaps_buffered(
                extra_width=0.5,
                min_overlap_ratio=0.1,
                show=False,   # this already calls plt.show() if it created the axes
            )"""

            markers = [
                (292, 0.4, "ego_vehicle", "red"),
                (256, 0.1, "car_1", "blue"),
                (262, 0.6, "car_2", "green"),
            ]

            fig, ax = lg.plot_lane_polygons_zoom(
                lane_ids=[256,262,291,292,258,257,255,170],
                extra_width=0.3,   # optional
                margin=5.0,
                markers=markers,
                show=True,
            )

            sequences = scenario.lane_graph.sequences
            root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
            rad_seqs = [s['lane_ids'] for s in sequences]
            road_segs = scenario.lane_graph.build_global_road_segments(all_chains=rad_seqs, min_overlap=20)
            print(road_segs)
            processed_segs = run_for_all_segments(lg, road_segs, show_plot=True)

        """fig, ax = plot_single_segment(
            scenario.lane_graph,
            road_segs,
            segment_key="seg_0",
            compute_polygons=True,
            inflate_pct=0.20,
            plot_boundaries=True,
            show_reference_line=True,
            show_centerlines=False,
        )
        plt.show()"""
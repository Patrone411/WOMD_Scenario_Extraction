from __future__ import annotations
from waymo_osc_extractor.waymoScenarioMining import get_scenario_by_id, features_description, Scenario, get_stitched_json_by_id
import json
from typing import List, Tuple, Union, Iterator, Dict, Any
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def iter_index(path: str | Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Yields (tf_record_id, scene_id, entry_dict) for every entry in the file.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # data: dict[tf_record_id -> dict[scene_id -> list[entry_dict]]]
    for tf_record_id, scenes in data.items():
        if not isinstance(scenes, dict):
            continue
        for scene_id, entries in scenes.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    yield tf_record_id, scene_id, entry

def get_valid_mask(xs,ys):
    valid = (~np.isnan(xs)) & (~np.isnan(ys)) & (xs != -1) & (ys != -1)
    return valid

def get_vehicle_xy(source: Union[str, dict], vehicle_id: str) -> Tuple[List[float], List[float]]:
    """
    Return (xs, ys) for the given vehicle_id from the stitched JSON.
    `source` can be a file path or a preloaded dict.
    """
    # Load JSON if a path was passed
    data = json.load(open(source, "r")) if isinstance(source, str) else source

    # Build the activities key for this vehicle
    act_key = f"{vehicle_id}_activity"
    try:
        vehicle_act = data["activities"]["vehicle"][act_key]
    except KeyError as e:
        raise KeyError(
            f"Could not find coordinates for {vehicle_id}. "
            f"Looked under activities['vehicle']['{act_key}']."
        ) from e

    xs = vehicle_act.get("x", [])
    ys = vehicle_act.get("y", [])

    # Basic sanity check: lengths should match
    if len(xs) != len(ys):
        raise ValueError(
            f"x and y length mismatch for {vehicle_id}: {len(xs)} vs {len(ys)}"
        )
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    return xs, ys

def plot_single_entry(stitched_data, scenario: Scenario, entry, plot_windows: bool=True  ):
    ego = entry["ego"]
    npc = entry["npc"]
    xse, yse = get_vehicle_xy(stitched_data,ego)
    xsn, ysn = get_vehicle_xy(stitched_data,npc)
    valid_e = get_valid_mask(xs=xse,ys=yse)
    valid_n = get_valid_mask(xs=xsn,ys=ysn)

    fig, ax = scenario.lane_graph.plot_all_lane_polygons()

    if plot_windows:
        for t_start, t_end in entry["time_windows"]:
            xse_w = xse[t_start:t_end+1]
            yse_w = yse[t_start:t_end+1]
            xsn_w = xsn[t_start:t_end+1]
            ysn_w = ysn[t_start:t_end+1]
            valid_e_w = valid_e[t_start:t_end+1]
            valid_n_w = valid_n[t_start:t_end+1]
            ax.plot(xse_w[valid_e_w], yse_w[valid_e_w], color='green', linewidth=2, label='ego')
            ax.plot(xsn_w[valid_n_w], ysn_w[valid_n_w], color='red', linewidth=2, label='npc')
            plt.show()

    else:
        ax.plot(xse[valid_e], yse[valid_e], color='green', linewidth=2, label='ego')
        ax.plot(xsn[valid_n], ysn[valid_n], color='red', linewidth=2, label='npc')
        plt.show()

def plot_matches(file_name: str, plot_windows=True):
    current_key = None

    for tfrecord_id, scene_id, entry in iter_index(file_name):
        key = (tfrecord_id, scene_id)
        if key != current_key:
            # load only when we see a new pair
            stitched = get_stitched_json_by_id(tf_record_id=tfrecord_id, scenario_id= scene_id)
            parsed = get_scenario_by_id(
                features_description=features_description,
                tfrecord_number=tfrecord_id,
                scenario_id=scene_id
                )
            scenario = Scenario(parsed)
            current_key = key

        plot_single_entry(stitched_data=stitched,scenario=scenario,entry=entry, plot_windows=plot_windows)

#plot_matches("waymo_osc_extractor/blockmatch_results.json")







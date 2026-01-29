import json, re, traceback
from pathlib import Path

from waymo_osc_extractor.waymoScenarioMining import features_description, create_s3_client, tf_scenario_streamer_with_keys, Scenario
from rich.progress import track
from collections import Counter
from typing import Dict, List, Optional, Any, Set, Iterable
from collections import defaultdict
import os
import math
from copy import deepcopy


def _clean_chains(seg_info):
    raw = deepcopy(seg_info.get("chains", []))  # break shared refs
    clean = []
    for ch in raw if isinstance(raw, list) else []:
        if not isinstance(ch, dict):
            continue
        ch_id = ch.get("id")
        lids = ch.get("lane_ids", [])
        # keep only plain ints
        safe_lids = []
        if isinstance(lids, list):
            for x in lids:
                # skip nested dicts/lists accidentally inserted
                if isinstance(x, (dict, list)):
                    continue
                try:
                    safe_lids.append(int(x))
                except Exception:
                    pass
        clean.append({"id": ch_id, "lane_ids": safe_lids})
    return clean

def filter_nans(obj):
    """
    Recursively walk through dicts/lists and replace NaN or Inf with None.
    That way the JSON stays valid.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: filter_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [filter_nans(v) for v in obj]
    return obj

def extract_vehicle_primary_lane_ids_from_waymo_dict(
    data: dict,
    allowed=("freeway", "surface_street", "bike_lane"),
    priority=("freeway", "surface_street", "bike_lane"),
    min_valid_id: int = 1,
    unknown_value: int = -99,
) -> Dict[str, List[int]]:
    """
    Returns: { 'vehicle_0': [lane_id_t0, lane_id_t1, ...], ... } with *int* lane IDs.
    For each timestep, take the first lane id (by `priority`) that is >= min_valid_id.
    Otherwise use `unknown_value` (default -99).
    """
    vehicles = data.get("actors_environment_element_intersection", {}).get("vehicle", {})
    out: dict[str, list[int]] = {}

    for veh_id, env_elems in vehicles.items():
        # Pull only allowed elements that have current_lane_id
        seqs = {
            k: env_elems[k]["current_lane_id"]
            for k in allowed
            if k in env_elems and isinstance(env_elems[k], dict) and "current_lane_id" in env_elems[k]
        }
        if not seqs:
            continue

        # Equalize lengths (pad with unknown)
        T = max(len(v) for v in seqs.values())
        for k, v in seqs.items():
            if len(v) < T:
                seqs[k] = v + [unknown_value] * (T - len(v))

        merged: list[int] = []
        for t in range(T):
            chosen = unknown_value
            for key in priority:
                if key in seqs:
                    val = seqs[key][t]
                    # cast early so we treat 0.0 / -99.0 as ints
                    try:
                        ival = int(val)
                    except (TypeError, ValueError):
                        ival = unknown_value
                    if ival >= min_valid_id:
                        chosen = ival
                        break
            merged.append(int(chosen))
        out[veh_id] = merged

    return out

# --- 1) Build lane lookup that allows a lane_id to map to multiple segments ---
def build_lane_lookup(road_segments: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    lane_id -> list of mappings (because a lane can appear in multiple segments).
    Each mapping: {'segment': seg_id, 'chain_id': int, 'osc_index': int}
    """
    lane_lookup: Dict[int, List[Dict[str, Any]]] = {}
    for seg_id, seg_data in road_segments.items():
        chains = seg_data.get("chains", [])
        for osc_index, chain in enumerate(chains, start=1):  # left->right => 1..N
            chain_id = chain["id"]
            for lane_id in chain["lane_ids"]:
                lane_lookup.setdefault(lane_id, []).append({
                    "segment": seg_id,
                    "chain_id": chain_id,
                    "osc_index": osc_index
                })
    return lane_lookup
# --- 2) per actor: map lane_ids -> per-timestep list of candidate mappings (or None) ---
FIXED_T = 91  # Waymo timesteps
def map_actor_lane_path_multi(
    actor_lane_ids: List[int],
    lane_lookup: Dict[int, List[Dict[str, Any]]],
    unknown_value: int = -99,
    T: int = FIXED_T,
) -> List[Optional[List[Dict[str, Any]]]]:
    """
    Map an actor's lane_id sequence into candidate OpenSCENARIO mappings.

    Args:
        actor_lane_ids: list of raw lane_ids for the actor (any length).
        lane_lookup: mapping {lane_id -> [ {segment, chain_id, osc_index}, ... ]}.
        unknown_value: placeholder used when lane_id is invalid.
        T: fixed output length (Waymo always 90 timesteps).

    Returns:
        A list of length T.
          - Each entry is None if lane_id is unknown at that timestep.
          - Otherwise, it's a list of candidate dicts (one per matching segment).
    """

    # --- Step 1: normalize the lane_id sequence to exactly length T ---
    # If the input is too long, cut it.
    # If the input is too short, pad it with unknown_value (-99).
    padded_seq = list(actor_lane_ids[:T])  # truncate if longer
    while len(padded_seq) < T:
        padded_seq.append(unknown_value)

    # --- Step 2: map each lane_id to segment candidates ---
    mapped_seq: List[Optional[List[Dict[str, Any]]]] = []
    for lid in padded_seq:
        if lid == unknown_value:
            # Unknown or invalid lane → no mapping
            mapped_seq.append(None)
        else:
            # Look up candidates in lane_lookup.
            # Could return multiple mappings if lane_id belongs to multiple segments.
            candidates = lane_lookup.get(lid, None)
            mapped_seq.append(candidates)

    return mapped_seq

# 2) Build segment-centric views directly from the multi-candidate mapping
def build_segment_views_from_multi(
    actor_lane_paths: Dict[str, List[int]],   # {actor_id: [lane_id_t]}
    road_segments: Dict[str, Any],            # your preprocessed segments
    unknown_value: int = -99,
    min_timesteps_per_actor: int = 1,         # filter actors with too few frames in a segment
    T: int = FIXED_T,
) -> Dict[str, Dict[str, Any]]:
    """
    Output (one entry per segment):
    {
      seg_id: {
        "meta":     {"segment": seg_id, "num_lanes": int, "T": T},
        "actors":   [actor_ids...],                              # filtered by min_timesteps_per_actor
        "lane_index": {actor_id: [osc_index or None]*T},         # per-timestep lane index in this segment
        "present":    {actor_id: [0/1]*T},                       # per-timestep presence bit in this segment
      },
      ...
    }
    """
    # --- Step 1: build lane lookup once ---
    lane_lookup = build_lane_lookup(road_segments)

    # --- Step 2: map every actor to candidates per timestep (length T) ---
    # mapped_multi[actor_id][t] = None or [ {segment, chain_id, osc_index}, ... ]
    mapped_multi: Dict[str, List[Optional[List[Dict[str, Any]]]]] = {}
    for actor_id, lane_seq in actor_lane_paths.items():
        mapped_multi[actor_id] = map_actor_lane_path_multi(
            lane_seq, lane_lookup, unknown_value=unknown_value, T=T
        )

    # --- Step 3: accumulate per-segment, per-actor, per-time osc_index ---
    # We'll collect sparse writes in dictionaries, then densify into length-T arrays.
    # lane_index_seg[seg_id][actor_id][t] = osc_index
    lane_index_seg: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))

    # Track how many frames each actor appears in each segment (for filtering)
    counts_seg_actor: Dict[str, Counter] = defaultdict(Counter)

    # Walk all actors and timesteps, and write to the appropriate segments
    for actor_id, mapped in mapped_multi.items():
        for t in range(T):
            step = mapped[t]
            if not step:
                continue

            # A lane can map to multiple segments (overlap). We add this actor to
            # each such segment at timestep t. If multiple mappings happen to
            # point to the SAME segment at t, keep the first osc_index we see.
            seen_segments = set()
            for m in step:
                seg_id = m["segment"]
                if seg_id in seen_segments:
                    continue  # ignore duplicate same-segment candidates at this timestep
                seen_segments.add(seg_id)

                osc_index = int(m["osc_index"])  # 1..num_lanes
                lane_index_seg[seg_id][actor_id][t] = osc_index
                counts_seg_actor[seg_id][actor_id] += 1

    # --- Step 4: finalize fixed-length arrays per segment, filter by min_timesteps ---
    segment_views: Dict[str, Dict[str, Any]] = {}
    for seg_id, per_actor_sparse in lane_index_seg.items():
        num_lanes = int(road_segments.get(seg_id, {}).get("num_lanes", 0))

        # Which actors should we keep in this segment?
        # Keep those with >= min_timesteps_per_actor frames present in this segment.
        kept_actors: List[str] = [
            a for a, c in counts_seg_actor[seg_id].items() if c >= min_timesteps_per_actor
        ]
        kept_actors.sort()

        # Build dense arrays of length T for lane_index and present
        lane_index_out: Dict[str, List[Optional[int]]] = {}
        present_out: Dict[str, List[int]] = {}
        for actor_id in kept_actors:
            li = [None] * T
            pr = [0] * T
            # Sparse writes for this actor in this segment:
            filled = per_actor_sparse.get(actor_id, {})
            for t, idx in filled.items():
                li[t] = idx
                pr[t] = 1
            lane_index_out[actor_id] = li
            present_out[actor_id] = pr

        seg_info = road_segments.get(seg_id, {}) or {}
        clean_chains = _clean_chains(seg_info)
        segment_views[seg_id] = {
            "meta":     {"segment": seg_id, "num_lanes": num_lanes, "T": T},
            "lane_chains": clean_chains,
            "actors":   kept_actors,
            "lane_index": lane_index_out,
            "present":    present_out,
        }

    return segment_views

def all_actors_in_segments(segment_views: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return a sorted list of all unique actor IDs present in ANY segment.
    Works with segment_views built by your build_segment_views_from_multi (actors is a list).
    """
    seen: Set[str] = set()
    for seg_id, seg_data in segment_views.items():
        actors = seg_data.get("actors", [])
        if isinstance(actors, list):
            seen.update(a for a in actors if isinstance(a, str))
    return sorted(seen)

def pretty_print_segment_views(segment_views, max_timesteps=10):
    """
    Nicely print segment_views in a human-friendly way.

    Args:
        segment_views: dict produced by build_segment_views_from_multi
        max_timesteps: truncate long sequences for display
    """
    for seg_id, seg_data in segment_views.items():
        print(f"\n=== Segment {seg_id} ===")
        meta = seg_data.get("meta", {})
        if meta:
            print(f"  Meta: {meta}")

        actors = seg_data.get("actors", [])
        print(f"  Actors ({len(actors)}): {actors}")

        lane_index = seg_data.get("lane_index", {})
        present = seg_data.get("present", {})

        for actor in actors:
            li = lane_index.get(actor, [])
            pr = present.get(actor, [])

            # Trim sequences for display
            li_short = li[:max_timesteps]
            pr_short = pr[:max_timesteps]

            li_str = ", ".join(str(x) if x is not None else "-" for x in li_short)
            pr_str = "".join("✓" if x else "." for x in pr_short)

            print(f"    Actor {actor}:")
            print(f"      LaneIdx: [{li_str}{'...' if len(li) > max_timesteps else ''}]")
            print(f"      Present:  {pr_str}{'...' if len(pr) > max_timesteps else ''}")

def extract_actor_subset(tag: Dict[str, Any], actors: Iterable[str]) -> Dict[str, Any]:
    """
    Given a parsed tag JSON dict and a list of actors (like 'vehicle_0', 'pedestrian_3'),
    return a dict with:
      - 'activities':  subset of actors_activity for those actors
      - 'relations':   subset of inter_actor_relation where src and dst are both in the list
      - 'env_xsects':  subset of actors_environment_element_intersection for those actors only
    """
    wanted: Set[str] = set(actors)

    # ---------- activities ----------
    activities_out: Dict[str, Dict[str, Any]] = {}
    activities = tag.get("actors_activity", {})
    for actor_type, per_type in activities.items():
        subset = {}
        for k, v in per_type.items():
            # keys look like "vehicle_0_activity"
            base_name = k.replace("_activity", "")
            if base_name in wanted:
                subset[k] = v
        if subset:
            activities_out[actor_type] = subset

    # ---------- inter-actor relations ----------
    relations = tag.get("inter_actor_relation", {})
    relations_out: Dict[str, Dict[str, Any]] = {}
    for src, dsts in relations.items():
        if src not in wanted:
            continue
        filtered_dsts = {dst: payload for dst, payload in dsts.items() if dst in wanted}
        if filtered_dsts:
            relations_out[src] = filtered_dsts

    # ---------- environment intersections ----------
    env = tag.get("actors_environment_element_intersection", {})
    env_out: Dict[str, Dict[str, Any]] = {}
    for actor_type, per_type in env.items():
        subset = {actor_key: data for actor_key, data in per_type.items() if actor_key in wanted}
        if subset:
            env_out[actor_type] = subset

    return {
        "activities": activities_out,
        "relations": relations_out,
        "env_xsects": env_out,
    }

def match_block_constraints(block_constraints, tagged_data, ego, npc, t_start, t_end, road_segments):
    ego_data = tagged_data["actors"][ego]
    npc_data = tagged_data["actors"][npc]
    relations = tagged_data["inter_actor_relation"][ego][npc]

    # --- Ego constraints ---
    if "speed_range" in block_constraints["ego"]:
        min_s, max_s = block_constraints["ego"]["speed_range"]
        speeds = ego_data["speed"][t_start:t_end]
        if not all(min_s <= s <= max_s for s in speeds):
            return False

    if "lane_at_start" in block_constraints["ego"]:
        required_lane = int(block_constraints["ego"]["lane_at_start"])
        actual_lane = ego_data["lane_id"][t_start]
        if actual_lane != required_lane:
            return False

    if "change_speed" in block_constraints["ego"]:
        delta = block_constraints["ego"]["change_speed"]
        # check average speed change over window
        start_speed = ego_data["speed"][t_start]
        end_speed = ego_data["speed"][t_end - 1]
        if abs((end_speed - start_speed) - delta) > 1e-3:  # tolerance
            return False

    if block_constraints["ego"].get("stay_in_lane"):
        lane_ids = ego_data["lane_id"][t_start:t_end]
        if not all(l == lane_ids[0] for l in lane_ids):
            return False

    # --- NPC constraints ---
    if "speed_range" in block_constraints["npc"]:
        min_s, max_s = block_constraints["npc"]["speed_range"]
        speeds = npc_data["speed"][t_start:t_end]
        if not all(min_s <= s <= max_s for s in speeds):
            return False

    if block_constraints["npc"].get("stay_in_lane"):
        lane_ids = npc_data["lane_id"][t_start:t_end]
        if not all(l == lane_ids[0] for l in lane_ids):
            return False

    if "relative_at_start" in block_constraints["npc"]:
        for rel in block_constraints["npc"]["relative_at_start"]:
            if rel["relation"] == "behind":
                dist = relations["distance"][t_start]
                if rel.get("distance") and dist < rel["distance"]:
                    return False
            if rel["relation"] == "right_of":
                pos = relations["position"][t_start]
                if pos != 3:  # "3" = right
                    return False

    if "relative_at_end" in block_constraints["npc"]:
        for rel in block_constraints["npc"]["relative_at_end"]:
            if rel["relation"] == "ahead_of":
                dist = relations["distance"][t_end - 1]
                if rel.get("distance") and dist < rel["distance"]:
                    return False

    # --- Map constraints ---
    if "min_lanes" in block_constraints["map"]:
        if road_segments["num_lanes"] < block_constraints["map"]["min_lanes"]:
            return False

    return True

def stitch_actor_and_map_data(result_dict,
                              scenario: Scenario,
                           ):
    
    try:

        lane_graph = scenario.lane_graph
        sequences = lane_graph.sequences
        root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
        road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)

        print(road_segs)
        lane_lookup = lane_graph.build_lane_lookup(road_segments=road_segs)
        vehicle_lane_ids = extract_vehicle_primary_lane_ids_from_waymo_dict(result_dict)

        segment_views = build_segment_views_from_multi(
            actor_lane_paths=vehicle_lane_ids,
            road_segments=road_segs,
            unknown_value=-99,
            min_timesteps_per_actor=5,   # require >= 5 frames in the segment
            T=91
        )

        actors = all_actors_in_segments(segment_views)
        subset = extract_actor_subset(tag=result_dict, actors=actors)
        
        out = {
        "segments": segment_views,
        "activities": subset.get("activities", {}),
        "relations": subset.get("relations", {}),   # <--- direct copy
        "env_xsects": subset.get("env_xsects", {}),
        }

        basename = os.path.basename(f'{scenario.scenario_id}.json')
        path = os.path.join(os.getcwd(), basename)
        print('saving to path: ', path)
        clean_out = filter_nans(out)
        #with open(path, "w") as f:
            #json.dump(clean_out, f, indent=2)
        #print(subset)
        #pretty_print_segment_views(segment_views)
        return clean_out

    except Exception as e:
            trace = traceback.format_exc()
            print(f"failed stitching sactor and map data")
            print(f"trace:{trace}")



    except Exception as e:
            trace = traceback.format_exc()
            print(f"failed stitching sactor and map data")
            print(f"trace:{trace}")

def run_data_stitching(   bucket_name: str = "waymo", 
                          prefix: str = "tfrecords/", 
                          result_prefix: str = "results/", 
                          time_of_tag_results: str = "2025-09-01-22_42",
                           ):
    
    s3 = create_s3_client()
    result_prefix = f"{result_prefix}{time_of_tag_results}/"

    # Stream scenarios with original TFRecord key for FILENUM
    for idx, (parsed, key) in enumerate(track(
        tf_scenario_streamer_with_keys(features_description, bucket_name, prefix),
        description=f"Processing scenarios from s3://{bucket_name}/{prefix}"
    )):
        try:
            #get road segments from scenario
            scenario = Scenario(parsed)
            print(f"handling scenario with ID = {scenario.scenario_id}")
            lane_graph = scenario.lane_graph
            sequences = lane_graph.sequences
            root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
            road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
            #print(road_segs)

            #get tags dict from s3
            FILE = Path(key).name  # e.g. training_tfexample.tfrecord-00000-of-01000
            
            # Extract correct Waymo FILENUM from filename
            match = re.search(r"-(\d{5})-of-\d{5}$", FILE)
            FILENUM = match.group(1) if match else str(idx).zfill(5)  # fallback just in case
            # Extract scenario ID
            scene_id = parsed['scenario/id'].numpy().item().decode("utf-8") #TODO use scenario id from Scenario class

            # Build result folder per TFRecord file number
            result_filename = f"Waymo_{FILENUM}_{scene_id}_tag.json"
            
            result_s3_key = f"{result_prefix}{FILENUM}/{result_filename}"
            print('result_prefix:', result_prefix)
            print('FILENUM' ,FILENUM)
            print('result_s3_key' , result_s3_key)
            obj = s3.get_object(Bucket=bucket_name, Key=result_s3_key)
            result_dict = json.loads(obj['Body'].read().decode("utf-8"))

            lane_lookup = lane_graph.build_lane_lookup(road_segments=road_segs)
            vehicle_lane_ids = extract_vehicle_primary_lane_ids_from_waymo_dict(result_dict)

            per_segment_data = stitch_actor_and_map_data(result_dict,scenario)
            print(scenario.scenario_id)
            stitch_key = f"{result_prefix}{FILENUM}/stitched/{scene_id}_stitched_data.json"
            s3.put_object(
                Bucket=bucket_name,
                Key=stitch_key,
                Body=json.dumps(per_segment_data, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
            print("stitched and saved data for scenario: ", scene_id )
            #segment_views = per_segment_data.get("segments", {})
            #pretty_print_segment_views(segment_views)


        except Exception as e:
                trace = traceback.format_exc()
                print(f"Scene {idx} error: {e}")
                print(f"trace:{trace}")

if __name__ == "__main__":
    #TODO add arg parser for tags result time etc
    run_data_stitching()

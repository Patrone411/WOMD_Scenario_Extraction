from waymo_osc_extractor.waymo_scenario_tools import (
    Scenario,
    
)
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.segment_polygon_handling import run_for_all_segments
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.stripped_actor_activities import per_actor_minimal
from waymo_osc_extractor.helpers.condense_actors import kept_actors_from_per_actor_minimal
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.stripped_inter_actor_tags import build_inter_actor_position_and_ttc
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.actor_per_segment import compute_segment_openscenario_coords
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.stripped_environ_elements import EnvironmentElementsWaymo
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.env_elements_per_segment import process_env_elements_segment_wise

def env_elements_per_segment(parsed, processed_segs, debug:bool= False):
    if debug:print("environment elements start")
    env = EnvironmentElementsWaymo(parsed)
    env(eval_mode=False) 
    env_res = process_env_elements_segment_wise(processed_segs, env)
    all_tl_ids = set.union(*(env["tl_ids"] for env in env_res.values())) if env_res else set()
    tl_states_per_id = {}
    if len(all_tl_ids) > 0:
        tl_state = env.traffic_lights['traffic_lights_state']    # np.ndarray [T, N]
        for id in all_tl_ids:
            tl_states_per_id[id] = tl_state[id]
    env_res["tl_states"] = tl_states_per_id

    if debug:print(env_res)
    if debug:print("environment elements done")
    return env_res

def process_actor_activities_per_segment(condensed_actors, processed_segs, debug: bool = False):
    if debug:print("actor_activities_per_segment start")
    per_segment_actor_activities = compute_segment_openscenario_coords(
        condensed=condensed_actors,
        processed_segs=processed_segs,
        dt=0.1,
        ds=0.5,           
    ) # process per segment actor data
    if debug:print("actor_activities_per_segment done")

    return per_segment_actor_activities

def process_inter_actor_activities(condensed_actors, debug: bool = False):
    if debug:print("inter actor start")

    inter_actor_relation = build_inter_actor_position_and_ttc(
        condensed_actors["all_payloads"],   
        dt=0.1,
    )
    if debug:print("inter actor relations done")

    return inter_actor_relation

def process_road_segments(lane_graph, road_segs, debug:bool = False):
    if debug:print("segments processing start ")
    processed_segs = run_for_all_segments(lane_graph, road_segs, show_plot=False)
    if debug:print("segments processing done ")

    return processed_segs


def result_dict_from_parsed(parsed, debug:bool = True):
    scene_id = parsed['scenario/id'].numpy().item().decode("utf-8")
    if debug: print("executing scene id :", scene_id)
    
    scenario = Scenario(parsed)
    lane_graph = scenario.lane_graph
    sequences = lane_graph.sequences
    root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
    road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)

    processed_segs = process_road_segments(lane_graph=lane_graph, road_segs=road_segs, debug=debug)
    actors = per_actor_minimal(parsed, eval_mode=False) # gets minimal preprocessed actor data from waymo data
    condensed = kept_actors_from_per_actor_minimal(road_segments=road_segs, per_actor=actors, min_steps=5) #filters out actors present in road segments
    
    #return TODO:
    #
    inter_actor_activities = process_inter_actor_activities(condensed, debug=debug)
    actor_activities_per_segment = process_actor_activities_per_segment(condensed_actors=condensed, processed_segs=processed_segs, debug=debug)
    per_segment_env = env_elements_per_segment(parsed, processed_segs)
    result_dict = {
        "road_segments": road_segs,
        "general_actor_activities": condensed,
        "inter_actor_activities": inter_actor_activities,
        "actor_activities_per_segment": actor_activities_per_segment,
        "per_segment_env_elements": per_segment_env,
    }
    
    return result_dict
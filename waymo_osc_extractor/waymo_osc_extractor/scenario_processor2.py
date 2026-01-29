""" module docstring """
# pylint: disable=import-error, no-name-in-module
from itertools import chain
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import Scenario
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.segment_polygon_handling import run_for_all_segments
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.stripped_actor_activities import per_actor_minimal
from waymo_osc_extractor.helpers.condense_actors import kept_actors_from_per_actor_minimal
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.stripped_inter_actor_tags import build_inter_actor_position_and_ttc
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.actor_per_segment import compute_segment_openscenario_coords
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.stripped_environ_elements import EnvironmentElementsWaymo
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.env_elements_per_segment import process_env_elements_segment_wise



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
        condensed_actors["actor_activities"],   
        dt=0.1,
    )
    if debug:print("inter actor relations done")

    return inter_actor_relation

def process_road_segments(lane_graph, road_segs, debug:bool = False):
    if debug:print("segments processing start ")
    processed_segs = run_for_all_segments(lane_graph, road_segs, show_plot=True)
    if debug:print("segments processing done ")

    return processed_segs


def result_dict_from_scenario(scenario, debug:bool = True):
    parsed = scenario.example
    scene_id = parsed['scenario/id'].numpy().item().decode("utf-8")
    if debug: print("executing scene id :", scene_id)
    scenario.plot_map(show=True)
    lane_graph = scenario.lane_graph
    sequences = lane_graph.sequences
    root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
    road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
    processed_segs = process_road_segments(lane_graph=lane_graph, road_segs=road_segs, debug=debug)

    valid_keys = [
        k for k, v in processed_segs.items()
        if v.get("valid", v.get("reference_line") is not None)
    ]
    if debug and len(valid_keys) != len(processed_segs):
        dropped = [k for k in processed_segs.keys() if k not in valid_keys]
        print(f"Dropping {len(dropped)} segments without a reference_line:", 
              dropped[:10], "..." if len(dropped) > 10 else "")

    # Keep keys in sync across *everything* downstream
    road_segs = {k: road_segs[k] for k in valid_keys}
    processed_segs = {k: processed_segs[k] for k in valid_keys}


    actors = per_actor_minimal(parsed, eval_mode=False) # gets minimal preprocessed actor data from waymo data
    condensed = kept_actors_from_per_actor_minimal(road_segments=road_segs, per_actor=actors, min_steps=5) #filters out actors present in road segments

    inter_actor_activities = process_inter_actor_activities(condensed, debug=debug)
    actor_activities_per_segment = process_actor_activities_per_segment(condensed_actors=condensed, processed_segs=processed_segs, debug=debug)
    per_segment_env = None
    result_dict = { 
    "road_segments": road_segs,
    "general_actor_data": {
        k: v for k, v in condensed.items() if k != "per_segment_payloads"
    },
    "inter_actor_activities": inter_actor_activities,
    "segment_actor_data": actor_activities_per_segment,
    "segment_env_elements": per_segment_env,
    "processed_road_segments": processed_segs,
    }
    
    return result_dict

class ScenarioProcessor2:
    """
    Processes a scenario using a tagging generator and stores the processed data.
    Attributes:
        scenario (Scenario): The scenario to be processed.
        scenario_index (int): The index of the scenario.
        processed_data (dict or None): The result of the processing, containing tagged information.
    Methods:
        process():
            Processes the scenario using the tagger, prints progress messages, and returns a dictionary with:
                - "general_info": General information about the scenario.
                - "inter_actor_relation": Relationships between actors.
                - "actors_activity": Activities of the actors.
                - "actors_environment_element_intersection": Intersections between actors and environment elements.
    """
    def __init__(self, scenario: Scenario, scenario_index: int):
        self.scenario = scenario
        self.scenario_name = self.scenario.scenario_id[0].decode('utf-8')
        self.scenario_index = scenario_index
        self.processed_data = None

    def process(self):
        """
        Processes the current scenario by tagging it and extracting relevant information.
        This method performs the following steps:
            1. Logs the start of the scenario processing.
            2. Uses the tagger to extract:
                - General information about the scenario.
                - Inter-actor relationships.
                - Activities of actors.
                - Intersections between actors and environmental elements.
            3. Stores the extracted data in the `processed_data` attribute.
            4. Logs the completion of the scenario processing.
            5. Returns the processed data as a dictionary.
        Returns:
            dict: A dictionary containing the following keys:
                - "general_info": General information about the scenario.
                - "inter_actor_relation": Relationships between actors.
                - "actors_activity": Activities performed by actors.
                - "actors_environment_element_intersection": Intersections between actors and environmental elements.
        """
        print(f"[tag] {self.scenario_index} - {self.scenario_name} start")
        if self.scenario.lane_graph.is_empty:
            print('faulty scenario data (probably missing key lane typed(1,2,3))')
            return None
        else:
            print('creating data')
            self.processed_data = result_dict_from_scenario(self.scenario, debug=True)
            print('done creating data')

        return self.processed_data

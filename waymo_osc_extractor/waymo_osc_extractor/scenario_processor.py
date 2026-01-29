""" module docstring """
# pylint: disable=import-error, no-name-in-module

from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.tags_generator import TagsGenerator
from waymo_osc_extractor.scenario_handling.scenario import Scenario


class ScenarioProcessor:
    """
    Processes a scenario using a tagging generator and stores the processed data.
    Attributes:
        scenario (Scenario): The scenario to be processed.
        tagger (TagsGenerator): The tagging generator used for processing.
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
    def __init__(self, scenario: Scenario, tagger: TagsGenerator, scenario_index: int):
        self.scenario = scenario
        self.tagger = TagsGenerator
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
        print(f"[tag] {self.scenario_index} - {self.scenario.name} start")
        if self.scenario.lane_graph.is_empty:
            print('faulty scenario data (probably missing key lane typed(1,2,3))')
            return None
        gi, iar, aa, aeei = self.tagger.tagging(
            self.scenario.example, f"S3_scene_{self.scenario.name}", eval_mode=False)
        self.processed_data = {
            "general_info": gi,
            "inter_actor_relation": iar,
            "actors_activity": aa,
            "actors_environment_element_intersection": aeei,
        }
        print(f"[tag] {self.scenario_index} - {self.scenario.name} done")
        return self.processed_data

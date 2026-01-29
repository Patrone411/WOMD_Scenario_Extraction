from .srunner.tools.osc2_helper import OSC2Helper
from .srunner.osc2_stdlib.path import Path

class MiniOSC2ScenarioConfig:
    def __init__(self, filename):
        self.filename = filename
        self.ast_tree = OSC2Helper.gen_osc2_ast(self.filename)
        # Required attributes accessed by ConfigInit
        self.variables = {}
        self.unit_dict = {}
        self.physical_dict = {}
        self.ego_vehicles = []
        self.other_actors = []
        self.all_actors = {}
        self.struct_declaration = {}
        self.scenario_declaration = {}

        self.path = Path
        self.store_variable(self.variables)

    def store_variable(self, vary):
        pass

    def add_ego_vehicles(self, vehicle):
        self.ego_vehicles.append(vehicle)
        self.all_actors[vehicle.get_name()] = vehicle

    def add_other_actors(self, actor):
        self.other_actors.append(actor)
        self.all_actors[actor.get_name()] = actor
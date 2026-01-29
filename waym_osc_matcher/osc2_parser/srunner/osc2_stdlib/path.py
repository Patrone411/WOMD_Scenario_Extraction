#import srunner.scenariomanager.carla_data_provider as carla_data
from osc2_parser.srunner.osc2_dm.physical_types import Physical


class Path:
    map_name = None
    min_driving_lanes = None
    max_lanes = None
    _length = None
    sign_type = None
    sign_types = None
    over_junction_check = None
    over_lanes_decrease_check = None
    is_path_dest = None
    is_path_origin = None
    is_explicit = None
    over_different_marks = None

    @classmethod
    def set_map(cls, map_name: str) -> None:
        cls.map_name = map_name

    @classmethod
    def get_map(cls) -> str:
        return cls.map_name

    @classmethod
    def path_length(cls, length: str) -> None:
        cls._length = float(length)

    @classmethod
    def path_min_driving_lanes(cls, min_lanes: str) -> None:
        cls.min_driving_lanes = float(min_lanes)

    @classmethod
    def path_max_lanes(cls, max_lanes: str) -> None:
        cls.max_lanes = float(max_lanes)

    @classmethod
    def path_different_dest(cls):
        cls.is_dest = True

    @classmethod
    def path_different_origin(cls):
        cls.is_path_origin = True

    @classmethod
    def path_has_sign(cls, sign_type: str):
        if sign_type == "speed_limit":
            cls.sign_type ="speed_limit" #carla.LandmarkType.MaximumSpeed
        elif sign_type == "stop_sign":
            cls.sign_type = "stop_sign" #carla.LandmarkType.StopSign
        elif sign_type == "yield":
            cls.sign_type = "yield" #carla.LandmarkType.YieldSign
        elif sign_type == "roundabout":
            cls.sign_type = "roundabout" #carla.LandmarkType.Roundabout

    @classmethod
    def path_has_no_signs(cls):
        cls.sign_types = [
            #carla.LandmarkType.MaximumSpeed,
            #carla.LandmarkType.StopSign,
            #carla.LandmarkType.YieldSign,
            #carla.LandmarkType.Roundabout,
            'MaximumSpeed',
            'StopSign',
            'YieldSign',
            'Roundabout',
        ]


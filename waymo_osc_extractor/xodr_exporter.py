#konvertiert road segmente zu xodr 

from waymo_osc_extractor.waymo_scenario_tools.s3_handlers.s3_tfrecord_streamer import get_scenario_by_id
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import features_description
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import Scenario
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.segment_polygon_handling import run_for_all_segments
from waymo_osc_extractor.carla_exporter.segment_to_opendrive import export_segment_to_xodr
from waymo_osc_extractor.scenario_processor2 import ScenarioProcessor2
import tensorflow as tf

import boto3, pickle, io

bucket = "waymo"
scenario_id="104b4a3e67b26ce1" #gewünschte scenario id
tf_path = "tf_records/training_tfexample.tfrecord-00000-of-01000" #loakler pfad zum tfrecord
dataset = tf.data.TFRecordDataset(tf_path)

for raw in dataset:
        parsed = tf.io.parse_single_example(raw, features_description)
        parsed_id = parsed["scenario/id"].numpy().item().decode("utf-8")
        if parsed_id != scenario_id:
            continue


        scenario = Scenario(parsed)
        scenario.setup()
        lane_graph = scenario.lane_graph
        sequences = lane_graph.sequences
        root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
        road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
        seg_key =  "seg_0" #n gewünschtes segment
        spec_seg = road_segs[seg_key] 
        print(spec_seg)
        processed_seg = run_for_all_segments(lane_graph, road_segs, show_plot=False)

        print(scenario.name)

        result_dict = ScenarioProcessor2(scenario,0).process()

        seg_block   = result_dict["road_segments"][seg_key]          
        results     = result_dict["processed_road_segments"][seg_key]

        xodr = export_segment_to_xodr(seg_key=seg_key, results=results)
        with open(f"{seg_key}.xodr", "w", encoding="utf-8") as f:
            f.write(xodr)
        print('xodr created from segment')


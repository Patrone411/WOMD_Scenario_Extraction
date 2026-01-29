from scenario_handling import Scenario
from scenario_handling import features_description
from s3_handlers import pb_scenario_streamer
from s3_handlers import tf_scenario_streamer

'''s3 streamer tf record example'''
for i, example in enumerate(tf_scenario_streamer(features_description)):
    scenario = Scenario(example)
    print(f"Example {i}: scenario ID = {scenario.scenario_id}")
    lane_graph = scenario.lane_graph
    sequences = lane_graph.sequences
    root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
    lane_graph.interactive_lane_neighborhood_viewer()
    road_segs = lane_graph.build_global_road_segments(all_chains=root_seqs, min_overlap=20)
    print(road_segs)
    for i in range (0, len(road_segs)):
        seg_id = f"seg_{i}" 
        lane_graph.plot_segment_highlight(road_segs, seg_id)

    if i >= 1:
        break  # only show a few for demo

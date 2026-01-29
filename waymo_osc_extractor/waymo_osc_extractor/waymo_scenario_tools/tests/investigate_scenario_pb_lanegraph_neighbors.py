import numpy as np
from scenario_handling import LaneGraph
from s3_handlers import get_scenarios_from_all_files


"""shows neighbor lanes of pb file computed in our own lanegraph class using 3 diffrent methods, 
text box lets you chose lane ids of which you want to compute neighbors, radio toggle button lets you chose from
neighbor computation methods. 
proto format is converted from pb scenario fomat into tfrecord format which are used by waymoscenarionmining thesis"""

for j, scenario in enumerate(get_scenarios_from_all_files()):
    print(f"{j}: Scenario ID = {scenario.scenario_id}")
    xyz_list = []
    dir_list = []
    id_list = []
    type_list = []
    valid_list = []   
    # --- Process map features ---
    for feature in scenario.map_features:
        if not feature.HasField("lane"):
            continue  # you can include road edges, crosswalks, etc. if needed

        lane = feature.lane
        points = lane.polyline
        lane_id = feature.id
        lane_type = lane.type

        for i in range(len(points)):
            p = points[i]
            xyz = [p.x, p.y, p.z]
            xyz_list.append(xyz)
            id_list.append([lane_id])
            type_list.append([lane_type])
            valid_list.append([1])  # all real points are valid

            # Compute direction (forward difference)
            if i < len(points) - 1:
                dx = points[i + 1].x - p.x
                dy = points[i + 1].y - p.y
                dz = points[i + 1].z - p.z
            else:
                dx = dy = dz = 0.0
            dir_list.append([dx, dy, dz])


    MAX_POINTS = 20000  # Same as TFExample shape

    def pad_array(arr, shape, dtype):
        arr = np.asarray(arr, dtype=dtype)
        pad_len = MAX_POINTS - len(arr)
        if pad_len < 0:
            raise ValueError(f"Too many samples: {len(arr)} > {MAX_POINTS}")
        return np.pad(arr, ((0, pad_len), (0, 0)), mode='constant')

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////TODO
    # === Parameters ===
    DIST_THRESHOLD = 1.5  # Meters to consider two segments connected

    semantic_types = {
        1: 'LaneCenter-Freeway',
        2: 'LaneCenter-SurfaceStreet',
        3: 'LaneCenter-BikeLane',
        15: 'RoadEdgeBoundary',
    }

    desired_type_ids = list(semantic_types.keys())

    # === Read One Scenario ===
    #TF_EXAMPLE_PATH = "/mnt/d/WSL/Ubuntu/waymoExampleSet/uncompressed_tf_example_training_training_tfexample.tfrecord-00000-of-01000"

    valid = np.array(valid_list, dtype=np.int64).reshape(-1)
    types = np.array(type_list, dtype=np.int64).reshape(-1)
    ids   = np.array(id_list, dtype=np.int64).reshape(-1)
    xyz = np.array(xyz_list, dtype=np.float32)
    dirs = np.array(dir_list, dtype=np.float32)
    # === Filter Valid Points of Target Types ===
    #mask = (valid == 1) & np.isin(types, desired_type_ids)
    #ids, types, xyz = ids[mask], types[mask], xyz[mask]
    lane_types_to_use = [1,2,3]
    lane_graph = LaneGraph(
                xyz,
                dirs,
                types,
                ids,
                valid,
                lane_types=lane_types_to_use
            )

    lane_graph.interactive_lane_neighborhood_viewer()    

    


    '''sequences, lane_to_seq = lane_graph.build_aligned_lane_sequences()
    root_seqs = [s['lane_ids'] for s in sequences if not s["is_branch_root"]]
    for seq in root_seqs:
        if 470 in seq:
            result = lane_graph.longest_neighbor_chain_left_and_right(lane_chain=seq, min_overlap=30)
            neighbors_left = lane_graph.longest_neighbor_chain_one_side(lane_chain=seq,side='left', min_overlap= 10, enforce_continuity=True)
            neighbors_right = lane_graph.longest_neighbor_chain_one_side(lane_chain=seq,side='right', min_overlap= 10,enforce_continuity=True)
            if neighbors_left:
                lane_graph.plot_neighbor_chain_with_both_sides(neighbors_left)
            if neighbors_right:
                lane_graph.plot_neighbor_chain_with_both_sides(neighbors_right)

            if result:
                lane_graph.plot_segment_buffers_with_neighbors(lane_id=330, buffer_width_percentage=1.5)
                lane_graph.plot_neighbor_chain_with_both_sides(result)
    if j >= 2:  # Just demo first few
        break'''
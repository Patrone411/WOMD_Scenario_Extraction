import numpy as np
from s3_handlers import pb_scenario_streamer
from scenario_handling import LaneGraph

"""computes errors of lanegraph class lane neighbor calculations and waymo pb neighbor info"""

for j, scenario in enumerate(pb_scenario_streamer()):
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

    fix, ax = lane_graph.plot_all_polygons_by_lane_type()
    
    #plt.show()


    # --- Build Lane Dictionary ---
    lane_dict = {f.id: f.lane for f in scenario.map_features if f.HasField("lane")}
    lane_neighbors_from_proto = {}

    for lane_id, lane in lane_dict.items():
        neighbor_ids = set()
        for n in list(lane.left_neighbors) + list(lane.right_neighbors):
            neighbor_ids.add(n.feature_id)
        lane_neighbors_from_proto[lane_id] = neighbor_ids




    # Store overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-lane error logs
    type_ii_errors = {}  # false negatives (missed neighbors)
    type_i_errors = {}   # false positives (extra neighbors)
    true_positives = {}

    for lane_id in lane_neighbors_from_proto:
        gt_neighbors = lane_neighbors_from_proto.get(lane_id, set())
        pred_neighbors = lane_graph.segment_neighbors.get(lane_id, set())

        tp = gt_neighbors & pred_neighbors
        fp = pred_neighbors - gt_neighbors
        fn = gt_neighbors - pred_neighbors

        # Log errors
        if fn:
            type_ii_errors[lane_id] = fn
        if fp:
            type_i_errors[lane_id] = fp
        if tp:
            true_positives[lane_id] = tp

        # Accumulate counts
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

    # Compute overall precision, recall, and F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Output results
    print("==== Performance Metrics ====")
    print(f"True Positives: {total_tp}")
    print(f"False Positives (Type I Errors): {total_fp}")
    print(f"False Negatives (Type II Errors): {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    print("\n==== Example Errors ====")
    print("Type II Errors (missed neighbors):")
    for k, v in list(type_ii_errors.items())[:5]:
        print(f"  Lane {k}: missed {v}")

    print("Type I Errors (extra neighbors):")
    for k, v in list(type_i_errors.items())[:5]:
        print(f"  Lane {k}: extra {v}")

    if j > 2:  # Just demo first few
        break

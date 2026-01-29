import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from s3_handlers import pb_scenario_streamer
import math

def collect_chains_from_lane(lane_id, current_chain, all_chains, visited, lane_map):
    #finds all unique longest chains of consecutive lanes
    if lane_id in visited:
        # Cycle detected — stop exploring this path
        return

    lane = lane_map.get(lane_id)
    if not lane:
        return

    current_chain.append(lane_id)
    visited.add(lane_id)

    if not lane.exit_lanes:
        all_chains.append(current_chain.copy())
    else:
        for next_lane_id in lane.exit_lanes:
            collect_chains_from_lane(next_lane_id, current_chain, all_chains, visited, lane_map)

    # Backtrack
    current_chain.pop()
    visited.remove(lane_id)

def find_longest_neighbor_stretch_flat(
    chain,
    lane_neighbors_by_index,
    lane_segments,
    min_length=1,
    direction="left",  # "left", "right", or "both"
):
    '''for a given chain of consecutive lanes, this code returns the longest stretch
    for which there is a continous neighbor on either left, right or both sides'''
    assert direction in {"left", "right", "both"}, "Invalid direction argument"

    flattened_indices = []
    for lane_id in chain:
        num_points = lane_segments.get(lane_id, 0)
        for idx in range(num_points):
            flattened_indices.append((lane_id, idx))

    max_stretch = {
        "start_global_index": None,
        "end_global_index": None,
        "start_lane_id": None,
        "start_lane_index": None,
        "length": 0,
        "chain": chain,
    }

    current_start_global = None
    current_start_lane = None
    current_start_index = None
    current_length = 0

    for global_idx, (lane_id, lane_index) in enumerate(flattened_indices):
        lane_data = lane_neighbors_by_index.get(lane_id, {}).get(lane_index, {})
        has_left = bool(lane_data.get("left"))
        has_right = bool(lane_data.get("right"))

        if (
            (direction == "left" and has_left)
            or (direction == "right" and has_right)
            or (direction == "both" and has_left and has_right)
        ):
            if current_length == 0:
                current_start_global = global_idx
                current_start_lane = lane_id
                current_start_index = lane_index
            current_length += 1
        else:
            if current_length >= min_length and current_length > max_stretch["length"]:
                max_stretch = {
                    "start_global_index": current_start_global,
                    "end_global_index": global_idx - 1,
                    "start_lane_id": current_start_lane,
                    "start_lane_index": current_start_index,
                    "length": current_length,
                    "chain": chain,
                }
            current_length = 0

    if current_length >= min_length and current_length > max_stretch["length"]:
        max_stretch = {
            "start_global_index": current_start_global,
            "end_global_index": len(flattened_indices) - 1,
            "start_lane_id": current_start_lane,
            "start_lane_index": current_start_index,
            "length": current_length,
            "chain": chain,
        }

    return max_stretch if max_stretch["length"] >= min_length else None

def plot_lane_chain_with_stretch(result, lane_polyline, color_chain='gray', color_stretch='red'):
    if not result:
        return
    print(result)
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Plot all lanes with labels
    for lane_id, polyline in lane_polyline.items():
        if not polyline:
            continue
        xs = [p.x for p in polyline]
        ys = [p.y for p in polyline]
        ax.plot(xs, ys, color='lightgray', linewidth=0.8, alpha=0.4)

        # Add lane ID label at the start
        ax.text(xs[0], ys[0], str(lane_id), fontsize=6, color='black', alpha=0.6)

    # 2. Highlight the lane chain
    for lane_id in result['chain']:
        polyline = lane_polyline.get(lane_id, [])
        if not polyline:
            continue
        xs = [p.x for p in polyline]
        ys = [p.y for p in polyline]
        ax.plot(xs, ys, color=color_chain, linewidth=1.5, alpha=0.7)

    # 3. Plot the longest stretch in the chain
    stretch_indices = []
    flattened = []
    for lane_id in result['chain']:
        poly = lane_polyline.get(lane_id, [])
        for i in range(len(poly)):
            flattened.append((lane_id, i))

    start = result['start_global_index']
    end = result['end_global_index']
    stretch_slice = flattened[start:end + 1]

    for lane_id, idx in stretch_slice:
        poly = lane_polyline.get(lane_id, [])
        if 0 <= idx < len(poly):
            pt = poly[idx]
            stretch_indices.append((pt.x, pt.y))

    if stretch_indices:
        xs, ys = zip(*stretch_indices)
        ax.plot(xs, ys, color=color_stretch, linewidth=3.0, label='Longest Neighbor Stretch')

    ax.set_aspect('equal')
    ax.set_title('All Lanes with Chain and Longest Neighbor Stretch')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

RoadLineType = {
            'TYPE_UNKNOWN': 0,
            'TYPE_BROKEN_SINGLE_WHITE' : 1,
            'TYPE_SOLID_SINGLE_WHITE' : 2,
            'TYPE_SOLID_DOUBLE_WHITE' : 3,
            'TYPE_BROKEN_SINGLE_YELLOW' : 4,
            'TYPE_BROKEN_DOUBLE_YELLOW' : 5,
            'TYPE_SOLID_SINGLE_YELLOW' : 6,
            'TYPE_SOLID_DOUBLE_YELLOW' : 7,
            'TYPE_PASSING_DOUBLE_YELLOW' : 8,
        }
broken_line_types = [1,4,5,8]
solid_line_types = [2,3,6,7,0]

for i, scenario in enumerate(pb_scenario_streamer()):
    show_pb_neighbors_interactive = False
    investigate_neighbors_further = False
    investigate_neighbors_with_opposing_dir = False
    print(f"{i}: Scenario ID = {scenario.scenario_id}")

    if show_pb_neighbors_interactive:
        """shows neighbor lanes taken from the official waymo proto, 
        text box lets you chose lane ids of which you want to compute neighbors
        same as interactive neighborhoood viewer in lane_graph class"""

        # --- Build Lane Dictionary ---
        lane_dict = {f.id: f.lane for f in scenario.map_features if f.HasField("lane")}
        # --- Plotting function ---
        def plot_lanes(ax, target_lane_id):
            ax.clear()
            neighbor_ids = set()

            target_lane = lane_dict.get(target_lane_id, None)
            if target_lane:
                for n in list(target_lane.left_neighbors) + list(target_lane.right_neighbors):
                    neighbor_ids.add(n.feature_id)

            for feature in scenario.map_features:
                if not feature.HasField("lane"):
                    continue

                lane = feature.lane
                polyline = lane.polyline
                if not polyline:
                    continue

                xs = [p.x for p in polyline]
                ys = [p.y for p in polyline]

                # Determine lane color
                if feature.id == target_lane_id:
                    color = 'red'
                    lw = 2
                elif feature.id in neighbor_ids:
                    color = 'green'
                    lw = 2
                else:
                    color = 'gray'
                    lw = 1

                # Plot lane
                ax.plot(xs, ys, color=color, linewidth=lw)

                # Direction arrow
                if len(xs) >= 2:
                    mid = len(xs) // 2
                    if mid + 1 < len(xs):
                        dx = xs[mid + 1] - xs[mid]
                        dy = ys[mid + 1] - ys[mid]
                        ax.arrow(xs[mid], ys[mid], dx, dy,
                                head_width=0.5, head_length=0.7, fc=color, ec=color, length_includes_head=True)

                # Lane start/end markers
                ax.plot(xs[0], ys[0], marker='o', color='blue', markersize=4)  # Start
                ax.plot(xs[-1], ys[-1], marker='o', color='red', markersize=4)  # End

                # Show lane ID
                mid = len(xs) // 2
                ax.text(xs[mid], ys[mid], str(feature.id), fontsize=6, color='black')

            ax.set_title(f"Lane ID: {target_lane_id} (Red), Neighbors (Green)")
            ax.axis('equal')
            ax.grid(True)
            plt.draw()


        # --- Setup interactive plot ---
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Leave space for the text box

        # Initial lane to display
        initial_lane_id = next(iter(lane_dict))
        plot_lanes(ax, initial_lane_id)

        # Add text box widget
        axbox = plt.axes([0.25, 0.05, 0.5, 0.075])
        text_box = TextBox(axbox, 'Enter Lane ID: ')

        def on_submit(text):
            try:
                lane_id = int(text)
                if lane_id in lane_dict:
                    plot_lanes(ax, lane_id)
                else:
                    print(f"Lane ID {lane_id} not found.")
            except ValueError:
                print("Invalid input. Please enter an integer lane ID.")

        text_box.on_submit(on_submit)

        plt.show()

    if investigate_neighbors_further:
        
        allowed_line_types = broken_line_types #+ solid_line_types 

        lane_neighbors_by_index = {}
        lane_map = {}
        root_lanes = []
        lane_segments = {}
        lane_polyline = {}

        #collect lane and neighbor neighbor information 
        for feature in scenario.map_features:
            if feature.WhichOneof("feature_data") != "lane":
                continue

            lane = feature.lane
            lane_id = feature.id
            num_points = len(lane.polyline)
            lane_segments[lane_id] = num_points
            lane_polyline[lane_id] = lane.polyline
            lane_map[lane_id] = lane

            if not lane.entry_lanes:
                root_lanes.append(lane_id)
            # Initialize the dict for this lane with empty lists
            lane_neighbors_by_index[lane_id] = {
                idx: {'left': [], 'right': []} for idx in range(num_points)
            }

            def process_neighbors(neighbors, direction):
                '''gets all lane neighbors with the previously set boundary type'''
                for neighbor in neighbors:
                    # Check each boundary of this neighbor to see if it's valid
                    for boundary in neighbor.boundaries:
                        if boundary.boundary_type not in allowed_line_types:
                            continue

                        # Only fill neighbor data if boundary is within polyline index range
                        for idx in range(boundary.lane_start_index, boundary.lane_end_index + 1):
                            if 0 <= idx < num_points:
                                lane_neighbors_by_index[lane_id][idx][direction].append(neighbor.feature_id)
            
            # Process left and right neighbors
            process_neighbors(lane.left_neighbors, direction='left')
            process_neighbors(lane.right_neighbors, direction='right')

        target_boundary_types = [1, 4, 5]  # example: dashed lines, solid lines, etc.

        for feature in scenario.map_features:
            if feature.WhichOneof("feature_data") != "lane":
                continue

            lane = feature.lane
            lane_id = feature.id

            #Find lanes that have a dashed boundary line but no neighbor lanes on either side, left and right
            
            # Check LEFT boundaries
            for boundary in lane.left_boundaries:
                if boundary.boundary_type in target_boundary_types:
                    has_left_neighbor = any(
                        lane.left_neighbors
                    )
                    if not has_left_neighbor:
                        print(f"[LEFT] Lane {lane_id} has boundary type {boundary.boundary_type} but NO left neighbors.")

            # Check RIGHT boundaries
            for boundary in lane.right_boundaries:
                if boundary.boundary_type in target_boundary_types:
                    has_right_neighbor = any(
                        lane.right_neighbors
                    )
                    if not has_right_neighbor:
                        print(f"[RIGHT] Lane {lane_id} has boundary type {boundary.boundary_type} but NO right neighbors.")

        # Example usage of collected neighbor info:
        # Print neighbors for lane 99 at index 10
        '''
        lane_id_to_inspect = 99
        for index, neighbor_data in lane_neighbors_by_index[lane_id_to_inspect].items():
            print(f"Index {index}:")
            print(f"  Left neighbors: {neighbor_data['left']}")
            print(f"  Right neighbors: {neighbor_data['right']}")
        '''

        # find lane chains, aka successive lanes
        all_lane_chains = []
        for root_id in root_lanes:
            collect_chains_from_lane(root_id, [], all_lane_chains, set(), lane_map)
        #could use an arbitrary chain from the collection, or specify an example chain as list like so_

        #find longest stretch along a lane chain that has a neighbor and plots it:
        chain = [307,274,270]
        for lane_id in chain:
            print('id: ',lane_id)
            print('lane_length: ', lane_segments[lane_id])
            print('neighbors: ', lane_neighbors_by_index[lane_id])
        result = find_longest_neighbor_stretch_flat(
            chain,
            lane_neighbors_by_index,
            lane_segments,
            min_length=100,
            direction="both"
            )

        plot_lane_chain_with_stretch(result, lane_polyline)
    
    if investigate_neighbors_with_opposing_dir:
        '''check if there is an occasion of a neighbor that runs in
        the opposite direction within the waymo neighbor information.
        so far i have not found a single occasion'''
        def compute_heading(polyline):
            if len(polyline) < 2:
                return None
            dx = polyline[-1].x - polyline[0].x
            dy = polyline[-1].y - polyline[0].y
            norm = math.hypot(dx, dy)
            if norm == 0:
                return None
            return (dx / norm, dy / norm)

        def dot_product(v1, v2):
            return v1[0]*v2[0] + v1[1]*v2[1]

        broken_line_types = [1,4,5,8]
        solid_line_types = [2,3,6,7,0]
        allowed_line_types = broken_line_types #+ solid_line_types 

        
        lane_neighbors_by_index = {}
        lane_map = {}
        root_lanes = []
        lane_segments = {}
        lane_polyline = {}

        for feature in scenario.map_features:
            if feature.WhichOneof("feature_data") != "lane":
                continue

            lane = feature.lane
            lane_id = feature.id
            num_points = len(lane.polyline)
            lane_segments[lane_id] = num_points
            lane_polyline[lane_id] = lane.polyline
            lane_map[lane_id] = lane

            if not lane.entry_lanes:
                root_lanes.append(lane_id)
            # Initialize the dict for this lane with empty lists
            lane_neighbors_by_index[lane_id] = {
                idx: {'left': [], 'right': []} for idx in range(num_points)
            }

            def process_neighbors(neighbors, direction):
                for neighbor in neighbors:
                    # Check each boundary of this neighbor to see if it's valid
                    for boundary in neighbor.boundaries:
                        if boundary.boundary_type not in allowed_line_types:
                            continue

                        # Only fill neighbor data if boundary is within polyline index range
                        for idx in range(boundary.lane_start_index, boundary.lane_end_index + 1):
                            if 0 <= idx < num_points:
                                lane_neighbors_by_index[lane_id][idx][direction].append(neighbor.feature_id)

            # Process left and right neighbors
            process_neighbors(lane.left_neighbors, direction='left')
            process_neighbors(lane.right_neighbors, direction='right')

            for lane_id, lane_data in lane_neighbors_by_index.items():
                #print(lane_id, " checked")
                main_heading = compute_heading(lane_polyline.get(lane_id, []))
                if not main_heading:
                    continue

                for idx, neighbor_info in lane_data.items():
                    for direction in ['left', 'right']:
                        for neighbor_id in neighbor_info[direction]:
                            neighbor_polyline = lane_polyline.get(neighbor_id, [])
                            neighbor_heading = compute_heading(neighbor_polyline)
                            if not neighbor_heading:
                                continue

                            dp = dot_product(main_heading, neighbor_heading)
                            if dp < -0.5:
                                print(f"Lane {lane_id} and its {direction} neighbor {neighbor_id} are in opposite directions at index {idx} (dot={dp:.2f})")
        
    if i > 2:
        break #show first 2 scenarios
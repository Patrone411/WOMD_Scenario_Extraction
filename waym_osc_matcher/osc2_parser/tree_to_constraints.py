
def pytree_to_actor_constraints(pytree_blocks):
    """early version, constraints per action+modifier combos needs to be build upon for complex scenarios"""
    constraints = []
    for block in pytree_blocks:
        if block is None:
            continue  # skip empty blocks

        block_constraints = {"duration_s": block.get("duration"), "ego": {}, "npc": {}}

        actors = block.get("actors", {})

        # Ego modifiers
        ego_actor = actors.get("ego_vehicle", {})
        for mod in ego_actor.get("modifiers", []):
            if "speed" in mod:
                block_constraints["ego"]["speed_range"] = mod["speed"]
            if "lane" in mod:
                block_constraints["ego"]["lane_at_start"] = mod["lane"]
            if "change_speed" in mod:
                block_constraints["ego"]["change_speed"] = mod["change_speed"]

        # NPC modifiers
        npc_actor = actors.get("npc", {})
        start_rel = []
        end_rel = []
        for mod in npc_actor.get("modifiers", []):
            if "right_of" in mod:
                start_rel.append({"relation": "right_of", "actor": mod["right_of"], "distance": mod.get("distance")})
            if "behind" in mod:
                start_rel.append({"relation": "behind", "actor": mod["behind"], "distance": mod["distance"]})
            if "ahead_of" in mod and mod.get("at") == "end":
                end_rel.append({"relation": "ahead_of", "actor": mod["ahead_of"], "distance": mod["distance"]})

        if start_rel:
            block_constraints["npc"]["relative_at_start"] = start_rel
        if end_rel:
            block_constraints["npc"]["relative_at_end"] = end_rel

        constraints.append(block_constraints)

    return constraints


"""
def pytree_to_actor_constraints(pytree_blocks):
    constraints = []
    for block in pytree_blocks:
        if block is None:
            continue  # skip empty blocks

        block_constraints = {"duration_s": block.get("duration"), "ego": {}, "npc": {}}

        actors = block.get("actors", {})

        # Ego modifiers
        ego_actor = actors.get("ego_vehicle", {})
        for mod in ego_actor.get("modifiers", []):
            if "speed" in mod:
                block_constraints["ego"]["speed_range"] = mod["speed"]
            if "lane" in mod:
                block_constraints["ego"]["lane_at_start"] = mod["lane"]
            if "change_speed" in mod:
                block_constraints["ego"]["change_speed"] = mod["change_speed"]

        # NPC modifiers
        npc_actor = actors.get("npc", {})
        start_rel = []
        end_rel = []
        for mod in npc_actor.get("modifiers", []):
            if "right_of" in mod:
                start_rel.append({"relation": "right_of", "actor": mod["right_of"], "distance": mod.get("distance")})
            if "behind" in mod:
                start_rel.append({"relation": "behind", "actor": mod["behind"], "distance": mod["distance"]})
            if "ahead_of" in mod and mod.get("at") == "end":
                end_rel.append({"relation": "ahead_of", "actor": mod["ahead_of"], "distance": mod["distance"]})

        if start_rel:
            block_constraints["npc"]["relative_at_start"] = start_rel
        if end_rel:
            block_constraints["npc"]["relative_at_end"] = end_rel

        constraints.append(block_constraints)

    return constraints



def pytree_to_actor_constraints(pytree_blocks):
    constraints = []
    for block in pytree_blocks:
        block_constraints = {
            "duration_s": block.duration,
            "ego": {},
            "npc": {}
        }

        # --- Ego modifiers ---
        has_lane_or_change = False
        for mod in block.ego.modifiers:
            if "speed" in mod:
                block_constraints["ego"]["speed_range"] = mod["speed"]
            elif "lane" in mod:
                block_constraints["ego"]["lane_at_start"] = mod["lane"]
                has_lane_or_change = True
            elif "change_speed" in mod:
                block_constraints["ego"]["change_speed"] = mod["change_speed"]
            elif "change_lane" in mod:
                block_constraints["ego"]["change_lane"] = mod["change_lane"]
                has_lane_or_change = True

        # If ego drives without lane change/lane assignment → must stay in same lane
        if block.ego.action == "drive" and not has_lane_or_change:
            block_constraints["ego"]["stay_in_lane"] = True

        # --- NPC modifiers ---
        start_rel = []
        end_rel = []
        has_lane_or_change_npc = False
        for mod in block.npc.modifiers:
            if "right_of" in mod:
                start_rel.append({
                    "relation": "right_of",
                    "actor": mod["right_of"],
                    "distance": mod.get("distance")
                })
            elif "behind" in mod:
                start_rel.append({
                    "relation": "behind",
                    "actor": mod["behind"],
                    "distance": mod.get("distance")
                })
            elif "ahead_of" in mod:
                if mod.get("at") == "end":
                    end_rel.append({
                        "relation": "ahead_of",
                        "actor": mod["ahead_of"],
                        "distance": mod.get("distance")
                    })
            elif "lane" in mod or "change_lane" in mod:
                has_lane_or_change_npc = True

        if start_rel:
            block_constraints["npc"]["relative_at_start"] = start_rel
        if end_rel:
            block_constraints["npc"]["relative_at_end"] = end_rel

        # If NPC drives without lane change/lane assignment → must stay in same lane
        if block.npc.action == "drive" and not has_lane_or_change_npc:
            block_constraints["npc"]["stay_in_lane"] = True

        constraints.append(block_constraints)
    return constraints
"""
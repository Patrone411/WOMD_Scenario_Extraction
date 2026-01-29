def flat_list(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists

    if isinstance(list_of_lists[0], list):
        return flat_list(list_of_lists[0]) + flat_list(list_of_lists[1:])

    return list_of_lists[:1] + flat_list(list_of_lists[1:])

def print_pytree(pytree):
    for i, block in enumerate(pytree):
        if not block:
            continue

        # --- Setup/header row ---
        if block.get("type") == "Setup" or "block_name" not in block:
            print("Map Details:")
            paths = block.get("paths", {})
            if not paths:
                print("  (no paths)")
            else:
                for pname, pdata in paths.items():
                    print(f"  path '{pname}':")
                    if "min_lanes" in pdata:
                        print(f"    min_lanes: {pdata['min_lanes']}")
                    # optional: show call log for debugging
                    calls = pdata.get("calls", [])
                    for call in calls:
                        args = ", ".join(repr(a) for a in call.get("args", []))
                        kwargs = ", ".join(f"{k}={repr(v)}" for k, v in call.get("kwargs", {}).items())
                        sep = ", " if args and kwargs else ""
                        print(f"    call: {call.get('fn','?')}({args}{sep}{kwargs})")
            print()
            continue

        # --- Normal behavior blocks ---
        bname = block.get("block_name", f"Block#{i}")
        btype = block.get("type", "?")
        dur   = block.get("duration")
        print(f"Block: {bname} (Type: {btype}, Duration: {dur})")

        for actor, details in (block.get("actors") or {}).items():
            print(f"  Actor: {actor}")
            print(f"    Action: {details.get('action')}")
            print(f"    Args: {details.get('args')}")
            mods = details.get("modifiers") or []
            if mods:
                print("    Modifiers:")
                for mod in mods:
                    print(f"      - {mod}")
        print()
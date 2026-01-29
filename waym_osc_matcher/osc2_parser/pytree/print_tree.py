from typing import Any, List
from .pytree import ScenarioNode, SerialBlock, ParallelBlock, ActionCall, OneOfBlock, ModifierCall
from ..srunner.osc2_dm.physical_types import Physical, Range

try:
    from osc2_parser.config_init import _GenericPath
except Exception:
    _GenericPath = None

def _unit_abbr(u) -> str:
    # Try common abbreviations; fall back to unit's name
    name = getattr(u, "unit_name", None) or getattr(u, "name", None) or str(u)
    table = {
        "meter": "m", "second": "s", "kilometer_per_hour": "kph",
        "meter_per_second": "mps", "mile_per_hour": "mph",
        "kilometer_per_hour_per_sec": "kph/s",
        "mile_per_hour_per_sec": "mph/s",
        "degree": "deg", "radian": "rad",
        "millisecond": "ms", "minute": "min", "hour": "h",
        "millimeter": "mm", "centimeter": "cm", "kilometer": "km",
    }
    return table.get(name, name)

def _num_to_str(x: Any) -> str:
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        s = f"{x}"
        return s[:-2] if s.endswith(".0") else s
    return str(x)

def _fmt_value(v: Any) -> str:

     # Pretty physical values (and ranges of them)
    if isinstance(v, Physical):
        unit = _unit_abbr(v.unit)
        if isinstance(v.num, Range):
            return f"[low: {_num_to_str(v.num.start)} {unit}, high: {_num_to_str(v.num.end)} {unit}]"
        return f"{_num_to_str(v.num)} {unit}"
    if isinstance(v, Range):
        # was: return f"[{_num_to_str(v.start)}..{_num_to_str(v.end)}]"
        return f"[low: {_num_to_str(v.start)}, high: {_num_to_str(v.end)}]"
    # Pretty-print your temporary path objects
    if isinstance(v, _GenericPath):                  # <-- add this block
        name = v.get_name() or "path"
        ml = getattr(v, "min_lanes_required", None)
        extra = f", min_lanes={ml}" if ml is not None else ""
        return f"{name}{extra}"
    
    # (rest unchanged)
    if isinstance(v, Range): ...
    if isinstance(v, str):  return repr(v)
    return _num_to_str(v)


def _print_block(block, indent="  "):
    pad = indent
    if isinstance(block, SerialBlock):
        label = f"{block.label}:" if block.label else ""
        print(f"{pad}serial{(', ' + label) if label else ':'}")
        for ch in block.children:
            _print_block(ch, indent + "  ")

    elif isinstance(block, ParallelBlock):
        head = "parallel"
        if block.duration is not None:
            head += f", duration: {_fmt_value(block.duration)}"
        label = f" ({block.label})" if block.label else ""
        print(f"{pad}{head}:{label}")
        for ch in block.children:
            _print_block(ch, indent + "  ")
    
    elif isinstance(block, OneOfBlock):
        label = f" ({block.label})" if block.label else ""
        print(f"{pad}one_of:{label}")
        for ch in block.children:
            _print_block(ch, indent + "  ")

    elif isinstance(block, ActionCall):
        # --- Print action with its args ---
        parts: List[str] = []
        # In your IR, block.args can be a mixed list:
        #   - positional values
        #   - ("name", value) tuples for named args
        if getattr(block, "args", None):
            for a in block.args:
                if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
                    k, v = a
                    parts.append(f"{k}: {_fmt_value(v)}")  # OSC style "k: v"
                else:
                    parts.append(_fmt_value(a))
        inside = ", ".join(parts)
        print(f"{pad}{block.actor}.{block.action}({inside})")

        # --- Print modifiers (keep OSC style for named args) ---
        for m in block.modifiers:
            mparts: List[str] = []
            if getattr(m, "args", None):
                mparts.extend(_fmt_value(a) for a in m.args)
            if getattr(m, "kwargs", None):
                mparts.extend(f"{k}: {_fmt_value(v)}" for k, v in m.kwargs.items())
            mins = ", ".join(mparts)
            print(f"{pad}  with {m.name}({mins})")

    else:
        print(f"{pad}{block}")

def print_ir(scenarios: List[ScenarioNode]) -> None:
    for scn in scenarios:
        print(f"Scenario: {scn.name}")
        # Actors
        if scn.actors:
            print("  Actors:")
            for a in scn.actors.values():
                print(f"    - {a.name}: {a.type}")
        # Vars
        if scn.vars:
            print("  Vars:")
            for v in scn.vars.values():
                ty = f": {v.type}" if getattr(v, "type", None) else ""
                print(f"    - {v.name}{ty} = {_fmt_value(v.value)}")
        # Events
        if scn.events:
            print("  Events:")
            for e in scn.events:
                print(f"    - {e.name}")
        # Do / blocks
        if scn.blocks:
            print("  Do:")
            for b in scn.blocks:
                _print_block(b, indent="    ")

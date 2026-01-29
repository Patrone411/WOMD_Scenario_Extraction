# osc2_parser/constraints.py
from typing import Any, Dict, List
from osc2_parser.pytree.pytree import (
    ScenarioNode, SerialBlock, ParallelBlock, ActionCall, ModifierCall
)
try:
    # Optional: only if you already introduced OneOfBlock
    from osc2_parser.pytree.pytree import OneOfBlock
    _HAS_ONEOF = True
except Exception:
    OneOfBlock = tuple()  # harmless sentinel
    _HAS_ONEOF = False

from osc2_parser.srunner.osc2_dm.physical_types import Physical, Range

# First positional → canonical param name for legacy/positional uses
_POS0_PARAM = {
    "speed": "speed",
    "change_speed": "speed",
    "acceleration": "acceleration",
    "position": "distance",
    "distance": "distance",
    "lateral": "distance",
    "yaw": "angle",
    "lane": "lane",
    "change_lane": "lane",
}


def _normalize(v: Any) -> Any:
    """
    Physical/Range → JSON-friendly dict; recurse through dicts/lists/tuples.
    Leaves primitives and strings as-is (actor names, enums, etc.).
    """
    if isinstance(v, Physical):
        unit = getattr(v.unit, "unit_name", None) or getattr(v.unit, "name", None) or str(v.unit)
        if isinstance(v.num, Range):
            return {"range": [v.num.start, v.num.end], "unit": unit}
            
        return {"value": v.num, "unit": unit}
    if isinstance(v, Range):
        return {"range": [v.start, v.end]}
    if isinstance(v, dict):
        return {k: _normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_normalize(x) for x in v]
    return v

# Back-compat alias: some code may still call _value_to_filter
def _value_to_filter(v: Any) -> Any:
    return _normalize(v)

def _mod_to_named(m: ModifierCall) -> Dict[str, Any]:
    """
    Normalize a ModifierCall into named args (merge pos0 -> canonical).
    We only map the first positional; additional positionals are uncommon in OSC movement modifiers.
    """
    named: Dict[str, Any] = dict(getattr(m, "kwargs", {}) or {})
    pos = list(getattr(m, "args", []) or [])

    if pos:
        p0 = _POS0_PARAM.get(m.name)
        if p0 and p0 not in named:
            named[p0] = pos[0]
    return named

def _action_args_to_named(a: ActionCall) -> Dict[str, Any]:
    """
    Action args in your IR are provided as a list of (key, value) tuples.
    Convert to dict and normalize values (e.g., duration → {"value": 12, "unit": "second"}).
    """
    out: Dict[str, Any] = {}
    for item in (getattr(a, "args", []) or []):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            out[item[0]] = _normalize(item[1])
    return out

def _call_dict(ch: ActionCall) -> Dict[str, Any]:
    """Build a call dict WITHOUT block_duration (we keep duration at the block)."""
    # Modifiers → named args + normalized values
    mods: List[Dict[str, Any]] = []
    for m in ch.modifiers:
        m_named = _mod_to_named(m)
        mods.append({"name": m.name, "args": _normalize(m_named)})

    # Action args (e.g., duration) normalized
    a_named = _action_args_to_named(ch)

    return {
        "actor": ch.actor,
        "action": ch.action,
        "action_args": a_named,
        "modifiers": mods,
    }

def _block_type(b) -> str:
    if isinstance(b, SerialBlock):
        return "serial"
    if isinstance(b, ParallelBlock):
        return "parallel"
    if _HAS_ONEOF and isinstance(b, OneOfBlock):
        return "one_of"
    # fallback
    return b.__class__.__name__.lower()

def _collect_block(b) -> Dict[str, Any]:
    """
    Preserve block hierarchy:
      { type, label, duration, calls: [...], children: [...] }
    """
    block_dict: Dict[str, Any] = {
        "type": _block_type(b),
        "label": getattr(b, "label", None),
        "duration": _normalize(getattr(b, "duration", None)),
        "calls": [],
        "children": [],
    }

    for ch in getattr(b, "children", []):
        if isinstance(ch, ActionCall):
            block_dict["calls"].append(_call_dict(ch))
        elif isinstance(ch, (SerialBlock, ParallelBlock) + ((OneOfBlock,) if _HAS_ONEOF else ())):
            block_dict["children"].append(_collect_block(ch))
    return block_dict

def _flatten_blocks(block: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    """
    Optional flat view: collect all calls into a single list,
    annotating each call with its block label/type (but NOT duplicating duration).
    """
    for call in block.get("calls", []):
        out.append({
            **call,
            "block_label": block.get("label"),
            "block_type": block.get("type"),
            # If you *do* want to include the block duration per-call, uncomment the next line:
            # "block_duration": block.get("duration"),
        })
    for child in block.get("children", []):
        _flatten_blocks(child, out)

def constraints_from_ir(scenarios):
    """
    Returns a per-scenario dict with preserved block structure:

    {
      "<scenario>": {
        "actors": {"ego_vehicle":{"type":"vehicle"}, ...},
        "blocks": [
          {
            "type": "parallel",
            "label": "get_ahead",
            "duration": {"value": 15, "unit": "second"},
            "calls": [
              {"actor":"ego_vehicle","action":"drive","action_args":{"duration":{"value":12,"unit":"second"}}, "modifiers":[...]}
            ],
            "children": [...]
          }
        ],
        "calls_flat": [  # optional convenience
          {"actor":"ego_vehicle","action":"drive", "action_args": {...}, "modifiers":[...], "block_label":"get_ahead", "block_type":"parallel"}
        ]
      }
    }
    """

    if isinstance(scenarios, dict):
        scn_list = list(scenarios.values())
    elif isinstance(scenarios, (list, tuple)):
        scn_list = scenarios
    else:
        raise TypeError("scenarios must be list/tuple of ScenarioNode or dict[str, ScenarioNode]")
    
    out: Dict[str, Any] = {}
    for scn in scn_list:
        #for name, inst in scn.actors.items():
            #print("name: ", name, " type: ", inst.type)
        scn_dict: Dict[str, Any] = {
            "actors": {name: {"type": inst.type} for name, inst in scn.actors.items()},
            "blocks": [],
            "calls_flat": [],  # optional; remove if you don't need flat view
        }
        # top-level blocks
        for blk in scn.blocks:
            bdict = _collect_block(blk)
            scn_dict["blocks"].append(bdict)
            _flatten_blocks(bdict, scn_dict["calls_flat"])
        out[scn.name] = scn_dict
    return out

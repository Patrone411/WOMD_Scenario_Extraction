# osc2_parser/temporal_resolver.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional
from copy import deepcopy

def _unwrap_quantity(q: Optional[Dict[str, Any]]):
    if not q: return None, None
    if "value" in q:  return q["value"], q.get("unit")
    if "range" in q:  return q["range"], q.get("unit")
    return q, None

def _to_seconds(q: Optional[Dict[str, Any]]) -> Optional[float]:
    if not q: return None
    val, unit = _unwrap_quantity(q)
    if val is None: return None
    if isinstance(val, (list, tuple)):
        # pick upper bound as the window length; adjust if your IR means something else
        val = float(val[-1])
    else:
        val = float(val)
    u = (unit or "s").lower()
    if u in {"s","sec","secs","second","seconds"}: return val
    if u in {"ms","millisecond","milliseconds"}:   return val/1000.0
    if u in {"min","mins","minute","minutes"}:    return val*60.0
    return val

def _normalize_seconds(s: float) -> Dict[str, Any]:
    # keep normalized quantity format; you can also carry a frames field later if you like
    return {"value": float(s), "unit": "s"}

def resolve_durations_in_constraints(
    constraints_by_scenario: Mapping[str, Any],
    *,
    policy: str = "strict",   # "strict" | "prefer_call" | "prefer_block"
) -> Dict[str, Any]:
    """
    Returns a deep-copied constraints dict where:
      - Each block carries 'effective_duration' if it or an ancestor had a duration.
      - For PARALLEL blocks, any child call missing duration inherits block effective_duration.
      - Optionally reconcile conflicts per 'policy'.
    """
    root = deepcopy(constraints_by_scenario)

    def walk_block(b: Dict[str, Any], inherited_q: Optional[Dict[str, Any]]) -> None:
        this_q = b.get("duration") or inherited_q
        if this_q is not None:
            # store effective duration on the block for downstream consumers
            s = _to_seconds(this_q)
            if s is not None:
                b["effective_duration"] = _normalize_seconds(s)

        btype = str(b.get("type", "")).lower()

        # forward to calls for PARALLEL blocks
        if btype == "parallel":
            for call in b.get("calls", []):
                aa = call.setdefault("action_args", {})
                call_q = aa.get("duration")
                if call_q is None and b.get("effective_duration") is not None:
                    aa["duration"] = deepcopy(b["effective_duration"])
                elif call_q is not None and b.get("effective_duration") is not None:
                    # conflict resolution
                    if policy == "prefer_block":
                        aa["duration"] = deepcopy(b["effective_duration"])
                    elif policy == "strict":
                        # quick numeric check; relax with tolerances if needed
                        cs = _to_seconds(call_q)
                        bs = _to_seconds(b["effective_duration"])
                        if cs is not None and bs is not None and abs(cs - bs) > 1e-6:
                            raise ValueError(
                                f"Duration mismatch in parallel block '{b.get('label')}' "
                                f"(call={cs}s, block={bs}s)."
                            )
                    # prefer_call: do nothing

        # recurse
        for c in b.get("children", []):
            walk_block(c, this_q)

    for scn in root.values():
        for blk in scn.get("blocks", []):
            walk_block(blk, inherited_q=None)

        # OPTIONAL: patch the flat view too so single-call matchers “just work”
        flat = scn.get("calls_flat", [])
        # Build a quick lookup of block_label -> effective_duration
        eff_by_label = {}
        def collect(b):
            if "effective_duration" in b and b.get("label") is not None:
                eff_by_label[b["label"]] = b["effective_duration"]
            for ch in b.get("children", []):
                collect(ch)
        for b in scn.get("blocks", []):
            collect(b)

        for call in flat:
            aa = call.setdefault("action_args", {})
            if "duration" not in aa:
                bl = call.get("block_label")
                if bl in eff_by_label:
                    aa["duration"] = deepcopy(eff_by_label[bl])

    return root

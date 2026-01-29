# srunner/osc2/semantics/ir_adapter.py
from typing import Dict, List, Tuple
from osc2_parser.pytree.pytree import ScenarioNode, SerialBlock, ParallelBlock, ActionCall, ModifierCall
from .validator import ActionCall as VActionCall, ModifierAttach as VModifierAttach, ArgValue
from osc2_parser.srunner.osc2_dm.physical_types import Physical
from osc2_parser.config_init import _GenericPath
from typing import Any, Dict, Iterable, List, Optional, Union

def _extract_ttc_rise_predicate(pred_like) -> Tuple[Optional[str], Optional[object], Optional[object]]:
    """
    Try to pull out:
        reference (actor name), threshold_time (Physical or dict), sample_every (Physical or dict)
    from a few likely shapes:
      • {"op":"<", "left": <call>, "right": <time>}
      • {"<":[ <call>, <time> ]}
      • ("<", <call>, <time>)
      • {"lt":[...]}  (alias)
      • optional sampling wrapper:
           {"sample": <call>, "every": <time>} as the *left* side
    where <call> looks like:
      • {"method":"time_to_collision", "reference":"npc"}  (kwargs)
      • {"method":"time_to_collision", "args":[ego, npc]}  (pos)
      • ("time_to_collision", ego, npc)  (tuple form)
    Returns (reference, threshold_time, sample_every) or (None, None, None).
    """
    def _is_lt(d):
        if isinstance(d, dict):
            if "op" in d and str(d["op"]).lower() in ("<", "lt", "less_than"):
                return d.get("left"), d.get("right")
            if "<" in d and isinstance(d["<"], (list, tuple)) and len(d["<"]) >= 2:
                return d["<"][0], d["<"][1]
            if "lt" in d and isinstance(d["lt"], (list, tuple)) and len(d["lt"]) >= 2:
                return d["lt"][0], d["lt"][1]
        if isinstance(d, (list, tuple)) and len(d) >= 3 and str(d[0]).lower() in ("<","lt"):
            return d[1], d[2]
        return None, None

    def _extract_method_ref(call_like):
        """Return (method_name, reference, sampling_every_opt)."""
        # sampling wrapper on the left? {"sample": <call>, "every": <time>}
        if isinstance(call_like, dict) and "sample" in call_like:
            inner = call_like.get("sample")
            m, ref, _ = _extract_method_ref(inner)
            return m, ref, call_like.get("every")

        # dict call with explicit method name
        if isinstance(call_like, dict):
            mname = str(call_like.get("method", "")).lower() or str(call_like.get("name", "")).lower()
            ref = call_like.get("reference")
            if ref is None:
                # try kwargs bag
                kwargs = call_like.get("kwargs") or {}
                if isinstance(kwargs, dict):
                    ref = kwargs.get("reference")
            if ref is None:
                # try args (positional)
                args = call_like.get("args")
                if isinstance(args, (list, tuple)) and len(args) >= 2:
                    ref = args[1]  # assume args=[ego, ref]
            return mname, ref, None

        # tuple-ish call: ("time_to_collision", ego, ref)
        if isinstance(call_like, (list, tuple)) and call_like:
            tag = str(call_like[0]).lower()
            if tag in ("time_to_collision", "ttc"):
                ref = call_like[2] if len(call_like) >= 3 else None
                return tag, ref, None

        return None, None, None

    left, right = _is_lt(pred_like)
    if left is None:
        return None, None, None

    mname, ref, every = _extract_method_ref(left)
    if mname not in ("time_to_collision", "ttc"):
        return None, None, None

    # threshold time is right
    thr = right
    return ref, thr, every

def _collect_symbols(scn) -> dict:
    """Gather name→value bindings defined in the scenario (best-effort)."""
    env = {}
    for attr in ("symbols", "variables", "consts", "constants", "params"):
        src = getattr(scn, attr, None)
        if isinstance(src, dict):
            # allow objects with .value
            for k, v in src.items():
                env[k] = getattr(v, "value", v)
        elif isinstance(src, (list, tuple)):
            for item in src:
                name = getattr(item, "name", None)
                val  = getattr(item, "value", None)
                if name is not None:
                    env[name] = val
    return env

def _resolve_names(obj, env: dict):
    """Recursively replace strings that are variable names with their bound values."""
    if not env:
        return obj
    if isinstance(obj, str) and obj in env:
        return _resolve_names(env[obj], env)  # handle chained aliases
    if isinstance(obj, dict):
        return {k: _resolve_names(v, env) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        T = type(obj)
        return T(_resolve_names(v, env) for v in obj)
    return obj

# Map each modifier name to the canonical param its *first positional* represents
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

def _flatten(seq):
    """Flatten arbitrarily nested lists/tuples."""
    if isinstance(seq, (list, tuple)):
        for x in seq:
            yield from _flatten(x)
    else:
        yield seq

def _normalize_odr_point(v):
    """
    Canonicalize an ODR point to:
        {"road": <id>, "lane": <id>, "s": <length>, ["t": <length>]}
    Accepts:
      • dict with any of the common key spellings (road/road_id, lane/lane_id; s,t)
      • tagged list/tuple forms (possibly nested):
            ["odr_point", road, lane, s] or ["odr_point", road, lane, s, t]
        and the same without the tag if you already know it’s odr-like.
    """
    if isinstance(v, dict):
        keys = {str(k).lower(): k for k in v.keys()}
        def _get(*alts):
            for a in alts:
                if a in keys:
                    return v[keys[a]]
            return None
        road = _get("road", "road_id", "roadid")
        lane = _get("lane", "lane_id", "laneid")
        s    = _get("s")
        t    = _get("t")
        out = {}
        if road is not None: out["road"] = road
        if lane is not None: out["lane"] = lane
        if s    is not None: out["s"]    = s
        if t    is not None: out["t"]    = t
        return out if ("road" in out and "lane" in out and "s" in out) else {}

    if isinstance(v, (list, tuple)):
        flat = list(_flatten(v))
        if not flat:
            return {}
        tag = str(flat[0]).lower()
        items = flat[1:] if tag in ("odr_point", "odrpoint", "odr") else flat
        # road, lane, s[, t]
        if len(items) >= 3:
            out = {"road": items[0], "lane": items[1], "s": items[2]}
            if len(items) >= 4:
                out["t"] = items[3]
            return out
    return {}

def _normalize_position_3d(v):
    """
    Canonicalize a position to {"x","y"[,"z"]} OR {"s","t"}.
    Accepts dicts and tagged list/tuple forms like ["position", x, y, z?] or ["position", s, t].
    """
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            lk = str(k).lower()
            if lk in ("x", "y", "z", "s", "t"):
                out[lk] = val
        # Only accept if it's clearly XY or ST
        if {"x","y"}.issubset(out.keys()) or {"s","t"}.issubset(out.keys()):
            return out
        return {}

    if isinstance(v, (list, tuple)):
        flat = list(_flatten(v))
        if not flat:
            return {}
        tag = str(flat[0]).lower()
        items = flat[1:] if tag in ("pos", "position", "position_3d") else flat
        if len(items) >= 2:
            # Heuristic: if items look like (x,y[,z]) use xy; otherwise treat as (s,t[,z])
            out = {}
            # We can't reliably distinguish XY vs ST just by numbers; prefer XY default.
            out["x"], out["y"] = items[0], items[1]
            if len(items) >= 3:
                out["z"] = items[2]
            return out
    return {}

def _normalize_orientation_3d(v):
    """
    Canonicalize to {"yaw":..., "pitch":..., "roll":...} (any subset).
    Accepts:
      • dict with yaw/pitch/roll or {"angle": ...} alias for yaw
      • scalar/Physical → yaw-only
      • tagged list/tuple ["orientation", yaw, pitch?, roll?]
    """
    try:
        from osc2_parser.srunner.osc2_dm.physical_types import Physical
        _Physical = Physical
    except Exception:
        _Physical = ()

    if isinstance(v, (int, float, _Physical)):
        return {"yaw": v}

    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            lk = str(k).lower()
            if lk in ("yaw","pitch","roll"):
                out[lk] = val
            elif lk == "angle":  # alias for yaw
                out["yaw"] = val
        return out

    if isinstance(v, (list, tuple)):
        flat = list(_flatten(v))
        if not flat:
            return {}
        tag = str(flat[0]).lower()
        items = flat[1:] if tag in ("ori", "orientation", "orientation_3d") else flat
        out = {}
        if len(items) >= 1: out["yaw"]   = items[0]
        if len(items) >= 2: out["pitch"] = items[1]
        if len(items) >= 3: out["roll"]  = items[2]
        return out
    return {}

def _first_positional_param_for(mod_name: str) -> str:
    return _POS0_PARAM.get(mod_name)

def _to_argdict(action_name: str, args_list):
    """
    Convert IR ActionCall.args (list of scalars or (k,v)) into named args.

    Heuristics:
      • assign_position:
            - First positional can be: odr_point (tagged list or dict), position_3d (xy or st), or route_point.
            - We normalize to canonical dicts and choose the proper param name.
      • assign_orientation:
            - First positional can be a scalar/Physical angle (yaw-only) or orientation dict/tag → normalize to {"yaw":..[, "pitch","roll"]}.
      • assign_speed / assign_acceleration:
            - First positional becomes named 'speed' / 'acceleration' if not already provided.
      • Otherwise:
            - Only map first positional to 'duration' if it's clearly a time Physical.
    """
    named = {}
    pos = []
    for a in args_list:
        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
            k, v = a
            lk = k.lower()
            # Normalize known struct-like values written as named args too
            if lk in ("odr_point", "odrpoint", "odr"):
                v = _normalize_odr_point(v) or v
            elif lk in ("position", "position_3d"):
                v = _normalize_position_3d(v) or v
            elif lk in ("orientation", "orientation_3d"):
                v = _normalize_orientation_3d(v) or v
            named[k] = v
        else:
            pos.append(a)

    if not pos:
        # Nothing positional to map
        return named

    v0 = pos[0]
    an = (action_name or "").lower()

    # ---------- Special-cases for actions with positional short-hands ----------
    if an == "change_lane":
        # Support both signatures:
        #   change_lane(num_of_lanes, side, reference, [...])
        #   change_lane(target=lane, [...])  (or positional first value treated as target if it looks like a lane handle)
        # 1) Copy any explicit named keys (already in `named`)
        # 2) Map positionals if present and not already named
        if pos:
            # Heuristic: if first positional is an int → treat as num_of_lanes signature
            if isinstance(pos[0], int):
                named.setdefault("num_of_lanes", pos[0])
                if len(pos) >= 2:
                    named.setdefault("side", pos[1])
                if len(pos) >= 3:
                    named.setdefault("reference", pos[2])
            else:
                # Otherwise assume it's a target lane handle/id provided positionally
                # (e.g., change_lane(my_lane))
                named.setdefault("target", pos[0])

        # Normalize a few synonyms that show up in the wild
        if "num_lanes" in named and "num_of_lanes" not in named:
            named["num_of_lanes"] = named.pop("num_lanes")
        if "ref" in named and "reference" not in named:
            named["reference"] = named.pop("ref")

        return named
    
    if an == "assign_position":
        # 1) Try odr_point first (handles tagged/nested lists)
        odr = _normalize_odr_point(v0)
        if odr:
            named.setdefault("odr_point", odr)
            return named

        # 2) Then try position_3d (xy or st)
        pos3 = _normalize_position_3d(v0)
        if pos3:
            named.setdefault("position", pos3)
            return named

        # 3) Route-like object
        try:
            from osc2_parser.config_init import _GenericPath
            if isinstance(v0, _GenericPath):
                named.setdefault("route_point", v0)
                return named
        except Exception:
            pass

        # 4) Fallback: treat unknown dicts as position, everything else as odr_point
        if isinstance(v0, dict):
            named.setdefault("position", v0)
        else:
            named.setdefault("odr_point", _normalize_odr_point(v0) or v0)
        return named

    if an in ("assign_speed", "assign_acceleration", "assign_orientation"):
        # map first positional → target (if provided)
        if pos and "target" not in named:
            named["target"] = pos[0]

        # accept common synonyms and normalize to 'target'
        for syn in ("speed", "acceleration", "orient", "orientation", "orientation_3d"):
            if syn in named and "target" not in named:
                named["target"] = named.pop(syn)

        # don't auto-map duration unless it's clearly a time Physical (you already do that below)
        return named

    # ---------- Generic: map first positional to duration only if clearly 'time' ----------
    if "duration" not in named:
        try:
            from osc2_parser.srunner.osc2_dm.physical_types import Physical
            if isinstance(v0, Physical):
                phys_name = getattr(getattr(v0.unit, "physical", None), "name", None) \
                            or getattr(getattr(v0.unit, "physical", None), "type_name", None)
                uname = (getattr(v0.unit, "unit_name", None) or getattr(v0.unit, "name", None) or "").lower()
                if phys_name == "time" or uname in {
                    "s","sec","second","seconds","ms","millisecond","milliseconds",
                    "min","minute","minutes","h","hr","hour","hours"
                }:
                    named["duration"] = v0
        except Exception:
            pass

    return named

def _mod_to_named(m: ModifierCall) -> Dict[str, object]:
    named, pos = {}, []
    for a in m.args:
        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
            named[a[0]] = a[1]
        else:
            pos.append(a)

    if pos:
        if m.name in ("position", "lateral", "distance"):
            named.setdefault("distance", pos[0])
        elif m.name in ("speed", "change_speed"):
            named.setdefault("speed", pos[0])
        elif m.name in ("acceleration", "change_acceleration"):
            named.setdefault("acceleration", pos[0])
        elif m.name == "along":
            named.setdefault("route", pos[0])
        elif m.name == "yaw":
            named.setdefault("angle", pos[0])
        #TODO: check true lane counts using topology
        elif m.name == "lane":
            # Positional handling first (index / "from" shorthand)
            # e.g. lane(3), lane(2, "left")
            if pos and "lane" not in named and all(k not in named for k in ("side_of", "same_as", "right_of", "left_of")):
                named.setdefault("lane", pos[0])
            if len(pos) >= 2 and "side" not in named and isinstance(pos[1], str):
                named.setdefault("side", pos[1])

            # --- normalize synonyms to canonical keys expected by validator/spec ---
            # right_of / left_of -> side_of + side
            if "right_of" in named:
                named.setdefault("side_of", named.pop("right_of"))
                named.setdefault("side", "right")
            if "left_of" in named:
                named.setdefault("side_of", named.pop("left_of"))
                named.setdefault("side", "left")

            # 'from' is an alias for 'side' in the "index" + side distance-1 case
            if "from" in named and "side" not in named:
                named["side"] = named.pop("from")

            # Finally merge kwargs (so explicit kwargs win over positional defaults)
            for k, v in (m.kwargs or {}).items():
                named[k] = v

        elif m.name == "change_lane":
            # pos[0] → lane delta (number), pos[1] → side/from
            if pos:
                named.setdefault("lane", pos[0])
            if len(pos) >= 2:
                if isinstance(pos[1], str):
                    named.setdefault("side", pos[1])

            # allow 'from' as an alias for 'side'
            if "from" in named and "side" not in named:
                named["side"] = named.pop("from")

            # DEFAULT: if side provided but no lane delta → 1 lane
            if "side" in named and "lane" not in named:
                named["lane"] = 1

        elif m.name == "until":
            # Expect a single positional predicate dict from visit_modifier_invocation('rise')
            named, pos = {}, []
            for a in m.args:
                if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
                    named[a[0]] = a[1]
                else:
                    pos.append(a)

            pred = pos[0] if pos else named.get("predicate")
            # pred should look like {"op":"<", "left": {...}, "right": Physical(time)}
            if isinstance(pred, dict) and pred.get("op") in ("<", "<=", ">", ">=", "==", "!="):
                L, R = pred.get("left"), pred.get("right")

                # Support both orders: TTC < T  or  T > TTC
                def _is_ttc(x):
                    return isinstance(x, dict) and x.get("method") == "time_to_collision"

                if _is_ttc(L) and isinstance(R, Physical):
                    ttc = L
                    named = {
                        "reference": ttc.get("reference"),
                        "time": R
                    }
                elif _is_ttc(R) and isinstance(L, Physical):
                    ttc = R
                    named = {
                        "reference": ttc.get("reference"),
                        "time": L
                    }
                else:
                    # Not the TTC pattern → leave as-is so validator can fail cleanly
                    named["predicate"] = pred
            else:
                named["predicate"] = pred

            return named

    for k, v in (m.kwargs or {}).items():
        named[k] = v
    return named

def _infer_type_for_value(v, actor_types: dict) -> str:
    # Physical -> "length"/"time"/"speed"/...
    if isinstance(v, Physical):
        try:
            return v.unit.physical.name
        except Exception:
            return None

    # Route path object
    if isinstance(v, _GenericPath):
        return "route"

    # Strings (actor names, simple enums)
    if isinstance(v, str):
        if v in actor_types:
            return "physical_object"
        low = v.lower()
        if low in ("start", "end"):
            return "at"
        if low in ("left", "right"):
            return "side_left_right"
        return None

    # Dict-shaped domain valuesf
    if isinstance(v, (list, tuple)):
        flat = list(_flatten(v))
        if flat:
            tag = str(flat[0]).lower()
            if tag in ("odr_point", "odrpoint", "odr"):
                return "odr_point"
            if tag in ("pos", "position", "position_3d"):
                return "position_3d"
            if tag in ("ori", "orientation", "orientation_3d"):
                return "orientation_3d"

    # Plain numerics / bools
    if isinstance(v, bool):  return "bool"
    if isinstance(v, int):   return "int"
    if isinstance(v, float): return "float"
    return None

def _collect_symbols_deep(root) -> dict:
    """
    Walk the ScenarioNode tree and gather name→value bindings.
    Robust against different IR shapes:
      - objects with .name and a value in one of {value, val, literal, number, expr}
      - dicts / lists / tuples (recursive)
    """
    env = {}
    seen = set()

    def _unwrap(v):
        # unwrap single-layer wrappers that carry .value
        return getattr(v, "value", v)

    def visit(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        # primitives
        if isinstance(obj, (str, int, float, bool, type(None))):
            return

        # named binding objects
        nm = getattr(obj, "name", None)
        if isinstance(nm, str) and nm and nm not in env:
            for attr in ("value", "val", "literal", "number", "expr"):
                if hasattr(obj, attr):
                    env[nm] = _unwrap(getattr(obj, attr))
                    break

        # dict
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(k); visit(v)
            return

        # sequences
        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                visit(v)
            return

        # generic object: recurse into public attrs
        d = getattr(obj, "__dict__", {})
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ("parent", "token", "span"):  # avoid cycles/noise
                    continue
                visit(v)

    visit(root)
    return env



from typing import List, Tuple

def validate_from_ir(scenarios: List[ScenarioNode], validator) -> Tuple[bool, List[str]]:
    """
    Validate lowered IR with `validator`.

    Returns:
        (passed, errors)
        - passed=True and errors=[] if everything validated
        - passed=False and errors=[...] with human-readable messages otherwise
    """
    any_fail = False
    error_log: List[str] = []

    def _log(msg: str):
        nonlocal any_fail
        any_fail = True
        error_log.append(msg)

    def _stringify_issue(iss) -> str:
        # Be tolerant of many shapes (str, objects with .message, .level, .code, etc.)
        if isinstance(iss, str):
            return iss
        level = getattr(iss, "level", None) or getattr(iss, "severity", None)
        code  = getattr(iss, "code",  None)
        msg   = getattr(iss, "message", None) or str(iss)
        parts = []
        if level: parts.append(str(level))
        if code:  parts.append(f"[{code}]")
        parts.append(msg)
        return " ".join(parts)

    for scn in scenarios:
        actor_types = {name: inst.type for name, inst in scn.actors.items()}
        name_env = _collect_symbols_deep(scn)

        def walk_block(block):
            for ch in getattr(block, "children", []):
                if isinstance(ch, ActionCall):
                    inv_name = ch.actor
                    inv_type = actor_types.get(inv_name)
                    if not inv_type:
                        _log(f"[{scn.name}] Unknown actor '{inv_name}' used by action '{getattr(ch, 'action', '?')}'.")
                        continue

                    # --- ACTION ARGS ---
                    a_named = _to_argdict(ch.action, ch.args)
                    a_named = _resolve_names(a_named, name_env)

                    v_args = {
                        k: ArgValue(k, v, type_name=_infer_type_for_value(v, actor_types))
                        for k, v in a_named.items()
                    }

                    # --- MODIFIERS ---
                    v_mods = []
                    for m in ch.modifiers:
                        m_named = _mod_to_named(m)
                        m_named = _resolve_names(m_named, name_env)
                        v_mods.append(
                            VModifierAttach(
                                name=m.name,
                                args={k: ArgValue(k, v, type_name=_infer_type_for_value(v, actor_types))
                                      for k, v in m_named.items()},
                                token=getattr(m, "token", None),
                            )
                        )

                    vcall = VActionCall(
                        invoker_name=inv_name,
                        invoker_type=inv_type,
                        method_name=ch.action,
                        args=v_args,
                        modifiers=v_mods,
                        token=getattr(ch, "token", None),
                    )

                    # --- VALIDATE ---
                    try:
                        result = validator.validate_action_call(vcall)
                    except Exception as e:
                        _log(f"[{scn.name}] {inv_name}.{ch.action}: validator raised {type(e).__name__}: {e}")
                        continue

                    # Detect pass/fail from result with broad compatibility
                    passed = None
                    for attr in ("ok", "success", "valid", "passed"):
                        if hasattr(result, attr):
                            passed = bool(getattr(result, attr))
                            break
                    if passed is None:
                        if hasattr(result, "errors"):
                            passed = len(getattr(result, "errors") or []) == 0
                        elif hasattr(result, "issues"):
                            issues = getattr(result, "issues") or []
                            passed = not any(getattr(i, "level", "").lower() == "error" for i in issues)
                        else:
                            passed = True  # assume success if no signal

                    if not passed:
                        # Gather messages
                        msgs = []
                        if hasattr(result, "errors") and result.errors:
                            msgs.extend(result.errors)
                        elif hasattr(result, "issues") and result.issues:
                            msgs.extend(result.issues)
                        elif hasattr(result, "messages") and result.messages:
                            msgs.extend(result.messages)
                        else:
                            msgs = ["validation failed (no details provided)"]

                        for iss in msgs:
                            _log(f"[{scn.name}] {inv_name}.{ch.action}: {_stringify_issue(iss)}")

                    # --- WRITE-BACK (keep behavior even if failed; guarded) ---
                    try:
                        ch.args = []
                        for k, v in getattr(result, "resolved_args", {}).items():
                            ch.args.append((k, v))
                    except Exception:
                        # ignore if result doesn't expose resolved_args
                        pass

                    # Normalize modifiers back onto the IR as named kwargs
                    try:
                        new_mods = []
                        for vm in v_mods:
                            named = {k: av.value for k, av in vm.args.items()}
                            try:
                                mc = ModifierCall(name=vm.name, args=[], kwargs=named)
                            except TypeError:
                                pairs = list(named.items())
                                try:
                                    mc = ModifierCall(name=vm.name, args=pairs)
                                except TypeError:
                                    mc = ModifierCall(vm.name, pairs)
                            try:
                                if getattr(vm, "token", None) is not None and hasattr(mc, "token"):
                                    mc.token = vm.token
                            except Exception:
                                pass
                            new_mods.append(mc)
                        ch.modifiers = new_mods
                    except Exception as e:
                        _log(f"[{scn.name}] {inv_name}.{ch.action}: failed to write back modifiers: {e}")

                elif isinstance(ch, (SerialBlock, ParallelBlock)):
                    walk_block(ch)

        for blk in scn.blocks:
            walk_block(blk)

    return (not any_fail, error_log)


def _iter_scenarios(scenarios: Union[Dict[str, Any], Iterable[Any], Any]) -> Iterable[Any]:
    """
    Yield ScenarioNode objects from:
      - a dict {name: ScenarioNode}
      - a list/iterable of ScenarioNode
      - a single ScenarioNode
    """
    if scenarios is None:
        return []
    if isinstance(scenarios, dict):
        return scenarios.values()
    if isinstance(scenarios, (list, tuple, set)):
        return scenarios
    # assume single ScenarioNode
    return [scenarios]

def _pick_scenario(
    scenarios: Union[Dict[str, Any], Iterable[Any], Any],
    scenario_name: Optional[str]
) -> Optional[Any]:
    """
    Pick the requested scenario by name (if given), else the first one.
    """
    for scn in _iter_scenarios(scenarios):
        if scenario_name is None:
            return scn
        name = getattr(scn, "name", None)
        if name == scenario_name:
            return scn
    return None

def get_min_lanes(
    scenarios: Union[Dict[str, Any], Iterable[Any], Any],
    *,
    scenario_name: Optional[str] = None,
    default: Optional[int] = None,
) -> Optional[int]:
    """
    Return `path.min_lanes_required` from the given IR scenarios.

    Works regardless of whether your IR is a dict ({name: ScenarioNode}),
    a list of ScenarioNode, or a single ScenarioNode.

    It looks in this order:
      1) The merged name environment from `_collect_symbols_deep(scn)` for 'path'
      2) `scn.vars['path'].value` if present
    and accepts any of the attributes: `min_lanes_required`, `min_lanes`, `lanes_required`.

    Args:
        scenarios: IR from IRLowering.lower(...) (dict/list/ScenarioNode)
        scenario_name: pick a specific scenario (e.g., "top"); if None, use the first
        default: returned if nothing is found

    Returns:
        int or `default` if not present.
    """
    scn = _pick_scenario(scenarios, scenario_name)
    if scn is None:
        return default

    # 1) Try the merged symbol environment that validate_from_ir already uses
    try:
        env = _collect_symbols_deep(scn)  # existing helper in this module
    except Exception:
        env = {}

    path_obj = env.get("path")

    # 2) Fallback to the IR variable table
    if path_obj is None:
        path_var = getattr(scn, "vars", {}).get("path")
        if path_var is not None:
            # VarNode has a .value; if path_var is already the object, getattr returns itself
            path_obj = getattr(path_var, "value", path_var)

    if path_obj is None:
        return default

    # Accept common attribute spellings
    for attr in ("min_lanes_required", "min_lanes", "lanes_required"):
        val = getattr(path_obj, attr, None)
        if val is not None:
            try:
                return int(val)
            except Exception:
                pass

    # Also handle dict-like path objects, just in case
    if isinstance(path_obj, dict):
        for key in ("min_lanes_required", "min_lanes", "lanes_required"):
            if key in path_obj:
                try:
                    return int(path_obj[key])
                except Exception:
                    pass

    return default
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from osc2_parser.srunner.osc2_dm.physical_types import Physical, Range

from .registry import SemanticsRegistry, ActionSpec, OverloadSpec, ModifierSpec, ModifierVariantSpec

# Fallback logging if your project logger isn't available
try:
    from osc2_parser.srunner.osc2.utils.log_manager import LOG_ERROR, LOG_WARN
except Exception:
    def LOG_ERROR(msg, token=None): raise ValueError(msg)
    def LOG_WARN(msg, token=None): print("WARN:", msg)

def _flatten(seq):
    """Flatten arbitrarily nested lists/tuples."""
    if isinstance(seq, (list, tuple)):
        for x in seq:
            yield from _flatten(x)
    else:
        yield seq
        
# ---------- Public API types ----------

@dataclass
class ArgValue:
    """One actual argument in a call/modifier clause."""
    name: str
    value: Any
    type_name: Optional[str] = None  # optional; filled by type_of_expr hook if needed

@dataclass
class ModifierAttach:
    name: str
    args: Dict[str, ArgValue]
    token: Any = None

@dataclass
class ActionCall:
    invoker_name: str
    invoker_type: str
    method_name: str
    args: Dict[str, ArgValue]
    modifiers: List[ModifierAttach]
    token: Any = None

@dataclass
class ValidationResult:
    resolved_action_qname: str
    resolved_action_overload: str  # overload name or "<default>"
    resolved_args: Dict[str, Any]
    resolved_modifiers: List[Tuple[str, str, Dict[str, Any]]]  # (mod_name, variant_name, args)

def infer_type(arg: ArgValue):
    v = arg.value
    if isinstance(v, Physical):
        u = v.unit

        # Try common attribute names for the physical dimension object
        phys_obj = (
            getattr(u, "physical", None)
            or getattr(u, "physical_object", None)
            or getattr(u, "physical_type", None)
        )

        # Try to pull a name off the physical object
        phys_name = (
            getattr(phys_obj, "type_name", None)
            or getattr(phys_obj, "name", None)
            or getattr(phys_obj, "physical_name", None)
        )

        if phys_name:
            # e.g. "length", "time", "speed", "acceleration"
            #print(f"[infer_type] {arg.name}={v} -> {phys_name} (unit={getattr(u,'unit_name',getattr(u,'name','?'))})") #debug
            return phys_name

        # Fallback: infer by canonical unit name
        uname = getattr(u, "unit_name", None) or getattr(u, "name", None)

        time_units = {"second","millisecond","minute","hour"}
        length_units = {"meter","millimeter","centimeter","kilometer","inch","feet","mile","micrometer"}
        speed_units = {"meter_per_second","kilometer_per_hour","mile_per_hour"}
        accel_units = {"meter_per_sec_sqr","kilometer_per_hour_per_sec","mile_per_hour_per_sec","feet_per_sec_sqr"}
        angle_units = {"degree","radian"}
        if uname in time_units:   return "time"
        if uname in length_units: return "length"
        if uname in speed_units:  return "speed"
        if uname in accel_units:  return "acceleration"
        if uname in angle_units: return "angle"

        # Last resort: unknown physical
        # print(f"[infer_type] {arg.name}: unknown unit '{uname}' -> None")
        return None
    
    if isinstance(v, dict):
        ks = {str(k).lower() for k in v.keys()}
        # odr_point: road/lane (+ optional s/t)
        if ({"road", "road_id", "roadid"} & ks) and ({"lane", "lane_id", "laneid"} & ks):
            return "odr_point"
        # position_3d: (x,y[,z]) OR (s,t[,h])
        if {"x","y"} <= ks or {"s","t"} <= ks:
            return "position_3d"
        # orientation_3d: any of yaw/pitch/roll
        if {"yaw","pitch","roll"} & ks:
            return "orientation_3d"
        # route_point-ish
        if {"route_point","route","path"} & ks:
            return "route_point"

    if isinstance(v, (list, tuple)):
        # Accept tagged tuples like ["odr_point", road, lane, s, t]
        def _flatten(x):
            if isinstance(x, (list, tuple)):
                for y in x: 
                    yield from _flatten(y)
            else:
                yield x
        flat = list(_flatten(v))
        if flat and isinstance(flat[0], str):
            tag = flat[0].lower()
            if tag in ("odr_point","odrpoint","odr"):
                return "odr_point"
            if tag in ("position","position_3d","pos"):
                return "position_3d"
            if tag in ("orientation","orientation_3d","ori"):
                return "orientation_3d"
            if tag in ("route_point","route","path"):
                return "route_point"

    if isinstance(v, bool):  return "bool"
    if isinstance(v, int):
        return "uint" if v >= 0 else "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str):   return "string"
    return None

def _extract_enums_from_registry(registry):
    """Return { enum_name: set(literals) } regardless of how the registry is wrapped."""
    src = None
    if isinstance(registry, dict):
        src = registry.get("enums")
    if src is None:
        src = getattr(registry, "enums", None)
    if src is None:
        for attr in ("data", "raw", "spec", "_data", "__dict__"):
            d = getattr(registry, attr, None)
            if isinstance(d, dict) and "enums" in d:
                src = d["enums"]
                break
    if not isinstance(src, dict):
        return {}
    out = {}
    for name, edef in src.items():
        # edef may be a list (literals) or dict like {"literals":[...]}
        if isinstance(edef, dict):
            lits = edef.get("literals") or edef.get("values") or []
        else:
            lits = edef or []
        out[name] = set(map(lambda x: str(x).lower(), lits))
    return out

def _seed_required_enums(enum_map: dict):
    """Ensure required enums exist (ASAM common)."""
    enum_map.setdefault("gap_direction", {"ahead", "behind", "left", "right"})
    enum_map.setdefault("distance_direction", {"ahead", "behind"})
    enum_map.setdefault("headway_direction", {"increase", "decrease"})
    enum_map.setdefault("side_left_right", {"left", "right"})
    enum_map.setdefault("lane_change_side", {"left", "right", "same", "same_as"})
    enum_map.setdefault("at", {"start", "end"})


# ---------- Validator ----------

class SemanticValidator:
    """
    Stateless semantic checker using a SemanticsRegistry.
    Host feeds it ActionCall records (from IR adapter or AST adapter).
    """

    def __init__(self, registry: SemanticsRegistry, type_of_expr: Optional[Callable[[ArgValue], Optional[str]]] = None, debug_types: bool=False):
        self.registry = registry
        # Hook to infer type name from ArgValue if not given (e.g. Physical -> "length")
        self.type_of_expr = type_of_expr or (lambda a: a.type_name)
        self.debug_types = debug_types
        self._ENUM_LITERALS = _extract_enums_from_registry(registry)
        _seed_required_enums(self._ENUM_LITERALS)
    

    def _is_numeric_range(self, v):
        # duck-typing for your Range class
        return hasattr(v, "start") and hasattr(v, "end")

    def _dbg(self, msg: str):
        if getattr(self, "debug_types", False):
            print(msg)
    # =====================================================================
    # Normalization helpers (dicts vs. objects)
    # =====================================================================
    def _arg_type_ok(self, expected: str, arg) -> bool:
        v = getattr(arg, "value", arg)
        actual = getattr(arg, "type_name", None) or (self.type_of_expr(arg) if callable(self.type_of_expr) else None)

        # exact match
        if actual and actual == expected:
            return True

        if expected in ("odr_point", "position_3d", "orientation_3d", "route_point"):
            if isinstance(v, dict):
                keys = {str(k).lower() for k in v.keys()}
                if expected == "odr_point"     and {"road","lane","s"} <= keys: return True
                if expected == "position_3d"   and ({"x","y"} <= keys or {"s","t"} <= keys): return True
                if expected == "orientation_3d" and ({"yaw"} & keys or {"angle"} & keys): return True
                if expected == "route_point"   and ({"route","path","route_point"} & keys): return True
            if isinstance(v, (list, tuple)):
                flat = list(_flatten(v))
                if flat:
                    tag = str(flat[0]).lower()
                    if expected == "odr_point" and tag in ("odr_point", "odrpoint", "odr") and len(flat) >= 4:
                        return True
                    if expected == "position_3d" and tag in ("pos","position","position_3d") and len(flat) >= 3:
                        return True
                    if expected == "orientation_3d" and tag in ("ori","orientation","orientation_3d") and len(flat) >= 1:
                        return True
            
        # -------------------------------------------------------------------------------

        # enums...
        lits = self._ENUM_LITERALS.get(expected)
        if lits is not None:
            if not lits: return True
            return isinstance(v, str) and v.lower() in lits

        # ----- physical categories by unit -----
        try:
            from osc2_parser.srunner.osc2_dm.physical_types import Physical
        except Exception:
            Physical = ()  # if types not importable here

        if isinstance(v, Physical):
            unit = getattr(v.unit, "unit_name", None) or getattr(v.unit, "name", None)
            if unit:
                uname = str(unit)
                time_units   = {"second","millisecond","minute","hour"}
                length_units = {"meter","millimeter","centimeter","kilometer","inch","feet","mile","micrometer"}
                speed_units  = {"meter_per_second","kilometer_per_hour","mile_per_hour"}
                accel_units  = {"meter_per_sec_sqr","kilometer_per_hour_per_sec","mile_per_hour_per_sec","feet_per_sec_sqr"}
                angle_units  = {"degree","radian"}

                if expected == "time"         and uname in time_units:   return True
                if expected == "length"       and uname in length_units: return True
                if expected == "speed"        and uname in speed_units:  return True
                if expected == "acceleration" and uname in accel_units:  return True
                if expected == "angle"        and uname in angle_units:  return True

        # ----- ints / uints -----
        # Accept numeric ranges for int where appropriate
        if expected == "int":
            if self._is_numeric_range(v):
                return isinstance(v.start, (int, float)) and isinstance(v.end, (int, float))
            return isinstance(v, int)
        if expected == "uint":
            if isinstance(v, int) and v >= 0:
                return True
            # allow hook to say it's an int-ish
            if actual in ("int", "uint"):
                return True
            return False

        # ----- physical_object (actor references, route refs, etc.) -----
        if expected == "physical_object":
            # In this validator stage we don’t resolve names; accept strings.
            return isinstance(v, str)

        # ----- fallback -----
        return False
    
    def _get_action_spec(self, qname: str):
        """
        Return the action spec (dict or object) for a qualified action name.
        Works with several registry shapes.
        """
        # Preferred explicit getter
        if hasattr(self.registry, "get_action"):
            try:
                spec = self.registry.get_action(qname)
                if spec is not None:
                    return spec
            except Exception:
                pass

        # Common direct mapping attribute
        actions = getattr(self.registry, "actions", None)
        if isinstance(actions, dict) and qname in actions:
            return actions[qname]

        # JSON-esque document storage
        doc = getattr(self.registry, "doc", None)
        if isinstance(doc, dict):
            return doc.get("actions", {}).get(qname)

        # Last resort: registry could itself be dict-like
        if isinstance(self.registry, dict):
            return self.registry.get("actions", {}).get(qname, {})

        return None

    def _spec_inherits(self, spec) -> Optional[str]:
        """Read 'inherits' from dict or object spec."""
        if spec is None:
            return None
        if isinstance(spec, dict):
            return spec.get("inherits")
        return getattr(spec, "inherits", None)

    def _spec_overloads(self, spec):
        """Read 'overloads' from dict or object spec; always returns a list."""
        if spec is None:
            return []
        if isinstance(spec, dict):
            return spec.get("overloads", []) or []
        return getattr(spec, "overloads", []) or []

    def _overload_params_map(self, overload) -> Dict[str, Dict[str, Any]]:
        """
        Normalize an overload's params into a dict:
            name -> { "type": str|None, "optional": bool, ["default": Any] }
        Handles dict- or object-style overloads and param specs.
        """
        # 1) load raw container
        if isinstance(overload, dict):
            params = overload.get("params", {}) or {}
        else:
            params = getattr(overload, "params", {}) or {}

        out: Dict[str, Dict[str, Any]] = {}

        # Mapping: {name -> (dict|ParamSpec)}
        if isinstance(params, dict):
            for name, p in params.items():
                if isinstance(p, dict):
                    entry = {
                        "type": p.get("type"),
                        "optional": p.get("optional", False),
                    }
                    if "default" in p:
                        entry["default"] = p["default"]
                    out[name] = entry
                else:
                    # object-like ParamSpec
                    p_type = getattr(p, "type", None) or getattr(p, "param_type", None)
                    p_opt  = getattr(p, "optional", False) or getattr(p, "is_optional", False)
                    entry = {"type": p_type, "optional": p_opt}
                    if hasattr(p, "default"):
                        entry["default"] = getattr(p, "default")
                    out[name] = entry
            return out

        # Iterable: [ParamSpec or dict] with explicit 'name'
        if isinstance(params, (list, tuple)):
            for p in params:
                if isinstance(p, dict):
                    name = p.get("name")
                    if not name:
                        continue
                    entry = {
                        "type": p.get("type"),
                        "optional": p.get("optional", False),
                    }
                    if "default" in p:
                        entry["default"] = p["default"]
                    out[name] = entry
                else:
                    name = getattr(p, "name", None)
                    if not name:
                        continue
                    p_type = getattr(p, "type", None) or getattr(p, "param_type", None)
                    p_opt  = getattr(p, "optional", False) or getattr(p, "is_optional", False)
                    entry = {"type": p_type, "optional": p_opt}
                    if hasattr(p, "default"):
                        entry["default"] = getattr(p, "default")
                    out[name] = entry
            return out

        return out

    def _extract_candidate(self, item) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """
        Normalize a candidate entry to (qualified_action_name, overload, overload_name).
        Supports:
          - (qname, overload) tuples
          - (qname, overload, overload_name) tuples
          - object with attributes: qualified_name/qname/name, overload/spec/overload_spec, overload_name
        """
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                return item[0], item[1], item[2]
            if len(item) == 2:
                return item[0], item[1], None
            return None, None, None

        # object-like
        qname = (
            getattr(item, "qualified_name", None)
            or getattr(item, "qname", None)
            or getattr(item, "name", None)
        )
        ov = (
            getattr(item, "overload", None)
            or getattr(item, "spec", None)
            or getattr(item, "overload_spec", None)
        )
        ovn = getattr(item, "overload_name", None)
        # fall back to overload.name if present
        if ovn is None and ov is not None:
            ovn = getattr(ov, "name", None)
        return qname, ov, ovn

    # =====================================================================
    # Inheritance parameter merge
    # =====================================================================

    def _params_from_inherits(self, action_qname: str) -> Dict[str, Dict[str, Any]]:
        """
        Walk 'inherits' chain and merge all ancestor overload params into one map.
        Child definitions will override parents later (we merge child last).
        """
        merged: Dict[str, Dict[str, Any]] = {}
        seen: Set[str] = set()
        cur = action_qname
        while cur and cur not in seen:
            seen.add(cur)
            spec = self._get_action_spec(cur)
            if not spec:
                break
            for ov in self._spec_overloads(spec):
                merged.update(self._overload_params_map(ov))
            cur = self._spec_inherits(spec)
        return merged

    # =====================================================================
    # Top-level
    # =====================================================================

    def validate_action_call(self, call: ActionCall) -> ValidationResult:
        # 1) candidates by method + invoker ancestry
        candidates = self.registry.find_actions_for_method_on(call.invoker_type, call.method_name)
        if not candidates:
            LOG_ERROR(
                f"No action '{call.method_name}' is defined for actor type '{call.invoker_type}' (or its bases).",
                call.token
            )

        # 2) choose concrete overload
        chosen_qname, chosen_spec, ov_name, resolved_args = self._choose_action_overload(call, candidates)

        # 3) modifiers
        resolved_mods: List[Tuple[str, str, Dict[str, Any]]] = []
        for m in call.modifiers:
            mod_spec: ModifierSpec = self.registry.get_modifier(m.name)
            if not mod_spec:
                LOG_ERROR(f"Modifier '{m.name}' is not defined.", m.token)
            family = self.registry.action_family_of(chosen_qname)
            if not self.registry.is_action_in_family(chosen_qname, mod_spec.applies_to):
                LOG_ERROR(
                    f"Modifier '{m.name}' does not apply to action '{chosen_qname}' "
                    f"(expected family '{mod_spec.applies_to}', got '{family}').",
                    m.token
                )
            v_name, v_args = self._choose_modifier_variant(m, mod_spec, call)
            resolved_mods.append((m.name, v_name, v_args))

        return ValidationResult(
            resolved_action_qname=chosen_qname,
            resolved_action_overload=ov_name or "<default>",
            resolved_args=resolved_args,
            resolved_modifiers=resolved_mods,
        )

    # =====================================================================
    # Overload resolution (actions)
    # =====================================================================

    def _choose_action_overload(self, call: ActionCall, candidates):
        """
        candidates: iterable of (qname, action_spec[, overload_name]) OR already (qname, overload[, name])
        We evaluate each overload separately:
        effective_params = params_from_inherits(qname)  +  params_of_this_overload
        """
        supplied_names = set(call.args.keys())
        failures = []

        def _rules_of(overload) -> Dict[str, list]:
            if isinstance(overload, dict):
                return overload.get("rules") or {}
            return getattr(overload, "rules", {}) or {}

        for item in candidates:
            qname, ov, ov_name = self._extract_candidate(item)
            if not qname or ov is None:
                continue

            # Expand a candidate ActionSpec into its concrete overloads
            cand_overloads: List[Tuple[object, str]] = []
            if hasattr(ov, "overloads"):  # ActionSpec
                olist = self._spec_overloads(ov) or []
                for idx, child_ov in enumerate(olist):
                    name = getattr(child_ov, "name", None) or f"overload[{idx}]"
                    cand_overloads.append((child_ov, name))
            else:
                # Already an overload-like object/dict
                cand_overloads.append((ov, ov_name or "<default>"))

            # Try each overload independently
            for child_ov, child_name in cand_overloads:
                parent_params = self._params_from_inherits(qname)     # ancestors (e.g., duration)
                own_params    = self._overload_params_map(child_ov)   # THIS overload only
                effective_params = {**parent_params, **own_params}

                param_names = set(effective_params.keys())
                if call.method_name == "assign_orientation":
                    print(f"[assign_orientation] supplied= {sorted(supplied_names)}")
                    print(f"[assign_orientation] params= {sorted(param_names)}")
                # unknown args?
                unknown = supplied_names - param_names
                if unknown:
                    failures.append((f"{qname}:{child_name}", f"unknown args: {sorted(unknown)}"))
                    continue

                # missing required? (required = not optional, and no default)
                required = {
                    k for k, v in effective_params.items()
                    if not v.get("optional", False) and ("default" not in v)
                }
                missing = required - supplied_names
                if missing:
                    failures.append((f"{qname}:{child_name}", f"missing required: {sorted(missing)}"))
                    continue

                # type checks
                type_mismatch = []
                for k in supplied_names:
                    expected = effective_params.get(k, {}).get("type")
                    if expected and not self._arg_type_ok(expected, call.args[k]):
                        got = call.args[k].type_name or self.type_of_expr(call.args[k])
                        type_mismatch.append((k, expected, got))
                if type_mismatch:
                    failures.append((f"{qname}:{child_name}", f"type mismatch: {type_mismatch}"))
                    continue

                # enforce action-level rules like exactly_one_of (if provided in this overload)
                rules = _rules_of(child_ov)
                if not self._check_rules_groups(rules, resolved={}, provided=supplied_names):
                    failures.append((f"{qname}:{child_name}", f"rule violation: {rules}"))
                    continue

                # SUCCESS for this overload
                resolved_args = {k: v.value for k, v in call.args.items()}

                # fill defaults from this overload+ancestors
                for k, ps in effective_params.items():
                    if k not in resolved_args and "default" in ps:
                        dv = ps["default"]
                        if dv == "<actor>":
                            dv = call.invoker_name
                        resolved_args[k] = dv

                return qname, child_ov, child_name, resolved_args

        # No overload matched → keep your error
        raise ValueError(f"No overload of '{call.method_name}' matches the supplied arguments: {sorted(supplied_names)}.")


    # =====================================================================
    # Variant resolution (modifiers)
    # =====================================================================

    def _choose_modifier_variant(self, m: ModifierAttach, spec: ModifierSpec, call: ActionCall) -> Tuple[str, Dict[str, Any]]:
        matches: List[Tuple[str, Dict[str, Any]]] = []
        var_list = spec.variants or [ModifierVariantSpec(None, {}, {})]
        for var in var_list:
            ok, resolved = self._args_match_variant(m, var, call)
            if ok:
                matches.append((var.name or "<default>", resolved))
        if not matches:
            LOG_ERROR(f"No variant of modifier '{m.name}' matches supplied arguments: {list(m.args.keys())}", m.token)
        if len(matches) > 1 and spec.rules.get("overloads_mutually_exclusive", False):
            LOG_ERROR(f"Ambiguous modifier '{m.name}': multiple variants match: {[x[0] for x in matches]}", m.token)
        return matches[0]

    def _args_match_variant(self, m: ModifierAttach, var: ModifierVariantSpec, call: ActionCall) -> Tuple[bool, Dict[str, Any]]:
        provided: Set[str] = set(m.args.keys())
        resolved: Dict[str, Any] = {}

        for pn, ps in var.params.items():
            if not ps.optional and pn not in provided:
                return False, {}

        unknown = provided - set(var.params.keys())
        if unknown:
            return False, {}

        for pn, ps in var.params.items():
            if pn in m.args:
                arg = m.args[pn]
                if ps.type and not self._arg_type_ok(ps.type, arg):
                    # (optional) debug
                    # print(f"[mod-type-of] mod={m.name} param='{pn}' expected='{ps.type}' actual={getattr(arg,'type_name',None) or self.type_of_expr(arg)} value={getattr(arg,'value',arg)}")
                    return False, {}
                resolved[pn] = getattr(arg, "value", arg)
                if ps.ignored_if_present:
                    for bad in ps.ignored_if_present:
                        if bad in provided:
                            LOG_WARN(f"Modifier '{m.name}': parameter '{pn}' is ignored when '{bad}' is present.", m.token)
            else:
                if ps.optional and ps.default is not None:
                    resolved[pn] = call.invoker_name if ps.default == "<actor>" else ps.default

        if not self._check_rules_groups(var.rules, resolved, provided):
            return False, {}
        return True, resolved

    # =====================================================================
    # Common rules
    # =====================================================================

    def _check_rules_groups(self, rules: Dict[str, List[List[str]]], resolved: Dict[str, Any], provided: Set[str]) -> bool:
        for group in rules.get("exactly_one_of", []):
            cnt = sum(1 for k in group if k in provided)
            if cnt != 1:
                return False
        for group in rules.get("at_least_one_of", []):
            cnt = sum(1 for k in group if k in provided)
            if cnt < 1:
                return False
        for pair in rules.get("requires", []):
            if len(pair) == 2:
                a, b = pair
                if a in provided and b not in provided:
                    return False
        return True

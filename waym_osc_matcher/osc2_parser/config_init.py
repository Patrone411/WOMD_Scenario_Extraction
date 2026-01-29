import sys
from typing import List, Tuple

from .srunner.osc2_dm.physical_object import *
from .srunner.osc2_dm.physical_types import Physical, Range
from .srunner.osc2.ast_manager import ast_node
from .srunner.tools.osc2_helper import OSC2Helper
from .srunner.osc2.ast_manager.ast_vistor import ASTVisitor
from .utils import flat_list

# --- Minimal fallback actor classes (generic OSC types) ---
class _GenericActor:
    def __init__(self):
        self._name = None
        self._category = self.__class__.__name__

    def set_name(self, n: str):
        self._name = n

    # >>> missing in your code; add this <<<
    def get_name(self) -> str:
        return self._name

    # optional, but often useful for logging/pipelines
    def get_category(self) -> str:
        return self._category

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self._name!r}>"

class _GenericPhysicalObject(_GenericActor):
    # placeholder fields commonly present
    box = None
    obj_color = None
    geometry_reference = None
    center_of_gravity = None
    pose = None

class _GenericMovableObject(_GenericPhysicalObject):
    velocity = None
    acceleration = None
    speed_value = None

class _GenericTrafficParticipant(_GenericMovableObject):
    intended_infrastructure = None

class _GenericVehicle(_GenericTrafficParticipant):
    vehicle_category = None
    axles = None
    rear_overhang = None

class _GenericPerson(_GenericTrafficParticipant):
    pass

class _GenericPath:
    def __init__(self):
        self._name = None
        self.min_lanes_required = None  # store whatever constraints you care about

    def set_name(self, n: str):
        self._name = n

    def get_name(self) -> str:
        return self._name

    # methods you call from OSC
    def min_lanes(self, n):
        self.min_lanes_required = int(n)

def _split_pos_kw(arguments):
    pos, kw = [], {}
    from .utils import flat_list
    if isinstance(arguments, list):
        for a in flat_list(arguments):
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
                kw[a[0]] = a[1]
            else:
                pos.append(a)
    elif isinstance(arguments, tuple) and len(arguments) == 2 and isinstance(arguments[0], str):
        kw[arguments[0]] = arguments[1]
    elif arguments is not None:
        pos.append(arguments)
    return pos, kw


debug = False
class ConfigInit(ASTVisitor):
    def __init__(self, configInstance) -> None:
        super().__init__()
        self.father_ins = configInstance
        self.pytree = []  # This will be a list of blocks
        self._pytree_header = {"type": "Setup", "paths": {}}  # add more sections later if you want
        self.pytree = [self._pytree_header]
        self.current_block = None  # Temporary reference to current block
        self.current_actor = None  # Tracks actor during behavior invocation
        # NEW: discovered OSC actor types + inheritance
        # e.g. {"vehicle": {"base": "traffic_participant"}, ...}
        self.objects = {}  # name -> instance (paths, future map objects, etc.)
        if not hasattr(self.father_ins, "paths"):
            self.father_ins.paths = {}  # optional: expose to the rest of your pipeline
        
        self.actor_registry = {}
        self._actor_decl_stack = []      # track which actor we're currently visiting (for ActorInherts)

        # NEW: fallback Python classes to instantiate if no brand class is found
        self._actor_fallback = {
            "osc_actor": _GenericActor,
            "physical_object": _GenericPhysicalObject,
            "movable_object": _GenericMovableObject,
            "traffic_participant": _GenericTrafficParticipant,
            "vehicle": _GenericVehicle,
            "person": _GenericPerson,
        }

    def _as_ttc_call(self, x, invoker_guess=None):
        # Accept either dict {"method":"time_to_collision", ...}
        if isinstance(x, dict) and x.get("method") == "time_to_collision":
            return x

        # Or list/tuple like ["ego.time_to_collision", ('reference','other_car')] (your print)
        if isinstance(x, (list, tuple)) and x:
            head = x[0]
            if isinstance(head, str) and (head.endswith(".time_to_collision") or head == "time_to_collision"):
                ref = None
                for e in x[1:]:
                    if isinstance(e, tuple) and e[0] == "reference":
                        ref = e[1]
                    elif ref is None and not isinstance(e, tuple):
                        # allow positional fallback
                        ref = e
                out = {"method": "time_to_collision"}
                if isinstance(head, str) and "." in head:
                    out["invoker"] = head.split(".", 1)[0]
                elif invoker_guess:
                    out["invoker"] = invoker_guess
                if ref is not None:
                    out["reference"] = ref
                return out
        return None

    def _normalize_until_value(self, pred):
        from .srunner.osc2_dm.physical_types import Physical

        # already-normalized
        if isinstance(pred, dict) and "reference" in pred and "time" in pred:
            return ("until", {"reference": pred["reference"], "time": pred["time"]})

        # op-dict (if your BinaryExpression emitted {"op":..., "left":..., "right":...})
        if isinstance(pred, dict) and pred.get("op") in ("<", "<=", ">", ">=", "==", "!="):
            L, R = pred.get("left"), pred.get("right")
            Lttc = self._as_ttc_call(L)
            Rttc = self._as_ttc_call(R)
            if Lttc and isinstance(R, Physical):
                return ("until", {"reference": Lttc.get("reference"), "time": R})
            if Rttc and isinstance(L, Physical):
                return ("until", {"reference": Rttc.get("reference"), "time": L})
            return ("until", {"predicate": pred})

        # <<< NEW: handle your printed list-shape >>> 
        # e.g. [ ["ego.time_to_collision", ('reference','other_car')], Physical(10s) ]
        if isinstance(pred, (list, tuple)) and len(pred) == 2:
            L, R = pred
            Lttc = self._as_ttc_call(L)
            Rttc = self._as_ttc_call(R)
            if Lttc and isinstance(R, Physical):
                return ("until", {"reference": Lttc.get("reference"), "time": R})
            if Rttc and isinstance(L, Physical):
                return ("until", {"reference": Rttc.get("reference"), "time": L})
            return ("until", {"predicate": pred})

        # fallback
        return ("until", {"predicate": pred})

    def visit_until_directive(self, node: ast_node.UntilDirective):
        """
        Grammar node for:   until <something>
        <something> is typically rise(<binary_expr>)
        We visit the child; visit_modifier_invocation('rise') already unwraps to the predicate.
        """
        pred = None
        for ch in node.get_children():
            pred = ch.accept(self) if hasattr(ch, "accept") else ch
            break
        return self._normalize_until_value(pred)

    def _is_path_type(self, t: str) -> bool:
        return str(t).lower() == "path"

    def _register_object(self, name: str, inst, kind: str = None):
        self.objects[name] = inst
        if kind == "path":
            self.father_ins.paths[name] = inst
            self._pytree_header["paths"].setdefault(name, {"name": name})
        # also expose to variables to allow reference resolution
        self.father_ins.variables[name] = inst

    def _install_unit_aliases(self):
        """Create friendly aliases to canonical unit names, only if the canonical exists."""
        u = self.father_ins.unit_dict

        def alias(alias_name: str, canonical: str):
            if alias_name not in u and canonical in u:
                u[alias_name] = u[canonical]

        # --- speed (you already had these) ---
        alias("kph",  "kilometer_per_hour")
        alias("kmph", "kilometer_per_hour")
        alias("kmh",  "kilometer_per_hour")
        alias("mps",  "meter_per_second")
        alias("mph",  "mile_per_hour")

        # --- time ---
        alias("s",    "second")
        alias("sec",  "second")
        alias("ms",   "millisecond")
        alias("min",  "minute")
        alias("h",    "hour")
        alias("hr",   "hour")

        # --- length ---
        alias("m",    "meter")
        alias("mm",   "millimeter")
        alias("cm",   "centimeter")
        alias("km",   "kilometer")
        alias("in",   "inch")
        alias("ft",   "feet")
        alias("mi",   "mile")
        alias("um",   "micrometer")
        alias("µm",   "micrometer")

        # --- angle ---
        alias("deg",  "degree")
        alias("rad",  "radian")

        # --- acceleration ---
        alias("mps2",   "meter_per_sec_sqr")
        alias("m/s2",   "meter_per_sec_sqr")
        alias("m/s^2",  "meter_per_sec_sqr")
        alias("kphps",  "kilometer_per_hour_per_sec")
        alias("kmhps",  "kilometer_per_hour_per_sec")
        alias("km/h/s", "kilometer_per_hour_per_sec")
        alias("mphps",  "mile_per_hour_per_sec")
        alias("ftps2",  "feet_per_sec_sqr")
        alias("fps2",   "feet_per_sec_sqr")

    def _normalize_unit_name(self, name: str) -> str:
        table = {
            # speed
            "kph": "kilometer_per_hour", "kmph": "kilometer_per_hour", "kmh": "kilometer_per_hour",
            "mps": "meter_per_second",   "mph":  "mile_per_hour",
            # time
            "s": "second", "sec": "second", "ms": "millisecond", "min": "minute", "h": "hour", "hr": "hour",
            # length
            "m": "meter", "mm": "millimeter", "cm": "centimeter", "km": "kilometer",
            "in": "inch", "ft": "feet", "mi": "mile", "um": "micrometer", "µm": "micrometer",
            # angle
            "deg": "degree", "rad": "radian",
            # accel
            # acceleration
            "mps2": "meter_per_sec_sqr", "m/s2": "meter_per_sec_sqr", "m/s^2": "meter_per_sec_sqr",
            "kphps": "kilometer_per_hour_per_sec", "kmhps": "kilometer_per_hour_per_sec", "km/h/s": "kilometer_per_hour_per_sec",
            "mphps": "mile_per_hour_per_sec", "ftps2": "feet_per_sec_sqr", "fps2": "feet_per_sec_sqr",
        }
        return table.get(name, name)

    def _is_actor_type(self, t: str) -> bool:
        return t in self.actor_registry
    
    def visit_compilation_unit(self, node: ast_node.CompilationUnit):
        # ---- PASS 1a: physical types ----
        for c in node.get_children():
            if isinstance(c, ast_node.PhysicalTypeDeclaration):
                c.accept(self)

        # ---- PASS 1b: units (depend on physical types) ----
        for c in node.get_children():
            if isinstance(c, ast_node.UnitDeclaration):
                c.accept(self)

        self._install_unit_aliases()

        # ---- PASS 1c: other global declarations (order-insensitive now) ----
        for c in node.get_children():
            if isinstance(c, (
                ast_node.EnumDeclaration,
                ast_node.StructDeclaration, ast_node.StructInherts,
                ast_node.ActorDeclaration, ast_node.ActorInherts,
                ast_node.ActionDeclaration, ast_node.ActionInherts,
                ast_node.StructuredTypeExtension, ast_node.EnumTypeExtension,
                ast_node.ModifierDeclaration,
            )):
                c.accept(self)

        # (Optional) quick sanity check
        if debug:
            print("[DEBUG] Units loaded:", sorted(self.father_ins.unit_dict.keys()))

        # ---- PASS 2: scenarios, globals, variables, etc. ----
        for c in node.get_children():
            if isinstance(c, (
                ast_node.GlobalParameterDeclaration,
                ast_node.VariableDeclaration,
                ast_node.ScenarioDeclaration,
                ast_node.coverDeclaration,
                ast_node.recordDeclaration,
                ast_node.RemoveDefaultDeclaration,
            )):
                c.accept(self)

    def _create_actor_instance(self, type_name: str, instance_name: str):
        """
        Build a runtime instance purely from declarations, choosing a generic fallback
        by walking up the inheritance chain (type -> base -> ...).
        """
        # find the first known fallback along the chain
        t = type_name
        chosen_cls = None
        visited = set()
        while t and t not in visited:
            visited.add(t)
            if t in self._actor_fallback:
                chosen_cls = self._actor_fallback[t]
                break
            t = self.actor_registry.get(t, {}).get("base")

        if chosen_cls is None:
            chosen_cls = _GenericActor  # ultimate fallback

        inst = chosen_cls()
        if hasattr(inst, "set_name"):
            inst.set_name(instance_name)
        return inst
    
    def _add_actor_instance(self, name: str, inst):
        # keep your existing wiring to father_ins
        if name == OSC2Helper.ego_name and hasattr(self.father_ins, "add_ego_vehicles"):
            self.father_ins.add_ego_vehicles(inst)
        elif hasattr(self.father_ins, "add_other_actors"):
            self.father_ins.add_other_actors(inst)

     # small helper: turn Identifier or str into a plain name
    def _as_name(self, maybe_node):
        if isinstance(maybe_node, str):
            return maybe_node
        if hasattr(maybe_node, "accept"):
            return maybe_node.accept(self)
        return str(maybe_node)

    def _resolve_vars(self, obj):
        if isinstance(obj, str):
            # If it's a declared variable AND it isn't an actor type, resolve it.
            val = self.father_ins.variables.get(obj, None)
            if val is not None and not self._is_actor_type(val):
                return self._resolve_vars(val)
            # Otherwise leave the symbol as-is (e.g., keep 'other_car' as a name)
            return obj
        if isinstance(obj, tuple):
            k, v = obj
            return (k, self._resolve_vars(v))
        if isinstance(obj, list):
            return [self._resolve_vars(x) for x in obj]
        return obj
    
    # NEW: recursively resolve strings that are variable names to their stored values
    def visit_global_parameter_declaration(self, node: ast_node.GlobalParameterDeclaration):
        para_name = node.field_name[0]
        args = self.visit_children(node)

        if isinstance(args, list) and len(args) == 2:
            declared_type, default_value = args
            # store default (resolving a variable reference if needed)
            self.father_ins.variables[para_name] = self.father_ins.variables.get(str(default_value), default_value)
            # instantiate if it's an actor type
            if isinstance(declared_type, str) and self._is_actor_type(declared_type):
                inst = self._create_actor_instance(declared_type, para_name)
                self._add_actor_instance(para_name, inst)
        elif isinstance(args, str):
            declared_type = args
            if self._is_actor_type(declared_type):
                inst = self._create_actor_instance(declared_type, para_name)
                self._add_actor_instance(para_name, inst)
            # store the declared type (so later references resolve)
            self.father_ins.variables[para_name] = declared_type

        self.father_ins.store_variable(self.father_ins.variables)

    def visit_struct_declaration(self, node: ast_node.StructDeclaration):
        struct_name = node.struct_name
        self.father_ins.struct_declaration[struct_name] = node
        for child in node.get_children():
            if isinstance(child, ast_node.MethodDeclaration):
                self.visit_method_declaration(child)

    def visit_string_literal(self, node: ast_node.StringLiteral):
        # Strip quotes from around the literal
        value = node.value.strip('"').strip("'")
        #print(f"[StringLiteral] Parsed string: {value}")

        return value
    
    def visit_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
        scenario_name = node.qualified_behavior_name
        self.father_ins.scenario_declaration[scenario_name] = node
        if scenario_name != "top":
            return

        for child in node.get_children():
            if isinstance(child, ast_node.ParameterDeclaration):
                self.visit_parameter_declaration(child)
            elif isinstance(child, ast_node.ModifierInvocation):
                self.visit_modifier_invocation(child)
            elif isinstance(child, ast_node.VariableDeclaration):
                self.visit_variable_declaration(child)
            elif isinstance(child, ast_node.EventDeclaration):
                pass
                # self.visit_event_declaration(child)
            elif isinstance(child, ast_node.DoDirective):
                self.visit_do_directive(child)

    def visit_do_directive(self, node: ast_node.DoDirective):
        if debug: print("Visiting DoDirective")  # 🟢 sanity check
        for child in node.get_children():
            if isinstance(child, ast_node.DoMember):
                self.visit_do_member(child)
    
    def visit_do_member(self, node: ast_node.DoMember):
        name = getattr(node, "label_name", None)
        composition = node.composition_operator

        # create a block even if there's no label
        block_name = name or "Anonymous Block"
        self.current_block = {
            "block_name": block_name,
            "type": composition.capitalize(),  # "Parallel" / "Serial"
            "duration": None,
            "actors": {}
        }

        for child in node.get_children():
            if isinstance(child, ast_node.NamedArgument):
                named_arg = self.visit_named_argument(child)
                if named_arg[0] == "duration":
                    duration = named_arg[1]
                    if isinstance(duration, Physical):
                        self.current_block["duration"] = duration.gen_physical_value()

            elif isinstance(child, ast_node.BehaviorInvocation):
                self.visit_behavior_invocation(child)

            elif isinstance(child, ast_node.DoMember):
                self.visit_do_member(child)

        self.pytree.append(self.current_block)
        self.current_block = None



     # --- NEW: record actor declarations from std osc files ---
    def visit_actor_declaration(self, node: ast_node.ActorDeclaration):
        """
        Handles:   actor <name> ...
        Your node: ActorDeclaration(actor_name)
        Children:  the parser will typically push ActorInherts + members under this node.
        """
        name = self._as_name(node.actor_name)
        # ensure registry entry exists
        self.actor_registry.setdefault(name, {"base": None, "fields": {}, "methods": {}})

        # enter scope (so ActorInherts knows which child to attach to)
        self._actor_decl_stack.append(name)
        try:
            return self.visit_children(node)
        finally:
            self._actor_decl_stack.pop()

    def visit_actor_inherts(self, node: ast_node.ActorInherts):
        """
        Handles:   inherits <base>
        Your node: ActorInherts(actor_name)  <-- this 'actor_name' is actually the BASE type name
        We attach it to the most recent actor_declaration on the stack.
        """
        base = self._as_name(node.actor_name)

        if self._actor_decl_stack:
            child = self._actor_decl_stack[-1]
            self.actor_registry.setdefault(child, {"base": None, "fields": {}, "methods": {}})
            self.actor_registry[child]["base"] = base
        else:
            # Defensive: if inherits shows up outside an actor context, you could log or stash for later.
            if debug:
                print(f"[WARN] ActorInherts('{base}') without an active ActorDeclaration")

        return self.visit_children(node)
        
    
    def visit_behavior_invocation(self, node: ast_node.BehaviorInvocation):
        actor = node.actor
        behavior = node.behavior_name

        if self.current_block is not None:
            self.current_block["actors"].setdefault(actor, {
                "action": None, "args": None, "target": None, "modifiers": []
            })
            self.current_block["actors"][actor]["action"] = behavior
        else:
            # safety net (shouldn't happen now that do_member always creates a block)
            self.current_block = {"block_name":"Anonymous Block","type":"Parallel","duration":None,"actors":{actor:{
                "action": behavior, "args": None, "target": None, "modifiers": []
            }}}

        pos_args, kw_args = [], {}

        for child in node.get_children():
            if isinstance(child, ast_node.PositionalArgument):
                arg = self.visit_positional_argument(child)
                pos_args.append(arg)

            elif isinstance(child, ast_node.NamedArgument):
                # let the modifier handling collect 'until' via UntilDirective; keep other named args
                if getattr(child, "argument_name", "") != "until":
                    k, v = self.visit_named_argument(child)
                    kw_args[k] = v
                else:
                    child.accept(self)  # ensure inner expressions get visited

            elif isinstance(child, ast_node.ModifierInvocation):
                mod = self.visit_modifier_invocation(child)
                if isinstance(mod, tuple) and len(mod) == 2 and isinstance(mod[0], str):
                    mname, margs = mod
                    self.current_block["actors"][actor]["modifiers"].append({"name": mname, "args": margs})

            elif hasattr(ast_node, "UntilDirective") and isinstance(child, ast_node.UntilDirective):
                mname, margs = self.visit_until_directive(child)
                print("mname: ", mname," margs: ", margs)
                if self.current_block is not None:
                    self.current_block["actors"][actor]["modifiers"].append({"name": mname, "args": margs})

            else:
                if hasattr(child, "accept"):
                    child.accept(self)

        # store args (unchanged)
        if pos_args and not kw_args and len(pos_args) == 1:
            self.current_block["actors"][actor]["args"] = pos_args[0]
        elif not pos_args and kw_args:
            self.current_block["actors"][actor]["args"] = kw_args
        else:
            combined = list(pos_args) + [(k, v) for k, v in kw_args.items()]
            self.current_block["actors"][actor]["args"] = combined
            
        

    
    def visit_positional_argument(self, node: ast_node.PositionalArgument):
        """Visit and evaluate a positional argument."""
        # Visit children and collect the values they produce
        argument_values = []
        for child in node.get_children():
            if hasattr(child, 'accept'):
                result = child.accept(self)
                argument_values.append(result)
            else:
                argument_values.append(child)  # fallback for primitive types

        # Typically, you expect just one value
        if len(argument_values) == 1:
            return argument_values[0]
        return argument_values

    def visit_parameter_declaration(self, node: ast_node.ParameterDeclaration):
        """
        Handles both:
        1) fields inside an actor declaration (register them on the actor type),
        2) scenario-level parameters (instantiate actors or store values).
        """
        # ---- helpers to normalize names ----
        def _name_of(x):
            if isinstance(x, str):
                return x
            if hasattr(x, "type_name"):   # ast_node.Type
                return x.type_name
            if hasattr(x, "name"):        # ast_node.Identifier / IdentifierReference
                return x.name
            return str(x)

        # field_name can be a list in your grammar
        if isinstance(node.field_name, list):
            param_names = [_name_of(n) for n in node.field_name]
        else:
            param_names = [_name_of(node.field_name)]

        declared_type = _name_of(node.field_type) if getattr(node, "field_type", None) else None

        # Evaluate raw children (as in your original code)
        arguments = self.visit_children(node)

        # ===== A) Inside an ACTOR declaration: treat as actor field =====
        in_actor_decl = bool(getattr(self, "_actor_decl_stack", []))
        if in_actor_decl:
            actor_name = self._actor_decl_stack[-1]
            self.actor_registry.setdefault(actor_name, {"base": None, "fields": {}, "methods": {}})

            field_type = declared_type
            if field_type is None and isinstance(arguments, str):
                field_type = arguments

            for pname in param_names:
                self.actor_registry[actor_name]["fields"][pname] = field_type
            # Done for actor-field case
            return arguments

        # ===== B) Scenario-level parameters (no actor-decl context) =====
        updated_vars = False

        if isinstance(arguments, list) and len(arguments) == 2:
            declared_type_from_args, default_value = arguments
            val = self.father_ins.variables.get(str(default_value), default_value)

            for pname in param_names:
                # If this is an actor type, instantiate and register the actor instance
                if isinstance(declared_type_from_args, str) and self._is_actor_type(declared_type_from_args):
                    inst = self._create_actor_instance(declared_type_from_args, pname)
                    self._add_actor_instance(pname, inst)
                    # store the type name for later resolution
                    self.father_ins.variables[pname] = declared_type_from_args
                elif self._is_path_type(declared_type_from_args):
                    p = _GenericPath()
                    p.set_name(pname)
                    self._register_object(pname, p, kind="path")
                else:
                    self.father_ins.variables[pname] = val
                updated_vars = True

        elif isinstance(arguments, str):
            declared_type_from_args = arguments
            for pname in param_names:
                if self._is_actor_type(declared_type_from_args):
                    inst = self._create_actor_instance(declared_type_from_args, pname)
                    self._add_actor_instance(pname, inst)
                elif self._is_path_type(declared_type_from_args):
                    p = _GenericPath()
                    p.set_name(pname)
                    self._register_object(pname, p, kind="path")
                else:
                    self.father_ins.variables[pname] = declared_type_from_args
                updated_vars = True

        else:
            # fall back to declared_type (from the node) if present
            if declared_type is not None and not updated_vars:
                for pname in param_names:
                    if self._is_actor_type(declared_type):
                        inst = self._create_actor_instance(declared_type, pname)
                        self._add_actor_instance(pname, inst)
                    elif self._is_path_type(declared_type):
                        p = _GenericPath()
                        p.set_name(pname)
                        self._register_object(pname, p, kind="path")
                    self.father_ins.variables[pname] = declared_type
                    updated_vars = True

        if updated_vars:
            self.father_ins.store_variable(self.father_ins.variables)

        return arguments

    def visit_variable_declaration(self, node: ast_node.VariableDeclaration):
        variable_name = node.field_name[0]
        # variable_type = ""
        variable_value = ""
        arguments = self.visit_children(node)
        if isinstance(arguments, list) and len(arguments) == 2:
            # variable_type = arguments[0]
            variable_value = arguments[1]
            if self.father_ins.variables.get(str(variable_value)) is not None:
                variable_value = self.father_ins.variables.get(str(variable_value))
            self.father_ins.variables[variable_name] = variable_value
        elif isinstance(arguments, str):
            # variable_type = arguments
            self.father_ins.variables[variable_name] = variable_value
        self.father_ins.store_variable(self.father_ins.variables)

    def visit_modifier_invocation(self, node: ast_node.ModifierInvocation):
        function_name = node.modifier_name
        modifier_name = node.modifier_name
        target_name   = node.actor

        # evaluate raw children then resolve variables
        arguments = self.visit_children(node)
        arguments = self._resolve_vars(arguments)
        # 1) If this is foo.bar(...), try to call method on the named instance
        target_obj = None
        if target_name:
            # look up in (a) objects, (b) father_ins.paths, (c) variables
            target_obj = self.objects.get(target_name)
            if target_obj is None and hasattr(self.father_ins, "paths"):
                target_obj = self.father_ins.paths.get(target_name)
            if target_obj is None:
                target_obj = self.father_ins.variables.get(target_name)

        if target_obj is not None and hasattr(target_obj, function_name):
            # normalize args
            pos_args, kw_args = [], {}
            if isinstance(arguments, list):
                for a in flat_list(arguments):
                    if isinstance(a, tuple) and len(a) == 2:
                        kw_args[a[0]] = a[1]
                    else:
                        pos_args.append(a)
            elif isinstance(arguments, tuple) and len(arguments) == 2 and isinstance(arguments[0], str):
                kw_args[arguments[0]] = arguments[1]
            else:
                pos_args.append(arguments)

            # call the method on the instance
            getattr(target_obj, function_name)(*pos_args, **kw_args)

            is_path = hasattr(self.father_ins, "paths") and target_name in self.father_ins.paths
            if is_path:
                entry = self._pytree_header["paths"].setdefault(target_name, {"name": target_name})
                # capture specific fields you care about
                if function_name == "min_lanes" and pos_args:
                    entry["min_lanes"] = int(pos_args[0])
    

            return modifier_name, arguments

        # 2) Backwards compat: call father_ins.path.<fn>(...) if available
        if hasattr(self.father_ins, "path") and hasattr(self.father_ins.path, function_name):
            path_function = getattr(self.father_ins.path, function_name)
            pos_args, kw_args = [], {}
            if isinstance(arguments, list):
                for a in flat_list(arguments):
                    if isinstance(a, tuple) and len(a) == 2:
                        kw_args[a[0]] = a[1]
                    else:
                        pos_args.append(a)
            elif isinstance(arguments, tuple) and len(arguments) == 2 and isinstance(arguments[0], str):
                kw_args[arguments[0]] = arguments[1]
            else:
                pos_args.append(arguments)
            path_function(*pos_args, **kw_args)
            return modifier_name, arguments
    
        # Split positional/named for the remaining normalization
        pos_args, kw_args = _split_pos_kw(arguments)

        # --- time_to_collision normalization (already in your code) ---
        if function_name == "time_to_collision":
            out = {"method": "time_to_collision"}
            if target_name:
                out["invoker"] = target_name
            if "reference" in kw_args:
                out["reference"] = kw_args["reference"]
            elif pos_args:
                out["reference"] = pos_args[0]
            return out

        # --- rise(...) just unwraps the predicate (already in your code) ---
        if function_name == "rise":
            pred = pos_args[0] if pos_args else kw_args.get("predicate")
            return pred

        # --- NEW: until rise( ego.time_to_collision(...) < 10s ) ---
        if function_name == "until":
            # unwrap the predicate produced by rise(...)
            pred = pos_args[0] if pos_args else kw_args.get("predicate")

            ref = None
            thr = None
            if isinstance(pred, dict) and pred.get("op") in ("<", "<=", ">", ">=", "==", "!="):
                L, R = pred.get("left"), pred.get("right")

                def _is_ttc(x):
                    return isinstance(x, dict) and x.get("method") == "time_to_collision"

                # TTC < T  or  T > TTC
                from .srunner.osc2_dm.physical_types import Physical
                if _is_ttc(L) and isinstance(R, Physical):
                    ref, thr = L.get("reference"), R
                elif _is_ttc(R) and isinstance(L, Physical):
                    ref, thr = R.get("reference"), L

            if ref is not None and thr is not None:
                return ("until", {"reference": ref, "time": thr})
            else:
                # Not a TTC comparison → keep the raw predicate so downstream can handle/fail cleanly
                return ("until", {"predicate": pred})

        # default passthrough
        return modifier_name, arguments

    def visit_event_declaration(self, node: ast_node.EventDeclaration):
        event_name = node.field_name
        arguments = self.visit_children(node)
        if hasattr(self.father_ins.event, event_name):
            # event_function = getattr(self.father_ins.event, event_name)
            position_args = []
            keyword_args = {}
            if isinstance(arguments, List):
                arguments = flat_list(arguments)
                for arg in arguments:
                    if isinstance(arg, Tuple):
                        if self.father_ins.variables.get(arg[0]) is not None:
                            keyword_args[arg[0]] = self.father_ins.variables.get(
                                arg[0]
                            )
                        else:
                            keyword_args[arg[0]] = arg[1]
                    else:
                        if self.father_ins.variables.get(arg) is not None:
                            position_args.append(self.father_ins.variables.get(arg))
                        else:
                            position_args.append(arg)
            else:
                if self.father_ins.variables.get(arguments) is not None:
                    position_args.append(self.father_ins.variables.get(arguments))
                else:
                    position_args.append(arguments)

    def visit_method_declaration(self, node: ast_node.MethodDeclaration):
        for child in node.get_children():
            if isinstance(child, ast_node.MethodBody):
                self.visit_method_body(child)

    def visit_method_body(self, node: ast_node.MethodBody):
        # type = node.type
        for child in node.get_children():
            if isinstance(child, ast_node.BinaryExpression):
                self.visit_binary_expression(child)

    def visit_binary_expression(self, node: ast_node.BinaryExpression):
        # First try the simple structured path
        lr = self.visit_children(node)
        op = node.operator

        left = right = None
        if isinstance(lr, (list, tuple)) and len(lr) == 2:
            left, right = lr

        # 1) Structured comparisons (new behavior — previously broken)
        if op in ('<', '>', '<=', '>=', '==', '!=') and left is not None and right is not None:
            return {"op": op, "left": left, "right": right}

        # 2) Normal arithmetic (unchanged behavior)
        if left is not None and right is not None and op in ('+', '-', '*', '/', '%'):
            if op == '+':  return left + right
            if op == '-':  return left - right
            if op == '*':  return left * right
            if op == '/':  return left / right
            if op == '%':  return left % right

        # 3) Legacy fallback — your original stack-machine path
        #    (keeps any quirky cases working exactly as before)
        args = [self.visit_children(node), node.operator]
        flat = flat_list(args)
        stack = []
        for token in flat:
            if token in ('+', '-', '*', '/', '%'):
                right = stack.pop()
                left = stack.pop()
                if token == '+': stack.append(left + right)
                elif token == '-': stack.append(left - right)
                elif token == '*': stack.append(left * right)
                elif token == '/': stack.append(left / right)
                else:             stack.append(left % right)
            else:
                stack.append(token)
        return stack.pop()

    def visit_named_argument(self, node):
        """
        Safely evaluate a NamedArgument's value (which may be a primitive or AST),
        unwrap singletons, and resolve variable references.
        Returns: (name, value)
        """
        val = self.visit_children(node)  # already handles AST vs primitives
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        return node.argument_name, self._resolve_vars(val)

    def visit_range_expression(self, node: ast_node.RangeExpression):
        start, end = self.visit_children(node)
        if type(start) != type(end):
            print("[Error] different types between start and end of the range")
            sys.exit(1)

        start_num = None
        end_num = None
        start_unit = None
        end_unit = None
        unit_name = None

        if isinstance(start, Physical):
            start_num = start.num
            end_num = end.num

            start_unit = start.unit
            end_unit = end.unit
        else:
            start_num = start
            end_num = end
        if start_unit is not None and end_unit is not None:
            if start_unit == end_unit:
                unit_name = start_unit
            else:
                print("[Error] wrong unit in the range")
                sys.exit(1)
        
        if start_num >= end_num:
            print("[Error] wrong start and end in the range")
            sys.exit(1)

        var_range = Range(start_num, end_num)
        if unit_name:
            return Physical(var_range, unit_name)
        else:
            return var_range

    def visit_physical_literal(self, node: ast_node.PhysicalLiteral):
        unit_name = self._normalize_unit_name(node.unit_name)
        unit_obj = self.father_ins.unit_dict.get(unit_name)
        if unit_obj is None:
            available = ", ".join(sorted(self.father_ins.unit_dict.keys()))
            raise KeyError(
                f"Unit '{unit_name}' not defined. Available units: [{available}]. "
                "Ensure units are parsed before scenarios or add a prelude/alias."
            )
        return Physical(self.visit_children(node), unit_obj)

    def visit_integer_literal(self, node: ast_node.IntegerLiteral):
        return int(node.value)

    def visit_float_literal(self, node: ast_node.FloatLiteral):
        return float(node.value)

    def visit_bool_literal(self, node: ast_node.BoolLiteral):
        return node.value

    def visit_identifier(self, node: ast_node.Identifier):
        return node.name

    def visit_identifier_reference(self, node: ast_node.IdentifierReference):
        return node.name

    def visit_type(self, node: ast_node.Type):
        return node.type_name

    def visit_physical_type_declaration(
            self, node: ast_node.PhysicalTypeDeclaration
        ):
            si_base_exponent = {}
            arguments = self.visit_children(node)
            arguments = flat_list(arguments)
            #print('args: ', arguments)

            if isinstance(arguments, Tuple):
                si_base_exponent[arguments[0]] = arguments[1]
            else:
                for elem in arguments:
                    si_base_exponent[elem[0]] = elem[1]
            self.father_ins.physical_dict[node.type_name] = PhysicalObject(
                node.type_name, si_base_exponent
            )

    def visit_unit_declaration(self, node: ast_node.UnitDeclaration):
        arguments = self.visit_children(node)
        arguments = flat_list(arguments)
        factor = 1.0
        offset = 0
        for elem in arguments:
            if elem[0] == "factor":
                factor = elem[1]
            elif elem[0] == "offset":
                offset = elem[1]
        self.father_ins.unit_dict[node.unit_name] = UnitObject(
            node.unit_name,
            self.father_ins.physical_dict[node.physical_name],
            factor,
            offset,
        )

    def visit_si_base_exponent(self, node: ast_node.SIBaseExponent):
        return node.unit_name, self.visit_children(node)
from typing import Any, Dict, List, Optional, Set, Tuple

from ..srunner.osc2.ast_manager import ast_node
from ..srunner.osc2.ast_manager.ast_vistor import ASTVisitor   # note: module name is 'ast_vistor' in the repo
from ..srunner.osc2_dm.physical_types import Physical, Range
from .pytree import (
    ScenarioNode, EventNode, VarNode, ActorInst,
    SerialBlock, ParallelBlock, OneOfBlock, ActionCall, ModifierCall,
)

MOD_ALIAS = {
    "position": "mod_position",
    "distance": "mod_distance",
    "lane": "mod_lane",
    "keep_lane": "mod_keep_lane",
    "lateral": "mod_lateral",
    "speed": "mod_speed",
    "acceleration": "mod_acceleration",
    "change_speed": "mod_change_speed",
    "change_lane": "mod_change_lane",
}

KW_ALIAS = {
    "at": "at_pos",
    "movement_mode": "move_mode",
    "track": "track_mode",
    "lane": "lane_num",
}

class IRLowering(ASTVisitor):
    """
    Pass 2: Lower AST to a small runtime IR (PyTree) of class instances.
    This pass assumes Pass 1 (ConfigInit) has already populated:
        - config.unit_dict, physical_dict
        - config.variables (names -> values/types/instances)
        - actor type registry (we take it as ctor arg: actor_registry)
    """
    def __init__(self, config, actor_registry: Dict[str, Dict], entry_names: Optional[Set[str]] = None) -> None:
        super().__init__()
        self.config = config
        self.actor_registry = actor_registry or {}
        self.entry_names = entry_names or {"top"}   # set() to process all scenarios
        self.scenarios: List[ScenarioNode] = []

        # State
        self._cur_scn: Optional[ScenarioNode] = None
        self._block_stack: List[Any] = []

        # Optional: cache commonly used dicts
        self.unit_dict = getattr(config, "unit_dict", None)
        self.variables = getattr(config, "variables", {})

    # ---- public API ----
    def lower(self, ast_root) -> List[ScenarioNode]:
        self.visit(ast_root)
        return {scn.name: scn for scn in self.scenarios}


    # ---- helpers ----
    def _get_unit_dict(self):
        """
        Prefer a unit dict carried on this pass, else from the config that constructed it.
        """
        if hasattr(self, "unit_dict") and self.unit_dict:
            return self.unit_dict
        if hasattr(self, "config") and hasattr(self.config, "unit_dict"):
            return self.config.unit_dict
        if hasattr(self, "father_ins") and hasattr(self.father_ins, "unit_dict"):
            return self.father_ins.unit_dict
        raise RuntimeError("No unit dictionary available in IRLowering")

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
            # acceleration
            "mps2": "meter_per_sec_sqr", "m/s2": "meter_per_sec_sqr", "m/s^2": "meter_per_sec_sqr",
            "kphps": "kphps", "kmhps": "kphps", "km/h/s": "kphps",
            "mphps": "mile_per_hour_per_sec", "ftps2": "feet_per_sec_sqr", "fps2": "feet_per_sec_sqr", "g": "g",
        }
        return table.get(name, name)

    def _resolve_pass2(self, val):
        """
        Use this pass's resolver if present; otherwise fall back gracefully.
        """
        if hasattr(self, "_resolve"):
            return self._resolve(val)
        if hasattr(self, "_resolve_vars"):
            return self._resolve_vars(val)
        return val

    def _is_actor_type(self, t: Optional[str]) -> bool:
        return bool(t) and (t in self.actor_registry)

    def _push_block(self, block):
        if self._block_stack:
            parent = self._block_stack[-1]
            parent.children.append(block)
        else:
            self._cur_scn.blocks.append(block)
        self._block_stack.append(block)

    def _pop_block(self):
        if self._block_stack:
            self._block_stack.pop()

    def _resolve(self, v):
        # resolve simple variable references to their stored values when useful
        if isinstance(v, str) and v in getattr(self.config, "variables", {}):
            return self.config.variables[v]
        return v

    # ---- compilation unit ----
    def visit_compilation_unit(self, node: ast_node.CompilationUnit):
        # process all children; scenario filtering happens in visit_scenario_declaration
        for c in node.get_children():
            c.accept(self)

    # ---- scenarios ----
    def visit_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
        scn_name = node.qualified_behavior_name
        if self.entry_names and len(self.entry_names) > 0 and scn_name not in self.entry_names:
            # still collect declaration map on config for reference
            self.config.scenario_declaration[scn_name] = node
            return

        self._cur_scn = ScenarioNode(scn_name)
        self.config.scenario_declaration[scn_name] = node

        # Walk its members
        for ch in node.get_children():
            if isinstance(ch, ast_node.ParameterDeclaration):
                self._lower_param_decl(ch)
            elif isinstance(ch, ast_node.VariableDeclaration):
                self._lower_var_decl(ch)
            elif isinstance(ch, ast_node.EventDeclaration):
                self._lower_event_decl(ch)
            elif isinstance(ch, ast_node.DoDirective):
                ch.accept(self)

        # finalize
        self.scenarios.append(self._cur_scn)
        self._cur_scn = None
    
    def _is_time_physical(self, p) -> bool:
        if not isinstance(p, Physical):
            return False
        try:
            # preferred: physical type name is 'time'
            if getattr(p.unit.physical, "name", "").lower() == "time":
                return True
        except Exception:
            pass
        uname = (getattr(p.unit, "unit_name", None) or getattr(p.unit, "name", None) or "").lower()
        return uname in {
            "s","sec","second","seconds","ms","millisecond","milliseconds",
            "min","minute","minutes","h","hr","hour","hours"
        }

    def _lower_until_directive(self, node: ast_node.UntilDirective):
        """
        Best-effort: find TTC 'reference' (from a NamedArgument) and a time-threshold Physical.
        If both are found, return kwargs={"reference": <name>, "time": <Physical>}.
        Otherwise return kwargs={"predicate": <raw>} so later passes can handle it.
        """
        ref = None
        thr = None

        # DFS over the subtree; use existing visitors to eval leaves
        def walk(n):
            nonlocal ref, thr
            # Try to catch 'reference: <actor>' inside the TTC call
            if isinstance(n, ast_node.NamedArgument) and getattr(n, "argument_name", None) == "reference":
                _, v = self.visit_named_argument(n)
                if isinstance(v, str):
                    ref = v
            # First time-like Physical we see becomes the threshold
            if isinstance(n, ast_node.PhysicalLiteral) and thr is None:
                p = self.visit_physical_literal(n)
                if self._is_time_physical(p):
                    thr = p
            # Recurse
            if hasattr(n, "get_children"):
                for ch in n.get_children():
                    walk(ch)

        walk(node)

        if ref is not None and thr is not None:
            return {"reference": ref, "time": thr}

        # Fallback: keep a raw predicate value so the adapter can try to normalize later.
        # We’ll evaluate the child to *something* by visiting its children.
        raw = getattr(node, "predicate", None)
        if raw is not None and hasattr(raw, "accept"):
            try:
                raw = raw.accept(self)
            except Exception:
                raw = None
        return {"predicate": raw}

    def _mk_modifier_call(self, name: str, *, kwargs: dict):
        """Create a ModifierCall regardless of the class constructor signature."""
        try:
            return ModifierCall(name=name, args=[], kwargs=kwargs)  # modern
        except TypeError:
            pairs = list(kwargs.items())
            try:
                return ModifierCall(name=name, args=pairs)          # older (no kwargs kwarg)
            except TypeError:
                return ModifierCall(name, pairs)                    # very old positional
        
    # ---- params / vars / events ----
    def _name_of(self, x) -> str:
        if isinstance(x, str): return x
        if hasattr(x, "type_name"): return x.type_name
        if hasattr(x, "name"): return x.name
        return str(x)

    def _lower_param_decl(self, node: ast_node.ParameterDeclaration):
        # Names can be a list in your grammar.
        names = [self._name_of(n) for n in (node.field_name if isinstance(node.field_name, list) else [node.field_name])]
        # Prefer the explicit field_type on node if present (your AST sets it)
        declared_type = self._name_of(getattr(node, "field_type", None)) if getattr(node, "field_type", None) else None

        for pname in names:
            val = self.config.variables.get(pname)
            # If Pass1 stored the type name instead of a value, keep it as type
            vtype = declared_type
            if vtype is None:
                # guess: if val is a string and is an actor type, treat it as type
                if isinstance(val, str) and self._is_actor_type(val):
                    vtype = val
            if vtype and self._is_actor_type(vtype):
                self._cur_scn.actors[pname] = ActorInst(pname, vtype)
            else:
                self._cur_scn.vars[pname] = VarNode(name=pname, type=vtype, value=val)

    def _lower_var_decl(self, node: ast_node.VariableDeclaration):
        vname = node.field_name[0] if isinstance(node.field_name, list) else node.field_name
        vname = self._name_of(vname)
        val = self.config.variables.get(vname)
        self._cur_scn.vars[vname] = VarNode(name=vname, value=val)

    def _lower_event_decl(self, node: ast_node.EventDeclaration):
        self._cur_scn.events.append(EventNode(node.field_name))

    # ---- do / blocks / behaviors ----
    def visit_do_directive(self, node: ast_node.DoDirective):
        for ch in node.get_children():
            if isinstance(ch, ast_node.DoMember):
                ch.accept(self)

    def visit_do_member(self, node: ast_node.DoMember):
        label = getattr(node, "label_name", None)
        kind = node.composition_operator  # "serial" or "parallel"

        # Create the correct block first (we may fill duration later)
        if kind == "serial":
            blk = SerialBlock(label=label)
        elif kind == "parallel":
            blk = ParallelBlock(label=label, duration=None)
        elif kind == "one_of":
            blk = OneOfBlock(label=label)
        else:
            # fallback: treat unknown as serial
            blk = SerialBlock(label=label)

        self._push_block(blk)

        # First pass through children to pick up named args like duration
        for ch in node.get_children():
            if isinstance(ch, ast_node.NamedArgument):
                k, v = self.visit_named_argument(ch)
                if isinstance(blk, ParallelBlock) and k == "duration":
                    blk.duration = v

        # Then behaviors / nested blocks
        for ch in node.get_children():
            if isinstance(ch, ast_node.BehaviorInvocation):
                ch.accept(self)
            elif isinstance(ch, ast_node.DoMember):
                ch.accept(self)

        self._pop_block()

    def visit_behavior_invocation(self, node: ast_node.BehaviorInvocation):
        actor = node.actor
        action = node.behavior_name

        # Gather action args (positional + named)
        action_args: List[Any] = []
        for ch in node.get_children():
            if isinstance(ch, ast_node.PositionalArgument):
                val = ch.accept(self)  # returns scalar/Physical/list
                action_args.append(val)
            elif isinstance(ch, ast_node.NamedArgument):
                k, v = self.visit_named_argument(ch)
                action_args.append((k, v))
        call = ActionCall(actor=actor, action=action, args=action_args)

        # Lower modifiers
        for ch in node.get_children():
            if isinstance(ch, ast_node.ModifierInvocation):
                mname, margs = self._lower_modifier_invocation(ch)
                # margs can be a list of mixed positional/("k",v) tuples.
                pos: List[Any] = []
                kw: Dict[str, Any] = {}
                if isinstance(margs, list):
                    for a in margs:
                        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
                            kw[a[0]] = a[1]
                        else:
                            pos.append(a)
                elif isinstance(margs, tuple) and len(margs) == 2 and isinstance(margs[0], str):
                    kw[margs[0]] = margs[1]
                else:
                    pos.append(margs)
                call.modifiers.append(ModifierCall(name=mname, args=pos, kwargs=kw))

            elif hasattr(ast_node, "UntilDirective") and isinstance(ch, ast_node.UntilDirective):
                kwargs = self._lower_until_directive(ch)
                call.modifiers.append(self._mk_modifier_call("until", kwargs=kwargs))

        # Attach to current block
        self._block_stack[-1].children.append(call)

    # --- leaf visitors reused ---
    def visit_positional_argument(self, node: ast_node.PositionalArgument):
        vals = []
        for ch in node.get_children():
            vals.append(ch.accept(self) if hasattr(ch, "accept") else ch)
        if len(vals) == 1:
            return self._resolve(vals[0])
        return [self._resolve(v) for v in vals]

    def visit_named_argument(self, node):
        """
        Safely evaluate a NamedArgument's value (primitive or AST), unwrap singletons,
        and resolve through pass-2 resolver if available.
        Returns: (name, value)
        """
        val = self.visit_children(node)
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        return node.argument_name, self._resolve_pass2(val)

    def visit_range_expression(self, node):
        start, end = self.visit_children(node)

        # unwrap [x] -> x
        if isinstance(start, list) and len(start) == 1:
            start = start[0]
        if isinstance(end, list) and len(end) == 1:
            end = end[0]

        # If either side is Physical, enforce same unit and return Physical(Range, unit)
        if isinstance(start, Physical) and isinstance(end, Physical):
            if start.unit != end.unit:
                raise ValueError("Mismatched units in range expression")
            if start.num >= end.num:
                raise ValueError("Range start must be < end")
            return Physical(Range(start.num, end.num), start.unit)

        # plain numeric range
        if start >= end:
            raise ValueError("Range start must be < end")
        return Range(start, end)

    # Simple literal/identifier visitors so lowering doesn’t depend on pass1’s return values
    def visit_integer_literal(self, node: ast_node.IntegerLiteral): return int(node.value)
    def visit_float_literal(self, node: ast_node.FloatLiteral):     return float(node.value)
    def visit_bool_literal(self, node: ast_node.BoolLiteral):       return node.value
    def visit_identifier(self, node: ast_node.Identifier):          return self._resolve(node.name)
    def visit_identifier_reference(self, node: ast_node.IdentifierReference): return self._resolve(node.name)
    def visit_type(self, node: ast_node.Type):                      return node.type_name

    def visit_physical_literal(self, node):
        """
        Build a Physical(value, unit) from a literal like `10 m` or `15s`.
        Children may already be primitives; never call .accept on them directly.
        """
        # numeric part: could be int/float or a Range already lowered
        num = self.visit_children(node)
        if isinstance(num, list) and len(num) == 1:
            num = num[0]

        unit_name = getattr(node, "unit_name", None)
        if unit_name is None:
            raise ValueError("PhysicalLiteral missing unit_name")

        unit_name = self._normalize_unit_name(unit_name)
        unit_dict = self._get_unit_dict()
        unit_obj = unit_dict.get(unit_name)
        if unit_obj is None:
            available = ", ".join(sorted(unit_dict.keys()))
            raise KeyError(
                f"Unit '{unit_name}' not defined. Available units: [{available}]"
            )

        return Physical(num, unit_obj)

    # Map a modifier invocation to (name, list-of-args/tuples)
    def _lower_modifier_invocation(self, node):
        name = node.modifier_name
        #name = MOD_ALIAS.get(node.modifier_name, node.modifier_name)

        args_raw = []
        for ch in node.get_children():
            if isinstance(ch, ast_node.PositionalArgument):
                args_raw.append(ch.accept(self))
            elif isinstance(ch, ast_node.NamedArgument):
                k, v = self.visit_named_argument(ch)
                #k = KW_ALIAS.get(k, k)
                args_raw.append((k, v))
        return name, args_raw

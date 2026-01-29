# osc2_parser/srunner/osc2/ast_manager/post_checks.py
from osc2_parser.srunner.osc2.utils.log_manager import LOG_ERROR
from osc2_parser.srunner.osc2.symbol_manager.struct_symbol import StructSymbol
from osc2_parser.srunner.osc2.symbol_manager.enum_symbol import EnumSymbol, EnumMemberSymbol
from osc2_parser.srunner.osc2.symbol_manager.actor_symbol import ActorSymbol
from osc2_parser.srunner.osc2.symbol_manager.action_symbol import ActionSymbol
from osc2_parser.srunner.osc2.symbol_manager.scenario_symbol import ScenarioSymbol
from osc2_parser.srunner.osc2.symbol_manager.method_symbol import MethodSymbol
from osc2_parser.srunner.osc2.symbol_manager.modifier_symbol import ModifierSymbol
from osc2_parser.srunner.osc2.symbol_manager.event_symbol import EventSymbol
from osc2_parser.srunner.osc2.symbol_manager.variable_symbol import VariableSymbol
from osc2_parser.srunner.osc2.symbol_manager.parameter_symbol import ParameterSymbol
from osc2_parser.srunner.osc2.symbol_manager.physical_type_symbol import PhysicalTypeSymbol
from osc2_parser.srunner.osc2.symbol_manager.unit_symbol import UnitSymbol
from osc2_parser.srunner.osc2.symbol_manager.si_exponent_symbol import SiExpSymbol, SiBaseExponentListScope

def _namespace_of(sym) -> str:
    if isinstance(sym, (StructSymbol, EnumSymbol, ActorSymbol, PhysicalTypeSymbol, UnitSymbol)):
        return "type"
    if isinstance(sym, (VariableSymbol, ParameterSymbol, EnumMemberSymbol)):
        return "value"
    if isinstance(sym, (ActionSymbol, ScenarioSymbol, MethodSymbol, ModifierSymbol, EventSymbol)):
        return "behavior"
    if isinstance(sym, (SiExpSymbol, SiBaseExponentListScope)):
        return "unitmeta"
    return "other"

def _child_scopes(scope):
    for s in getattr(scope, "symbols", {}).values():
        if hasattr(s, "symbols"):
            yield s

def _scope_label(scope):
    # best-effort for readable printing
    name = getattr(scope, "name", None)
    if name:
        try:
            # Action/Scenario sometimes have a QualifiedBehaviorSymbol in .name
            b = getattr(name, "behavior", None)
            a = getattr(name, "actor", None)
            if b is not None:
                return f"{a + '.' if a else ''}{b}"
        except Exception:
            pass
        return str(name)
    return scope.__class__.__name__

def check_namespace_collisions(global_scope, emit_errors=True):
    """
    Run the collision check. If emit_errors=True, LOG_ERROR is called for
    illegal redefinitions. Returns a stats dict you can print.
    """
    stats = {
        "scopes_scanned": 0,
        "names_with_redefs": 0,
        "issues": []  # list of dicts with details
    }

    def _check_scope(scope):
        stats["scopes_scanned"] += 1
        redefs = getattr(scope, "redefinitions", {}) or {}
        if redefs:
            stats["names_with_redefs"] += len(redefs)

        for name, dup_list in redefs.items():
            canonical = scope.symbols.get(name)
            if canonical is None:
                continue
            all_syms = [canonical] + [sym for (sym, _ctx) in dup_list]

            # group by namespace
            buckets = {}
            for s in all_syms:
                buckets.setdefault(_namespace_of(s), []).append(s)

            for ns, items in buckets.items():
                if ns == "other":
                    continue
                if len(items) > 1:
                    first_dup_ctx = dup_list[0][1] if dup_list else None
                    issue = {
                        "scope": _scope_label(scope),
                        "name": name,
                        "namespace": ns,
                        "line": getattr(first_dup_ctx, "line", None),
                        "column": getattr(first_dup_ctx, "column", None)
                    }
                    stats["issues"].append(issue)
                    if emit_errors:
                        LOG_ERROR(f"{name} is redefined in '{ns}' namespace.", first_dup_ctx)

        for child in _child_scopes(scope):
            _check_scope(child)

    if global_scope is not None:
        _check_scope(global_scope)
    return stats

def iter_children(node):
    # ASTBuilder uses node.set_children(child), which usually builds a 'children' list.
    for name in ("children", "_children", "nodes"):
        kids = getattr(node, name, None)
        if kids:
            if isinstance(kids, (list, tuple)):
                for k in kids: 
                    if k is not None: 
                        yield k
            else:
                yield kids

def find_any_scope(root):
    stack = [root]
    visited = set()
    while stack:
        n = stack.pop()
        if id(n) in visited: 
            continue
        visited.add(id(n))
        sc = getattr(n, "scope", None)
        if sc is not None:
            return sc
        # try getter if present
        if hasattr(n, "get_scope"):
            try:
                sc = n.get_scope()
                if sc is not None:
                    return sc
            except Exception:
                pass
        stack.extend(_iter_children(n))
    return None

def climb_to_global(scope):
    cur = scope
    top = cur
    while cur is not None and hasattr(cur, "get_enclosing_scope"):
        top = cur
        cur = cur.get_enclosing_scope()
    return top

def global_scope_from_ast_tree(ast_tree):
    any_scope = find_any_scope(ast_tree)
    global_scope = climb_to_global(any_scope) if any_scope is not None else None
    return global_scope
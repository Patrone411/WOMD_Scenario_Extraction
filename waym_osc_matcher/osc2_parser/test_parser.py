from . import MiniOSC2ScenarioConfig, ConfigInit, print_pytree, pytree_to_actor_constraints
from .pytree.ir_lowering import IRLowering
from .pytree.print_tree import print_ir

from osc2_parser.srunner.osc2.semantics.registry import SemanticsRegistry
from osc2_parser.srunner.osc2.semantics.validator import SemanticValidator, infer_type
from osc2_parser.srunner.osc2.semantics.ir_adapter import validate_from_ir
from osc2_parser.srunner.osc2.ast_manager.post_checks import check_namespace_collisions, global_scope_from_ast_tree


PREFIX = "osc2_parser/osc/"
osc_file = "test_actions.osc"
osc_file = "relative.osc"


config = MiniOSC2ScenarioConfig(PREFIX + osc_file)
# PASS 1: symbols/units/vars/actor-registry/instances
pass1 = ConfigInit(config)
pass1.visit(config.ast_tree)


global_scope = global_scope_from_ast_tree(config.ast_tree)
stats = check_namespace_collisions(global_scope, emit_errors=False)
print(
    f"[namespace check] scopes_scanned={stats['scopes_scanned']} "
    f"names_with_redefs={stats['names_with_redefs']} "
    f"illegal_collisions={len(stats['issues'])}"
)


# PASS 2: lower to IR (choose entry scenarios, e.g., {"top"} or set() for all)
lower = IRLowering(config, actor_registry=pass1.actor_registry, entry_names={"top"})
scenarios = lower.lower(config.ast_tree)

# Load registry + validator (optionally pass a better type hook)
REGISTRY_PATH = "osc2_parser/srunner/osc2/semantics/osc_semantics_registry.json"
sem_registry = SemanticsRegistry.from_file(REGISTRY_PATH)
print('test1')
ms = sem_registry.get_modifier("lane")
print("lane variants:", [getattr(v, "name", "<default>") for v in ms.variants])
for v in ms.variants:
    print("  params:", list(v.params.keys()))
print('test2')



#validator    = SemanticValidator(sem_registry)  # or with a type hook
validator = SemanticValidator(sem_registry, type_of_expr=infer_type, debug_types=True)
# Validate semantics using the IR
validate_from_ir(scenarios, validator)
print("Semantic validation completed.")

print_ir(scenarios)

"""
config = MiniOSC2ScenarioConfig(PREFIX + osc_file)
for scen in config.pytree.values():
    print(render_scenario(scen))"""

"""visitor = ConfigInit(config)
visitor.visit(config.ast_tree)
py_tree = visitor.pytree

print("tree:")
print_pytree(py_tree)

map_constraints = {}
map_constraints["min_lanes"] = config.path.min_driving_lanes
print("map_constraints: ", map_constraints)

actor_constraints = pytree_to_actor_constraints(py_tree)
print("actor constraints: ", actor_constraints)"""


"""from osc2_parser.srunner.osc2.ast_manager import ast_node

def _name_of(x):
    # Works for Identifier nodes or plain strings
    return getattr(x, "name", x) if not isinstance(x, str) else x
# Buckets we care about (seed with what Pass1 already collected)
buckets = {
    "physical_type": set(getattr(pass1.father_ins, "physical_dict", {}).keys()),
    "unit":          set(getattr(pass1.father_ins, "unit_dict", {}).keys()),
    "struct":        set(getattr(pass1.father_ins, "struct_declaration", {}).keys()),
    "actor":         set(pass1.actor_registry.keys()),
    "enum":          set(),
    "action":        set(),
    "modifier":      set(),
    # Optional: scenario-level variables (to detect shadowing)
    "variable":      set(getattr(pass1.father_ins, "variables", {}).keys()),
}

# Collect anything Pass1 didn’t store explicitly by scanning the AST once
for c in ast_root.get_children():
    if isinstance(c, ast_node.EnumDeclaration):
        buckets["enum"].add(_name_of(getattr(c, "enum_name", None)))
    elif isinstance(c, ast_node.StructDeclaration):
        buckets["struct"].add(_name_of(getattr(c, "struct_name", None)))
    elif isinstance(c, ast_node.ActorDeclaration):
        buckets["actor"].add(_name_of(getattr(c, "actor_name", None)))
    elif isinstance(c, (ast_node.ActionDeclaration, getattr(ast_node, "ActionInherts", tuple()))):
        # most grammars use qualified_behavior_name for actions
        n = getattr(c, "qualified_behavior_name", None)
        if n: buckets["action"].add(_name_of(n))
    elif isinstance(c, ast_node.ModifierDeclaration):
        buckets["modifier"].add(_name_of(getattr(c, "modifier_name", None)))

print("units: ")
units = buckets.get("unit", set())
if not units:
    print("(no modifiers found)")
else:
    for name in sorted(units):
        print(name)

print("physical_types: ")
physical_types= buckets.get("physical_types", set())
if not physical_types:
    print("(no modifiers found)")
else:
    for name in sorted(physical_types):
        print(name)"""
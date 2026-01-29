from osc2_parser.matching.constraints import constraints_from_ir
from . import MiniOSC2ScenarioConfig, ConfigInit, print_pytree, pytree_to_actor_constraints
from .pytree.ir_lowering import IRLowering
from .pytree.print_tree import print_ir
import json

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

#validator    = SemanticValidator(sem_registry)  # or with a type hook
validator = SemanticValidator(sem_registry, type_of_expr=infer_type, debug_types=True)
# Validate semantics using the IR
validate_from_ir(scenarios, validator)
print("Semantic validation completed.")


constraints_by_scenario = constraints_from_ir(scenarios)
# Example filter usage:
top = constraints_by_scenario["top"]
with open("./osc2_parser/constraints.json", "w", encoding="utf-8") as f:
        json.dump(constraints_by_scenario, f, ensure_ascii=False, indent=2)
print(top)
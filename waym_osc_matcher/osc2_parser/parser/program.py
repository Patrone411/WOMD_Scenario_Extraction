# osc2_parser/parser/program.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

from osc2_parser.scenario_config import MiniOSC2ScenarioConfig as ScenarioCfg
from osc2_parser.config_init import ConfigInit
from osc2_parser.pytree.ir_lowering import IRLowering
from osc2_parser.srunner.osc2.semantics.registry import SemanticsRegistry
from osc2_parser.srunner.osc2.semantics.validator import SemanticValidator, infer_type
from osc2_parser.srunner.osc2.semantics.ir_adapter import validate_from_ir, get_min_lanes
from osc2_parser.matching.constraints import constraints_from_ir  # this is OK: it only builds role/type constraints, no runtime matching

def collect_block_durations(blocks):
    out = {}
    stack = list(blocks or [])
    while stack:
        b = stack.pop()
        lbl = b.get("label")
        dur = b.get("duration")
        if lbl is not None and dur is not None:
            out[lbl] = dur
        stack.extend(b.get("children") or [])
    return out

@dataclass
class CompiledOSC:
    calls: List[dict]
    constraints_by_scenario: Dict
    min_lanes: int
    validation_result: bool  # optional to expose; remove if you prefer
    validation_errors: list
    block_durations: Dict

class OSCProgram:
    def __init__(self, osc_path: str, entry_names: Set[str] | None = None):
        self.osc_path = osc_path
        self.entry_names = entry_names or {"top"}

    def compile(self) -> CompiledOSC:
        config = ScenarioCfg(self.osc_path)

        # PASS 1: symbols/units/vars/actor-registry/instances
        pass1 = ConfigInit(config)
        pass1.visit(config.ast_tree)

        # PASS 2: lower to IR
        lower = IRLowering(config, actor_registry=pass1.actor_registry, entry_names=self.entry_names)
        scenarios = lower.lower(config.ast_tree)
        scenarios_list = list(scenarios.values())
        #print(scenarios_list)

        # Semantics registry + validator
        REGISTRY_PATH = "osc2_parser/srunner/osc2/semantics/osc_semantics_registry.json"
        sem_registry = SemanticsRegistry.from_file(REGISTRY_PATH)
        validator = SemanticValidator(sem_registry, type_of_expr=infer_type, debug_types=False)

        # Validate semantics using IR
        passed, errors = validate_from_ir(scenarios_list, validator)

        # Per-scenario constraints (role typing, domains, flat calls)
        constraints_by_scenario = constraints_from_ir(scenarios_list)
        calls = list(constraints_by_scenario["top"]["calls_flat"])
        print(calls)
        # Lanes
        min_lanes = get_min_lanes(scenarios_list, scenario_name="top", default=0)

        block_durations = collect_block_durations(constraints_by_scenario["top"].get("blocks"))
        # NOTE: do NOT build plans here anymore
        return CompiledOSC(
            calls=calls,
            constraints_by_scenario=constraints_by_scenario,
            min_lanes=min_lanes,
            validation_result=passed,
            validation_errors=errors,
            block_durations=block_durations,
        )

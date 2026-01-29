from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import os

# ---- Specs ----

@dataclass
class ParamSpec:
    type: Optional[str] = None
    optional: bool = False
    default: Any = None
    ignored_if_present: List[str] = field(default_factory=list)

@dataclass
class OverloadSpec:
    name: Optional[str]  # None => default
    params: Dict[str, ParamSpec]
    rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionSpec:
    qname: str
    invoker_type: str
    inherits: Optional[str]
    abstract: bool = False
    overloads: List[OverloadSpec] = field(default_factory=list)

@dataclass
class ModifierVariantSpec:
    name: Optional[str]  # None => default
    params: Dict[str, ParamSpec]
    rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModifierSpec:
    name: str
    applies_to: str  # action family qname, e.g. "movable_object.action_for_movable_object"
    variants: List[ModifierVariantSpec] = field(default_factory=list)
    rules: Dict[str, Any] = field(default_factory=dict)

class SemanticsRegistry:
    """
    Loads the machine-readable semantics JSON (actors/actions/modifiers)
    and provides queries used by the validator.
    """
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        self.actors: Dict[str, Dict[str, Any]] = data.get("actor_types", {})
        self.actions: Dict[str, ActionSpec] = {}
        self.modifiers: Dict[str, ModifierSpec] = {}
        self._build_actions()
        self._build_modifiers()

    @classmethod
    def from_file(cls, path: str) -> "SemanticsRegistry":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    # ---------- builders ----------

    def _paramspec_from(self, d: Dict[str, Any]) -> ParamSpec:
        return ParamSpec(
            type=d.get("type"),
            optional=bool(d.get("optional", False)),
            default=d.get("default"),
            ignored_if_present=list(d.get("ignored_if_present", []) or []),
        )

    def _overloadspec_from(self, od: Dict[str, Any]) -> OverloadSpec:
        params: Dict[str, ParamSpec] = {}
        for pname, ps in (od.get("params") or {}).items():
            params[pname] = self._paramspec_from(ps or {})
        return OverloadSpec(
            name=od.get("name"),
            params=params,
            rules=od.get("rules") or {},
        )

    def _build_actions(self):
        for qname, spec in (self.data.get("actions") or {}).items():
            overloads = [self._overloadspec_from(od) for od in (spec.get("overloads") or [])]
            self.actions[qname] = ActionSpec(
                qname=qname,
                invoker_type=spec.get("invoker_type"),
                inherits=spec.get("inherits"),
                abstract=bool(spec.get("abstract", False)),
                overloads=overloads,
            )

    def _variant_from(self, vd: Dict[str, Any]) -> ModifierVariantSpec:
        params: Dict[str, ParamSpec] = {}
        for pname, ps in (vd.get("params") or {}).items():
            params[pname] = self._paramspec_from(ps or {})
        return ModifierVariantSpec(
            name=vd.get("name"),
            params=params,
            rules=vd.get("rules") or {},
        )

    def _build_modifiers(self):
        for name, spec in (self.data.get("modifiers") or {}).items():
            variants = [self._variant_from(vd) for vd in (spec.get("variants") or [])]
            self.modifiers[name] = ModifierSpec(
                name=name,
                applies_to=spec.get("applies_to"),
                variants=variants,
                rules=spec.get("rules") or {},
            )

    # ---------- queries ----------

    def get_modifier(self, name: str) -> Optional[ModifierSpec]:
        return self.modifiers.get(name)

    def action_family_of(self, qname: str) -> str:
        ancestors = self.action_ancestors(qname)
        for anc in ancestors[1:]:
            spec = self.actions.get(anc)
            if spec and getattr(spec, "abstract", False):
                return anc
        return ancestors[1] if len(ancestors) > 1 else qname

    def _actor_ancestors(self, actor_type: Optional[str]) -> List[str]:
        """
        Return [actor_type, base, base_of_base, ...] up to root.
        """
        out: List[str] = []
        cur = actor_type
        guard = 0
        while cur and guard < 64:
            out.append(cur)
            parent = self.actors.get(cur, {}).get("inherits")
            cur = parent
            guard += 1
        return out
    
    def action_ancestors(self, qname: str):
        """Return [qname, parent, grandparent, ...] following 'inherits'."""
        out = []
        cur = qname
        while cur:
            out.append(cur)
            spec = self.actions.get(cur)
            cur = getattr(spec, "inherits", None) or None
        return out
    
    def is_action_in_family(self, action_q: str, family_q: str) -> bool:
        """True if 'family_q' is in the inheritance chain of 'action_q'."""
        return family_q in self.action_ancestors(action_q)
    
    def find_actions_for_method_on(self, invoker_type: str, method_name: str) -> List[Tuple[str, ActionSpec]]:
        """
        Find all actions whose simple method name matches and whose invoker_type
        is a base of (or equal to) the provided invoker_type.
        """
        results: List[Tuple[str, ActionSpec]] = []
        inv_anc = set(self._actor_ancestors(invoker_type))
        for q, spec in self.actions.items():
            simple = q.split(".")[-1]
            if simple != method_name:
                continue
            # spec.invoker_type must be a base (or same) of invoker_type
            if spec.invoker_type in inv_anc:
                results.append((q, spec))
        return results

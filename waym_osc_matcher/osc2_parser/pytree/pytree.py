# osc2_parser/srunner/ir/pytree.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---- IR node classes (runtime-friendly PyTree) ----

@dataclass
class EventNode:
    name: str

@dataclass
class VarNode:
    name: str
    type: Optional[str] = None
    value: Any = None  # can be scalar or Physical

@dataclass
class ActorInst:
    name: str
    type: str  # actor type name

@dataclass
class ModifierCall:
    name: str
    args: List[Any] = field(default_factory=list)          # positional
    kwargs: Dict[str, Any] = field(default_factory=dict)   # named

@dataclass
class ActionCall:
    actor: str
    action: str
    args: List[Any] = field(default_factory=list)           # action args (positional and/or ("key",val))
    modifiers: List[ModifierCall] = field(default_factory=list)

@dataclass
class SerialBlock:
    label: Optional[str] = None
    children: List[Any] = field(default_factory=list)       # ActionCall or nested blocks

@dataclass
class ParallelBlock:
    label: Optional[str] = None
    duration: Optional[Any] = None                          # Physical or scalar
    children: List[Any] = field(default_factory=list)

@dataclass
class OneOfBlock:
    label: Optional[str] = None
    children: List[Any] = field(default_factory=list)

@dataclass
class ScenarioNode:
    name: str
    events: List[EventNode] = field(default_factory=list)
    vars: Dict[str, VarNode] = field(default_factory=dict)
    actors: Dict[str, ActorInst] = field(default_factory=dict)
    blocks: List[Any] = field(default_factory=list)         # top-level blocks
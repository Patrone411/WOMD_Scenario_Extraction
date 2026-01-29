from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class Route:
    length: Optional[Any] = None         # Physical length or None
    directionality: Optional[str] = None # e.g., 'forward' | 'bidirectional' (string or enum)
    min_lanes: Optional[int] = None
    max_lanes: Optional[int] = None
    anchors: Optional[List[str]] = None

@dataclass
class Path(Route):
    points: List[Any] = field(default_factory=list)  # List[pose_3d]-ish
    interpolation: Optional[str] = None              # e.g., 'linear' (string or enum)
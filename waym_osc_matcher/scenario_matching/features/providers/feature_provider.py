from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from scenario_matching.features.adapters import TagFeatures

@dataclass
class SegmentMeta:
    scene_id: Optional[str] = None
    folder:   Optional[str] = None
    source_uri: Optional[str] = None

# What every provider returns (one or many of these)
@dataclass
class FeatureLoadResult:
    feats_by_seg: Dict[str, TagFeatures]                 # REQUIRED
    seg_meta_by_id: Dict[str, SegmentMeta] = field(default_factory=dict)
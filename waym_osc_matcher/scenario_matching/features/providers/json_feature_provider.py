from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict
import json

from scenario_matching.features.adapters import TagFeatures
from scenario_matching.features.providers.feature_provider import FeatureLoadResult, SegmentMeta

@dataclass
class JsonFeatureProvider:
    json_path: str
    min_lanes: int | None = None

    def load(self) -> FeatureLoadResult:
        with open(self.json_path, "r") as f:
            data = json.load(f)
        feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=self.min_lanes)
        # Optional: fill SegmentMeta if you have it (scene_id/folder/uri)
        seg_meta_by_id: Dict[str, SegmentMeta] = {}
        return FeatureLoadResult(feats_by_seg=feats_by_seg, seg_meta_by_id=seg_meta_by_id)
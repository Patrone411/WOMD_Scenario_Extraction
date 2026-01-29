# osc_parser/features/source.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, List, Dict

from scenario_matching.features.providers.s3_pickle_feature_provider import S3PickleFeatureProvider
from scenario_matching.features.providers.feature_provider import FeatureLoadResult

@dataclass
class S3PickleSource:
    bucket: str
    base_prefix: str
    endpoint_url: Optional[str] = None
    verify: Optional[str | bool] = None
    only_scene_ids: Optional[List[str]] = None
    min_lanes: Optional[int] = None
    max_segments: Optional[int] = None

    def __iter__(self) -> Iterable[FeatureLoadResult]:
        prov = S3PickleFeatureProvider(
            bucket=self.bucket,
            base_prefix=self.base_prefix,
            only_scene_ids=self.only_scene_ids,
            endpoint_url=self.endpoint_url,
            verify=self.verify,
            min_lanes=self.min_lanes,
            max_segments=self.max_segments,
        )
        yield from prov.load()

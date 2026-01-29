from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict, Generator, Optional
import os
import sys
import pickle
import boto3

from scenario_matching.features.adapters import TagFeatures
from .feature_provider import FeatureLoadResult, SegmentMeta
import hashlib

@dataclass
class S3PickleFeatureProvider:
    bucket: str
    base_prefix: str
    only_scene_ids: list[str] | None = None
    endpoint_url: str | None = None
    verify: str | bool | None = None
    min_lanes: int | None = None
    max_segments: int | None = None  # cap number of yielded (non-empty) pickles
    log_skipped: bool = False        # optional: log when a pickle yields no segments

    def _s3(self):
        return boto3.client("s3", endpoint_url=self.endpoint_url, verify=self.verify)

    def _iter_keys(self) -> Generator[str, None, None]:
        """Yield S3 object keys under base_prefix, optionally filtered by scene ids."""
        s3 = self._s3()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.base_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.only_scene_ids:
                    # keep key if it contains *any* requested scene id
                    if not any(sid in key for sid in self.only_scene_ids):
                        continue
                yield key

    @staticmethod
    def _scene_id_from_key(key: str) -> str:
        """Derive a scene id from the pickle filename (e.g., '.../abcd1234.pkl' -> 'abcd1234')."""
        base = os.path.basename(key)
        name, _ext = os.path.splitext(base)
        return name or "scene"

    def load(self) -> Iterable[FeatureLoadResult]:
        """
        Yield one FeatureLoadResult per NON-EMPTY pickle:
          - If the pickle decodes to a dict: pass through TagFeatures.load_all_segments(...).
          - If the pickle is a single TagFeatures: wrap into a 1-item mapping.
          - If, after filtering (e.g. min_lanes) there are 0 segments, skip yielding.
        """
        s3 = self._s3()
        yielded = 0

        for key in self._iter_keys():
            try:
                resp = s3.get_object(Bucket=self.bucket, Key=key)
                body = resp["Body"].read()
                obj = pickle.loads(body)
            except Exception as ex:
                print(f"[S3PickleFeatureProvider] ERROR reading {key}: {ex}", file=sys.stderr)
                continue

            # Build per-pickle segment map
            feats_by_seg: Dict[str, TagFeatures]
            if isinstance(obj, dict):
                feats_by_seg = TagFeatures.load_all_segments(obj, min_lanes=self.min_lanes)
            elif isinstance(obj, TagFeatures):
                feats_by_seg = {obj.segment_id: obj}
            else:
                print(f"[S3PickleFeatureProvider] Skip {key}: unexpected payload type {type(obj)}", file=sys.stderr)
                continue

            # SKIP LOGIC: if this pickle contributes no segments (after filtering), skip it
            if not feats_by_seg:
                if self.log_skipped:
                    print(f"[S3PickleFeatureProvider] Skip empty pickle (no segments after filtering): {key}", file=sys.stderr)
                continue

            # Metadata: same scene/folder/source for all segments from this pickle
            scene_id = self._scene_id_from_key(key)
            meta = SegmentMeta(
                scene_id=scene_id,
                folder=self.base_prefix,
                source_uri=f"s3://{self.bucket}/{key}",
            )
            seg_meta_by_id: Dict[str, SegmentMeta] = {seg_id: meta for seg_id in feats_by_seg.keys()}

            yield FeatureLoadResult(
                feats_by_seg=feats_by_seg,
                seg_meta_by_id=seg_meta_by_id,
            )

            yielded += 1
            if self.max_segments and yielded >= self.max_segments:
                break

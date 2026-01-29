""" Kubernetes runner for parallel processing of Waymo TFRecord files."""
import os
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor

# pylint: disable=import-error, no-name-in-module

from waymo_osc_extractor.s3_tfrecord_accessor import S3TFRecordAccessor
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import Scenario
from waymo_osc_extractor.scenario_processor2 import ScenarioProcessor2
from waymo_osc_extractor.external.waymo_motion_scenario_mining.utils.tags_generator import TagsGenerator
import numpy as np
import math

def json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    # last resort
    try:
        return json_sanitize(obj.__dict__)
    except Exception:
        return str(obj)
    
def tag_worker(scenario: Scenario, n: int):
    """Process a single scenario.

    Args:
        scenario (Scenario): An instance of the Scenario class to process.
    """
    tagger = TagsGenerator()

    # Placeholder for actual processing logic
    print(f"[setup] {n} - {scenario.name} start")
    scenario.setup()
    print(f"[setup] {n} - {scenario.name} done")
    gi, iar, aa, aeei = tagger.tagging(
            scenario.example, f"S3_scene_{scenario.name}")
        

    result = ScenarioProcessor2(scenario,n).process()
    print(result["inter_actor_activities"])

    return result, iar
    #result = ScenarioProcessor(scenario, tagger, n).process()

def tag_tfrecord(accessor: S3TFRecordAccessor, s3_url, result_prefix= None):
    """Process one TFRecord file from S3 and upload results back to S3.

    Args:
        accessor (S3TFRecordAccessor): An instance of S3TFRecordAccessor to handle S3 operations.
        key (str): The S3 key of the TFRecord file to process.
        result_prefix (str): The S3 prefix where results will be uploaded.
    Returns:
        tuple: A tuple containing counts of processed, skipped, and error records.
    """
    print(f"[input] {s3_url} - loading ...")
    accessor.load_dataset(s3_url)
    print(f"[loaded] {s3_url}")

    # read all scenarios
    scenarios = list(accessor.enumerate_scenarios())
    print(f"[processed] {s3_url} - {len(scenarios)} scenarios found")

    # with ThreadPoolExecutor(max_workers=10) as upool:
    #     for n, scenario in enumerate(scenarios):
    #         upool.submit(setup_worker, scenario, n, tags_generator)
    import json
    inter_actors = {}
    per_actor = {}
    for n, scenario in enumerate(scenarios):
        if n == 1:
            break
        result,wsm = tag_worker(scenario, n)
        inter_actors[f"woe_{scenario.name}_iar"] = result["inter_actor_activities"]
        inter_actors[f"wsm_{scenario.name}_iar"] = wsm
        per_actor[f"{scenario.name}_general_actor_activities"] = result["general_actor_activities"]
        per_actor[f"{scenario.name}_actor_per_segment"] = result["actor_activities_per_segment"]

        
    with open("inter_actors.json", "w", encoding="utf-8") as f:
        json.dump(inter_actors, f, ensure_ascii=False, indent=2)
    with open("per_actor.json", "w", encoding="utf-8") as g:
        json.dump(json_sanitize(per_actor), g, ensure_ascii=False, indent=2)
    with open("tag_result.json", "w", encoding="utf-8") as x:
        json.dump(json_sanitize(result), x, ensure_ascii=False, indent=2)


def main():
    bucket = os.getenv("BUCKET", "waymo")
    in_prefix = os.getenv("INPUT_PREFIX", "tfrecords/training_tfexample.tfrecord")
    result_prefix = os.getenv("RESULT_PREFIX", f"results/dow-{datetime.now(tz=timezone.utc).isoformat()}/")
    shard_count = int(os.getenv("SHARD_COUNT", "100"))
    # Kubernetes Indexed Job (K8s >=1.27) provides this field; fallback to SHARD_INDEX if needed
    shard_index = int(os.getenv("JOB_COMPLETION_INDEX", os.getenv("SHARD_INDEX", "0")))


    # use with to ensure proper cleanup of resources
    with S3TFRecordAccessor() as accessor:
        s3_urls = accessor.enumerate_tfrecords(bucket, in_prefix)
        for i, s3_url in enumerate(s3_urls):
            if i % shard_count != shard_index: 
                print(f"[input] {s3_url} not in shard {shard_index}/{shard_count}, skipping")
                continue
            t0 = time.perf_counter()
            tag_tfrecord(accessor, s3_url, result_prefix)
            dt = time.perf_counter() - t0
            print(f"{s3_url} - Completed in {dt:.1f} seconds")


if __name__ == "__main__":
    main()
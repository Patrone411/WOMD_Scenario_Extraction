""" Kubernetes runner for parallel processing of Waymo TFRecord files. """
import os
import time
from datetime import datetime, timezone

# pylint: disable=import-error, no-name-in-module
from waymo_osc_extractor.s3_tfrecord_accessor import S3TFRecordAccessor
from waymo_osc_extractor.s3_scenario_accessor import S3ScenarioAccessor
from waymo_osc_extractor.waymo_scenario_tools.scenario_handling.scenario import Scenario
from waymo_osc_extractor.scenario_processor2 import ScenarioProcessor2

def safe_tag_worker(scenario, n, accessor):
    try:
        return tag_worker(scenario, n, accessor)
    except Exception as e:
        # soft-fail the single scenario, hard-log the info
        scen_name = getattr(scenario, "name", None)
        if scen_name is None:
            try:
                scen_name = scenario.scenario_id[0].decode('utf-8')
            except:
                scen_name = "<unknown>"

        print(f"[error] scenario {n} ({scen_name}) crashed with {repr(e)}")
        return False

def tag_worker(scenario: Scenario, n: int, accessor: S3ScenarioAccessor):
    print(f"[setup] {n} - {scenario.name} start")
    scenario.setup()
    print(f"[setup] {n} - {scenario.name} done")

    result = ScenarioProcessor2(scenario, n).process()
    if result:
        print(f"[process] {n} - {scenario.name} done")
        print(f"[save] {n} - {scenario.name} saving ...")
        saved = accessor.save_pck(scenario.name, result)
        if saved:
            print(f"[save] {n} - {scenario.name} done")
            return True
        else:
            print(f"[save] {n} - {scenario.name} failed to save")
    else:
        print(f"[process] {n} - {scenario.name} - no result generated")
    return False

def tag_tfrecord(record_accessor: S3TFRecordAccessor, s3_url, results_prefix):
    print(f"[input] {s3_url} - loading ...")
    record_accessor.load_dataset(s3_url)

    scenario_accessor = S3ScenarioAccessor(
        bucket=record_accessor.bucket,
        key_prefix=results_prefix,
        record_name=str(record_accessor.file_num),
    )
    print(f"[loaded] {s3_url}")

    processed_ok = 0
    processed_fail = 0

    for n, scenario in enumerate(record_accessor.enumerate_scenarios()):
        ok = safe_tag_worker(scenario, n, scenario_accessor)
        if ok:
            processed_ok += 1
        else:
            processed_fail += 1

    print(f"[summary] {s3_url}: {processed_ok} scenarios saved, {processed_fail} failed")

def main():
    bucket = os.getenv("BUCKET", "waymo")
    in_prefix = os.getenv("INPUT_PREFIX", "tfrecords/training_tfexample.tfrecord")

    # new run folder so we don't collide with last run
    result_prefix = os.getenv(
        "RESULT_PREFIX",
        f"results/continue-{datetime.now(tz=timezone.utc).isoformat()}/"
    )

    # completion index from the Job
    shard_index = int(os.getenv("JOB_COMPLETION_INDEX", os.getenv("SHARD_INDEX", "0")))

    # <-- NEW: where to start in the global TFRecord list
    start_offset = int(os.getenv("START_OFFSET", "0"))

    with S3TFRecordAccessor() as accessor:
        all_urls = list(accessor.enumerate_tfrecords(bucket, in_prefix))
        all_urls.sort()

        total = len(all_urls)

        global_index = start_offset + shard_index

        print(f"[debug] total TFRecords: {total}")
        print(f"[debug] start_offset={start_offset}")
        print(f"[debug] shard_index (pod index)={shard_index}")
        print(f"[debug] global_index={global_index}")

        if global_index >= total:
            print(f"[debug] global_index {global_index} is out of range (0..{total-1}), exiting cleanly.")
            return

        s3_url = all_urls[global_index]
        print(f"[debug] this pod will process ONLY global index {global_index}: {s3_url}")

        t0 = time.perf_counter()
        tag_tfrecord(accessor, s3_url, result_prefix)
        dt = time.perf_counter() - t0
        print(f"{s3_url} - Completed in {dt:.1f} seconds")

    print("[done] pod finished normally")


if __name__ == "__main__":
    main()

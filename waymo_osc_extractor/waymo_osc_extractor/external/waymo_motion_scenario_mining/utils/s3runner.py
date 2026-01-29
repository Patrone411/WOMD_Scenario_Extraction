from pathlib import Path
import json, re, time, traceback
from rich.progress import track
import tensorflow as tf
from helpers.create_rect_from_file import features_description, get_parsed_carla_data
from logger.logger import *
from scenario_miner import ScenarioMiner
from tags_generator import TagsGenerator
from warnings import simplefilter
simplefilter('error')
import sys
from tqdm import tqdm  # progress bar
# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).parents[2].parent  # Adjusts for your nested structure
sys.path.append(str(PROJECT_ROOT))

from waymoScenarioMining import features_description, tf_scenario_streamer, create_s3_client, tf_scenario_streamer_with_keys

def run_scenario_miner_s3(bucket_name: str = "waymo", prefix: str = "tfrecords/", result_prefix: str = "results/"):
    s3 = create_s3_client()
    
    RESULT_TIME = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
    RESULT_TIME = "2025-09-01-22_42"
    result_prefix = f"{result_prefix}{RESULT_TIME}/"
    
    time_start = time.perf_counter()

    # Stream scenarios with original TFRecord key for FILENUM
    for idx, (parsed, key) in enumerate(track(
        tf_scenario_streamer_with_keys(features_description, bucket_name, prefix),
        description=f"Processing scenarios from s3://{bucket_name}/{prefix}"
    )):
        try:
            FILE = Path(key).name  # e.g. training_tfexample.tfrecord-00000-of-01000
            
            # Extract correct Waymo FILENUM from filename
            match = re.search(r"-(\d{5})-of-\d{5}$", FILE)
            FILENUM = match.group(1) if match else str(idx).zfill(5)  # fallback just in case
            
            # Extract scenario ID
            scene_id = parsed['scenario/id'].numpy().item().decode("utf-8")

            # Build result folder per TFRecord file number
            result_filename = f"Waymo_{FILENUM}_{scene_id}_tag.json"
            result_s3_key = f"{result_prefix}{FILENUM}/{result_filename}"

            # ✅ Check if result already exists in S3
            try:
                s3.head_object(Bucket=bucket_name, Key=result_s3_key)
                print(f"Skipping {scene_id} (already exists in s3://{bucket_name}/{result_s3_key})")
                continue
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] != "404":
                    raise  # real error (permissions, etc.), not just missing file
            
            print(f"Processing scene: {scene_id} from file {FILE} → {result_s3_key}")


            # Run tagging
            tags_generator = TagsGenerator()
            general_info, inter_actor_relation, actors_activity, actors_environment_element_intersection = \
                tags_generator.tagging(parsed, f"S3_scene_{scene_id}")

            result_dict = {
                "general_info": general_info,
                "inter_actor_relation": inter_actor_relation,
                "actors_activity": actors_activity,
                "actors_environment_element_intersection": actors_environment_element_intersection,
            }
            #print(result_dict)

            # Upload result JSON back to S3
            s3.put_object(
                Bucket=bucket_name,
                Key=result_s3_key,
                Body=json.dumps(result_dict, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
            print(f"Uploaded result to s3://{bucket_name}/{result_s3_key}")

        except Exception as e:
            trace = traceback.format_exc()
            print(f"Scene {idx} error: {e}")
            print(f"trace:{trace}")

    time_end = time.perf_counter()
    print(f"Time cost: {time_end - time_start:.2f}s. Results saved under s3://{bucket_name}/{result_prefix}")



def run_scenario_miner_filewise(
    bucket_name: str = "waymo",
    prefix: str = "tfrecords/",
    result_prefix: str = "results/"
):
    import tempfile
    import os

    s3 = create_s3_client()

    # Fixed result time
    RESULT_TIME = "2025-09-01-22_42"
    result_prefix = f"{result_prefix}{RESULT_TIME}/"

    time_start = time.perf_counter()

    # 1️⃣ List all TFRecord files in S3
    resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    tfrecord_keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".tfrecord")]

    for file_idx, tf_key in enumerate(track(tfrecord_keys, description="Processing TFRecord files")):
        FILE = Path(tf_key).name  # e.g. training_tfexample.tfrecord-00000-of-01000

        # Extract Waymo file number (FILENUM)
        match = re.search(r"-(\d{5})-of-\d{5}$", FILE)
        FILENUM = match.group(1) if match else str(file_idx).zfill(5)

        print(f"\n📥 Downloading {tf_key} ...")
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local_path = tmp.name
            s3.download_file(bucket_name, tf_key, local_path)

        try:
            # 2️⃣ Iterate over scenarios in this TFRecord
            for idx, parsed in enumerate(
                tf_scenario_streamer(features_description, local_path)  # local file
            ):
                try:
                    # Extract scenario ID
                    scene_id = parsed['scenario/id'].numpy().item().decode("utf-8")

                    # Build result filename
                    result_filename = f"Waymo_{FILENUM}_{scene_id}_tag.json"
                    result_s3_key = f"{result_prefix}{FILENUM}/{result_filename}"

                    # ✅ Skip if already exists
                    try:
                        s3.head_object(Bucket=bucket_name, Key=result_s3_key)
                        print(f"Skipping {scene_id} (already in s3://{bucket_name}/{result_s3_key})")
                        continue
                    except s3.exceptions.ClientError as e:
                        if e.response['Error']['Code'] != "404":
                            raise

                    print(f"Processing scene {scene_id} from {FILE} → {result_s3_key}")

                    # Run tagging
                    tags_generator = TagsGenerator()
                    general_info, inter_actor_relation, actors_activity, actors_environment_element_intersection = \
                        tags_generator.tagging(parsed, f"S3_scene_{scene_id}")

                    result_dict = {
                        "general_info": general_info,
                        "inter_actor_relation": inter_actor_relation,
                        "actors_activity": actors_activity,
                        "actors_environment_element_intersection": actors_environment_element_intersection,
                    }

                    # Upload result JSON
                    s3.put_object(
                        Bucket=bucket_name,
                        Key=result_s3_key,
                        Body=json.dumps(result_dict, indent=2).encode("utf-8"),
                        ContentType="application/json"
                    )
                    print(f"✅ Uploaded result to s3://{bucket_name}/{result_s3_key}")

                except Exception as e:
                    trace = traceback.format_exc()
                    print(f"Scene {idx} in {FILE} error: {e}")
                    print(trace)

        finally:
            # 3️⃣ Clean up local TFRecord file
            os.remove(local_path)
            print(f"🗑️ Deleted local file {local_path}")

    time_end = time.perf_counter()
    print(f"\n⏱️ Total time: {time_end - time_start:.2f}s. Results saved under s3://{bucket_name}/{result_prefix}")
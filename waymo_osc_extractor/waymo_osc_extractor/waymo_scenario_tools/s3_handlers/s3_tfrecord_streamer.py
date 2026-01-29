import subprocess
import boto3
import struct
import tensorflow as tf
import os
import re
import json
from smart_open import open



def create_s3_client():
    aws_access_key_id = os.environ.get('S3_ACCESS_KEY')
    aws_secret_access_key = os.environ.get('S3_SECRET_KEY')

    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url='https://gif.s3.iavgroup.local',
        verify="/mnt/c/Users/I010444/source/repos/googledrivePy/IAV-CA-Bundle.pem"
    )

def list_tfrecord_keys(s3, bucket_name, prefix='tfrecords/'):
    """List all .tfrecord files in the specified prefix of the bucket."""
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.tfrecord') or 'tfrecord' in key:
                keys.append(key)
    if not keys:
        raise FileNotFoundError(f"No TFRecord files found in s3://{bucket_name}/{prefix}")
    return keys

def read_tfrecord_stream(file_obj):
    """Generator that yields raw serialized TFRecord example bytes from a file-like object."""
    while True:
        length_bytes = file_obj.read(8)
        if len(length_bytes) < 8:
            break  # EOF

        length = struct.unpack('<Q', length_bytes)[0]
        file_obj.read(4)  # Skip length CRC
        data = file_obj.read(length)
        if len(data) < length:
            break
        file_obj.read(4)  # Skip data CRC

        yield data

def parse_example(serialized_example, features_description):
    return tf.io.parse_single_example(serialized_example, features_description)

def stream_and_parse_tfrecord_s3(s3_uri, features_description, s3_client):
    transport_params = {'client': s3_client}
    with open(s3_uri, 'rb', transport_params=transport_params) as f:
        for raw_example in read_tfrecord_stream(f):
            yield parse_example(raw_example, features_description)

def tf_scenario_streamer(features_description, bucket_name='waymo', prefix='tfrecords/'):
    """
    Generator that yields all parsed examples across all TFRecord files
    in the specified bucket and prefix.
    """
    s3 = create_s3_client()
    tfrecord_keys = list_tfrecord_keys(s3, bucket_name, prefix)
    try:
        for key in tfrecord_keys:
            s3_uri = f's3://{bucket_name}/{key}'
            print(f"Processing: {s3_uri}")
            yield from stream_and_parse_tfrecord_s3(s3_uri, features_description, s3_client=s3)
    finally:
        s3.close()  # explicitly close sockets

def local_tf_scenario_streamer(features_description, local_tf_record_path):
    """
    Generator that yields all parsed examples across all TFRecord files
    in the specified bucket and prefix.
    """

    dataset = tf.data.TFRecordDataset(local_tf_record_path)
    # Build a generator expression that parses each raw record:
    parsed_iter = (
        tf.io.parse_single_example(raw, features_description)
        for raw in dataset
    )
# Delegate yielding to that iterable:
    yield from parsed_iter


def tf_scenario_streamer_with_keys(features_description, bucket_name='waymo', prefix='tfrecords/'):
    """
    Generator that yields (parsed_example, s3_key) for each scenario,
    preserving the originating TFRecord file key.
    """
    s3 = create_s3_client()
    tfrecord_keys = list_tfrecord_keys(s3, bucket_name, prefix)
    try:
        for key in tfrecord_keys:
            s3_uri = f's3://{bucket_name}/{key}'
            print(f"Processing: {s3_uri}")
            for parsed in stream_and_parse_tfrecord_s3(s3_uri, features_description, s3_client=s3):
                yield parsed, key   # return both parsed scenario and the key
    finally:
        s3.close()

def get_scenario_by_id(features_description, tfrecord_number, scenario_id, bucket_name="waymo"):
    """
    Fetch a single scenario by its scenario_id and tfrecord key.
    """
    s3 = create_s3_client()
    try:
        filename = f'training_tfexample.tfrecord-{tfrecord_number}-of-01000'
        s3_uri = f"s3://{bucket_name}/tfrecords/{filename}"
        for parsed in stream_and_parse_tfrecord_s3(s3_uri, features_description, s3_client=s3):
            # Assuming scenario_id is stored under "scenario/id" or "scenario/scenario_id"
            
            parsed_id = parsed['scenario/id'].numpy().item().decode("utf-8")
            if parsed_id == scenario_id:
                return parsed  # return the parsed scenario dict/tensors
    finally:
        s3.close()
    return None  # not found

def stream_stitched_jsons(bucket_name: str, result_prefix: str):
    """
    Streams one stitched JSON at a time from S3.
    For each file, prints (folder_number, file_prefix) and yields the parsed JSON.
    """
    s3 = create_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=result_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_stitched_data.json"):
                parts = key.split("/")
                folder_number = parts[-3]   # e.g. "0000"
                filename = parts[-1]        # e.g. "1dde730d8ab3ae4a_stitched_data.json"

                match = re.match(r"(.+?)_stitched_data\.json$", filename)
                file_prefix = match.group(1) if match else filename

                # Download and parse
                obj_data = s3.get_object(Bucket=bucket_name, Key=key)
                data = json.loads(obj_data["Body"].read().decode("utf-8"))

                # Print and yield
                #print(f"Folder: {folder_number}, File prefix: {file_prefix}")
                yield folder_number,file_prefix,data

def get_stitched_json_by_id(tf_record_id: str,
                          scenario_id: str,
                          bucket_name: str = "waymo", 
                          result_prefix: str = "results", 
                          time_of_tag_results: str = "2025-09-01-22_42",
                          ):
    """
    Streams one stitched JSON at a time from S3.
    For each file, prints (folder_number, file_prefix) and yields the parsed JSON.
    """
    s3 = create_s3_client()

    key = f'{result_prefix}/{time_of_tag_results}/{tf_record_id}/stitched/{scenario_id}_stitched_data.json'
    obj_data = s3.get_object(Bucket=bucket_name, Key=key)
    data = json.loads(obj_data["Body"].read().decode("utf-8"))
    return data
import subprocess
import struct
import boto3
from smart_open import open
from waymo_open_dataset.protos import scenario_pb2
import os

def create_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('S3_SECRET_KEY'),
        endpoint_url='https://gif.s3.iavgroup.local',
        verify="/mnt/c/Users/I010444/source/repos/googledrivePy/IAV-CA-Bundle.pem"
    )

def list_pb_record_keys(s3, bucket_name, prefix='pb/'):
    keys = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    # Skip the first item if it's just the folder name (ends with '/')
    all_keys = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
    
    for file_key in all_keys:
        keys.append(file_key)
    if not keys:
        raise FileNotFoundError(f"No TFRecord files found in s3://{bucket_name}/{prefix}")
    return keys

# --------- STREAMING LOGIC ---------
def read_tfrecord_stream(file_obj):
    """Yields raw TFRecord bytes from a file-like object (local or S3)."""
    while True:
        length_bytes = file_obj.read(8)
        if len(length_bytes) < 8:
            break
        length = struct.unpack('<Q', length_bytes)[0]
        file_obj.read(4)  # skip length CRC
        data = file_obj.read(length)
        if len(data) < length:
            break
        file_obj.read(4)  # skip data CRC
        yield data

def stream_scenarios_from_tfrecord(s3_uri, transport_params=None):
    """Stream parsed Scenario protos from a single TFRecord file."""
    with open(s3_uri, 'rb', transport_params=transport_params) as f:
        for raw_record in read_tfrecord_stream(f):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_record)
            yield scenario

def pb_scenario_streamer(bucket_name='waymo', prefix='pb/'):
    print("get_scenarios_from_all_files called")
    """
    Generator that yields Scenario protos across all .tfrecord files in the given S3 prefix.
    """
    s3 = create_s3_client()
    transport_params = {'client': s3}
    tfrecord_keys = list_pb_record_keys(s3, bucket_name, prefix)

    for key in tfrecord_keys:
        s3_uri = f's3://{bucket_name}/{key}'
        print(f"Streaming from: {s3_uri}")
        yield from stream_scenarios_from_tfrecord(s3_uri, transport_params=transport_params)
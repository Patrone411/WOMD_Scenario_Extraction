"""Module to access TFRecord files stored in S3."""
import os
import re
import tempfile
from typing import Iterator
from urllib.parse import urlparse
import boto3
import botocore
import tensorflow as tf

# pylint: disable=import-error, no-name-in-module

from waymo_osc_extractor.waymo_scenario_tools.scenario_handling import Scenario, features_description


class S3TFRecordAccessor:
    """Class to access S3 and enumerate TFRecord files. Implements context manager protocol."""

    def __enter__(self):
        return self

    def __init__(self):
        self.client = boto3.client('s3', endpoint_url='https://gif.s3.iavgroup.local',verify="certs/IAV-CA-Bundle.pem" )
        self.paginator = self.client.get_paginator("list_objects_v2")
        self.bucket = None
        self.key = None
        self.file_num = None
        self.local_path = None
        self.dataset = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.clean_up()

    def clean_up(self):
        """
        Removes the temporary local file if it exists.

        Attempts to delete the file specified by `self.local_path`. 
        If an error occurs during deletion, it prints an error message.

        Exceptions:
            OSError: If the file cannot be deleted.
        """
        try:
            if self.local_path:
                os.remove(self.local_path)
        except OSError as e:
            print(f"Error deleting temporary file {self.local_path}: {e}")

    def load_dataset(self, s3_uri):
        """
        Loads a TensorFlow dataset from a specified S3 URI.
        This method downloads the TFRecord file from the given S3 URI to a temporary local path,
        then creates and returns a `tf.data.TFRecordDataset` from the downloaded file.
        Args:
            s3_uri (str): The S3 URI pointing to the TFRecord file.
        Returns:
            tf.data.TFRecordDataset: A TensorFlow dataset created from the downloaded TFRecord file.
        """
        self._parse_s3_uri(s3_uri)
        self._download_to_tmp()

        self.dataset = tf.data.TFRecordDataset(
            self.local_path, compression_type="")
        return self.dataset

    def _download_to_tmp(self):
        try:
            # remove any existing temp file
            self.clean_up()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                self.local_path = tmp.name
                # Use boto3's download_fileobj to stream directly into the temp file
                self.client.download_fileobj(self.bucket, self.key, tmp)

        except botocore.exceptions.ClientError:
            self.clean_up()
            raise

    def _file_num_from_key(self):
        m = re.search(r"-(\d{5})-of-\d{5}$", self.key)
        filenum = m.group(1) if m else "00000"
        return filenum

    def _parse_s3_uri(self, s3_uri):
        """
        Parses an S3 URI and extracts the bucket and key.
        Args:
            s3_uri (str): The S3 URI to parse, expected in the format 's3://bucket/key'.
        Raises:
            ValueError: If the URI scheme is not 's3', or if the bucket or key is missing.
        Returns:
            tuple: A tuple containing the bucket name and key as strings.
        """
        parsed = urlparse(s3_uri)
        # If a scheme is provided it must be "s3"
        if parsed.scheme and parsed.scheme != "s3":
            raise ValueError(f"Invalid URL scheme for S3 URI: {parsed.scheme}")

        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if not bucket or not key:
                raise ValueError(
                    f"Invalid S3 URI (must be s3://bucket/key): {s3_uri}")

            self.bucket = bucket
            self.key = key
            self.file_num = self._file_num_from_key()

        return bucket, key

    def enumerate_tfrecords(self,
                            bucket: str,
                            prefix: str = "",
                            ) -> Iterator[str]:
        """
        Feature description:
        The system is able to access an S3 bucket and enumerate TFRecord files under a given prefix.
        This function yields S3 URIs for objects whose keys end with common TFRecord file extensions.

        Parameters:
        - bucket: name of the S3 bucket to query.
        - prefix: key prefix under which to search for TFRecord files.

        Yields:
        - s3://{bucket}/{key} for each matching object.

        Behavior:
        - Uses boto3 paginator to list objects efficiently.
        - Raises botocore.exceptions.ClientError for AWS/client errors.
        """
        try:
            for page in self.paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if not key:
                        continue
                    yield f"s3://{bucket}/{key}"

        except botocore.exceptions.ClientError as e:
            # Log the error and return an empty iterator
            print(f"Error accessing S3: {e}")
            return iter([])

    def enumerate_scenarios(self) -> Iterator[Scenario]:
        """
        Iterates over all scenarios in the loaded dataset.

        Yields:
            Scenario: An instance of the Scenario class for each parsed example in the dataset.

        Raises:
            AssertionError: If the dataset has not been loaded prior to calling this method.
        """
        assert self.dataset is not None, 'Dataset not loaded. Call load_dataset first.'
        for raw_record in self.dataset:
            example = tf.io.parse_single_example(
                raw_record, features_description)
            scenario = Scenario(example=example)
            yield scenario

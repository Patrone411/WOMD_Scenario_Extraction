"""Module to access TFRecord files stored in S3.

Improved error handling and verification for uploads so failures are logged
with useful context and uploads are verified using a head_object call.
"""
import io
import pickle
from typing import Iterator, Optional
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
import json
from typing import Optional
# pylint: disable=import-error, no-name-in-module

def _format_client_error(e: Exception) -> str:
    if isinstance(e, ClientError):
        return str(e.response.get("Error", {}))
    return str(e)

class S3ScenarioAccessor:
    """Class to access S3 and enumerate/process Scenario files.

    Notes:
    - key_prefix may or may not include a trailing '/'. The code normalizes it.
    - The save() method attempts to verify the object exists after upload.
    """

    def __enter__(self):
        return self

    def __init__(self, bucket: str, key_prefix: Optional[str], record_name: str,
                 endpoint_url: Optional[str] = 'https://gif.s3.iavgroup.local', 
                 verify: Optional[str] = "certs/IAV-CA-Bundle.pem"):
        """Initialize the accessor.

        Args:
            bucket: S3 bucket name.
            key_prefix: Prefix within the bucket where records are stored.
            record_name: A subfolder/name under the prefix to store records.
            endpoint_url: Optional S3 endpoint override (for testing/local S3).
            verify: Optional path to CA bundle or boolean for SSL verification.
        """
        print(f"Initializing S3ScenarioAccessor for bucket:%s, prefix: %s record_name: %s",
                    bucket, key_prefix, record_name)
        
        try:
            client_kwargs = {"service_name": "s3"}
            # boto3.client signature: boto3.client('s3', ...). We build kwargs for flexibility.
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            if verify is not None:
                client_kwargs["verify"] = verify

            # Create the client in a try/except to catch configuration/connection issues early.
            self.client = boto3.client('s3', **{k: v for k, v in client_kwargs.items() if k != "service_name"})
        except Exception as exc:  # broad catch to present a clear error message
            print(f"Failed to create S3 client: %s", exc)
            raise RuntimeError(f"Could not initialize S3 client: {exc}") from exc

        self.bucket = bucket
        # Normalize key_prefix to empty string or ensure trailing slash
        if key_prefix is None:
            self.key_prefix = ""
        else:
            self.key_prefix = key_prefix if key_prefix.endswith("/") else f"{key_prefix}/"
        self.record_name = record_name.strip("/")  # avoid double slashes

    def save(self, scenario_name: str, data: str) -> bool:
        """Save the provided data to S3 and verify that it exists.

        Returns True if upload and verification succeeded, False otherwise.
        """
        key = f"{self.key_prefix}{self.record_name}/{scenario_name}.json"
        try:
            data_short = (data[:200] + "...") if len(data) > 200 else data
            print(f"Saving data to s3://%s/%s with content (truncated): %s", self.bucket, key, data_short)

            # boto3 accepts str or bytes for Body; use bytes to be explicit
            body = data.encode("utf-8") if isinstance(data, str) else data

            response = self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status not in (200, 201):
                print(f"PutObject returned non-success status %s for s3://%s/%s - full response: %s",
                             status, self.bucket, key, response)
                return False

            # Verify the object exists with a HeadObject call
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
                print(f"HeadObject response for s3://%s/%s: %s", self.bucket, key, head)
                return True
            except ClientError as head_err:
                # If the object is not found or another client error occurred, log it.
                print(f"Failed to verify object at s3://%s/%s after upload: %s",
                                 self.bucket, key, _format_client_error(head_err))
                return False

        except (ClientError, EndpointConnectionError, NoCredentialsError) as e:
            print(f"Error saving data to s3://%s/%s - %s", self.bucket, key, _format_client_error(e))
            return False
        except Exception as e:  # catch-all to prevent silent failures
            print(f"Unexpected error while saving data to s3://%s/%s - %s", self.bucket, key, e)
            return False
        
    def save_pck(self, scenario_name: str, obj) -> bool:
        """Save the provided data to S3 and verify that it exists.

        Returns True if upload and verification succeeded, False otherwise.
        """
        key = f"{self.key_prefix}{self.record_name}/{scenario_name}.pkl"
        try:
            print(f"Saving data to s3://%s/%s", self.bucket, key)

            buf = io.BytesIO()
            pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
            buf.seek(0)


            response = self.client.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue(),
              ContentType="application/octet-stream")
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status not in (200, 201):
                print(f"PutObject returned non-success status %s for s3://%s/%s - full response: %s",
                             status, self.bucket, key, response)
                return False

            # Verify the object exists with a HeadObject call
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
                print(f"HeadObject response for s3://%s/%s: %s", self.bucket, key, head)
                return True
            except ClientError as head_err:
                # If the object is not found or another client error occurred, log it.
                print(f"Failed to verify object at s3://%s/%s after upload: %s",
                                 self.bucket, key, _format_client_error(head_err))
                return False

        except (ClientError, EndpointConnectionError, NoCredentialsError) as e:
            print(f"Error saving data to s3://%s/%s - %s", self.bucket, key, _format_client_error(e))
            return False
        except Exception as e:  # catch-all to prevent silent failures
            print(f"Unexpected error while saving data to s3://%s/%s - %s", self.bucket, key, e)
            return False
        
    def save_json(self, scenario_name: str, obj, indent: Optional[int] = None) -> bool:
        """Save provided data as JSON to S3 and verify it exists."""
        
        key = f"{self.key_prefix}{self.record_name}/{scenario_name}.json"
        try:
            print(f"Saving data to s3://%s/%s", self.bucket, key)

            def default(o):
                # numpy support
                try:
                    import numpy as np
                    if isinstance(o, np.integer):
                        return int(o)
                    if isinstance(o, np.floating):
                        return float(o)
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                except Exception:
                    pass

                # dataclasses support
                try:
                    import dataclasses
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                except Exception:
                    pass

                # general fallback
                if hasattr(o, "to_dict") and callable(o.to_dict):
                    return o.to_dict()

                raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

            json_bytes = json.dumps(
                obj,
                ensure_ascii=False,
                indent=indent,
                separators=None if indent else (",", ":"),
                default=default,
            ).encode("utf-8")

            response = self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_bytes,
                ContentType="application/json; charset=utf-8",
            )

            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status not in (200, 201):
                print(
                    f"PutObject returned non-success status %s for s3://%s/%s - full response: %s",
                    status, self.bucket, key, response
                )
                return False

            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
                print(f"HeadObject response for s3://%s/%s: %s", self.bucket, key, head)
                return True
            except ClientError as head_err:
                print(
                    f"Failed to verify object at s3://%s/%s after upload: %s",
                    self.bucket, key, _format_client_error(head_err)
                )
                return False

        except (ClientError, EndpointConnectionError, NoCredentialsError) as e:
            print(f"Error saving data to s3://%s/%s - %s", self.bucket, key, _format_client_error(e))
            return False
        except Exception as e:
            print(f"Unexpected error while saving data to s3://%s/%s - %s", self.bucket, key, e)
            return False

        except (ClientError, EndpointConnectionError, NoCredentialsError) as e:
            print(f"Error saving data to s3://%s/%s - %s", self.bucket, key, _format_client_error(e))
            return False
        except Exception as e:
            print(f"Unexpected error while saving data to s3://%s/%s - %s", self.bucket, key, e)
            return False
    def enumerate_scenarios(self) -> Iterator[str]:
        """Enumerates scenario files stored in the specified S3 bucket and key prefix.

        Yields:
            str: The S3 key of each scenario file found.
        """
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.key_prefix):
                for obj in page.get("Contents", []):
                    yield obj.get("Key")
        except (ClientError, EndpointConnectionError, NoCredentialsError) as e:
            print(f"Error enumerating scenarios in s3://%s/%s - %s",
                             self.bucket, self.key_prefix, _format_client_error(e))
        except Exception as e:
            print(f"Unexpected error enumerating scenarios in s3://%s/%s - %s",
                             self.bucket, self.key_prefix, e)


def _format_client_error(err: Exception) -> str:
    """Return a compact, informative string for botocore client exceptions."""
    if isinstance(err, ClientError):
        try:
            errinfo = err.response.get("Error", {})
            return f"{errinfo.get('Code')}: {errinfo.get('Message')}"
        except Exception:
            return str(err)
    return str(err)

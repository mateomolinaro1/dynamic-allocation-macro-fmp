import io
import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Union
from urllib.parse import urlparse
import polars as pl
import boto3
import pandas as pd
from botocore.client import BaseClient

logger = logging.getLogger(__name__)


DEFAULT_PROFILE = os.getenv("AWS_PROFILE", "team-dev-aia")
DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")


class s3Utils:
    # ---------- helpers ----------
    @staticmethod
    def _parse_s3_uri(uri: str) -> Tuple[str, str]:
        u = urlparse(uri)
        if u.scheme != "s3":
            raise ValueError("uri must start with s3://")
        bucket = u.netloc
        key = u.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {uri}")
        return bucket, key

    @staticmethod
    def get_s3_client(profile: Optional[str] = DEFAULT_PROFILE,
                      region: Optional[str] = DEFAULT_REGION) -> BaseClient:
        """
        Returns a boto3 S3 client using a named AWS profile (recommended).
        """
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            # falls back to env/instance role credentials
            session = boto3.Session(region_name=region)
        return session.client("s3")

    # ---------- parquet: PUSH ----------
    @staticmethod
    def push_object_to_s3_parquet(
        object_to_push: pd.DataFrame,
        path: str,
        profile: Optional[str] = DEFAULT_PROFILE,
        region: Optional[str] = DEFAULT_REGION,
        index: bool = False,
        compression: Optional[str] = "snappy",
    ) -> None:
        """
        Upload a DataFrame to S3 as a parquet file.
        No s3fs/fsspec: uses boto3 + in-memory buffer.
        """
        if not isinstance(object_to_push, pd.DataFrame):
            logger.error("object_to_push must be a pd.DataFrame")
            raise ValueError("object_to_push must be a pd.DataFrame")
        if not isinstance(path, str):
            logger.error("path must be a str")
            raise ValueError("path must be a str")
        if not path.startswith("s3://"):
            raise ValueError("path must be an S3 URI like s3://bucket/key.parquet")

        bucket, key = s3Utils._parse_s3_uri(path)
        s3 = s3Utils.get_s3_client(profile=profile, region=region)

        buffer = io.BytesIO()
        object_to_push.to_parquet(
            buffer,
            engine="pyarrow",
            index=index,
            compression=compression,
        )
        buffer.seek(0)

        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"Uploaded parquet to s3://{bucket}/{key}")

    @staticmethod
    def push_object_to_s3(
            object_to_push,
            path: str,
            file_type: str,
            profile: Optional[str] = DEFAULT_PROFILE,
            region: Optional[str] = DEFAULT_REGION,
            index: bool = True,
            compression: Optional[str] = "snappy",
            pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        """
        Upload an object to S3 as parquet or pickle using boto3 + in-memory buffer.

        Parameters
        ----------
        object_to_push : object
            Object to upload (DataFrame for parquet, any pickleable object for pickle)
        path : str
            S3 path (s3://bucket/key.parquet or s3://bucket/key.pkl)
        file_type : str
            One of {"parquet", "pickle"}
        """
        if not isinstance(path, str):
            raise ValueError("path must be a str")

        if not path.startswith("s3://"):
            raise ValueError("path must be an S3 URI like s3://bucket/key")

        file_type = file_type.lower()
        if file_type not in {"parquet", "pickle", "pkl"}:
            raise ValueError("file_type must be one of {'parquet', 'pickle', 'pkl'}")

        if file_type == "parquet" and not isinstance(object_to_push, pd.DataFrame):
            raise ValueError("For parquet, object_to_push must be a pd.DataFrame")

        bucket, key = s3Utils._parse_s3_uri(path)
        s3 = s3Utils.get_s3_client(profile=profile, region=region)

        buffer = io.BytesIO()

        # ---- Write to buffer ----
        if file_type == "parquet":
            object_to_push.to_parquet(
                buffer,
                engine="pyarrow",
                index=index,
                compression=compression,
            )

        elif file_type == "pickle" or file_type == "pkl":
            pickle.dump(
                object_to_push,
                buffer,
                protocol=pickle_protocol,
            )

        buffer.seek(0)

        # ---- Upload ----
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
        )

        logger.info(f"Uploaded {file_type} to s3://{bucket}/{key}")

    @staticmethod
    def push_objects_to_s3_parquet(
        objects_dct: dict,
        profile: Optional[str] = DEFAULT_PROFILE,
        region: Optional[str] = DEFAULT_REGION,
        index: bool = False,
        compression: Optional[str] = "snappy",
    ) -> None:
        """
        Upload several DataFrames to S3 as parquet files.
        objects_dct: { "s3://bucket/key.parquet": df, ... }
        """
        if not isinstance(objects_dct, dict):
            logger.error("objects_dct must be a dict.")
            raise ValueError("objects_dct must be a dict.")

        for path, obj in objects_dct.items():
            s3Utils.push_object_to_s3_parquet(
                object_to_push=obj,
                path=path,
                profile=profile,
                region=region,
                index=index,
                compression=compression,
            )

    @staticmethod
    def pull_file_from_s3(
            path: str,
            profile: Optional[str] = DEFAULT_PROFILE,
            region: Optional[str] = DEFAULT_REGION,
            to_polars: bool = False,
            file_type: Optional[str] = None,  # "parquet" | "pickle"
    ):
        """
        Download a parquet or pickle file from S3.

        Parameters
        ----------
        path : str
            S3 URI (s3://bucket/key)
        profile : str, optional
            AWS profile
        region : str, optional
            AWS region
        to_polars : bool, default False
            If True and parquet → return polars.DataFrame
        file_type : {"parquet", "pickle"}, optional
            Force file type. If None, inferred from extension.

        Returns
        -------
        object
            pd.DataFrame, pl.DataFrame, or arbitrary Python object (pickle)
        """
        if not isinstance(path, str):
            raise ValueError("path must be a str")
        if not path.startswith("s3://"):
            raise ValueError("path must be an S3 URI like s3://bucket/key")

        # Infer file type if not provided
        if file_type is None:
            if path.endswith(".parquet"):
                file_type = "parquet"
            elif path.endswith((".pkl", ".pickle")):
                file_type = "pickle"
            else:
                raise ValueError("Cannot infer file type from path; specify file_type")

        bucket, key = s3Utils._parse_s3_uri(path)
        s3 = s3Utils.get_s3_client(profile=profile, region=region)

        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())

        if file_type == "parquet":
            if to_polars:
                return pl.read_parquet(buf)
            return pd.read_parquet(buf, engine="pyarrow")

        if file_type == "pickle" or file_type == "pkl":
            return pickle.loads(buf.getvalue())

        raise ValueError(f"Unsupported file_type: {file_type}")

    # ---------- parquet: PULL ----------
    @staticmethod
    def pull_parquet_file_from_s3(
        path: str,
        profile: Optional[str] = DEFAULT_PROFILE,
        region: Optional[str] = DEFAULT_REGION,
        to_polars: bool = False,
    ) -> pd.DataFrame:
        """
        Download a parquet from S3 into a DataFrame.
        No s3fs/fsspec: uses boto3 + in-memory buffer.
        """
        if not isinstance(path, str):
            logger.error("path must be a str.")
            raise ValueError("path must be a str.")
        if not path.startswith("s3://"):
            raise ValueError("path must be an S3 URI like s3://bucket/key.parquet")

        bucket, key = s3Utils._parse_s3_uri(path)
        s3 = s3Utils.get_s3_client(profile=profile, region=region)

        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())

        if to_polars:
            return pl.read_parquet(buf)
        return pd.read_parquet(buf, engine="pyarrow")

    @staticmethod
    def pull_parquet_files_from_s3(
        paths: List[str],
        profile: Optional[str] = DEFAULT_PROFILE,
        region: Optional[str] = DEFAULT_REGION,
    ) -> dict:
        """
        Given a list of S3 parquet paths returns dfs in a dict
        res[name] = df, where name is the filename without extension
        """
        if not isinstance(paths, list):
            logger.error("paths must be a list.")
            raise ValueError("paths must be a list.")
        for p in paths:
            if not isinstance(p, str):
                logger.error("all paths must be strings.")
                raise ValueError("all paths must be strings.")

        res_dct = {}
        for p in paths:
            name = p.split("/")[-1].split(".")[0]
            res_dct[name] = s3Utils.pull_parquet_file_from_s3(
                path=p, profile=profile, region=region
            )
        return res_dct

    # ---------- generic replace (parquet/pkl) ----------
    @staticmethod
    def replace_existing_files_in_s3(
        s3: BaseClient,
        bucket_name: str,
        files_dct: dict
    ) -> None:
        """
        Given a dict with the file keys (as in S3) as keys and the file content as values,
        replace the existing files in S3 with the new content.
        - keys in files_dct are S3 keys (NOT s3:// URIs): e.g. "data/foo.parquet"
        - values are pandas DataFrames for parquet, or python objects for pkl
        """
        if not hasattr(s3, 'put_object'):
            logger.error("s3 must be a boto3 BaseClient instance.")
            raise ValueError("s3 must be a boto3 BaseClient instance.")
        if not isinstance(bucket_name, str):
            logger.error("bucket_name must be a string.")
            raise ValueError("bucket_name must be a string.")
        if not isinstance(files_dct, dict):
            logger.error("files_dct must be a dictionary.")
            raise ValueError("files_dct must be a dictionary with file names as keys and file content as values.")

        # Check that the file keys exist
        for file_key in files_dct.keys():
            try:
                s3.head_object(Bucket=bucket_name, Key=file_key)
            except Exception as e:
                logger.error(f"File {file_key} does not exist in S3 bucket {bucket_name}: {e}")
                raise ValueError(f"File {file_key} does not exist in S3 bucket {bucket_name}.")

        # Upload new versions first
        for file_key, content in files_dct.items():
            ext = file_key.split('.')[-1].lower()

            if ext == "parquet":
                if not isinstance(content, pd.DataFrame):
                    raise ValueError(f"Content for {file_key} must be a pd.DataFrame")
                buffer = io.BytesIO()
                content.to_parquet(buffer, engine="pyarrow", index=True)
                buffer.seek(0)
                body = buffer.getvalue()

            elif ext == "pkl":
                body = pickle.dumps(content)

            else:
                raise ValueError(f"Unsupported extension: {file_key}")

            s3.put_object(Bucket=bucket_name, Key=file_key, Body=body)
            logger.info(f"Uploaded new version of {file_key} to S3 bucket {bucket_name}.")

        # Delete all non-latest versions + delete markers
        for file_key in files_dct.keys():
            paginator = s3.get_paginator("list_object_versions")
            to_delete = []

            for page in paginator.paginate(Bucket=bucket_name, Prefix=file_key):
                for v in page.get("Versions", []):
                    if not v["IsLatest"]:
                        to_delete.append({"Key": file_key, "VersionId": v["VersionId"]})
                for d in page.get("DeleteMarkers", []):
                    if not d["IsLatest"]:
                        to_delete.append({"Key": file_key, "VersionId": d["VersionId"]})

            if to_delete:
                s3.delete_objects(Bucket=bucket_name, Delete={"Objects": to_delete})
            logger.info(f"Deleted previous versions of {file_key} in S3 bucket {bucket_name}.")

    # ---------- legacy env creds (optional / keep if you want) ----------
    @staticmethod
    def get_credentials(return_bool: bool = False) -> Dict[str, str]:
        """
        Legacy: Load credentials from .env.
        Prefer using AWS profiles instead (recommended).
        """
        creds = {}
        for key in ["KEY", "SECRET_KEY", "REGION", "OUTPUT_FORMAT"]:
            val = os.getenv(key)
            if val is not None:
                creds[key] = val

        required = ["KEY", "SECRET_KEY", "REGION", "OUTPUT_FORMAT"]
        missing = [k for k in required if k not in creds]
        if missing:
            logger.error(f"Missing credentials: {missing}")
            raise RuntimeError(f"Missing credentials: {missing}")

        if return_bool:
            return creds
        return creds

    @staticmethod
    def connect_aws_s3(creds) -> BaseClient:
        """
        Legacy: Connect to AWS S3 using access keys from .env.
        Prefer using get_s3_client(profile=...) instead.
        """
        return boto3.client(
            "s3",
            aws_access_key_id=creds["KEY"],
            aws_secret_access_key=creds["SECRET_KEY"],
            region_name=creds["REGION"],
        )

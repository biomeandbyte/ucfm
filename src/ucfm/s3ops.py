"""S3 sync helpers."""
from __future__ import annotations

import os

from botocore.exceptions import ClientError

from ucfm.storage import Store


def require_s3(store: Store) -> None:
    if store.cfg.storage.backend != "s3":
        raise RuntimeError("S3 not configured. Set UCFM_S3_BUCKET in .env.")


def aws_region() -> str:
    return os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"


def check_credentials() -> dict:
    import boto3

    sts = boto3.client("sts", region_name=aws_region())
    ident = sts.get_caller_identity()
    return {"account": ident["Account"], "arn": ident["Arn"], "region": aws_region()}


def _s3_prefix(store: Store, prefix: str) -> str:
    p = prefix.strip("/")
    base = store.cfg.storage.prefix.strip("/")
    if base:
        return f"{base}/{p}" if p else base
    return p


def sync_prefix_to_s3(store: Store, prefix: str) -> int:
    require_s3(store)
    root = store.local_path(prefix)
    if not root.exists():
        raise FileNotFoundError(f"local path missing: {root}")
    s3 = store._s3()
    bucket = store.cfg.storage.bucket
    n = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(store.root).as_posix()
        key = store._s3_key(rel)
        s3.upload_file(str(path), bucket, key)
        n += 1
    return n


def sync_prefix_from_s3(store: Store, prefix: str) -> int:
    require_s3(store)
    s3 = store._s3()
    bucket = store.cfg.storage.bucket
    s3_prefix = _s3_prefix(store, prefix)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key
            if store.cfg.storage.prefix.strip("/"):
                rel = key[len(store.cfg.storage.prefix.strip("/")) + 1 :]
            dest = store.local_path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            n += 1
    return n


def s3_prefix_exists(store: Store, prefix: str) -> bool:
    require_s3(store)
    s3 = store._s3()
    bucket = store.cfg.storage.bucket
    s3_prefix = _s3_prefix(store, prefix)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix.rstrip("/") + "/", MaxKeys=1)
    return resp.get("KeyCount", 0) > 0

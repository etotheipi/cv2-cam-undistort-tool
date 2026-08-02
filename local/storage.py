"""Calibration storage backends for local (lab) mode.

DirStorage  — plain JSON files in a local directory.
S3Storage   — s3://multi-cam-calibration-files-{account_id}/{iam_username}/
              The bucket is derived from the AWS account and the prefix from
              the caller's IAM username (via STS GetCallerIdentity), so the
              credentials alone determine where files live: no separate
              location config, no cross-deployment overlap, and cycling the
              user's access keys changes nothing about the data location.

Credential precedence (S3): process env vars > .env file (never overrides
existing env) > ~/.aws/credentials — i.e. boto3's default chain, with the
.env loaded first without clobbering.
"""

import json
import os
import re
import time
from pathlib import Path

SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def load_env_file(path):
    """Parse KEY=VALUE lines into os.environ WITHOUT overriding existing
    variables (process env always wins)."""
    loaded = []
    p = Path(path).expanduser()
    if not p.is_file():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


class DirStorage:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)

    def status(self):
        return {"type": "dir", "path": str(self.path), "ok": True}

    def list(self):
        out = []
        for f in sorted(self.path.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                out.append({"slug": f.stem, "name": data.get("name"),
                            "modified": f.stat().st_mtime})
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def get(self, slug):
        f = self.path / f"{slug}.json"
        if not f.is_file():
            return None
        return json.loads(f.read_text())

    def put(self, slug, data):
        (self.path / f"{slug}.json").write_text(json.dumps(data, indent=2))
        return {"ok": True, "location": str(self.path / f"{slug}.json")}

    def delete(self, slug):
        f = self.path / f"{slug}.json"
        if not f.is_file():
            return {"error": "not found"}
        f.unlink()
        return {"ok": True}

    def rename(self, old, new):
        src = self.path / f"{old}.json"
        dst = self.path / f"{new}.json"
        if not src.is_file():
            return {"error": "not found"}
        if dst.exists():
            return {"error": "a calibration with that label already exists"}
        src.rename(dst)
        return {"ok": True}


class S3Storage:
    BUCKET_PREFIX = "multi-cam-calibration-files-"

    def __init__(self, env_file=None, region_default="us-east-1"):
        import boto3
        self.env_loaded = load_env_file(env_file) if env_file else []
        os.environ.setdefault("AWS_DEFAULT_REGION", region_default)
        self.session = boto3.session.Session()
        self.error = None
        self.bucket = self.prefix = self.identity_arn = None
        try:
            ident = self.session.client("sts").get_caller_identity()
            self.identity_arn = ident["Arn"]
            account = ident["Account"]
            self.bucket = f"{self.BUCKET_PREFIX}{account}"
            # arn:aws:iam::acct:user/NAME  -> NAME ; other principals use the
            # last ARN path segment (still stable per identity)
            self.prefix = self.identity_arn.split("/")[-1]
            self.s3 = self.session.client("s3")
        except Exception as e:
            self.error = str(e)

    def status(self):
        st = {"type": "s3", "bucket": self.bucket, "prefix": self.prefix,
              "identity": self.identity_arn,
              "region": self.session.region_name,
              "env_file_vars": self.env_loaded,
              "ok": self.error is None, "error": self.error}
        creds = self.session.get_credentials()
        if creds:
            st["access_key_id"] = creds.access_key[:4] + "…" + creds.access_key[-4:]
        if self.error is None:
            try:  # cheap reachability probe
                self.s3.list_objects_v2(Bucket=self.bucket,
                                        Prefix=self.prefix + "/", MaxKeys=1)
            except Exception as e:
                st["ok"] = False
                st["error"] = str(e)
        return st

    def reveal(self):
        """Full credentials for transfer to a target system (localhost UI)."""
        creds = self.session.get_credentials()
        if not creds:
            return {"error": "no credentials resolved"}
        return {"access_key_id": creds.access_key,
                "secret_access_key": creds.secret_key,
                "region": self.session.region_name or "us-east-1",
                "identity": self.identity_arn, "bucket": self.bucket,
                "prefix": self.prefix,
                "note": "put these on the target system as env vars or a .env file"}

    def _key(self, slug):
        return f"{self.prefix}/{slug}.json"

    def list(self):
        out = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket,
                                       Prefix=self.prefix + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                out.append({"slug": Path(obj["Key"]).stem,
                            "modified": obj["LastModified"].timestamp(),
                            "name": None})
        return out

    def get(self, slug):
        import botocore.exceptions
        try:
            r = self.s3.get_object(Bucket=self.bucket, Key=self._key(slug))
            return json.loads(r["Body"].read())
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def put(self, slug, data):
        self.s3.put_object(Bucket=self.bucket, Key=self._key(slug),
                           Body=json.dumps(data, indent=2).encode(),
                           ContentType="application/json")
        return {"ok": True,
                "location": f"s3://{self.bucket}/{self._key(slug)}"}

    def delete(self, slug):
        if self.get(slug) is None:
            return {"error": "not found"}
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(slug))
        return {"ok": True}

    def rename(self, old, new):
        if self.get(old) is None:
            return {"error": "not found"}
        if self.get(new) is not None:
            return {"error": "a calibration with that label already exists"}
        self.s3.copy_object(Bucket=self.bucket, Key=self._key(new),
                            CopySource={"Bucket": self.bucket,
                                        "Key": self._key(old)})
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(old))
        return {"ok": True}


def make_storage(config):
    """config: {"type": "dir"|"s3", "dir_path": ..., "env_file": ...}"""
    if config.get("type") == "s3":
        return S3Storage(env_file=config.get("env_file"))
    return DirStorage(config.get("dir_path") or "camera_cal")


def valid_slug(slug):
    return bool(SLUG_RE.match(slug or ""))

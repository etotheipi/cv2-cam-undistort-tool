"""One-time AWS provisioning for calibration storage.

Creates (idempotently), using YOUR admin credentials from the default chain:
  1. Bucket  multi-cam-calibration-files-{account_id}   (public access blocked)
  2. Managed policy  multi-cam-calibration-rw  scoped by IAM policy variable
     to  s3://<bucket>/${aws:username}/*  — one policy serves every user.
  3. An IAM user (one per deployment; its USERNAME becomes its S3 prefix),
     with that policy attached and a permanent access key.

The printed key + secret go onto the deployment's systems (bench and
target). Cycling keys later keeps the username, prefix and files intact.

Usage: python local/provision_aws.py <deployment-user-name> [--env-file PATH]
       (user name: letters/digits/._- e.g. camcal-01)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import boto3
import botocore.exceptions

POLICY_NAME = "multi-cam-calibration-rw"
BUCKET_PREFIX = "multi-cam-calibration-files-"


def policy_document(bucket):
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListOwnPrefix",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {
                    "s3:prefix": ["${aws:username}/*", "${aws:username}"]}},
            },
            {
                "Sid": "RwOwnPrefix",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": f"arn:aws:s3:::{bucket}/${{aws:username}}/*",
            },
        ],
    }


def ensure_bucket(s3, bucket, region):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket exists:   s3://{bucket}")
    except botocore.exceptions.ClientError:
        kwargs = {"Bucket": bucket}
        if region and region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        print(f"bucket created:  s3://{bucket}")
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})


def ensure_policy(iam, account, bucket):
    arn = f"arn:aws:iam::{account}:policy/{POLICY_NAME}"
    try:
        iam.get_policy(PolicyArn=arn)
        print(f"policy exists:   {POLICY_NAME}")
    except iam.exceptions.NoSuchEntityException:
        iam.create_policy(PolicyName=POLICY_NAME,
                          PolicyDocument=json.dumps(policy_document(bucket)),
                          Description="Per-user prefix R/W for multi-cam "
                                      "calibration files")
        print(f"policy created:  {POLICY_NAME}")
    return arn


def ensure_user(iam, name, policy_arn):
    try:
        iam.get_user(UserName=name)
        print(f"user exists:     {name}")
    except iam.exceptions.NoSuchEntityException:
        iam.create_user(UserName=name,
                        Tags=[{"Key": "purpose",
                               "Value": "multi-cam-calibration"}])
        print(f"user created:    {name}")
    iam.attach_user_policy(UserName=name, PolicyArn=policy_arn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("user", help="deployment user name (becomes the S3 prefix)")
    ap.add_argument("--env-file", default=None,
                    help="also write the credentials to this .env file")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]{2,40}", args.user):
        sys.exit("user name must be 2-40 chars of letters/digits/._-")

    session = boto3.session.Session()
    region = session.region_name or "us-east-1"
    ident = session.client("sts").get_caller_identity()
    account = ident["Account"]
    bucket = f"{BUCKET_PREFIX}{account}"
    print(f"account:         {account} (as {ident['Arn']})")

    s3 = session.client("s3", region_name=region)
    iam = session.client("iam")
    ensure_bucket(s3, bucket, region)
    policy_arn = ensure_policy(iam, account, bucket)
    ensure_user(iam, args.user, policy_arn)

    keys = iam.list_access_keys(UserName=args.user)["AccessKeyMetadata"]
    if len(keys) >= 2:
        sys.exit(f"user {args.user} already has 2 access keys — delete one "
                 f"first (aws iam delete-access-key --user-name {args.user} "
                 f"--access-key-id ...)")
    key = iam.create_access_key(UserName=args.user)["AccessKey"]

    print("\n=== credentials for this deployment (store them safely) ===")
    print(f"AWS_ACCESS_KEY_ID={key['AccessKeyId']}")
    print(f"AWS_SECRET_ACCESS_KEY={key['SecretAccessKey']}")
    print(f"AWS_DEFAULT_REGION={region}")
    print(f"# files:  s3://{bucket}/{args.user}/*.json")
    if args.env_file:
        p = Path(args.env_file).expanduser()
        p.write_text(f"AWS_ACCESS_KEY_ID={key['AccessKeyId']}\n"
                     f"AWS_SECRET_ACCESS_KEY={key['SecretAccessKey']}\n"
                     f"AWS_DEFAULT_REGION={region}\n")
        p.chmod(0o600)
        print(f"written to {p} (mode 600)")


if __name__ == "__main__":
    main()

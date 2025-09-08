import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# ---------------- Configuration ----------------
LOCAL_DIST_FOLDER = r"A:\Coding\mtgenesis.ai\dist\mtgenesis.ai"  # Must point directly to dist
S3_BUCKET_NAME = "mtgenesis.ai"                     # Your bucket
S3_PREFIX = ""                                      # Keep empty for root
REGION_NAME = "us-east-1"                           # Your bucket region
CLOUDFRONT_DIST_ID = "EJMB7DAOJAMZY"              # Optional CloudFront distro ID
DEFAULT_ROOT_OBJECT = "index.html"                 # Default root file
# ------------------------------------------------

s3_client = boto3.client("s3", region_name=REGION_NAME)
cf_client = boto3.client("cloudfront")


def clear_bucket(bucket, prefix=""):
    """Delete all objects in the bucket (optionally under a prefix)."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        to_delete = []
        for page in pages:
            for obj in page.get('Contents', []):
                to_delete.append({'Key': obj['Key']})
        if to_delete:
            print(f"Deleting {len(to_delete)} objects from bucket...")
            s3_client.delete_objects(Bucket=bucket, Delete={'Objects': to_delete})
            print("Bucket cleared.")
    except ClientError as e:
        print(f"Error clearing bucket: {e}")


def upload_file(file_path, bucket, s3_key):
    """Upload a single file with public-read ACL."""
    try:
        s3_client.upload_file(
            Filename=file_path,
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={'ACL': 'public-read'}
        )
        print(f"Uploaded: {s3_key}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except ClientError as e:
        print(f"Error uploading {file_path}: {e}")


def upload_folder(local_folder, bucket, prefix=""):
    """
    Upload folder contents to S3.
    Files directly under local_folder go to bucket root.
    Subfolders (like assets/) preserve their structure.
    """
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
            upload_file(local_file_path, bucket, s3_key)


def set_bucket_index(bucket, index_file=DEFAULT_ROOT_OBJECT):
    """Configure S3 to serve a default root object."""
    try:
        s3_client.put_bucket_website(
            Bucket=bucket,
            WebsiteConfiguration={
                'IndexDocument': {'Suffix': index_file},
                'ErrorDocument': {'Key': index_file}
            }
        )
        print(f"Bucket {bucket} configured to serve {index_file} as default root.")
    except ClientError as e:
        print(f"Error setting bucket website config: {e}")


def invalidate_cloudfront(distribution_id):
    """Invalidate all objects in the CloudFront distribution."""
    try:
        response = cf_client.create_invalidation(
            DistributionId=distribution_id,
            InvalidationBatch={
                'Paths': {'Quantity': 1, 'Items': ['/*']},
                'CallerReference': str(os.urandom(16).hex())
            }
        )
        print(f"CloudFront invalidation created: {response['Invalidation']['Id']}")
    except ClientError as e:
        print(f"Error invalidating CloudFront: {e}")


if __name__ == "__main__":
    # 1. Clear bucket
    clear_bucket(S3_BUCKET_NAME, S3_PREFIX)
    
    # 2. Upload files
    upload_folder(LOCAL_DIST_FOLDER, S3_BUCKET_NAME, S3_PREFIX)
    
    # 3. Set default index.html
    set_bucket_index(S3_BUCKET_NAME)
    
    # 4. Optional: invalidate CloudFront
    if CLOUDFRONT_DIST_ID:
        invalidate_cloudfront(CLOUDFRONT_DIST_ID)

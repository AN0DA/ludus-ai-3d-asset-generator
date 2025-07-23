"""
Factory for creating storage instances.
"""

from src.storage.cloud_storage import StorageConfig
from src.storage.s3_storage import S3Storage
from src.utils.env_config import AppSettings


def create_storage(settings: AppSettings) -> S3Storage | None:
    """Create storage based on configuration."""
    config_dict = settings.get_storage_config()

    # Check if all required fields are present and not empty/whitespace
    access_key = config_dict["access_key_id"]
    secret_key = config_dict["secret_access_key"]
    bucket_name = config_dict["bucket_name"]

    if access_key and access_key.strip() and secret_key and secret_key.strip() and bucket_name and bucket_name.strip():
        config = StorageConfig(**config_dict)
        return S3Storage(config)
    return None

"""
Factory for creating storage instances.
"""

from src.storage.cloud_storage import StorageConfig
from src.storage.s3_storage import S3Storage
from src.utils.env_config import AppSettings


def create_storage(settings: AppSettings) -> S3Storage | None:
    """Create storage based on configuration."""
    config_dict = settings.get_storage_config()
    if config_dict["access_key_id"] and config_dict["secret_access_key"]:
        config = StorageConfig(**config_dict)
        return S3Storage(config)
    return None

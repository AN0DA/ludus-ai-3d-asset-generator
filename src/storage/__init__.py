"""
Universal S3-compatible cloud storage module for 3D asset management.

This module provides a comprehensive cloud storage abstraction layer
with universal S3-compatible implementation that works with any S3-compatible
storage service including AWS S3, MinIO, CloudFlare R2, DigitalOcean Spaces,
Wasabi, Backblaze B2, and others.
"""

from typing import Optional

from .cloud_storage import (
    CloudStorage,
    FileInfo,
    FileNotFoundError,
    FileType,
    NetworkError,
    PermissionError,
    QuotaExceededError,
    StorageConfig,
    StorageError,
    StoragePermission,
    StorageProvider,
    UploadProgress,
    ValidationError,
)
from .file_utils import FileUtils, file_utils
from .s3_storage import S3Storage


def create_storage(
    bucket_name: str,
    access_key_id: str,
    secret_access_key: str,
    region: str = "us-east-1",
    endpoint_url: Optional[str] = None,
    **kwargs
) -> S3Storage:
    """
    Create an S3-compatible storage instance with simplified configuration.
    
    This factory function makes it easy to create storage instances for any
    S3-compatible service by just providing the essential parameters.
    
    Args:
        bucket_name: Name of the storage bucket
        access_key_id: Access key ID for authentication
        secret_access_key: Secret access key for authentication
        region: Storage region (default: us-east-1)
        endpoint_url: Custom endpoint URL for non-AWS services (optional)
        **kwargs: Additional configuration options
    
    Returns:
        Configured S3Storage instance
    
    Examples:
        # AWS S3
        storage = create_storage(
            bucket_name="my-bucket",
            access_key_id="AKIA...",
            secret_access_key="...",
            region="us-west-2"
        )
        
        # MinIO
        storage = create_storage(
            bucket_name="my-bucket",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            endpoint_url="http://localhost:9000"
        )
        
        # CloudFlare R2
        storage = create_storage(
            bucket_name="my-bucket",
            access_key_id="...",
            secret_access_key="...",
            endpoint_url="https://account-id.r2.cloudflarestorage.com"
        )
    """
    config = StorageConfig(
        provider=StorageProvider.S3_COMPATIBLE,
        bucket_name=bucket_name,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=region,
        endpoint_url=endpoint_url,
        **kwargs
    )
    
    return S3Storage(config)


__all__ = [
    # Abstract interfaces and base classes
    "CloudStorage",
    "StorageConfig",
    
    # Concrete implementations
    "S3Storage",
    
    # Factory functions
    "create_storage",
    
    # Data models and enums
    "FileInfo",
    "UploadProgress",
    "StorageProvider",
    "FileType",
    "StoragePermission",
    
    # Exceptions
    "StorageError",
    "FileNotFoundError",
    "PermissionError",
    "QuotaExceededError",
    "NetworkError",
    "ValidationError",
    
    # Utilities
    "FileUtils",
    "file_utils",
]
"""
Universal S3-compatible cloud storage module for 3D asset management.

This module provides a comprehensive cloud storage abstraction layer
with universal S3-compatible implementation that works with any S3-compatible
storage service including AWS S3, MinIO, CloudFlare R2, DigitalOcean Spaces,
Wasabi, Backblaze B2, and others.
"""

from .cloud_storage import (
    CloudStorage,
    FileInfo,
    FileType,
    NetworkError,
    QuotaExceededError,
    StorageConfig,
    StorageError,
    StorageFileNotFoundError,
    StoragePermission,
    StoragePermissionError,
    StorageProvider,
    UploadProgress,
    ValidationError,
)
from .file_utils import FileUtils, file_utils
from .s3_storage import S3Storage


def create_storage(config: StorageConfig) -> S3Storage:
    """Create an S3-compatible storage instance."""
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
    "StorageFileNotFoundError",
    "StoragePermissionError",
    "QuotaExceededError",
    "NetworkError",
    "ValidationError",
    # Utilities
    "FileUtils",
    "file_utils",
]

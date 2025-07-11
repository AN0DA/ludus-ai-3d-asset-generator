"""
Abstract cloud storage interface for 3D asset management.

This module defines the abstract base class and protocols for cloud storage
implementations, providing a consistent interface for file operations
regardless of the underlying storage provider.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StorageProvider(str, Enum):
    """S3-compatible storage providers."""

    S3_COMPATIBLE = "s3_compatible"  # Universal S3-compatible storage


class FileType(str, Enum):
    """Supported file types for 3D assets."""

    # 3D Model formats
    OBJ = "obj"
    GLTF = "gltf"
    GLB = "glb"
    FBX = "fbx"
    STL = "stl"
    PLY = "ply"
    DAE = "dae"

    # Image formats
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEBP = "webp"

    # Data formats
    JSON = "json"
    XML = "xml"
    YAML = "yaml"

    # Archive formats
    ZIP = "zip"
    TAR = "tar"

    # Other
    UNKNOWN = "unknown"


class StoragePermission(str, Enum):
    """File access permissions."""

    PRIVATE = "private"
    PUBLIC_READ = "public-read"
    PUBLIC_READ_WRITE = "public-read-write"
    AUTHENTICATED_READ = "authenticated-read"


@dataclass
class UploadProgress:
    """Progress information for file uploads."""

    bytes_uploaded: int
    total_bytes: int
    percentage: float
    speed_mbps: float | None = None
    estimated_remaining: timedelta | None = None

    @property
    def is_complete(self) -> bool:
        """Check if upload is complete."""
        return self.bytes_uploaded >= self.total_bytes


@dataclass
class FileInfo:
    """Information about a stored file."""

    key: str
    size: int
    content_type: str
    last_modified: datetime
    etag: str
    metadata: dict[str, str]
    public_url: str | None = None
    storage_class: str | None = None


class StorageConfig(BaseModel):
    """Configuration for S3-compatible cloud storage."""

    provider: StorageProvider = StorageProvider.S3_COMPATIBLE
    bucket_name: str
    region: str | None = None
    endpoint_url: str | None = None  # Required for non-AWS S3 services
    access_key_id: str | None = None
    secret_access_key: str | None = None

    # Upload settings
    multipart_threshold: int = Field(default=64 * 1024 * 1024, ge=5 * 1024 * 1024)  # 64MB
    max_concurrency: int = Field(default=10, ge=1, le=50)
    chunk_size: int = Field(default=8 * 1024 * 1024, ge=1024 * 1024)  # 8MB

    # Security settings
    use_ssl: bool = True
    verify_ssl: bool = True
    default_permission: StoragePermission = StoragePermission.PRIVATE

    # Lifecycle settings
    enable_versioning: bool = False
    lifecycle_rules: dict[str, Any] = Field(default_factory=dict)

    # CDN settings
    cdn_domain: str | None = None

    class Config:
        use_enum_values = True


class StorageError(Exception):
    """Base exception for storage operations."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class FileNotFoundError(StorageError):
    """File not found in storage."""

    pass


class PermissionError(StorageError):
    """Permission denied for storage operation."""

    pass


class QuotaExceededError(StorageError):
    """Storage quota exceeded."""

    pass


class NetworkError(StorageError):
    """Network-related storage error."""

    pass


class ValidationError(StorageError):
    """File validation error."""

    pass


class CloudStorage(ABC):
    """
    Abstract base class for cloud storage implementations.

    This interface provides a consistent API for file operations
    across different cloud storage providers.
    """

    def __init__(self, config: StorageConfig):
        """Initialize the storage client with configuration."""
        self.config = config
        self._client = None

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the storage service."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the storage service."""
        pass

    @abstractmethod
    async def upload_file(
        self,
        file_path: str | Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        permission: StoragePermission | None = None,
        progress_callback: Callable | None = None,
    ) -> FileInfo:
        """
        Upload a file to cloud storage.

        Args:
            file_path: Local path to the file to upload
            key: Storage key (path) for the file
            content_type: MIME type of the file
            metadata: Additional metadata to store with the file
            permission: Access permission for the file
            progress_callback: Callback function for upload progress

        Returns:
            FileInfo object with upload details
        """
        pass

    @abstractmethod
    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        permission: StoragePermission | None = None,
    ) -> FileInfo:
        """
        Upload bytes data to cloud storage.

        Args:
            data: Bytes data to upload
            key: Storage key (path) for the file
            content_type: MIME type of the data
            metadata: Additional metadata to store with the file
            permission: Access permission for the file

        Returns:
            FileInfo object with upload details
        """
        pass

    @abstractmethod
    async def download_file(
        self,
        key: str,
        file_path: str | Path,
        progress_callback: Callable | None = None,
    ) -> None:
        """
        Download a file from cloud storage.

        Args:
            key: Storage key of the file to download
            file_path: Local path where to save the file
            progress_callback: Callback function for download progress
        """
        pass

    @abstractmethod
    async def download_bytes(self, key: str) -> bytes:
        """
        Download file content as bytes.

        Args:
            key: Storage key of the file to download

        Returns:
            File content as bytes
        """
        pass

    @abstractmethod
    async def get_file_info(self, key: str) -> FileInfo:
        """
        Get information about a stored file.

        Args:
            key: Storage key of the file

        Returns:
            FileInfo object with file details
        """
        pass

    @abstractmethod
    async def list_files(
        self,
        prefix: str = "",
        limit: int | None = None,
    ) -> list[FileInfo]:
        """
        List files in storage with optional prefix filter.

        Args:
            prefix: Key prefix to filter files
            limit: Maximum number of files to return

        Returns:
            List of FileInfo objects
        """
        pass

    @abstractmethod
    async def delete_file(self, key: str) -> None:
        """
        Delete a file from storage.

        Args:
            key: Storage key of the file to delete
        """
        pass

    @abstractmethod
    async def delete_files(self, keys: list[str]) -> dict[str, bool]:
        """
        Delete multiple files from storage.

        Args:
            keys: List of storage keys to delete

        Returns:
            Dictionary mapping keys to deletion success status
        """
        pass

    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            key: Storage key to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    async def generate_presigned_url(
        self,
        key: str,
        expiration: timedelta = timedelta(hours=1),
        method: str = "GET",
    ) -> str:
        """
        Generate a pre-signed URL for secure file access.

        Args:
            key: Storage key of the file
            expiration: URL expiration time
            method: HTTP method (GET, PUT, POST, etc.)

        Returns:
            Pre-signed URL string
        """
        pass

    @abstractmethod
    async def generate_public_url(self, key: str) -> str | None:
        """
        Generate a public URL for file access.

        Args:
            key: Storage key of the file

        Returns:
            Public URL string if file is publicly accessible, None otherwise
        """
        pass

    @abstractmethod
    async def copy_file(
        self,
        source_key: str,
        destination_key: str,
        metadata: dict[str, str] | None = None,
    ) -> FileInfo:
        """
        Copy a file within the storage.

        Args:
            source_key: Source file key
            destination_key: Destination file key
            metadata: Optional new metadata for the copied file

        Returns:
            FileInfo object for the copied file
        """
        pass

    @abstractmethod
    async def move_file(
        self,
        source_key: str,
        destination_key: str,
        metadata: dict[str, str] | None = None,
    ) -> FileInfo:
        """
        Move a file within the storage.

        Args:
            source_key: Source file key
            destination_key: Destination file key
            metadata: Optional new metadata for the moved file

        Returns:
            FileInfo object for the moved file
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the storage service is accessible.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to list files with limit 1 to test connectivity
            await self.list_files(limit=1)
            return True
        except Exception:
            return False

    async def get_storage_usage(self) -> dict[str, Any]:
        """
        Get storage usage statistics.

        Returns:
            Dictionary with usage information
        """
        try:
            files = await self.list_files()
            total_size = sum(f.size for f in files)
            file_count = len(files)

            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "average_file_size": total_size / file_count if file_count > 0 else 0,
            }
        except Exception:
            return {"error": "Unable to retrieve storage usage"}


# Export all public classes and types
__all__ = [
    "CloudStorage",
    "StorageConfig",
    "StorageProvider",
    "FileType",
    "StoragePermission",
    "UploadProgress",
    "FileInfo",
    "StorageError",
    "FileNotFoundError",
    "PermissionError",
    "QuotaExceededError",
    "NetworkError",
    "ValidationError",
]

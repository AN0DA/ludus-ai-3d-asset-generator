"""
Universal S3-compatible cloud storage implementation.

This module provides a concrete implementation of the CloudStorage interface
for any S3-compatible storage service including AWS S3, MinIO, CloudFlare R2,
DigitalOcean Spaces, Wasabi, Backblaze B2, Oracle Cloud Infrastructure (OCI)
Object Storage, and others.

All modern object storage services support the S3 API, making this a universal
solution for cloud storage needs.

Note: This implementation includes specific compatibility fixes for Oracle Cloud
Infrastructure (OCI) Object Storage, which requires explicit Content-Length
headers in upload operations.

mypy: disable-error-code=union-attr,return,assignment
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, NoReturn

import aiofiles
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

from .cloud_storage import (
    CloudStorage,
    FileInfo,
    NetworkError,
    QuotaExceededError,
    StorageConfig,
    StorageError,
    StoragePermission,
    StoragePermissionError,
    UploadProgress,
    ValidationError,
)
from .file_utils import FileUtils


class S3StorageError(StorageError):
    """S3-specific storage error."""


logger = logging.getLogger(__name__)


class S3Storage(CloudStorage):
    """
    Universal S3-compatible storage implementation.

    Supports any S3-compatible storage service including:
    - AWS S3
    - MinIO
    - CloudFlare R2
    - DigitalOcean Spaces
    - Wasabi
    - Backblaze B2
    - And any other S3-compatible service

    This implementation uses the boto3 library which works with any
    service that implements the S3 API standard.
    """

    def __init__(self, config: StorageConfig):
        """Initialize S3 storage with configuration."""
        super().__init__(config)
        self._s3_client: Any | None = None
        self._s3_resource: Any | None = None
        self._bucket: Any | None = None
        self._session: Any | None = None
        self.file_utils = FileUtils()

    def _ensure_connected(self) -> None:
        """Ensure S3 client is connected."""
        if self._s3_client is None or self._s3_resource is None or self._bucket is None:
            raise S3StorageError("S3 client not connected. Call connect() first.")

    def _get_client(self) -> Any:
        """Get S3 client, ensuring it's connected."""
        self._ensure_connected()
        assert self._s3_client is not None  # nosec
        return self._s3_client

    def _get_resource(self) -> Any:
        """Get S3 resource, ensuring it's connected."""
        self._ensure_connected()
        assert self._s3_resource is not None  # nosec
        return self._s3_resource

    def _get_bucket(self) -> Any:
        """Get S3 bucket, ensuring it's connected."""
        self._ensure_connected()
        assert self._bucket is not None  # nosec
        return self._bucket

    async def connect(self) -> None:
        """Initialize connection to S3-compatible storage."""
        try:
            # Create boto3 session with configuration
            self._session = boto3.Session(
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region,
            )

            # Configure boto3 client settings
            boto_config = Config(
                max_pool_connections=self.config.max_concurrency, retries={"max_attempts": 3, "mode": "adaptive"}
            )

            # Create S3 client
            client_kwargs = {
                "config": boto_config,
                "region_name": self.config.region,
                "use_ssl": self.config.use_ssl,
            }

            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            self._s3_client = self._session.client("s3", **client_kwargs)
            self._s3_resource = self._session.resource("s3", **client_kwargs)

            # Ensure these are not None after creation
            assert self._s3_client is not None  # nosec
            assert self._s3_resource is not None  # nosec

            self._bucket = self._s3_resource.Bucket(self.config.bucket_name)

            # Test connection
            client = self._get_client()
            await self._run_sync(client.head_bucket, Bucket=self.config.bucket_name)

            logger.info(f"Connected to S3 storage: {self.config.bucket_name}")

        except NoCredentialsError as e:
            raise StoragePermissionError(
                "AWS credentials not found", error_code="NO_CREDENTIALS", details={"error": str(e)}
            ) from e
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "UNKNOWN")
            status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            if error_code == "NoSuchBucket":
                raise StorageError(
                    f"Bucket '{self.config.bucket_name}' does not exist", error_code=error_code, status_code=status_code
                ) from e
            elif error_code in ["AccessDenied", "Forbidden"]:
                raise StoragePermissionError(
                    f"Access denied to bucket '{self.config.bucket_name}'",
                    error_code=error_code,
                    status_code=status_code,
                ) from e
            else:
                raise NetworkError(
                    f"Failed to connect to S3 storage: {e}", error_code=error_code, status_code=status_code
                ) from e
        except Exception as e:
            raise StorageError(f"Unexpected error connecting to S3: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to S3 storage."""
        if self._s3_client:
            await self._run_sync(self._get_client().close)
            self._s3_client = None
            self._s3_resource = None
            self._bucket = None
            self._session = None
            logger.info("Disconnected from S3 storage")

    async def upload_file(
        self,
        file_path: str | Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        permission: StoragePermission | None = None,
        progress_callback: Callable | None = None,
    ) -> FileInfo:
        """Upload a file to S3 storage."""
        file_path = Path(file_path)

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self.file_utils.get_content_type(file_path)

        # Validate file type
        file_type = self.file_utils.get_file_type(file_path)
        if not self.file_utils.is_file_type_allowed(file_type):
            raise ValidationError(f"File type not allowed: {file_type}")

        # Get file size for progress tracking
        file_size = file_path.stat().st_size

        try:
            # Prepare upload parameters
            upload_args = {
                "ContentType": content_type,
                "ContentLength": file_size,  # Required for OCI Object Storage
                "ACL": (permission or self.config.default_permission).value,
            }

            if metadata:
                upload_args["Metadata"] = metadata

            # Use multipart upload for large files
            if file_size >= self.config.multipart_threshold:
                return await self._multipart_upload(file_path, key, upload_args, progress_callback)
            else:
                return await self._simple_upload(file_path, key, upload_args, progress_callback)

        except ClientError as e:
            await self._handle_client_error(e, f"upload file {key}")
            # This will raise an exception, so this line is unreachable but satisfies mypy
            raise  # pragma: no cover
        except Exception as e:
            raise StorageError(f"Failed to upload file {key}: {e}") from e

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        permission: StoragePermission | None = None,
    ) -> FileInfo:
        """Upload bytes data to S3 storage."""
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            upload_args = {
                "Body": data,
                "ContentType": content_type,
                "ContentLength": len(data),  # Required for OCI Object Storage
                "ACL": (permission or self.config.default_permission).value,
            }

            if metadata:
                upload_args["Metadata"] = metadata

            # Upload the data
            await self._run_sync(self._get_client().put_object, Bucket=self.config.bucket_name, Key=key, **upload_args)

            # Get file info for the uploaded data
            return await self.get_file_info(key)

        except ClientError as e:
            await self._handle_client_error(e, f"upload bytes to {key}")
        except Exception as e:
            raise StorageError(f"Failed to upload bytes to {key}: {e}") from e

    async def download_file(
        self,
        key: str,
        file_path: str | Path,
        progress_callback: Callable | None = None,
    ) -> None:
        """Download a file from S3 storage."""
        file_path = Path(file_path)

        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get file info for progress tracking
            file_info = await self.get_file_info(key)
            total_size = file_info.size
            bytes_downloaded = 0

            # Download the file
            async with aiofiles.open(file_path, "wb") as f:
                response = await self._run_sync(self._get_client().get_object, Bucket=self.config.bucket_name, Key=key)

                # Stream the content in chunks
                chunk_size = self.config.chunk_size
                body = response["Body"]

                while True:
                    chunk = await self._run_sync(body.read, chunk_size)
                    if not chunk:
                        break

                    await f.write(chunk)
                    bytes_downloaded += len(chunk)

                    if progress_callback:
                        progress = UploadProgress(
                            bytes_uploaded=bytes_downloaded,
                            total_bytes=total_size,
                            percentage=(bytes_downloaded / total_size) * 100,
                        )
                        await self._run_sync(progress_callback, progress)

                body.close()

        except ClientError as e:
            await self._handle_client_error(e, f"download file {key}")
        except Exception as e:
            raise StorageError(f"Failed to download file {key}: {e}") from e

    async def download_bytes(self, key: str) -> bytes:
        """Download file content as bytes."""
        try:
            response = await self._run_sync(self._get_client().get_object, Bucket=self.config.bucket_name, Key=key)

            content = await self._run_sync(response["Body"].read)
            response["Body"].close()

            return content

        except ClientError as e:
            await self._handle_client_error(e, f"download bytes from {key}")
        except Exception as e:
            raise StorageError(f"Failed to download bytes from {key}: {e}") from e

    async def get_file_info(self, key: str) -> FileInfo:
        """Get information about a stored file."""
        try:
            response = await self._run_sync(self._get_client().head_object, Bucket=self.config.bucket_name, Key=key)

            # Generate public URL if file is publicly accessible
            public_url = None
            with contextlib.suppress(Exception):
                public_url = await self.generate_public_url(key)

            return FileInfo(
                key=key,
                size=response["ContentLength"],
                content_type=response.get("ContentType", "application/octet-stream"),
                last_modified=response["LastModified"],
                etag=response["ETag"].strip('"'),
                metadata=response.get("Metadata", {}),
                public_url=public_url,
                storage_class=response.get("StorageClass"),
            )

        except ClientError as e:
            await self._handle_client_error(e, f"get info for file {key}")
        except Exception as e:
            raise StorageError(f"Failed to get file info for {key}: {e}") from e

    async def list_files(
        self,
        prefix: str = "",
        limit: int | None = None,
    ) -> list[FileInfo]:
        """List files in S3 storage with optional prefix filter."""
        try:
            paginator = self._get_client().get_paginator("list_objects_v2")

            page_iterator = paginator.paginate(
                Bucket=self.config.bucket_name, Prefix=prefix, PaginationConfig={"MaxItems": limit} if limit else {}
            )

            files = []
            async for page in self._async_paginate(page_iterator):
                for obj in page.get("Contents", []):
                    # Get detailed info for each file
                    try:
                        file_info = await self.get_file_info(obj["Key"])
                        files.append(file_info)
                    except Exception as e:
                        logger.warning(f"Failed to get info for {obj['Key']}: {e}")
                        # Create basic file info from list response
                        files.append(
                            FileInfo(
                                key=obj["Key"],
                                size=obj["Size"],
                                content_type="application/octet-stream",
                                last_modified=obj["LastModified"],
                                etag=obj["ETag"].strip('"'),
                                metadata={},
                            )
                        )

            return files

        except ClientError as e:
            await self._handle_client_error(e, "list files")
        except Exception as e:
            raise StorageError(f"Failed to list files: {e}") from e

    async def delete_file(self, key: str) -> None:
        """Delete a file from S3 storage."""
        try:
            await self._run_sync(self._get_client().delete_object, Bucket=self.config.bucket_name, Key=key)

        except ClientError as e:
            await self._handle_client_error(e, f"delete file {key}")
        except Exception as e:
            raise StorageError(f"Failed to delete file {key}: {e}") from e

    async def delete_files(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple files from S3 storage."""
        if not keys:
            return {}

        try:
            # Prepare delete request
            delete_objects = {"Objects": [{"Key": key} for key in keys]}

            response = await self._run_sync(
                self._get_client().delete_objects, Bucket=self.config.bucket_name, Delete=delete_objects
            )

            # Process results
            results = {}

            # Mark successful deletions
            for deleted in response.get("Deleted", []):
                results[deleted["Key"]] = True

            # Mark failed deletions
            for error in response.get("Errors", []):
                results[error["Key"]] = False
                logger.error(f"Failed to delete {error['Key']}: {error['Message']}")

            # Mark keys not in response as failed
            for key in keys:
                if key not in results:
                    results[key] = False

            return results

        except ClientError as e:
            await self._handle_client_error(e, "delete multiple files")
        except Exception as e:
            raise StorageError(f"Failed to delete files: {e}") from e

    async def file_exists(self, key: str) -> bool:
        """Check if a file exists in S3 storage."""
        try:
            await self._run_sync(self._get_client().head_object, Bucket=self.config.bucket_name, Key=key)
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "UNKNOWN")
            if error_code in ["NoSuchKey", "404"]:
                return False
            await self._handle_client_error(e, f"check existence of file {key}")
        except Exception as e:
            raise StorageError(f"Failed to check if file {key} exists: {e}") from e

    async def generate_presigned_url(
        self,
        key: str,
        expiration: timedelta = timedelta(hours=1),
        method: str = "GET",
    ) -> str:
        """Generate a pre-signed URL for secure file access."""
        try:
            url = await self._run_sync(
                self._get_client().generate_presigned_url,
                ClientMethod=f"{method.lower()}_object",
                Params={"Bucket": self.config.bucket_name, "Key": key},
                ExpiresIn=int(expiration.total_seconds()),
            )

            return url

        except ClientError as e:
            await self._handle_client_error(e, f"generate presigned URL for {key}")
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL for {key}: {e}") from e

    async def generate_public_url(self, key: str) -> str | None:
        """Generate a public URL for file access."""
        try:
            # Check if bucket/object allows public access
            if self.config.cdn_domain:
                return f"https://{self.config.cdn_domain}/{key}"

            # Generate standard S3 URL
            if self.config.endpoint_url:
                # Custom endpoint (e.g., MinIO, R2)
                endpoint = self.config.endpoint_url.rstrip("/")
                return f"{endpoint}/{self.config.bucket_name}/{key}"
            else:
                # AWS S3 URL
                if self.config.region:
                    return f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{key}"
                else:
                    return f"https://{self.config.bucket_name}.s3.amazonaws.com/{key}"

        except Exception:
            return None

    async def copy_file(
        self,
        source_key: str,
        destination_key: str,
        metadata: dict[str, str] | None = None,
    ) -> FileInfo:
        """Copy a file within S3 storage."""
        try:
            copy_source = {"Bucket": self.config.bucket_name, "Key": source_key}

            copy_args: dict[str, Any] = {}
            if metadata:
                copy_args["Metadata"] = metadata
                copy_args["MetadataDirective"] = "REPLACE"

            await self._run_sync(
                self._get_client().copy_object,
                CopySource=copy_source,
                Bucket=self.config.bucket_name,
                Key=destination_key,
                **copy_args,
            )

            return await self.get_file_info(destination_key)

        except ClientError as e:
            await self._handle_client_error(e, f"copy file from {source_key} to {destination_key}")
        except Exception as e:
            raise StorageError(f"Failed to copy file from {source_key} to {destination_key}: {e}") from e

    async def move_file(
        self,
        source_key: str,
        destination_key: str,
        metadata: dict[str, str] | None = None,
    ) -> FileInfo:
        """Move a file within S3 storage."""
        # Copy the file to the new location
        file_info = await self.copy_file(source_key, destination_key, metadata)

        # Delete the original file
        await self.delete_file(source_key)

        return file_info

    # Private helper methods

    async def _run_sync(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous function in the async context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _async_paginate(self, page_iterator: Any) -> AsyncGenerator[Any]:
        """Convert sync paginator to async generator."""
        for page in page_iterator:
            yield page

    async def _simple_upload(
        self,
        file_path: Path,
        key: str,
        upload_args: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> FileInfo:
        """Perform a simple (non-multipart) upload."""
        file_size = file_path.stat().st_size

        # Call progress callback at start
        if progress_callback:
            progress = UploadProgress(bytes_uploaded=0, total_bytes=file_size, percentage=0.0)
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(progress)
            else:
                progress_callback(progress)

        # Don't add Callback to upload_args - it's not a valid S3 parameter
        with open(file_path, "rb") as f:
            upload_args["Body"] = f
            await self._run_sync(self._get_client().put_object, Bucket=self.config.bucket_name, Key=key, **upload_args)

        # Call progress callback at completion
        if progress_callback:
            progress = UploadProgress(bytes_uploaded=file_size, total_bytes=file_size, percentage=100.0)
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(progress)
            else:
                progress_callback(progress)

        return await self.get_file_info(key)

    async def _multipart_upload(
        self,
        file_path: Path,
        key: str,
        upload_args: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> FileInfo:
        """Perform a multipart upload for large files."""
        file_size = file_path.stat().st_size
        uploaded_bytes = 0

        # Remove Body from upload_args for multipart upload
        upload_args.pop("Body", None)

        # Start multipart upload
        response = await self._run_sync(
            self._get_client().create_multipart_upload, Bucket=self.config.bucket_name, Key=key, **upload_args
        )

        upload_id = response["UploadId"]
        parts = []

        try:
            with open(file_path, "rb") as f:
                part_number = 1

                while True:
                    chunk = f.read(self.config.chunk_size)
                    if not chunk:
                        break

                    # Upload part with Content-Length for OCI compatibility
                    part_response = await self._run_sync(
                        self._get_client().upload_part,
                        Bucket=self.config.bucket_name,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk,
                        ContentLength=len(chunk),  # Required for OCI Object Storage
                    )

                    parts.append({"ETag": part_response["ETag"], "PartNumber": part_number})

                    uploaded_bytes += len(chunk)

                    # Report progress
                    if progress_callback:
                        progress = UploadProgress(
                            bytes_uploaded=uploaded_bytes,
                            total_bytes=file_size,
                            percentage=(uploaded_bytes / file_size) * 100,
                        )
                        await self._run_sync(progress_callback, progress)

                    part_number += 1

            # Complete multipart upload
            await self._run_sync(
                self._get_client().complete_multipart_upload,
                Bucket=self.config.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            return await self.get_file_info(key)

        except Exception as e:
            # Abort multipart upload on error
            with contextlib.suppress(Exception):
                await self._run_sync(
                    self._get_client().abort_multipart_upload,
                    Bucket=self.config.bucket_name,
                    Key=key,
                    UploadId=upload_id,
                )

            raise e

    async def _handle_client_error(self, error: ClientError, operation: str) -> NoReturn:
        """Handle S3 client errors and convert to appropriate exceptions."""
        error_code = error.response.get("Error", {}).get("Code", "UNKNOWN")
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        message = error.response.get("Error", {}).get("Message", str(error))

        if error_code in ["NoSuchKey", "404"]:
            raise FileNotFoundError(f"File not found during {operation}")
        elif error_code in ["AccessDenied", "Forbidden", "403"]:
            raise PermissionError(f"Access denied during {operation}")
        elif error_code in ["QuotaExceeded", "RequestLimitExceeded"]:
            raise QuotaExceededError(
                f"Quota exceeded during {operation}", error_code=error_code, status_code=status_code
            )
        elif error_code in ["RequestTimeout", "ServiceUnavailable"]:
            raise NetworkError(
                f"Network error during {operation}: {message}", error_code=error_code, status_code=status_code
            )
        else:
            raise StorageError(
                f"S3 error during {operation}: {message}", error_code=error_code, status_code=status_code
            )

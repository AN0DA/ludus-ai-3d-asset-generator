"""
Meshy AI integration for 3D asset generation.

This module provides the MeshyAIIntegration class for text-to-3D generation using Meshy AI.
"""

import asyncio
import contextlib
import os
import ssl
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiofiles
import aiohttp
import structlog

from src.generators.base import APIError
from src.generators.base_integration import BaseServiceIntegration
from src.generators.models import GenerationRequest, GenerationResult, ProgressUpdate
from src.models.asset_model import FileFormat, GenerationStatus, QualityLevel, StylePreference

from .enums import GenerationMethod, ServiceProvider, ServiceStatus

logger = structlog.get_logger(__name__)


class MeshyAIIntegration(BaseServiceIntegration):
    """Meshy AI service integration for text-to-3D generation."""

    async def check_health(self) -> ServiceStatus:
        """Check service health status for Meshy AI."""
        # For Meshy AI, we don't have a dedicated health endpoint
        # So we'll assume it's available if we have an API key
        if self.config.api_key:
            self._health_status = ServiceStatus.AVAILABLE
            logger.info("Meshy AI service marked as available (API key present)")
        else:
            self._health_status = ServiceStatus.UNAVAILABLE
            logger.warning("Meshy AI service marked as unavailable (no API key)")

        self._last_health_check = datetime.utcnow()
        return self._health_status

    async def initialize(self) -> None:
        """Initialize the service integration."""
        if aiohttp:
            # Always create a new session in the current event loop
            if self.session and not self.session.closed:
                with contextlib.suppress(BaseException):
                    await self.session.close()
            self.session = None

            try:
                # Create SSL context that's more permissive for API calls
                # Create session in current event loop context
                timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

                if ssl:
                    # Create SSL context and disable verification for development
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    logger.info("SSL verification disabled for development")

                    connector = aiohttp.TCPConnector(ssl=ssl_context)
                    self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                else:
                    # Fallback if ssl module is not available - use default session
                    logger.warning("SSL module not available, using default session")
                    self.session = aiohttp.ClientSession(timeout=timeout)

                logger.info(f"Created new HTTP session for {self.__class__.__name__} in current event loop")
            except Exception as e:
                logger.error(f"Failed to create HTTP session: {e}")
                self.session = None

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

    def can_handle_request(self, request: GenerationRequest) -> bool:
        """Check if this service can handle the given request."""
        logger.info(
            "Checking if service can handle request",
            service_type=self.__class__.__name__,
            generation_method=request.generation_method,
            output_format=request.output_format,
            quality_level=request.quality_level,
        )

        # Check service capabilities
        if request.generation_method == GenerationMethod.TEXT_TO_3D:
            if not self.config.supports_text_to_3d:
                logger.info("Service rejected: doesn't support text-to-3D")
                return False
        elif request.generation_method == GenerationMethod.IMAGE_TO_3D and not self.config.supports_image_to_3d:
            logger.info("Service rejected: doesn't support image-to-3D")
            return False

        # Check output format support
        logger.info(
            f"Checking output format support: {request.output_format} in {self.config.supported_output_formats}"
        )
        if request.output_format not in self.config.supported_output_formats:
            logger.info(f"Service rejected: doesn't support format {request.output_format}")
            return False

        # Check quality level support
        logger.info(f"Checking quality level support: {request.quality_level} in {self.config.quality_levels}")
        if request.quality_level not in self.config.quality_levels:
            logger.info(f"Service rejected: doesn't support quality level {request.quality_level}")
            return False

        logger.info("Service CAN handle request - all checks passed!")
        return True

    async def generate_3d_asset(
        self, request: GenerationRequest, progress_callback: Callable[[ProgressUpdate], None] | None = None
    ) -> GenerationResult:
        """Generate 3D asset using Meshy AI."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            "Starting Meshy AI generation",
            request_id=request_id,
            description=request.description,
            quality_level=request.quality_level,
        )

        try:
            # Ensure session is initialized and not closed
            # Always reinitialize to make sure we're in the right event loop
            logger.info("Reinitializing session to ensure correct event loop context")
            await self.initialize()

            # Double-check the session is valid
            if not self.session:
                logger.error("Failed to create HTTP session")
                raise APIError("Failed to create HTTP session", error_code="SESSION_CREATION_FAILED")

            # Test if the session is actually usable
            try:
                # This will fail if the event loop is closed or incompatible
                await asyncio.sleep(0)
                # Test the session with a simple operation
                if self.session.closed:
                    raise RuntimeError("Session is closed")
            except RuntimeError as e:
                logger.warning(f"Session/event loop issue detected: {e}")
                # Force recreation of session
                await self.initialize()
            if not self.session:
                raise APIError(
                    "Failed to recreate HTTP session", error_code="SESSION_RECREATION_FAILED"
                ) from None  # Update progress
            if progress_callback:
                progress_callback(
                    ProgressUpdate(
                        request_id=request_id,
                        status=GenerationStatus.IN_PROGRESS,
                        progress_percentage=10.0,
                        current_step="Initializing Meshy AI request",
                    )
                )

            # Prepare request payload for Meshy API v2
            payload: dict[str, Any] = {
                "mode": "preview",  # Start with preview mode
                "prompt": request.description,
                "art_style": self._map_style_to_meshy(request.style_preference),
            }

            # Add optional parameters based on quality level
            if request.quality_level == QualityLevel.DRAFT:
                payload["target_polycount"] = 1000  # Lower polygon count for draft
            elif request.quality_level == QualityLevel.STANDARD:
                payload["target_polycount"] = 5000
            elif request.quality_level == QualityLevel.HIGH:
                payload["target_polycount"] = 50000
                payload["topology"] = "quad"
            elif request.quality_level == QualityLevel.ULTRA:
                payload["target_polycount"] = 100000
                payload["topology"] = "quad"
                payload["ai_model"] = "meshy-5"  # Use newer model for ultra quality

            # Add polygon count if specified (Meshy supports 100-300,000)
            if request.max_polygon_count:
                payload["target_polycount"] = min(max(request.max_polygon_count, 100), 300000)

            logger.info(f"Meshy payload: {payload}")

            # Start generation
            headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

            logger.info(f"Making request to: {self.config.base_url}/openapi/v2/text-to-3d")
            logger.info(f"Session status: {self.session is not None}")

            if not self.session:
                logger.error("HTTP session is None, cannot make request")
                raise APIError("HTTP session not available", error_code="NO_HTTP_SESSION")

            # Type assertion to help type checker
            assert self.session is not None  # nosec

            try:
                async with self.session.post(
                    f"{self.config.base_url}/openapi/v2/text-to-3d", headers=headers, json=payload
                ) as response:
                    logger.info(f"Meshy API response status: {response.status}")
                    # Meshy AI returns 202 (Accepted) for task creation, not 200
                    if response.status not in [200, 202]:
                        error_text = await response.text()
                        logger.error(f"Meshy API error: {response.status} - {error_text}")
                        raise APIError(
                            f"Meshy AI request failed: {response.status}",
                            error_code="MESHY_REQUEST_FAILED",
                            details={"status": response.status, "response": error_text},
                        )

                    result_data = await response.json()
                    task_id = result_data["result"]
                    logger.info(f"Meshy task created: {task_id}")
            except Exception as e:
                logger.error(f"HTTP request failed: {type(e).__name__}: {e}")
                raise

            # Update progress
            if progress_callback:
                progress_callback(
                    ProgressUpdate(
                        request_id=request_id,
                        status=GenerationStatus.IN_PROGRESS,
                        progress_percentage=30.0,
                        current_step="Meshy AI processing request",
                    )
                )

            # Poll for completion
            model_url = await self._poll_meshy_completion(task_id, request_id, request, progress_callback)

            # Update progress
            if progress_callback:
                progress_callback(
                    ProgressUpdate(
                        request_id=request_id,
                        status=GenerationStatus.IN_PROGRESS,
                        progress_percentage=90.0,
                        current_step="Downloading generated model",
                    )
                )

            # Download and process result
            model_path = await self._download_model(model_url, request.output_format)
            file_size = os.path.getsize(model_path) if model_path else 0

            generation_time = time.time() - start_time

            result = GenerationResult(
                request_id=request_id,
                status=GenerationStatus.COMPLETED,
                service_used=ServiceProvider.MESHY_AI,
                generation_method=request.generation_method,
                model_url=model_url,
                model_file_path=str(model_path) if model_path else None,
                file_format=request.output_format,
                file_size_bytes=file_size,
                generation_time_seconds=generation_time,
                completed_at=datetime.utcnow(),
            )

            logger.info("Meshy AI generation completed", request_id=request_id, generation_time=generation_time)

            return result

        except Exception as e:
            logger.error("Meshy AI generation failed", request_id=request_id, error=str(e))

            return GenerationResult(
                request_id=request_id,
                status=GenerationStatus.FAILED,
                service_used=ServiceProvider.MESHY_AI,
                generation_method=request.generation_method,
                error_message=str(e),
                error_code=getattr(e, "error_code", "UNKNOWN_ERROR"),
                generation_time_seconds=time.time() - start_time,
            )

    def _map_style_to_meshy(self, style: StylePreference | None) -> str:
        """Map our style preferences to Meshy AI styles."""
        if style is None:
            return "realistic"

        # Direct mapping since we only support Meshy's native styles
        style_mapping = {StylePreference.REALISTIC: "realistic", StylePreference.SCULPTURE: "sculpture"}
        return style_mapping.get(style, "realistic")

    async def _poll_meshy_completion(
        self,
        task_id: str,
        request_id: str,
        request: GenerationRequest,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> str:
        """Poll Meshy AI for task completion."""
        max_attempts = 120  # 10 minutes with smart backoff
        attempt = 0
        consecutive_failures = 0

        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        while attempt < max_attempts:
            try:
                # Ensure session is valid for this request
                if not self.session or self.session.closed:
                    await self.initialize()

                if not self.session:
                    raise APIError("HTTP session not available", error_code="NO_HTTP_SESSION")

                async with self.session.get(
                    f"{self.config.base_url}/openapi/v2/text-to-3d/{task_id}", headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Meshy status check failed: {response.status} - {error_text}")
                        consecutive_failures += 1

                        # If we get too many consecutive failures, give up
                        if consecutive_failures >= 5:
                            raise APIError(f"Too many consecutive failures checking Meshy status: {response.status}")

                        # Wait a bit longer after failure
                        await asyncio.sleep(min(10, 2**consecutive_failures))
                        attempt += 1
                        continue

                    # Reset failure counter on successful response
                    consecutive_failures = 0

                    data = await response.json()
                    status = data.get("status")
                    progress = data.get("progress", 0)

                    logger.info(f"Meshy task {task_id} status: {status}, progress: {progress}%")

                    if status == "SUCCEEDED":
                        # Check if we have model URLs
                        model_urls = data.get("model_urls", {})
                        if not model_urls:
                            raise APIError("Meshy task succeeded but no model URLs provided")

                        # Try to get the requested format first
                        format_key = request.output_format.value.lower()
                        if format_key in model_urls and model_urls[format_key]:
                            return model_urls[format_key]

                        # Fallback order: GLB -> OBJ -> any available
                        for fallback_format in ["glb", "obj", "fbx", "usdz"]:
                            if fallback_format in model_urls and model_urls[fallback_format]:
                                logger.info(f"Using fallback format {fallback_format} instead of {format_key}")
                                return model_urls[fallback_format]

                        raise APIError("Meshy task succeeded but no usable model URLs found")

                    elif status == "FAILED":
                        error_msg = data.get("task_error", {}).get("message", "Unknown error")
                        raise APIError(f"Meshy generation failed: {error_msg}")

                    elif status in ["PENDING", "IN_PROGRESS"]:
                        # Update progress based on actual progress from API
                        api_progress = 30.0 + (progress * 0.5)  # Map 0-100% to 30-80% of our progress
                        if progress_callback:
                            progress_callback(
                                ProgressUpdate(
                                    request_id=request_id,
                                    status=GenerationStatus.IN_PROGRESS,
                                    progress_percentage=api_progress,
                                    current_step=f"Meshy AI processing ({status.lower()}, {progress}%)",
                                )
                            )

                        # Smart backoff: shorter intervals initially, longer as time goes on
                        if attempt < 12:  # First minute: 5 second intervals
                            delay = 5
                        elif attempt < 36:  # Next 2 minutes: 8 second intervals
                            delay = 8
                        else:  # Remaining time: 10 second intervals
                            delay = 10

                        await asyncio.sleep(delay)
                        attempt += 1

                    elif status == "CANCELED":
                        raise APIError("Meshy generation was canceled")

                    else:
                        logger.warning(f"Unknown Meshy status: {status}, continuing to poll")
                        await asyncio.sleep(10)
                        attempt += 1

            except TimeoutError:
                logger.error(f"Timeout while polling Meshy AI status (attempt {attempt + 1})")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise APIError("Repeated timeouts while polling Meshy AI status") from None
                await asyncio.sleep(5)
                attempt += 1
            except APIError:
                # Re-raise API errors as-is
                raise
            except Exception as e:
                logger.error(f"Unexpected error while polling Meshy AI: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise APIError(f"Repeated errors while polling Meshy AI: {e}") from e
                await asyncio.sleep(5)
                attempt += 1

        raise APIError("Meshy AI generation timed out after maximum polling attempts")

    async def _download_model(self, model_url: str, output_format: FileFormat) -> Path | None:
        """Download the generated model from Meshy AI."""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Ensure session is valid for download
                if not self.session or self.session.closed:
                    await self.initialize()

                if not self.session:
                    logger.error("HTTP session not available for download")
                    return None

                logger.info(f"Downloading model from: {model_url}")

                async with self.session.get(model_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download model: HTTP {response.status}")
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(2**retry_count)  # Exponential backoff
                            continue
                        return None

                    # Get content length for validation
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        content_length = int(content_length)
                        logger.info(f"Expected download size: {content_length} bytes")

                    # Create temporary file with proper extension
                    temp_dir = Path(tempfile.gettempdir()) / "meshy_downloads"
                    temp_dir.mkdir(exist_ok=True, parents=True)

                    # Determine file extension from URL or format
                    url_path = urlparse(model_url).path
                    if url_path.endswith(".glb"):
                        file_ext = "glb"
                    elif url_path.endswith(".obj"):
                        file_ext = "obj"
                    elif url_path.endswith(".fbx"):
                        file_ext = "fbx"
                    elif url_path.endswith(".usdz"):
                        file_ext = "usdz"
                    else:
                        file_ext = output_format.value

                    temp_file = temp_dir / f"model_{int(time.time())}_{uuid.uuid4().hex[:8]}.{file_ext}"

                    downloaded_bytes = 0
                    async with aiofiles.open(temp_file, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            downloaded_bytes += len(chunk)

                    # Validate download
                    if content_length and downloaded_bytes != content_length:
                        logger.warning(f"Download size mismatch: expected {content_length}, got {downloaded_bytes}")

                    # Verify file exists and has content
                    if not temp_file.exists() or temp_file.stat().st_size == 0:
                        logger.error("Downloaded file is empty or missing")
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(2**retry_count)
                            continue
                        return None

                    logger.info(f"Successfully downloaded model: {temp_file} ({downloaded_bytes} bytes)")
                    return temp_file

            except Exception as e:
                logger.error(f"Error downloading model (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2**retry_count)
                else:
                    logger.error("Max download retries exceeded")
                    return None

        return None

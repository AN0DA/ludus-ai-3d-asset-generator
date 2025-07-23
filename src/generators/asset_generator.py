"""
Main 3D asset generation module.

This module provides the Asset3DGenerator class that supports multiple 3D generation
services, async workflows, error handling, retry logic, and result processing.
"""

import asyncio
import os
import uuid
from collections.abc import Callable
from pathlib import Path

import structlog

from src.models.asset_model import FileFormat, GenerationStatus
from src.utils.validators import AssetValidator, TextValidator, ValidationException

from .base_integration import BaseServiceIntegration
from .configs import ServiceConfig
from .enums import GenerationMethod, ServiceProvider, ServiceStatus
from .meshy_integration import MeshyAIIntegration
from .models import GenerationRequest, GenerationResult, ProgressUpdate

logger = structlog.get_logger(__name__)


class Asset3DGenerator:
    """
    Main 3D asset generator with multiple service integrations.

    Supports multiple services with automatic fallback and comprehensive error handling.
    """

    def __init__(self, configs: dict[ServiceProvider, ServiceConfig]) -> None:
        """Initialize the asset generator with service configurations."""
        self.configs = configs
        self.services: dict[ServiceProvider, BaseServiceIntegration] = {}
        self.generation_history: list[GenerationResult] = []

        # Initialize service integrations
        for provider, config in configs.items():
            if provider == ServiceProvider.MESHY_AI:
                self.services[provider] = MeshyAIIntegration(config)
            # Add other services as needed

    async def initialize(self) -> None:
        """Initialize all service integrations."""
        for service in self.services.values():
            await service.initialize()

    async def cleanup(self) -> None:
        """Clean up all service integrations."""
        for service in self.services.values():
            await service.cleanup()

    async def generate_asset(
        self, request: GenerationRequest, progress_callback: Callable[[ProgressUpdate], None] | None = None
    ) -> GenerationResult:
        """
        Generate a 3D asset using the best available service.

        Args:
            request: Generation request parameters
            progress_callback: Optional callback for progress updates

        Returns:
            GenerationResult with the generated asset or error information
        """
        logger.info("Starting asset generation", description=request.description, asset_type=request.asset_type)

        # Validate request
        try:
            await self._validate_request(request)
        except ValidationException as e:
            return GenerationResult(
                request_id=str(uuid.uuid4()),
                status=GenerationStatus.FAILED,
                service_used=ServiceProvider.MESHY_AI,  # Default
                generation_method=request.generation_method,
                error_message=f"Validation failed: {e.message}",
                error_code="VALIDATION_ERROR",
            )

        # Select best service
        selected_service, service_provider = await self._select_service(request)

        if not selected_service:
            return GenerationResult(
                request_id=str(uuid.uuid4()),
                status=GenerationStatus.FAILED,
                service_used=ServiceProvider.MESHY_AI,  # Default
                generation_method=request.generation_method,
                error_message="No available services can handle this request",
                error_code="NO_SERVICE_AVAILABLE",
            )

        # Attempt generation with retry logic
        max_retries = 3
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                result = await selected_service.generate_3d_asset(request, progress_callback)

                if result.status == GenerationStatus.COMPLETED:
                    # Post-process the result
                    result = await self._post_process_result(result, request)

                    # Track history
                    self.generation_history.append(result)

                    logger.info(
                        "Asset generation completed successfully",
                        request_id=result.request_id,
                        service=service_provider,
                    )

                    return result
                else:
                    # Generation failed, try fallback
                    logger.warning(
                        "Generation failed, trying fallback", service=service_provider, error=result.error_message
                    )
                    break

            except Exception as e:
                logger.error(
                    "Generation attempt failed", service=service_provider, attempt=retry_count + 1, error=str(e)
                )
                last_error = e
                retry_count += 1

                if retry_count < max_retries:
                    await asyncio.sleep(2**retry_count)  # Exponential backoff

        # Try fallback services
        fallback_result = await self._try_fallback_services(request, service_provider, progress_callback)

        if fallback_result and fallback_result.status == GenerationStatus.COMPLETED:
            return fallback_result

        # All attempts failed
        error_message = str(last_error) if last_error else "All generation attempts failed"
        return GenerationResult(
            request_id=str(uuid.uuid4()),
            status=GenerationStatus.FAILED,
            service_used=service_provider,
            generation_method=request.generation_method,
            error_message=error_message,
            error_code="GENERATION_FAILED",
        )

    async def _validate_request(self, request: GenerationRequest) -> None:
        """Validate the generation request."""
        # Use our validation system
        TextValidator.validate_description(request.description)
        AssetValidator.validate_asset_type(request.asset_type)

        # Additional 3D-specific validation
        if (
            request.generation_method == GenerationMethod.IMAGE_TO_3D
            and not request.reference_image_path
            and not request.reference_image_url
        ):
            raise ValidationException(
                "Image-to-3D generation requires a reference image",
                field="reference_image",
                code="MISSING_REFERENCE_IMAGE",
            )

        # Validate prompt length for Meshy API (600 character limit)
        if len(request.description) > 600:
            raise ValidationException(
                "Description must be 600 characters or less for Meshy AI",
                field="description",
                code="DESCRIPTION_TOO_LONG",
            )

        if request.max_polygon_count and (request.max_polygon_count < 100 or request.max_polygon_count > 300000):
            raise ValidationException(
                "Maximum polygon count must be between 100 and 300,000 (Meshy API limits)",
                field="max_polygon_count",
                code="POLYGON_COUNT_OUT_OF_RANGE",
            )

    async def _select_service(
        self, request: GenerationRequest
    ) -> tuple[BaseServiceIntegration | None, ServiceProvider]:
        """Select the best service for the given request."""

        logger.info(
            f"Selecting service for request. Method: {request.generation_method}, Format: {request.output_format}"
        )
        logger.info(f"Available services: {list(self.services.keys())}")

        # If user specified a preferred service, try it first
        if request.preferred_service and request.preferred_service in self.services:
            service = self.services[request.preferred_service]
            can_handle = service.can_handle_request(request)
            logger.info(f"Preferred service {request.preferred_service} can handle: {can_handle}")
            if can_handle:
                return service, request.preferred_service

        # Score services based on various factors
        service_scores = []

        for provider, service in self.services.items():
            can_handle = service.can_handle_request(request)
            logger.info(f"Service {provider} can handle request: {can_handle}")
            if not can_handle:
                continue

            # Calculate score based on:
            # - Service health
            # - Capabilities match
            # - Historical success rate

            score = 0.0

            # Health check
            health_status = await service.check_health()
            logger.info(f"Service {provider} health check result: {health_status}")
            if health_status == ServiceStatus.AVAILABLE:
                score += 100
            elif health_status == ServiceStatus.DEGRADED:
                score += 50
            else:
                logger.warning(f"Service {provider} marked as unavailable, skipping")
                continue  # Skip unavailable services

            # Capability match
            if provider == ServiceProvider.MESHY_AI and request.generation_method == GenerationMethod.TEXT_TO_3D:
                score += 20  # Meshy AI is good for text-to-3D

            service_scores.append((score, service, provider))

        if not service_scores:
            return None, ServiceProvider.MESHY_AI  # Default

        # Sort by score (highest first)
        service_scores.sort(key=lambda x: x[0], reverse=True)

        return service_scores[0][1], service_scores[0][2]

    async def _try_fallback_services(
        self,
        request: GenerationRequest,
        failed_service: ServiceProvider,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> GenerationResult | None:
        """Try fallback services if the primary service fails."""

        for provider, service in self.services.items():
            if provider == failed_service:
                continue  # Skip the failed service

            if not service.can_handle_request(request):
                continue

            logger.info("Trying fallback service", fallback_service=provider, failed_service=failed_service)

            try:
                result = await service.generate_3d_asset(request, progress_callback)
                if result.status == GenerationStatus.COMPLETED:
                    return result
            except Exception as e:
                logger.error("Fallback service failed", service=provider, error=str(e))
                continue

        return None

    async def _post_process_result(self, result: GenerationResult, request: GenerationRequest) -> GenerationResult:
        """Post-process the generation result."""

        if not result.model_file_path:
            return result

        try:
            # Generate thumbnail if possible
            if not result.thumbnail_file_path:
                thumbnail_path = await self._generate_thumbnail(result.model_file_path)
                result.thumbnail_file_path = str(thumbnail_path) if thumbnail_path else None

            # Convert format if needed
            if request.output_format != result.file_format:
                converted_path = await self._convert_format(result.model_file_path, request.output_format)
                if converted_path:
                    result.model_file_path = str(converted_path)
                    result.file_format = request.output_format

            # Optimize model if polygon count is too high
            if result.polygon_count and request.max_polygon_count and result.polygon_count > request.max_polygon_count:
                optimized_path = await self._optimize_model(result.model_file_path, request.max_polygon_count)
                if optimized_path:
                    result.model_file_path = str(optimized_path)
                    result.polygon_count = request.max_polygon_count

            # Update file size after processing
            if result.model_file_path and os.path.exists(result.model_file_path):
                result.file_size_bytes = os.path.getsize(result.model_file_path)

        except Exception as e:
            logger.error("Post-processing failed", error=str(e))
            # Don't fail the whole generation for post-processing errors

        return result

    async def _generate_thumbnail(self, model_path: str) -> Path | None:
        """Generate a thumbnail image for the 3D model."""
        try:
            # This would use a 3D rendering library like Open3D or PyOpenGL
            # For now, return a placeholder
            logger.info("Would generate thumbnail", model_path=model_path)
            return None
        except Exception as e:
            logger.error("Thumbnail generation failed", error=str(e))
            return None

    async def _convert_format(self, model_path: str, target_format: FileFormat) -> Path | None:
        """Convert model to target format."""
        try:
            # This would use conversion libraries like Open3D, Assimp, or FBX SDK
            logger.info("Would convert format", source=model_path, target=target_format)
            return None
        except Exception as e:
            logger.error("Format conversion failed", error=str(e))
            return None

    async def _optimize_model(self, model_path: str, target_polygon_count: int) -> Path | None:
        """Optimize model by reducing polygon count."""
        try:
            # This would use mesh simplification algorithms
            logger.info("Would optimize model", source=model_path, target_polygons=target_polygon_count)
            return None
        except Exception as e:
            logger.error("Model optimization failed", error=str(e))
            return None

    def get_service_health_status(self) -> dict[ServiceProvider, ServiceStatus]:
        """Get current health status of all services."""
        return {provider: service._health_status for provider, service in self.services.items()}


def create_default_configs() -> dict[ServiceProvider, ServiceConfig]:
    """Create default service configurations."""
    return {}

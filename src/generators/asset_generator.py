"""
Comprehensive 3D asset generation module with multiple service integrations.

This module provides the Asset3DGenerator class that supports multiple 3D generation
services, async workflows, error handling, retry logic, and result processing.
Includes integrations for Meshy AI, Kaedim API, and image-to-3D pipelines.
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from urllib.parse import urlparse
import uuid

import aiofiles
try:
    import aiohttp
except ImportError:
    aiohttp = None  # Will be handled gracefully

import structlog
try:
    from PIL import Image
except ImportError:
    Image = None

from pydantic import BaseModel, Field, validator

from ..models.asset_model import (
    AssetType, 
    StylePreference, 
    QualityLevel, 
    FileFormat,
    GenerationStatus
)
from ..utils.validators import (
    TextValidator,
    AssetValidator,
    FileValidator,
    ValidationException
)
from .base import GenerationError, APIError, ErrorSeverity


# Configure logging
logger = structlog.get_logger(__name__)


# Service Provider Enums

class ServiceProvider(str, Enum):
    """Available 3D generation service providers."""
    MESHY_AI = "meshy_ai"
    KAEDIM = "kaedim"
    TRIPO3D = "tripo3d"
    LUMA_AI = "luma_ai"
    OPENAI_DALLE = "openai_dalle"  # For image generation in image-to-3D pipeline


class ServiceStatus(str, Enum):
    """Service availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class GenerationMethod(str, Enum):
    """3D generation methods."""
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    HYBRID = "hybrid"  # Text + reference image


# Configuration Models

@dataclass
class ServiceConfig:
    """Configuration for a 3D generation service."""
    api_key: str
    base_url: str
    max_requests_per_minute: int = 10
    max_requests_per_day: int = 1000
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    cost_per_generation: float = 0.0
    supports_text_to_3d: bool = True
    supports_image_to_3d: bool = False
    supported_output_formats: List[FileFormat] = field(default_factory=lambda: [FileFormat.OBJ])
    max_polygon_count: int = 50000
    quality_levels: List[QualityLevel] = field(default_factory=lambda: [QualityLevel.STANDARD])


@dataclass
class RateLimitInfo:
    """Rate limiting information for a service."""
    requests_per_minute: int = 0
    requests_per_day: int = 0
    last_reset_minute: datetime = field(default_factory=datetime.utcnow)
    last_reset_day: datetime = field(default_factory=datetime.utcnow)
    
    def can_make_request(self, config: ServiceConfig) -> bool:
        """Check if a request can be made without exceeding rate limits."""
        now = datetime.utcnow()
        
        # Reset counters if needed
        if now - self.last_reset_minute >= timedelta(minutes=1):
            self.requests_per_minute = 0
            self.last_reset_minute = now
        
        if now - self.last_reset_day >= timedelta(days=1):
            self.requests_per_day = 0
            self.last_reset_day = now
        
        return (
            self.requests_per_minute < config.max_requests_per_minute and
            self.requests_per_day < config.max_requests_per_day
        )
    
    def record_request(self) -> None:
        """Record that a request was made."""
        self.requests_per_minute += 1
        self.requests_per_day += 1


# Request and Response Models

class GenerationRequest(BaseModel):
    """Request model for 3D asset generation."""
    
    # Required fields
    description: str = Field(..., min_length=10, max_length=2000)
    asset_type: AssetType
    
    # Optional fields
    style_preference: Optional[StylePreference] = None
    quality_level: QualityLevel = QualityLevel.STANDARD
    output_format: FileFormat = FileFormat.OBJ
    reference_image_url: Optional[str] = None
    reference_image_path: Optional[str] = None
    
    # Generation parameters
    max_polygon_count: Optional[int] = Field(None, ge=1000, le=100000)
    generation_method: GenerationMethod = GenerationMethod.TEXT_TO_3D
    preferred_service: Optional[ServiceProvider] = None
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = Field(1, ge=1, le=10)
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description using our text validator."""
        return TextValidator.validate_description(v)
    
    @validator('asset_type', pre=True)
    def validate_asset_type(cls, v):
        """Validate asset type."""
        return AssetValidator.validate_asset_type(v)


class GenerationResult(BaseModel):
    """Result model for 3D asset generation."""
    
    # Generation info
    request_id: str
    status: GenerationStatus
    service_used: ServiceProvider
    generation_method: GenerationMethod
    
    # Files and URLs
    model_url: Optional[str] = None
    model_file_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_file_path: Optional[str] = None
    
    # Metadata
    file_format: Optional[FileFormat] = None
    file_size_bytes: Optional[int] = None
    polygon_count: Optional[int] = None
    generation_time_seconds: Optional[float] = None
    
    # Cost and usage
    cost_usd: Optional[float] = None
    service_request_id: Optional[str] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressUpdate(BaseModel):
    """Progress update for generation tracking."""
    
    request_id: str
    status: GenerationStatus
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    current_step: str
    estimated_completion_time: Optional[datetime] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Service Integration Classes

class BaseServiceIntegration:
    """Base class for 3D generation service integrations."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.rate_limit_info = RateLimitInfo()
        self.session = None  # Will be initialized as needed
        self._health_status = ServiceStatus.AVAILABLE
        self._last_health_check = datetime.utcnow()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the service integration."""
        if aiohttp and not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def check_health(self) -> ServiceStatus:
        """Check service health status."""
        if not aiohttp:
            logger.warning("aiohttp not available, marking service as unavailable")
            self._health_status = ServiceStatus.UNAVAILABLE
            return self._health_status
            
        try:
            if not self.session:
                await self.initialize()
            
            # Make a simple health check request
            health_url = f"{self.config.base_url}/health"
            if self.session:
                async with self.session.get(health_url) as response:
                    if response.status == 200:
                        self._health_status = ServiceStatus.AVAILABLE
                    else:
                        self._health_status = ServiceStatus.DEGRADED
                        
        except asyncio.TimeoutError:
            self._health_status = ServiceStatus.UNAVAILABLE
        except Exception:
            self._health_status = ServiceStatus.UNAVAILABLE
        
        self._last_health_check = datetime.utcnow()
        return self._health_status
    
    def can_handle_request(self, request: GenerationRequest) -> bool:
        """Check if this service can handle the given request."""
        # Check rate limits
        if not self.rate_limit_info.can_make_request(self.config):
            return False
        
        # Check service capabilities
        if request.generation_method == GenerationMethod.TEXT_TO_3D:
            if not self.config.supports_text_to_3d:
                return False
        elif request.generation_method == GenerationMethod.IMAGE_TO_3D:
            if not self.config.supports_image_to_3d:
                return False
        
        # Check output format support
        if request.output_format not in self.config.supported_output_formats:
            return False
        
        # Check quality level support
        if request.quality_level not in self.config.quality_levels:
            return False
        
        return True
    
    async def generate_3d_asset(
        self, 
        request: GenerationRequest,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> GenerationResult:
        """Generate 3D asset - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_3d_asset")
    
    def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate the cost for generating this asset."""
        base_cost = self.config.cost_per_generation
        
        # Adjust cost based on quality level
        quality_multiplier = {
            QualityLevel.DRAFT: 0.5,
            QualityLevel.STANDARD: 1.0,
            QualityLevel.HIGH: 2.0,
            QualityLevel.ULTRA: 4.0
        }.get(request.quality_level, 1.0)
        
        return base_cost * quality_multiplier


class MeshyAIIntegration(BaseServiceIntegration):
    """Meshy AI service integration for text-to-3D generation."""
    
    async def generate_3d_asset(
        self, 
        request: GenerationRequest,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> GenerationResult:
        """Generate 3D asset using Meshy AI."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info("Starting Meshy AI generation", 
                   request_id=request_id, 
                   description=request.description)
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(ProgressUpdate(
                    request_id=request_id,
                    status=GenerationStatus.PROCESSING,
                    progress_percentage=10.0,
                    current_step="Initializing Meshy AI request"
                ))
            
            # Record rate limit usage
            self.rate_limit_info.record_request()
            
            # Prepare request payload
            payload = {
                "mode": "text",
                "prompt": request.description,
                "art_style": self._map_style_to_meshy(request.style_preference),
                "negative_prompt": "low quality, blurry, distorted",
            }
            
            # Quality settings
            if request.quality_level == QualityLevel.HIGH:
                payload["texture_richness"] = "high"
            elif request.quality_level == QualityLevel.ULTRA:
                payload["texture_richness"] = "high"
                payload["topology"] = "quad"
            
            # Start generation
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.session:
                async with self.session.post(
                    f"{self.config.base_url}/v2/text-to-3d",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(
                            f"Meshy AI request failed: {response.status}",
                            error_code="MESHY_REQUEST_FAILED",
                            details={"status": response.status, "response": error_text}
                        )
                    
                    result_data = await response.json()
                    task_id = result_data["result"]
            else:
                raise APIError("HTTP session not available", error_code="NO_HTTP_SESSION")
            
            # Update progress
            if progress_callback:
                progress_callback(ProgressUpdate(
                    request_id=request_id,
                    status=GenerationStatus.PROCESSING,
                    progress_percentage=30.0,
                    current_step="Meshy AI processing request"
                ))
            
            # Poll for completion
            model_url = await self._poll_meshy_completion(
                task_id, request_id, progress_callback
            )
            
            # Update progress
            if progress_callback:
                progress_callback(ProgressUpdate(
                    request_id=request_id,
                    status=GenerationStatus.PROCESSING,
                    progress_percentage=90.0,
                    current_step="Downloading generated model"
                ))
            
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
                cost_usd=self.estimate_cost(request),
                service_request_id=task_id,
                completed_at=datetime.utcnow()
            )
            
            logger.info("Meshy AI generation completed", 
                       request_id=request_id,
                       generation_time=generation_time)
            
            return result
            
        except Exception as e:
            logger.error("Meshy AI generation failed", 
                        request_id=request_id, 
                        error=str(e))
            
            return GenerationResult(
                request_id=request_id,
                status=GenerationStatus.FAILED,
                service_used=ServiceProvider.MESHY_AI,
                generation_method=request.generation_method,
                error_message=str(e),
                error_code=getattr(e, 'error_code', 'UNKNOWN_ERROR'),
                generation_time_seconds=time.time() - start_time
            )
    
    def _map_style_to_meshy(self, style: Optional[StylePreference]) -> str:
        """Map our style preferences to Meshy AI styles."""
        if style is None:
            return "realistic"
            
        style_mapping = {
            StylePreference.REALISTIC: "realistic",
            StylePreference.STYLIZED: "cartoon",
            StylePreference.CARTOON: "cartoon",
            StylePreference.LOW_POLY: "low-poly",
            StylePreference.FANTASY: "fantasy",
            StylePreference.SCI_FI: "sci-fi",
            StylePreference.MEDIEVAL: "medieval",
            StylePreference.MODERN: "modern",
            StylePreference.FUTURISTIC: "futuristic"
        }
        return style_mapping.get(style, "realistic")
    
    async def _poll_meshy_completion(
        self, 
        task_id: str, 
        request_id: str,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> str:
        """Poll Meshy AI for task completion."""
        max_attempts = 60  # 5 minutes with 5-second intervals
        attempt = 0
        
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        while attempt < max_attempts:
            try:
                if self.session:
                    async with self.session.get(
                        f"{self.config.base_url}/v2/text-to-3d/{task_id}",
                        headers=headers
                    ) as response:
                        if response.status != 200:
                            raise APIError(f"Failed to check Meshy status: {response.status}")
                        
                        data = await response.json()
                        status = data.get("status")
                        
                        if status == "SUCCEEDED":
                            return data["model_urls"]["glb"]
                        elif status == "FAILED":
                            error_msg = data.get("error", "Unknown error")
                            raise APIError(f"Meshy generation failed: {error_msg}")
                        elif status in ["PENDING", "IN_PROGRESS"]:
                            # Update progress
                            progress = 30.0 + (attempt / max_attempts) * 50.0
                            if progress_callback:
                                progress_callback(ProgressUpdate(
                                    request_id=request_id,
                                    status=GenerationStatus.PROCESSING,
                                    progress_percentage=progress,
                                    current_step=f"Meshy AI processing ({status.lower()})"
                                ))
                            
                            await asyncio.sleep(5)
                            attempt += 1
                        else:
                            raise APIError(f"Unknown Meshy status: {status}")
                else:
                    raise APIError("HTTP session not available", error_code="NO_HTTP_SESSION")
                        
            except asyncio.TimeoutError:
                raise APIError("Timeout while polling Meshy AI status")
        
        raise APIError("Meshy AI generation timed out")
    
    async def _download_model(self, model_url: str, output_format: FileFormat) -> Optional[Path]:
        """Download the generated model from Meshy AI."""
        try:
            if self.session:
                async with self.session.get(model_url) as response:
                    if response.status != 200:
                        logger.error("Failed to download model", status=response.status)
                        return None
                    
                    # Create temporary file
                    temp_dir = Path(tempfile.gettempdir()) / "meshy_downloads"
                    temp_dir.mkdir(exist_ok=True)
                    
                    temp_file = temp_dir / f"model_{int(time.time())}.{output_format.value}"
                    
                    async with aiofiles.open(temp_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    return temp_file
            else:
                logger.error("HTTP session not available for download")
                return None
                
        except Exception as e:
            logger.error("Error downloading model", error=str(e))
            return None


class KaedimIntegration(BaseServiceIntegration):
    """Kaedim API service integration."""
    
    async def generate_3d_asset(
        self, 
        request: GenerationRequest,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> GenerationResult:
        """Generate 3D asset using Kaedim API."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info("Starting Kaedim generation", 
                   request_id=request_id, 
                   description=request.description)
        
        try:
            # For Kaedim, we need to first generate an image if using text-to-3D
            image_path = None
            if request.generation_method == GenerationMethod.TEXT_TO_3D:
                if progress_callback:
                    progress_callback(ProgressUpdate(
                        request_id=request_id,
                        status=GenerationStatus.PROCESSING,
                        progress_percentage=20.0,
                        current_step="Generating reference image"
                    ))
                
                # Generate image using DALL-E or similar
                image_path = await self._generate_reference_image(request.description)
            else:
                image_path = request.reference_image_path
            
            if not image_path:
                raise APIError("No reference image available for Kaedim generation")
            
            # Record rate limit usage
            self.rate_limit_info.record_request()
            
            # Upload image to Kaedim
            if progress_callback:
                progress_callback(ProgressUpdate(
                    request_id=request_id,
                    status=GenerationStatus.PROCESSING,
                    progress_percentage=40.0,
                    current_step="Uploading image to Kaedim"
                ))
            
            kaedim_job_id = await self._upload_to_kaedim(image_path, request)
            
            # Poll for completion
            if progress_callback:
                progress_callback(ProgressUpdate(
                    request_id=request_id,
                    status=GenerationStatus.PROCESSING,
                    progress_percentage=60.0,
                    current_step="Kaedim processing 3D model"
                ))
            
            model_url = await self._poll_kaedim_completion(
                kaedim_job_id, request_id, progress_callback
            )
            
            # Download result
            model_path = await self._download_model(model_url, request.output_format)
            file_size = os.path.getsize(model_path) if model_path else 0
            
            generation_time = time.time() - start_time
            
            result = GenerationResult(
                request_id=request_id,
                status=GenerationStatus.COMPLETED,
                service_used=ServiceProvider.KAEDIM,
                generation_method=request.generation_method,
                model_url=model_url,
                model_file_path=str(model_path) if model_path else None,
                file_format=request.output_format,
                file_size_bytes=file_size,
                generation_time_seconds=generation_time,
                cost_usd=self.estimate_cost(request),
                service_request_id=kaedim_job_id,
                completed_at=datetime.utcnow()
            )
            
            logger.info("Kaedim generation completed", 
                       request_id=request_id,
                       generation_time=generation_time)
            
            return result
            
        except Exception as e:
            logger.error("Kaedim generation failed", 
                        request_id=request_id, 
                        error=str(e))
            
            return GenerationResult(
                request_id=request_id,
                status=GenerationStatus.FAILED,
                service_used=ServiceProvider.KAEDIM,
                generation_method=request.generation_method,
                error_message=str(e),
                error_code=getattr(e, 'error_code', 'UNKNOWN_ERROR'),
                generation_time_seconds=time.time() - start_time
            )
    
    async def _generate_reference_image(self, description: str) -> Optional[str]:
        """Generate a reference image using DALL-E for Kaedim."""
        # This would integrate with DALL-E or another image generation service
        # For now, return None as placeholder
        logger.info("Would generate reference image for Kaedim", description=description)
        return None
    
    async def _upload_to_kaedim(self, image_path: str, request: GenerationRequest) -> str:
        """Upload image to Kaedim and start 3D generation."""
        # Placeholder implementation
        logger.info("Would upload to Kaedim", image_path=image_path)
        return f"kaedim_job_{int(time.time())}"
    
    async def _poll_kaedim_completion(
        self, 
        job_id: str, 
        request_id: str,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> str:
        """Poll Kaedim for job completion."""
        # Placeholder implementation
        logger.info("Would poll Kaedim completion", job_id=job_id)
        await asyncio.sleep(2)  # Simulate processing time
        return f"https://example.com/model_{job_id}.obj"
    
    async def _download_model(self, model_url: str, output_format: FileFormat) -> Optional[Path]:
        """Download the generated model from Kaedim."""
        try:
            if self.session:
                async with self.session.get(model_url) as response:
                    if response.status != 200:
                        logger.error("Failed to download model from Kaedim", status=response.status)
                        return None
                    
                    # Create temporary file
                    temp_dir = Path(tempfile.gettempdir()) / "kaedim_downloads"
                    temp_dir.mkdir(exist_ok=True)
                    
                    temp_file = temp_dir / f"model_{int(time.time())}.{output_format.value}"
                    
                    async with aiofiles.open(temp_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    return temp_file
            else:
                logger.error("HTTP session not available for Kaedim download")
                return None
                
        except Exception as e:
            logger.error("Error downloading model from Kaedim", error=str(e))
            return None


# Main Asset Generator Class

class Asset3DGenerator:
    """
    Main 3D asset generator with multiple service integrations.
    
    Supports multiple services with automatic fallback, rate limiting,
    cost tracking, and comprehensive error handling.
    """
    
    def __init__(self, configs: Dict[ServiceProvider, ServiceConfig]):
        """Initialize the asset generator with service configurations."""
        self.configs = configs
        self.services: Dict[ServiceProvider, BaseServiceIntegration] = {}
        self.total_cost_tracking = 0.0
        self.generation_history: List[GenerationResult] = []
        
        # Initialize service integrations
        for provider, config in configs.items():
            if provider == ServiceProvider.MESHY_AI:
                self.services[provider] = MeshyAIIntegration(config)
            elif provider == ServiceProvider.KAEDIM:
                self.services[provider] = KaedimIntegration(config)
            # Add other services as needed
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize all service integrations."""
        for service in self.services.values():
            await service.initialize()
    
    async def cleanup(self):
        """Clean up all service integrations."""
        for service in self.services.values():
            await service.cleanup()
    
    async def generate_asset(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> GenerationResult:
        """
        Generate a 3D asset using the best available service.
        
        Args:
            request: Generation request parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            GenerationResult with the generated asset or error information
        """
        logger.info("Starting asset generation", 
                   description=request.description,
                   asset_type=request.asset_type)
        
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
                error_code="VALIDATION_ERROR"
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
                error_code="NO_SERVICE_AVAILABLE"
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
                    
                    # Track cost and history
                    if result.cost_usd:
                        self.total_cost_tracking += result.cost_usd
                    self.generation_history.append(result)
                    
                    logger.info("Asset generation completed successfully",
                               request_id=result.request_id,
                               service=service_provider)
                    
                    return result
                else:
                    # Generation failed, try fallback
                    logger.warning("Generation failed, trying fallback",
                                 service=service_provider,
                                 error=result.error_message)
                    break
                    
            except Exception as e:
                logger.error("Generation attempt failed",
                           service=service_provider,
                           attempt=retry_count + 1,
                           error=str(e))
                last_error = e
                retry_count += 1
                
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
        
        # Try fallback services
        fallback_result = await self._try_fallback_services(
            request, service_provider, progress_callback
        )
        
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
            error_code="GENERATION_FAILED"
        )
    
    async def _validate_request(self, request: GenerationRequest) -> None:
        """Validate the generation request."""
        # Use our validation system
        TextValidator.validate_description(request.description)
        AssetValidator.validate_asset_type(request.asset_type)
        
        # Additional 3D-specific validation
        if request.generation_method == GenerationMethod.IMAGE_TO_3D:
            if not request.reference_image_path and not request.reference_image_url:
                raise ValidationException(
                    "Image-to-3D generation requires a reference image",
                    field="reference_image",
                    code="MISSING_REFERENCE_IMAGE"
                )
        
        if request.max_polygon_count and request.max_polygon_count > 100000:
            raise ValidationException(
                "Maximum polygon count exceeds limit (100,000)",
                field="max_polygon_count",
                code="POLYGON_COUNT_TOO_HIGH"
            )
    
    async def _select_service(
        self, 
        request: GenerationRequest
    ) -> Tuple[Optional[BaseServiceIntegration], ServiceProvider]:
        """Select the best service for the given request."""
        
        # If user specified a preferred service, try it first
        if request.preferred_service and request.preferred_service in self.services:
            service = self.services[request.preferred_service]
            if service.can_handle_request(request):
                return service, request.preferred_service
        
        # Score services based on various factors
        service_scores = []
        
        for provider, service in self.services.items():
            if not service.can_handle_request(request):
                continue
            
            # Calculate score based on:
            # - Service health
            # - Cost
            # - Capabilities match
            # - Historical success rate
            
            score = 0.0
            
            # Health check
            health_status = await service.check_health()
            if health_status == ServiceStatus.AVAILABLE:
                score += 100
            elif health_status == ServiceStatus.DEGRADED:
                score += 50
            else:
                continue  # Skip unavailable services
            
            # Cost factor (lower cost = higher score)
            cost = service.estimate_cost(request)
            if cost > 0:
                score += max(0, 50 - cost * 10)  # Adjust scaling as needed
            
            # Capability match
            if provider == ServiceProvider.MESHY_AI and request.generation_method == GenerationMethod.TEXT_TO_3D:
                score += 20  # Meshy AI is good for text-to-3D
            elif provider == ServiceProvider.KAEDIM and request.generation_method == GenerationMethod.IMAGE_TO_3D:
                score += 20  # Kaedim is good for image-to-3D
            
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
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> Optional[GenerationResult]:
        """Try fallback services if the primary service fails."""
        
        for provider, service in self.services.items():
            if provider == failed_service:
                continue  # Skip the failed service
            
            if not service.can_handle_request(request):
                continue
            
            logger.info("Trying fallback service", 
                       fallback_service=provider,
                       failed_service=failed_service)
            
            try:
                result = await service.generate_3d_asset(request, progress_callback)
                if result.status == GenerationStatus.COMPLETED:
                    return result
            except Exception as e:
                logger.error("Fallback service failed", 
                           service=provider, 
                           error=str(e))
                continue
        
        return None
    
    async def _post_process_result(
        self, 
        result: GenerationResult, 
        request: GenerationRequest
    ) -> GenerationResult:
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
                converted_path = await self._convert_format(
                    result.model_file_path, 
                    request.output_format
                )
                if converted_path:
                    result.model_file_path = str(converted_path)
                    result.file_format = request.output_format
            
            # Optimize model if polygon count is too high
            if result.polygon_count and request.max_polygon_count:
                if result.polygon_count > request.max_polygon_count:
                    optimized_path = await self._optimize_model(
                        result.model_file_path,
                        request.max_polygon_count
                    )
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
    
    async def _generate_thumbnail(self, model_path: str) -> Optional[Path]:
        """Generate a thumbnail image for the 3D model."""
        try:
            # This would use a 3D rendering library like Open3D or PyOpenGL
            # For now, return a placeholder
            logger.info("Would generate thumbnail", model_path=model_path)
            return None
        except Exception as e:
            logger.error("Thumbnail generation failed", error=str(e))
            return None
    
    async def _convert_format(self, model_path: str, target_format: FileFormat) -> Optional[Path]:
        """Convert model to target format."""
        try:
            # This would use conversion libraries like Open3D, Assimp, or FBX SDK
            logger.info("Would convert format", 
                       source=model_path, 
                       target=target_format)
            return None
        except Exception as e:
            logger.error("Format conversion failed", error=str(e))
            return None
    
    async def _optimize_model(self, model_path: str, target_polygon_count: int) -> Optional[Path]:
        """Optimize model by reducing polygon count."""
        try:
            # This would use mesh simplification algorithms
            logger.info("Would optimize model", 
                       source=model_path, 
                       target_polygons=target_polygon_count)
            return None
        except Exception as e:
            logger.error("Model optimization failed", error=str(e))
            return None
    
    # Monitoring and Analytics Methods
    
    def get_service_health_status(self) -> Dict[ServiceProvider, ServiceStatus]:
        """Get current health status of all services."""
        return {
            provider: service._health_status 
            for provider, service in self.services.items()
        }
    
    def get_cost_tracking(self) -> Dict[str, Any]:
        """Get cost tracking information."""
        service_costs = {}
        for result in self.generation_history:
            if result.cost_usd:
                service = result.service_used
                if service not in service_costs:
                    service_costs[service] = {"total": 0.0, "count": 0}
                service_costs[service]["total"] += result.cost_usd
                service_costs[service]["count"] += 1
        
        return {
            "total_cost_usd": self.total_cost_tracking,
            "service_breakdown": service_costs,
            "total_generations": len(self.generation_history),
            "successful_generations": len([
                r for r in self.generation_history 
                if r.status == GenerationStatus.COMPLETED
            ])
        }
    
    def get_rate_limit_status(self) -> Dict[ServiceProvider, Dict[str, Any]]:
        """Get rate limit status for all services."""
        return {
            provider: {
                "requests_per_minute": service.rate_limit_info.requests_per_minute,
                "requests_per_day": service.rate_limit_info.requests_per_day,
                "can_make_request": service.rate_limit_info.can_make_request(service.config)
            }
            for provider, service in self.services.items()
        }


# Utility Functions

def create_default_configs() -> Dict[ServiceProvider, ServiceConfig]:
    """Create default service configurations."""
    return {
        ServiceProvider.MESHY_AI: ServiceConfig(
            api_key=os.getenv("MESHY_API_KEY", ""),
            base_url="https://api.meshy.ai",
            max_requests_per_minute=10,
            max_requests_per_day=100,
            timeout_seconds=600,
            cost_per_generation=0.50,
            supports_text_to_3d=True,
            supports_image_to_3d=True,
            supported_output_formats=[FileFormat.OBJ, FileFormat.GLTF, FileFormat.GLB],
            quality_levels=[QualityLevel.STANDARD, QualityLevel.HIGH, QualityLevel.ULTRA]
        ),
        ServiceProvider.KAEDIM: ServiceConfig(
            api_key=os.getenv("KAEDIM_API_KEY", ""),
            base_url="https://api.kaedim3d.com",
            max_requests_per_minute=5,
            max_requests_per_day=50,
            timeout_seconds=900,
            cost_per_generation=2.00,
            supports_text_to_3d=False,
            supports_image_to_3d=True,
            supported_output_formats=[FileFormat.OBJ, FileFormat.FBX],
            quality_levels=[QualityLevel.HIGH, QualityLevel.ULTRA]
        )
    }


# Export main classes and functions
__all__ = [
    'Asset3DGenerator',
    'GenerationRequest',
    'GenerationResult',
    'ProgressUpdate',
    'ServiceProvider',
    'ServiceConfig',
    'GenerationMethod',
    'ServiceStatus',
    'create_default_configs'
]

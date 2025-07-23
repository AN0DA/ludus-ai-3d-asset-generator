"""
Base integration for 3D generation services.

This module provides the BaseServiceIntegration class for 3D generation services.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from .configs import ServiceConfig
from .enums import ServiceStatus
from .models import GenerationRequest, GenerationResult, ProgressUpdate


class BaseServiceIntegration:
    """Base class for 3D generation service integrations."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.session: Any | None = None  # Will be initialized as needed
        self._health_status = ServiceStatus.AVAILABLE
        self._last_health_check = datetime.utcnow()

    async def __aenter__(self) -> "BaseServiceIntegration":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the service integration."""
        pass

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    async def check_health(self) -> ServiceStatus:
        """Check service health status."""
        return self._health_status

    def can_handle_request(self, request: GenerationRequest) -> bool:
        """Check if this service can handle the given request."""
        return True

    async def generate_3d_asset(
        self, request: GenerationRequest, progress_callback: Callable[[ProgressUpdate], None] | None = None
    ) -> GenerationResult:
        """Generate a 3D asset."""
        raise NotImplementedError("Subclasses must implement generate_3d_asset")

"""
Enums for asset generators.

This module contains enums for service providers, statuses, and generation methods.
"""

from enum import Enum


class ServiceProvider(str, Enum):
    """Available 3D generation service providers."""

    MESHY_AI = "meshy_ai"


class ServiceStatus(str, Enum):
    """Service availability status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class GenerationMethod(str, Enum):
    """3D generation methods."""

    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    HYBRID = "hybrid"  # Text + reference image

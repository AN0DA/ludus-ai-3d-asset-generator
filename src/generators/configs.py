"""
Configurations for asset generators.

This module contains configuration dataclasses for asset generation services.
"""

from dataclasses import dataclass, field

from src.models.asset_model import FileFormat, QualityLevel


@dataclass
class ServiceConfig:
    """Configuration for a 3D generation service."""

    api_key: str
    base_url: str
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    supports_text_to_3d: bool = True
    supports_image_to_3d: bool = False
    supported_output_formats: list[FileFormat] = field(default_factory=lambda: [FileFormat.OBJ])
    max_polygon_count: int = 300000  # Meshy AI supports up to 300,000 polygons
    quality_levels: list[QualityLevel] = field(default_factory=lambda: [QualityLevel.STANDARD])

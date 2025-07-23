"""
Models for asset generators.

This module contains Pydantic models for generation requests, results, and progress updates.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models.asset_model import AssetType, FileFormat, GenerationStatus, QualityLevel, StylePreference

from .enums import GenerationMethod, ServiceProvider


class GenerationRequest(BaseModel):
    """Request model for 3D asset generation."""

    # Required fields
    description: str = Field(..., min_length=10, max_length=2000)
    asset_type: AssetType

    # Optional fields
    style_preference: StylePreference | None = None
    quality_level: QualityLevel = QualityLevel.STANDARD
    output_format: FileFormat = FileFormat.OBJ
    reference_image_url: str | None = None
    reference_image_path: str | None = None

    # Generation parameters
    max_polygon_count: int | None = Field(None, ge=100, le=300000)
    generation_method: GenerationMethod = GenerationMethod.TEXT_TO_3D
    preferred_service: ServiceProvider | None = None

    # Metadata
    user_id: str | None = None
    session_id: str | None = None
    priority: int = Field(1, ge=1, le=10)


class GenerationResult(BaseModel):
    """Result model for 3D asset generation."""

    # Generation info
    request_id: str
    status: GenerationStatus
    service_used: ServiceProvider
    generation_method: GenerationMethod

    # Files and URLs
    model_url: str | None = None
    model_file_path: str | None = None
    thumbnail_url: str | None = None
    thumbnail_file_path: str | None = None

    # Metadata
    file_format: FileFormat | None = None
    file_size_bytes: int | None = None
    polygon_count: int | None = None
    generation_time_seconds: float | None = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Error information
    error_message: str | None = None
    error_code: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProgressUpdate(BaseModel):
    """Progress update for generation tracking."""

    request_id: str
    status: GenerationStatus
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    current_step: str
    estimated_completion_time: datetime | None = None
    message: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

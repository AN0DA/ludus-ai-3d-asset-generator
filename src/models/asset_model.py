"""
Pydantic data models for 3D asset generation application.

This module defines all the data models used throughout the asset generation
pipeline, from user input to final asset delivery.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, AnyUrl


# Enums for controlled vocabularies

class AssetType(str, Enum):
    """Supported asset types for generation."""
    WEAPON = "weapon"
    ARMOR = "armor"
    POTION = "potion"
    TOOL = "tool"
    ENVIRONMENT = "environment"


class StylePreference(str, Enum):
    """Visual style preferences for asset generation."""
    REALISTIC = "realistic"
    SCULPTURE = "sculpture"


class FileFormat(str, Enum):
    """Supported 3D file formats."""
    GLB = "glb"
    FBX = "fbx"
    OBJ = "obj"
    USDZ = "usdz"


class QualityLevel(str, Enum):
    """Quality levels for generation."""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"


class GenerationStatus(str, Enum):
    """Status of a generation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


# Core Data Models


class AssetMetadata(BaseModel):
    """Metadata for generated 3D asset."""

    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    original_description: str
    enhanced_description: dict[str, Any]
    asset_type: AssetType
    style_preferences: List[StylePreference]
    quality_level: QualityLevel
    generation_service: str
    session_id: str
    metadata: Dict[str, Any]

"""
Pydantic data models for 3D asset generation application.

This module defines all the data models used throughout the asset generation
pipeline, from user input to final asset delivery.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Annotated
from pathlib import Path

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import HttpUrl, AnyUrl


# Enums for controlled vocabularies

class AssetType(str, Enum):
    """Supported asset types for generation."""
    WEAPON = "weapon"
    ARMOR = "armor"
    POTION = "potion"
    TOOL = "tool"
    JEWELRY = "jewelry"
    FURNITURE = "furniture"
    BUILDING = "building"
    VEHICLE = "vehicle"
    CREATURE = "creature"
    ENVIRONMENT = "environment"
    PROP = "prop"
    OTHER = "other"


class StylePreference(str, Enum):
    """Visual style preferences for asset generation."""
    REALISTIC = "realistic"
    SCULPTURE = "sculpture"


class GenerationStatus(str, Enum):
    """Status of asset generation process."""
    PENDING = "pending"
    VALIDATING = "validating"
    ENHANCING_DESCRIPTION = "enhancing_description"
    PROCESSING = "processing"
    GENERATING_3D = "generating_3d"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileFormat(str, Enum):
    """Supported 3D file formats for Meshy AI."""
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


# Core Data Models

class AssetRequest(BaseModel):
    """User input model for asset generation request."""
    
    # Basic information
    description: str = Field(
        ..., 
        min_length=10, 
        max_length=1000,
        description="Detailed description of the asset to generate"
    )
    asset_type: AssetType = Field(
        default=AssetType.OTHER,
        description="Type of asset to generate"
    )
    
    # Style and preferences
    style_preferences: Annotated[List[StylePreference], Field(max_length=5)] = Field(
        default_factory=list,
        description="Visual style preferences for the asset"
    )
    
    # Technical requirements
    quality_level: QualityLevel = Field(
        default=QualityLevel.STANDARD,
        description="Desired quality level for generation"
    )
    preferred_formats: Annotated[List[FileFormat], Field(max_length=3)] = Field(
        default_factory=lambda: [FileFormat.GLB, FileFormat.OBJ],
        description="Preferred output file formats"
    )
    
    # Optional metadata
    reference_images: Optional[Annotated[List[HttpUrl], Field(max_length=5)]] = Field(
        default=None,
        description="Reference image URLs for inspiration"
    )
    additional_notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Additional notes or specific requirements"
    )
    
    # User information
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for tracking"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for tracking"
    )
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description content."""
        if not v.strip():
            raise ValueError("Description cannot be empty")
        
        # Check for potentially problematic content
        forbidden_words = ['nsfw', 'explicit', 'violent']
        if any(word in v.lower() for word in forbidden_words):
            raise ValueError("Description contains inappropriate content")
        
        return v.strip()
    
    @validator('style_preferences')
    def validate_style_preferences(cls, v):
        """Ensure no duplicate style preferences."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate style preferences not allowed")
        return v


class TechnicalSpecs(BaseModel):
    """Technical specifications for a 3D asset."""
    
    polygon_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of polygons in the model"
    )
    vertex_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of vertices in the model"
    )
    texture_resolution: Optional[int] = Field(
        default=None,
        ge=256,
        le=4096,
        description="Texture resolution in pixels (square)"
    )
    file_size_mb: Optional[float] = Field(
        default=None,
        ge=0,
        description="File size in megabytes"
    )
    bounding_box: Optional[Dict[str, float]] = Field(
        default=None,
        description="3D bounding box dimensions (x, y, z)"
    )
    materials_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of materials used"
    )
    has_animations: bool = Field(
        default=False,
        description="Whether the asset includes animations"
    )
    has_textures: bool = Field(
        default=True,
        description="Whether the asset includes textures"
    )
    
    @validator('bounding_box')
    def validate_bounding_box(cls, v):
        """Validate bounding box format."""
        if v is not None:
            required_keys = ['x', 'y', 'z']
            if not all(key in v for key in required_keys):
                raise ValueError("Bounding box must contain x, y, z dimensions")
            if any(val <= 0 for val in v.values()):
                raise ValueError("Bounding box dimensions must be positive")
        return v


class EnhancedDescription(BaseModel):
    """LLM enhanced description with detailed metadata."""
    
    # Enhanced content
    enhanced_description: str = Field(
        ...,
        min_length=50,
        max_length=2000,
        description="LLM-enhanced detailed description"
    )
    
    # Extracted attributes
    physical_properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physical properties (size, weight, material, etc.)"
    )
    visual_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Visual attributes (color, texture, style, etc.)"
    )
    functional_aspects: Dict[str, Any] = Field(
        default_factory=dict,
        description="Functional aspects and usage"
    )
    
    # Generation hints
    modeling_hints: List[str] = Field(
        default_factory=list,
        description="Hints for 3D modeling process"
    )
    texture_suggestions: List[str] = Field(
        default_factory=list,
        description="Texture and material suggestions"
    )
    
    # Metadata
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the enhancement (0-1)"
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time taken for enhancement in milliseconds"
    )
    llm_provider: Optional[str] = Field(
        default=None,
        description="LLM provider used for enhancement"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Specific LLM model used"
    )
    
    # Original request reference
    original_description: str = Field(
        ...,
        description="Original user description"
    )
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Ensure confidence score is reasonable."""
        if v < 0.1:
            raise ValueError("Confidence score too low, enhancement may be unreliable")
        return v


class CloudStorageInfo(BaseModel):
    """File storage details for cloud-stored assets."""
    
    # Storage location
    provider: str = Field(..., description="Storage provider (aws, minio, r2, etc.)")
    bucket: str = Field(..., description="Storage bucket name")
    key: str = Field(..., description="Object key/path in storage")
    region: Optional[str] = Field(default=None, description="Storage region")
    
    # Access information
    public_url: Optional[AnyUrl] = Field(
        default=None,
        description="Public URL for accessing the file"
    )
    presigned_url: Optional[AnyUrl] = Field(
        default=None,
        description="Temporary presigned URL for access"
    )
    presigned_url_expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration time for presigned URL"
    )
    
    # File information
    file_size: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )
    content_type: Optional[str] = Field(
        default=None,
        description="MIME content type"
    )
    etag: Optional[str] = Field(
        default=None,
        description="Entity tag for the file"
    )
    
    # Timestamps
    uploaded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Upload timestamp"
    )
    last_modified: Optional[datetime] = Field(
        default=None,
        description="Last modification timestamp"
    )
    
    # Metadata
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional storage metadata"
    )
    
    @validator('key')
    def validate_key(cls, v):
        """Validate storage key format."""
        if not v or v.startswith('/') or v.endswith('/'):
            raise ValueError("Storage key must not start or end with '/'")
        if '//' in v:
            raise ValueError("Storage key must not contain consecutive slashes")
        return v
    
    @validator('presigned_url_expires_at')
    def validate_presigned_expiry(cls, v, values):
        """Ensure presigned URL expiry is in the future."""
        if v is not None and v <= datetime.utcnow():
            raise ValueError("Presigned URL expiry must be in the future")
        return v


class AssetFile(BaseModel):
    """Information about a single asset file."""
    
    # File identification
    filename: str = Field(..., description="Original filename")
    file_format: FileFormat = Field(..., description="File format")
    file_path: Optional[str] = Field(
        default=None,
        description="Local file path (if stored locally)"
    )
    
    # File properties
    file_size: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )
    checksum: Optional[str] = Field(
        default=None,
        description="File checksum for integrity verification"
    )
    
    # Cloud storage info
    storage_info: Optional[CloudStorageInfo] = Field(
        default=None,
        description="Cloud storage information"
    )
    
    # Technical specs specific to this file
    technical_specs: Optional[TechnicalSpecs] = Field(
        default=None,
        description="Technical specifications for this file"
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="File creation timestamp"
    )
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename format."""
        if not v or '/' in v or '\\' in v:
            raise ValueError("Invalid filename format")
        
        # Check file extension
        allowed_extensions = [fmt.value for fmt in FileFormat]
        file_ext = Path(v).suffix[1:].lower()  # Remove the dot
        if file_ext not in allowed_extensions:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        return v


class GenerationProgress(BaseModel):
    """Detailed progress information for asset generation."""
    
    status: GenerationStatus = Field(
        default=GenerationStatus.PENDING,
        description="Current generation status"
    )
    progress_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)"
    )
    current_step: str = Field(
        default="",
        description="Description of current processing step"
    )
    
    # Timestamps
    started_at: Optional[datetime] = Field(
        default=None,
        description="Generation start timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp"
    )
    
    # Error information
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )
    
    # Processing details
    processing_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed log of processing steps"
    )
    estimated_completion_time: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )
    
    @validator('completed_at')
    def validate_completion_time(cls, v, values):
        """Ensure completion time is after start time."""
        started_at = values.get('started_at')
        if v is not None and started_at is not None and v < started_at:
            raise ValueError("Completion time cannot be before start time")
        return v


class AssetMetadata(BaseModel):
    """Complete metadata for a generated 3D asset."""
    
    # Identification
    asset_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique asset identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Asset name"
    )
    
    # Descriptions
    original_description: str = Field(
        ...,
        description="Original user description"
    )
    enhanced_description: Optional[EnhancedDescription] = Field(
        default=None,
        description="LLM-enhanced description with metadata"
    )
    
    # Asset properties
    asset_type: AssetType = Field(
        ...,
        description="Type of the generated asset"
    )
    style_preferences: List[StylePreference] = Field(
        default_factory=list,
        description="Applied style preferences"
    )
    quality_level: QualityLevel = Field(
        default=QualityLevel.STANDARD,
        description="Quality level of the generated asset"
    )
    
    # Files and storage
    files: List[AssetFile] = Field(
        default_factory=list,
        description="List of generated asset files"
    )
    thumbnail_url: Optional[AnyUrl] = Field(
        default=None,
        description="URL to asset thumbnail/preview"
    )
    
    # Technical information
    technical_specs: Optional[TechnicalSpecs] = Field(
        default=None,
        description="Overall technical specifications"
    )
    
    # Generation information
    generation_progress: GenerationProgress = Field(
        default_factory=GenerationProgress,
        description="Generation progress and status"
    )
    generation_service: Optional[str] = Field(
        default=None,
        description="3D generation service used (meshy, kaedim, etc.)"
    )
    generation_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for generation"
    )
    
    # User and session information
    user_id: Optional[str] = Field(
        default=None,
        description="User who requested the asset"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session in which asset was requested"
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Asset creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    # Additional metadata
    tags: Annotated[List[str], Field(max_length=20)] = Field(
        default_factory=list,
        description="Tags for categorization and search"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate asset name."""
        if not v.strip():
            raise ValueError("Asset name cannot be empty")
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format."""
        # Remove duplicates and empty tags
        cleaned_tags = list(set(tag.strip().lower() for tag in v if tag.strip()))
        return cleaned_tags
    
    def get_primary_file(self, format_preference: Optional[FileFormat] = None) -> Optional[AssetFile]:
        """Get the primary asset file, optionally filtered by format."""
        if not self.files:
            return None
        
        if format_preference:
            for file in self.files:
                if file.file_format == format_preference:
                    return file
        
        # Return first file if no preference or preference not found
        return self.files[0]
    
    def get_public_urls(self) -> Dict[str, str]:
        """Get public URLs for all files."""
        urls = {}
        for file in self.files:
            if file.storage_info and file.storage_info.public_url:
                urls[file.file_format.value] = str(file.storage_info.public_url)
        return urls


class AssetResponse(BaseModel):
    """Complete response model for frontend."""
    
    # Core asset information
    asset: AssetMetadata = Field(
        ...,
        description="Complete asset metadata"
    )
    
    # Status information
    status: GenerationStatus = Field(
        ...,
        description="Current generation status"
    )
    success: bool = Field(
        ...,
        description="Whether the generation was successful"
    )
    
    # Download information
    download_urls: Dict[str, str] = Field(
        default_factory=dict,
        description="Download URLs for each file format"
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="Thumbnail URL for preview"
    )
    
    # Error information (if applicable)
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed error information"
    )
    
    # Processing information
    processing_time_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total processing time in seconds"
    )
    estimated_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated cost for generation"
    )
    
    # Response metadata
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    response_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp"
    )
    api_version: str = Field(
        default="1.0",
        description="API version used"
    )
    
    @root_validator(pre=False, skip_on_failure=True)
    def validate_response_consistency(cls, values):
        """Validate response consistency."""
        success = values.get('success')
        error_message = values.get('error_message')
        asset = values.get('asset')
        
        # If not successful, should have error message
        if not success and not error_message:
            raise ValueError("Unsuccessful response must include error message")
        
        # If successful, should not have error message
        if success and error_message:
            raise ValueError("Successful response should not include error message")
        
        # If successful, asset should be completed or at least processed
        if success and asset:
            status = asset.generation_progress.status
            if status in [GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                raise ValueError("Successful response cannot have failed/cancelled asset")
        
        return values
    
    @classmethod
    def from_asset_metadata(
        cls, 
        asset_metadata: AssetMetadata,
        processing_time: Optional[float] = None,
        estimated_cost: Optional[float] = None
    ) -> "AssetResponse":
        """Create response from asset metadata."""
        status = asset_metadata.generation_progress.status
        success = status == GenerationStatus.COMPLETED
        
        return cls(
            asset=asset_metadata,
            status=status,
            success=success,
            download_urls=asset_metadata.get_public_urls(),
            thumbnail_url=str(asset_metadata.thumbnail_url) if asset_metadata.thumbnail_url else None,
            error_message=asset_metadata.generation_progress.error_message if not success else None,
            processing_time_seconds=processing_time,
            estimated_cost=estimated_cost
        )


# Utility Models

class AssetFilter(BaseModel):
    """Filter parameters for asset queries."""
    
    asset_type: Optional[AssetType] = None
    style_preferences: Optional[List[StylePreference]] = None
    quality_level: Optional[QualityLevel] = None
    status: Optional[GenerationStatus] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class AssetListResponse(BaseModel):
    """Response model for asset listing endpoints."""
    
    assets: List[AssetMetadata] = Field(default_factory=list)
    total_count: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    has_next: bool = False
    has_previous: bool = False


# Export all models
__all__ = [
    "AssetType",
    "StylePreference", 
    "GenerationStatus",
    "FileFormat",
    "QualityLevel",
    "AssetRequest",
    "TechnicalSpecs",
    "EnhancedDescription",
    "CloudStorageInfo",
    "AssetFile",
    "GenerationProgress",
    "AssetMetadata",
    "AssetResponse",
    "AssetFilter",
    "AssetListResponse",
]

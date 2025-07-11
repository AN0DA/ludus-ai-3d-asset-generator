"""
Core application for the AI 3D Asset Generator.

This module contains the main application logic, generator initialization,
and the complete asset generation pipeline.
"""

import json
import os
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.core.cache_manager import CacheManager
from src.core.session_manager import SessionManager
from src.core.task_manager import TaskManager
from src.generators.asset_generator import (
    Asset3DGenerator,
    GenerationRequest,
    ServiceConfig,
    ServiceProvider,
)
from src.generators.llm_generator import LLMConfig, LLMGenerator
from src.models.asset_model import AssetMetadata, AssetType, FileFormat, GenerationStatus, QualityLevel, StylePreference
from src.storage.cloud_storage import StorageConfig, StorageProvider
from src.storage.s3_storage import S3Storage
from src.utils.env_config import get_settings
from src.utils.validators import ValidationException

logger = structlog.get_logger(__name__)


class AssetGenerationApp:
    """Main application class for the AI 3D Asset Generator."""

    def __init__(self, config_path: Path | None = None):
        """Initialize the application."""
        try:
            # Load configuration from environment variables
            self.settings = get_settings()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Config loading failed, using defaults: {e}")
            self.settings = None

        # Initialize directories
        self.temp_dir = Path(tempfile.gettempdir()) / "asset_generator"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize managers
        self.task_manager = TaskManager()
        self.session_manager = SessionManager(self.temp_dir)
        self.cache_manager = CacheManager(self.temp_dir / "cache")

        # Initialize generators and storage
        self.llm_generator: LLMGenerator | None = None
        self.asset_generator: Asset3DGenerator | None = None
        self.cloud_storage: S3Storage | None = None

        # Application state
        self.is_initialized = False
        self.generation_queue: list[str] = []
        self.active_generations: dict[str, str] = {}  # task_id -> session_id

    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        if self.is_initialized:
            return

        try:
            logger.info("Initializing Asset Generation App")

            # Initialize LLM generator
            try:
                llm_config = self._create_llm_config()
                if llm_config.api_key:  # Only initialize if API key is available
                    self.llm_generator = LLMGenerator(llm_config)
                    logger.info("LLM generator initialized successfully")
                else:
                    logger.warning("No LLM API key provided, LLM features will be disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM generator: {e}")
                self.llm_generator = None

            # Initialize 3D asset generator
            try:
                service_configs = self._create_service_configs()
                logger.info(f"Service configs created: {list(service_configs.keys()) if service_configs else 'None'}")
                if service_configs:
                    self.asset_generator = Asset3DGenerator(service_configs)
                    await self.asset_generator.initialize()
                    logger.info("3D asset generator initialized successfully")
                else:
                    logger.warning("No 3D generation services configured")
            except Exception as e:
                logger.warning(f"Failed to initialize 3D asset generator: {e}")
                self.asset_generator = None

            # Initialize cloud storage
            try:
                storage_config = self._create_storage_config()
                if storage_config.access_key_id and storage_config.secret_access_key:
                    self.cloud_storage = S3Storage(storage_config)
                    # IMPORTANT: Call connect() to initialize the S3 client
                    await self.cloud_storage.connect()
                    logger.info("Cloud storage initialized and connected successfully")
                else:
                    logger.warning("No cloud storage credentials provided, file storage will be local only")
                    self.cloud_storage = None
            except Exception as e:
                logger.warning(f"Failed to initialize cloud storage: {e}")
                self.cloud_storage = None

            # Start background tasks
            self.task_manager.start_cleanup()

            self.is_initialized = True
            logger.info("App initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize app: {e}")
            raise

    def _create_llm_config(self) -> LLMConfig:
        """Create LLM configuration from environment variables."""
        if self.settings:
            llm_config = self.settings.get_llm_config()
            return LLMConfig(
                api_key=llm_config["api_key"] or "demo",
                model=llm_config["model"],
                base_url=llm_config["base_url"] or "https://api.openai.com/v1",
                max_tokens=llm_config["max_tokens"],
                temperature=llm_config["temperature"],
                timeout=llm_config["timeout"],
                max_retries=llm_config["max_retries"],
            )
        else:
            # Fallback to environment variables
            return LLMConfig(
                api_key=os.getenv("LLM_API_KEY", "demo"),
                model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                timeout=int(os.getenv("LLM_TIMEOUT", "60")),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            )

    def _create_service_configs(self) -> dict[ServiceProvider, ServiceConfig]:
        """Create 3D generation service configurations."""
        configs = {}

        if self.settings:
            # Use settings for service configuration
            meshy_key = self.settings.meshy_api_key
        else:
            # Fallback to environment variables
            meshy_key = os.getenv("MESHY_API_KEY")

        logger.info(f"MESHY_API_KEY found: {bool(meshy_key)}")
        if meshy_key:
            configs[ServiceProvider.MESHY_AI] = ServiceConfig(
                api_key=meshy_key,
                base_url="https://api.meshy.ai",
                timeout_seconds=600,
                supports_text_to_3d=True,
                supports_image_to_3d=True,
                supported_output_formats=[FileFormat.GLB, FileFormat.FBX, FileFormat.OBJ, FileFormat.USDZ],
                quality_levels=[QualityLevel.DRAFT, QualityLevel.STANDARD, QualityLevel.HIGH, QualityLevel.ULTRA],
            )
            logger.info("Meshy service config created with full capabilities")

        return configs

    def _create_storage_config(self) -> StorageConfig:
        """Create storage configuration from app settings."""
        if self.settings:
            storage_config = self.settings.get_storage_config()
            return StorageConfig(
                provider=StorageProvider.S3_COMPATIBLE,
                bucket_name=storage_config["bucket_name"],
                region=storage_config["region"],
                access_key_id=storage_config["access_key_id"],
                secret_access_key=storage_config["secret_access_key"],
                endpoint_url=storage_config["endpoint_url"],
            )
        else:
            # Use environment variables as fallback
            return StorageConfig(
                provider=StorageProvider.S3_COMPATIBLE,
                bucket_name=os.getenv("STORAGE_BUCKET_NAME", "ai-3d-assets"),
                region=os.getenv("STORAGE_REGION", "us-east-1"),
                access_key_id=os.getenv("STORAGE_ACCESS_KEY_ID", ""),
                secret_access_key=os.getenv("STORAGE_SECRET_ACCESS_KEY", ""),
                endpoint_url=os.getenv("STORAGE_ENDPOINT_URL"),
            )

    def _get_content_type_for_format(self, file_format: FileFormat) -> str:
        """Get MIME content type for a file format."""
        content_type_map = {
            FileFormat.GLB: "model/gltf-binary",
            FileFormat.FBX: "application/octet-stream",
            FileFormat.OBJ: "model/obj",
            FileFormat.USDZ: "model/usd",
        }
        return content_type_map.get(file_format, "application/octet-stream")

    async def generate_asset_pipeline(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None = None,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        session_id: str | None = None,
        progress_callback: Callable | None = None,
    ) -> tuple[str, str]:  # Returns (task_id, session_id)
        """
        Main asset generation pipeline.

        Returns task_id and session_id for tracking progress.
        """
        # Create or get session
        if session_id is None:
            session_id = self.session_manager.create_session()

        session = self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError("Invalid or expired session")

        # Create generation task with a pre-allocated task_id
        task_id = str(uuid.uuid4())

        async def task_wrapper():
            return await self._execute_generation_pipeline(
                description=description,
                asset_type=asset_type,
                style_preference=style_preference,
                quality_level=quality_level,
                session_id=session_id,
                progress_callback=progress_callback,
                task_id=task_id,
            )

        # Create and start the task
        actual_task_id = self.task_manager.create_task(task_wrapper(), task_id)

        self.active_generations[actual_task_id] = session_id
        return actual_task_id, session_id

    async def _execute_generation_pipeline(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None,
        quality_level: QualityLevel,
        session_id: str,
        progress_callback: Callable | None = None,
        task_id: str | None = None,
    ) -> AssetMetadata:
        """Execute the complete asset generation pipeline."""

        def update_progress(step: str, progress: float, message: str):
            if progress_callback:
                progress_callback(step, progress, message)
            # Update task manager status
            if task_id:
                self.task_manager.update_task_status(
                    task_id, status="in_progress", progress=progress, message=message, current_step=step
                )

        try:
            update_progress("validation", 0.05, "Validating input parameters")

            # Validate input
            if len(description.strip()) < 10:
                raise ValidationException("Description must be at least 10 characters")

            # Step 1: Enhance description with LLM
            update_progress("llm_enhancement", 0.15, "Enhancing description with AI")

            if self.llm_generator:
                try:
                    llm_result = await self.llm_generator.generate(
                        prompt=description,
                        asset_type=asset_type,
                        style_preferences=[style_preference] if style_preference else None,
                        quality_level=quality_level,
                    )

                    if llm_result.status == GenerationStatus.COMPLETED and llm_result.data:
                        enhanced_asset_data = llm_result.data.get("enhanced_asset", {})
                        enhanced_description = {
                            "enhanced_description": enhanced_asset_data.get(
                                "enhanced_description", f"Enhanced version of: {description}"
                            ),
                            "asset_name": enhanced_asset_data.get("asset_name", f"{asset_type.value} asset"),
                            "materials": enhanced_asset_data.get(
                                "materials", ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"]
                            ),
                            "style_notes": enhanced_asset_data.get(
                                "style_notes", [style_preference.value] if style_preference else ["generic"]
                            ),
                        }
                    else:
                        # Fallback if LLM fails
                        enhanced_description = {
                            "enhanced_description": f"Enhanced version of: {description}",
                            "asset_name": f"{asset_type.value} asset",
                            "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
                            "style_notes": [style_preference.value] if style_preference else ["generic"],
                        }
                        error_msg = llm_result.error.message if llm_result.error else "Unknown error"
                        logger.warning("LLM generation failed, using fallback description", error=error_msg)

                except Exception as e:
                    logger.error(f"LLM enhancement failed: {e}")
                    # Use fallback description
                    enhanced_description = {
                        "enhanced_description": f"Enhanced version of: {description}",
                        "asset_name": f"{asset_type.value} asset",
                        "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
                        "style_notes": [style_preference.value] if style_preference else ["generic"],
                    }
            else:
                # No LLM generator available, use basic enhancement
                enhanced_description = {
                    "enhanced_description": f"Enhanced version of: {description}",
                    "asset_name": f"{asset_type.value} asset",
                    "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
                    "style_notes": [style_preference.value] if style_preference else ["generic"],
                }

            update_progress("llm_enhancement", 0.25, "Description enhanced successfully")

            # Step 2: Generate 3D asset
            update_progress("asset_generation", 0.35, "Generating 3D model")

            # Prepare description for 3D generation - truncate if needed for service limits
            generation_description = enhanced_description["enhanced_description"]

            model_file_path = None
            if self.asset_generator:
                try:
                    # Create progress callback for 3D generation
                    def asset_progress_callback(progress_update):
                        # Convert 3D generation progress to overall progress (35% to 65% range)
                        overall_progress = 0.35 + (progress_update.progress_percentage * 0.30 / 100.0)
                        update_progress(
                            "asset_generation",
                            overall_progress,
                            progress_update.message or progress_update.current_step,
                        )

                    # Meshy AI has a 600 character limit - truncate smartly if needed
                    if len(generation_description) > 590:  # Leave some margin
                        # Try to truncate at sentence boundary
                        sentences = generation_description.split(". ")
                        truncated_description = ""
                        for sentence in sentences:
                            if len(truncated_description + sentence + ". ") <= 590:
                                truncated_description += sentence + ". "
                            else:
                                break

                        # If no complete sentences fit, just truncate at word boundary
                        if not truncated_description.strip():
                            words = generation_description.split()
                            truncated_description = ""
                            for word in words:
                                if len(truncated_description + word + " ") <= 590:
                                    truncated_description += word + " "
                                else:
                                    break

                        # Fallback to hard truncation if needed
                        if len(truncated_description) > 590:
                            truncated_description = truncated_description[:587] + "..."

                        generation_description = truncated_description.strip()
                        logger.info(
                            f"Truncated description from {len(enhanced_description['enhanced_description'])} to {len(generation_description)} characters"
                        )

                    # Create generation request
                    generation_request = GenerationRequest(
                        description=generation_description,
                        asset_type=asset_type,
                        style_preference=style_preference,
                        quality_level=quality_level,
                        session_id=session_id,
                        output_format=FileFormat.OBJ,  # Default to OBJ format
                        max_polygon_count=None,  # Use service default
                        priority=1,  # Standard priority
                    )

                    # Generate the 3D asset
                    asset_result = await self.asset_generator.generate_asset(
                        request=generation_request, progress_callback=asset_progress_callback
                    )

                    if asset_result.status == GenerationStatus.COMPLETED:
                        generation_result = {
                            "status": GenerationStatus.COMPLETED,
                            "file_format": asset_result.file_format or FileFormat.OBJ,
                            "file_size": asset_result.file_size_bytes or 0,
                            "polygon_count": asset_result.polygon_count or 0,
                            "generation_time": asset_result.generation_time_seconds or 0.0,
                        }

                        # Get the generated file path if available
                        model_file_path = asset_result.model_file_path

                    else:
                        # Fallback if 3D generation fails
                        generation_result = {
                            "status": GenerationStatus.FAILED,
                            "file_format": FileFormat.OBJ,
                            "file_size": 0,
                            "polygon_count": 0,
                            "generation_time": 0.0,
                            "error": asset_result.error_message or "Unknown error",
                        }
                        logger.warning("3D asset generation failed", error=asset_result.error_message)

                except Exception as e:
                    logger.error(f"3D asset generation failed: {e}")
                    generation_result = {
                        "status": GenerationStatus.FAILED,
                        "file_format": FileFormat.OBJ,
                        "file_size": 0,
                        "polygon_count": 0,
                        "generation_time": 0.0,
                        "error": str(e),
                    }
            else:
                # No 3D generator available, create placeholder result
                generation_result = {
                    "status": GenerationStatus.FAILED,
                    "file_format": FileFormat.OBJ,
                    "file_size": 0,
                    "polygon_count": 0,
                    "generation_time": 0.0,
                    "error": "No 3D generator configured",
                }
                logger.warning("No 3D asset generator available")

            update_progress("asset_generation", 0.65, "3D model generation completed")

            # Step 3: Upload to cloud storage
            update_progress("cloud_upload", 0.75, "Uploading to cloud storage")

            # Generate unique filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{asset_type.value}_{timestamp}_{str(uuid.uuid4())[:8]}"

            model_url = None
            metadata_url = None

            if self.cloud_storage and generation_result["status"] == GenerationStatus.COMPLETED and model_file_path:
                try:
                    # Upload the generated model file
                    model_key = f"assets/{filename}.{generation_result['file_format'].value.lower()}"

                    def upload_progress_callback(progress):
                        # progress is an UploadProgress object with bytes_uploaded, total_bytes, percentage
                        progress_pct = progress.percentage / 100.0 if progress.percentage else 0
                        # Map upload progress to 75% to 85% overall progress
                        overall_progress = 0.75 + (progress_pct * 0.10)
                        update_progress(
                            "cloud_upload", overall_progress, f"Uploading model file ({int(progress.percentage)}%)"
                        )

                    model_info = await self.cloud_storage.upload_file(
                        file_path=model_file_path,
                        key=model_key,
                        content_type=self._get_content_type_for_format(generation_result["file_format"]),
                        metadata={
                            "asset_type": asset_type.value,
                            "session_id": session_id,
                            "generation_time": str(generation_result["generation_time"]),
                            "polygon_count": str(generation_result["polygon_count"]),
                        },
                        progress_callback=upload_progress_callback,
                    )
                    model_url = model_info.public_url

                    # Upload metadata
                    metadata_content = {
                        "original_description": description,
                        "enhanced_description": enhanced_description,
                        "generation_description": generation_description,
                        "generation_request": {
                            "description": description,
                            "asset_type": asset_type.value,
                            "style_preference": style_preference.value if style_preference else None,
                            "quality_level": quality_level.value,
                        },
                        "generation_result": generation_result,
                        "generated_at": datetime.utcnow().isoformat(),
                        "session_id": session_id,
                        "model_url": model_url,
                    }

                    # Create temporary metadata file
                    metadata_key = f"metadata/{filename}.json"
                    temp_metadata_path = self.temp_dir / f"metadata_{filename}.json"

                    with open(temp_metadata_path, "w") as f:
                        json.dump(metadata_content, f, indent=2, default=str)

                    try:
                        metadata_info = await self.cloud_storage.upload_file(
                            file_path=temp_metadata_path,
                            key=metadata_key,
                            content_type="application/json",
                            metadata={"asset_type": asset_type.value, "session_id": session_id},
                        )
                        metadata_url = metadata_info.public_url
                    finally:
                        # Clean up temporary file
                        if temp_metadata_path.exists():
                            temp_metadata_path.unlink()

                    update_progress("cloud_upload", 0.85, "Files uploaded successfully")

                except Exception as e:
                    logger.error(f"Cloud storage upload failed: {e}")
                    # Create fallback URLs for local development
                    model_url = f"file://{model_file_path}" if model_file_path else None
                    metadata_url = None
                    update_progress("cloud_upload", 0.85, "Upload failed, using local files")

            else:
                # No cloud storage or no file to upload
                if not self.cloud_storage:
                    logger.warning("No cloud storage configured")
                elif generation_result["status"] != GenerationStatus.COMPLETED:
                    logger.warning("Skipping upload due to failed generation")
                elif not model_file_path:
                    logger.warning("No model file to upload")

                # Create fallback URLs
                model_url = f"file://{model_file_path}" if model_file_path else None
                metadata_url = None
                update_progress("cloud_upload", 0.85, "Skipped cloud upload")

            # Step 4: Create asset metadata
            service_used = "local" if not self.cloud_storage else "integrated"
            if generation_result["status"] == GenerationStatus.COMPLETED:
                service_used = "llm+3d+storage"
            elif self.llm_generator and not self.asset_generator:
                service_used = "llm_only"
            elif self.asset_generator and not self.llm_generator:
                service_used = "3d_only"

            asset_metadata = AssetMetadata(
                asset_id=str(uuid.uuid4()),
                name=enhanced_description.get("asset_name", f"Generated {asset_type.value}"),
                original_description=description,
                asset_type=asset_type,
                style_preferences=[style_preference] if style_preference else [],
                quality_level=quality_level,
                generation_service=service_used,
                session_id=session_id,
                metadata={
                    "model_url": model_url,
                    "metadata_url": metadata_url,
                    "enhanced_description": enhanced_description,
                    "generation_description": generation_description,
                    "generation_result": generation_result,
                    "model_file_path": model_file_path,
                },
            )

            # Add to session history
            self.session_manager.add_to_session_history(session_id, asset_metadata.dict())

            update_progress("completed", 1.0, "Asset generation completed successfully")

            # Mark task as completed
            if task_id:
                self.task_manager.update_task_status(
                    task_id,
                    status="completed",
                    progress=1.0,
                    message="Asset generation completed successfully",
                    result=asset_metadata,
                )

            logger.info(f"Asset generation completed successfully: {asset_metadata.asset_id}")
            return asset_metadata

        except Exception as e:
            error_message = f"Generation failed: {str(e)}"
            logger.error(error_message, error=str(e))
            update_progress("error", 0.0, error_message)

            # Mark task as failed
            if task_id:
                self.task_manager.update_task_status(
                    task_id, status="failed", progress=0.0, message=error_message, error=error_message
                )
            raise

    def get_generation_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a generation task."""
        return self.task_manager.get_task_status(task_id)

    def cancel_generation(self, task_id: str) -> bool:
        """Cancel a running generation task."""
        success = self.task_manager.cancel_task(task_id)
        if success and task_id in self.active_generations:
            del self.active_generations[task_id]
        return success

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the generation history for a session."""
        return self.session_manager.get_session_history(session_id)

    async def shutdown(self) -> None:
        """Shutdown the application and clean up resources."""
        logger.info("Shutting down Asset Generation App")

        # Shutdown task manager
        await self.task_manager.shutdown()

        # Clear active generations
        self.active_generations.clear()

        # Clean up generators and storage
        self.llm_generator = None
        self.asset_generator = None

        # Properly disconnect cloud storage
        if self.cloud_storage:
            try:
                await self.cloud_storage.disconnect()
                logger.info("Cloud storage disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting cloud storage: {e}")
        self.cloud_storage = None

        logger.info("Application shutdown completed")

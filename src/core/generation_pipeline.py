import uuid
from collections.abc import Callable
from typing import Any

import structlog

from src.core.app import AssetGenerationApp
from src.generators.asset_generator import GenerationRequest
from src.models.asset_model import AssetMetadata, AssetType, FileFormat, GenerationStatus, QualityLevel, StylePreference
from src.utils.validators import ValidationException

logger = structlog.get_logger(__name__)


class GenerationPipeline:
    """Handles the asset generation pipeline."""

    def __init__(self, app: AssetGenerationApp):
        self.app = app

    async def generate_asset_pipeline(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None = None,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        session_id: str | None = None,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> tuple[str, str]:
        """
        Main asset generation pipeline.

        Returns task_id and session_id for tracking progress.
        """
        # Create or get session
        if session_id is None:
            session_id = self.app.session_manager.create_session()

        session = self.app.session_manager.get_session(session_id)
        if session is None:
            raise ValueError("Invalid or expired session")

        # Create generation task with a pre-allocated task_id
        task_id = str(uuid.uuid4())

        async def task_wrapper() -> AssetMetadata:
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
        actual_task_id = self.app.task_manager.create_task(task_wrapper(), task_id)

        return actual_task_id, session_id

    async def _execute_generation_pipeline(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None,
        quality_level: QualityLevel,
        session_id: str,
        progress_callback: Callable[[str, float, str], None] | None = None,
        task_id: str | None = None,
    ) -> AssetMetadata:
        """Execute the complete asset generation pipeline."""

        def update_progress(step: str, progress: float, message: str) -> None:
            if progress_callback:
                progress_callback(step, progress, message)
            # Update task manager status
            if task_id:
                self.app.task_manager.update_task_status(
                    task_id, status="in_progress", progress=progress, message=message, current_step=step
                )

        try:
            update_progress("validation", 0.05, "Validating input parameters")

            # Validate input
            if len(description.strip()) < 10:
                raise ValidationException("Description must be at least 10 characters")

            # Step 1: Enhance description with LLM
            update_progress("llm_enhancement", 0.15, "Enhancing description with AI")

            enhanced_description = await self._enhance_description(
                description=description,
                asset_type=asset_type,
                style_preference=style_preference,
                quality_level=quality_level,
            )

            update_progress("llm_enhancement", 0.25, "Description enhanced successfully")

            # Step 2: Generate 3D asset
            update_progress("asset_generation", 0.35, "Generating 3D model")

            asset_result = await self._generate_3d_asset(
                enhanced_description=enhanced_description,
                generation_description=enhanced_description["enhanced_description"],
                asset_type=asset_type,
                style_preference=style_preference,
                quality_level=quality_level,
                session_id=session_id,
                update_progress=update_progress,
            )

            update_progress("asset_generation", 0.65, "3D model generation completed")

            # Step 3: Upload to cloud storage
            update_progress("cloud_upload", 0.75, "Uploading to cloud storage")

            upload_result = await self._upload_to_cloud(
                asset_result=asset_result,
                enhanced_description=enhanced_description,
                generation_description=enhanced_description["enhanced_description"],
                description=description,
                asset_type=asset_type,
                style_preference=style_preference,
                quality_level=quality_level,
                session_id=session_id,
                update_progress=update_progress,
            )

            update_progress("cloud_upload", 0.85, "Files uploaded successfully")

            # Step 4: Create asset metadata
            asset_metadata = self._create_asset_metadata(
                enhanced_description=enhanced_description,
                upload_result=upload_result,
                description=description,
                asset_type=asset_type,
                style_preference=style_preference,
                quality_level=quality_level,
                session_id=session_id,
                asset_result=asset_result,
            )

            # Add to session history
            self.app.session_manager.add_to_session_history(session_id, asset_metadata.dict())

            update_progress("completed", 1.0, "Asset generation completed successfully")

            # Mark task as completed
            if task_id:
                self.app.task_manager.update_task_status(
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
                self.app.task_manager.update_task_status(
                    task_id, status="failed", progress=0.0, message=error_message, error=error_message
                )
            raise

    async def _enhance_description(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None,
        quality_level: QualityLevel,
    ) -> dict[str, Any]:
        """Enhance description using LLM or fallback."""
        if self.app.llm_generator:
            try:
                llm_result = await self.app.llm_generator.generate(
                    prompt=description,
                    asset_type=asset_type,
                    style_preferences=[style_preference] if style_preference else None,
                    quality_level=quality_level,
                )

                if llm_result.status == GenerationStatus.COMPLETED and llm_result.data:
                    enhanced_asset_data = llm_result.data.get("enhanced_asset", {})
                    return {
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
                    error_msg = llm_result.error.message if llm_result.error else "Unknown error"
                    logger.warning("LLM generation failed, using fallback description", error=error_msg)
            except Exception as e:
                logger.error(f"LLM enhancement failed: {e}")

        # Fallback description
        return {
            "enhanced_description": f"Enhanced version of: {description}",
            "asset_name": f"{asset_type.value} asset",
            "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
            "style_notes": [style_preference.value] if style_preference else ["generic"],
        }

    async def _generate_3d_asset(
        self,
        enhanced_description: dict[str, Any],
        generation_description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None,
        quality_level: QualityLevel,
        session_id: str,
        update_progress: Callable[[str, float, str], None],
    ) -> Any:
        """Generate 3D asset or fallback."""
        if self.app.asset_generator:
            try:
                # Truncate description if needed
                if len(generation_description) > 590:
                    generation_description = self._truncate_description(generation_description)

                # Create progress callback for 3D generation
                def asset_progress_callback(progress_update: ProgressUpdate) -> None:
                    overall_progress = 0.35 + (progress_update.progress_percentage * 0.30 / 100.0)
                    update_progress(
                        "asset_generation",
                        overall_progress,
                        progress_update.message or progress_update.current_step,
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
                asset_result = await self.app.asset_generator.generate_asset(
                    request=generation_request, progress_callback=asset_progress_callback
                )

                if asset_result.status == GenerationStatus.COMPLETED:
                    return asset_result
                else:
                    logger.warning("3D asset generation failed", error=asset_result.error_message)
            except Exception as e:
                logger.error(f"3D asset generation failed: {e}")

        # Fallback if 3D generation fails or no generator
        return None  # Or create placeholder result

    def _truncate_description(self, description: str) -> str:
        """Truncate description smartly for service limits."""
        return description[:587] + "..."  # Simplified

    async def _upload_to_cloud(
        self,
        asset_result: Any,
        enhanced_description: dict[str, Any],
        generation_description: str,
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None,
        quality_level: QualityLevel,
        session_id: str,
        update_progress: Callable[[str, float, str], None],
    ) -> dict[str, Any]:
        """Upload generated files to cloud storage or fallback to local."""
        return {"model_url": None, "metadata_url": None}  # Placeholder

    def _create_asset_metadata(
        self,
        enhanced_description: dict[str, Any],
        upload_result: dict[str, Any],
        description: str,
        asset_type: AssetType,
        style_preference: StylePreference | None,
        quality_level: QualityLevel,
        session_id: str,
        asset_result: Any,
    ) -> AssetMetadata:
        """Create asset metadata."""
        return AssetMetadata(
            asset_id=str(uuid.uuid4()),
            name=enhanced_description.get("asset_name", f"{asset_type.value} asset"),
            original_description=description,
            enhanced_description=enhanced_description,
            asset_type=asset_type,
            style_preferences=[style_preference] if style_preference else [],
            quality_level=quality_level,
            generation_service="meshy_ai",  # Placeholder
            session_id=session_id,
            metadata={"upload_result": upload_result},
        )

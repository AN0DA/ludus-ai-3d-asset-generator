import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from src.core.cache_manager import CacheManager
from src.core.generation_pipeline import GenerationPipeline
from src.core.session_manager import SessionManager
from src.core.task_manager import TaskManager
from src.factories.asset_generator_factory import create_asset_generator
from src.factories.llm_factory import create_llm_generator
from src.factories.storage_factory import create_storage
from src.generators.asset_generator import Asset3DGenerator
from src.generators.llm_generator import LLMGenerator
from src.models.asset_model import AssetType, QualityLevel, StylePreference
from src.storage.s3_storage import S3Storage
from src.utils.env_config import AppSettings, get_settings

logger = structlog.get_logger(__name__)


class AssetGenerationApp:
    """Main application class for the AI 3D Asset Generator."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the application."""
        try:
            # Load configuration from environment variables
            self.settings: AppSettings = get_settings()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Config loading failed, using defaults: {e}")
            self.settings = AppSettings()  # Default instance

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
        self.pipeline: GenerationPipeline | None = None

        # Application state
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        if self.is_initialized:
            return

        try:
            logger.info("Initializing Asset Generation App")

            # Initialize LLM generator
            try:
                self.llm_generator = create_llm_generator(self.settings)
                if self.llm_generator:
                    logger.info("LLM generator initialized successfully")
                else:
                    logger.warning("No LLM API key provided, LLM features will be disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM generator: {e}")
                self.llm_generator = None

            # Initialize 3D asset generator
            try:
                self.asset_generator = create_asset_generator(self.settings)
                if self.asset_generator:
                    await self.asset_generator.initialize()
                    logger.info("3D asset generator initialized successfully")
                else:
                    logger.warning("No 3D generation services configured")
            except Exception as e:
                logger.warning(f"Failed to initialize 3D asset generator: {e}")
                self.asset_generator = None

            # Initialize cloud storage
            try:
                self.cloud_storage = create_storage(self.settings)
                if self.cloud_storage:
                    await self.cloud_storage.connect()
                    logger.info("Cloud storage initialized and connected successfully")
                else:
                    logger.warning("No cloud storage credentials provided, file storage will be local only")
            except Exception as e:
                logger.warning(f"Failed to initialize cloud storage: {e}")
                self.cloud_storage = None

            # Initialize generation pipeline
            self.pipeline = GenerationPipeline(self)

            # Start background tasks
            self.task_manager.start_cleanup()

            self.is_initialized = True
            logger.info("App initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize app: {e}")
            raise

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
        if self.pipeline is None:
            raise RuntimeError("Application not initialized")

        return await self.pipeline.generate_asset_pipeline(
            description=description,
            asset_type=asset_type,
            style_preference=style_preference,
            quality_level=quality_level,
            session_id=session_id,
            progress_callback=progress_callback,
        )

    def get_generation_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a generation task."""
        return self.task_manager.get_task_status(task_id)

    def cancel_generation(self, task_id: str) -> bool:
        """Cancel a running generation task."""
        return self.task_manager.cancel_task(task_id)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the generation history for a session."""
        return self.session_manager.get_session_history(session_id)

    async def shutdown(self) -> None:
        """Shutdown the application and clean up resources."""
        logger.info("Shutting down Asset Generation App")

        # Shutdown task manager
        await self.task_manager.shutdown()

        # Clean up generators and storage
        if self.asset_generator:
            await self.asset_generator.cleanup()
        if self.cloud_storage:
            await self.cloud_storage.disconnect()

        logger.info("Application shutdown completed")

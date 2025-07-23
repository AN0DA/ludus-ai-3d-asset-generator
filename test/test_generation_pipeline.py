from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.generation_pipeline import GenerationPipeline
from src.models.asset_model import AssetMetadata, AssetType, QualityLevel, StylePreference
from src.utils.validators import ValidationException


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock AssetGenerationApp."""
    app = MagicMock()
    app.session_manager = MagicMock()
    app.task_manager = MagicMock()
    app.llm_generator = AsyncMock()
    app.asset_generator = AsyncMock()
    app.cloud_storage = AsyncMock()
    return app


@pytest.fixture
def generation_pipeline(mock_app: MagicMock) -> GenerationPipeline:
    """Create a GenerationPipeline instance with mock app."""
    return GenerationPipeline(mock_app)


class TestGenerationPipeline:
    """Test suite for GenerationPipeline class."""

    @pytest.mark.asyncio
    async def test_generate_asset_pipeline_creates_session(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that generate_asset_pipeline creates a new session when none provided."""
        # Setup
        mock_app.session_manager.create_session.return_value = "test-session-id"
        mock_app.session_manager.get_session.return_value = {"id": "test-session-id"}
        mock_app.task_manager.create_task.return_value = "test-task-id"

        # Execute
        task_id, session_id = await generation_pipeline.generate_asset_pipeline(
            description="A medieval sword", asset_type=AssetType.WEAPON
        )

        # Assert
        assert task_id == "test-task-id"
        assert session_id == "test-session-id"
        mock_app.session_manager.create_session.assert_called_once()
        mock_app.task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_asset_pipeline_uses_existing_session(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that generate_asset_pipeline uses existing session when provided."""
        # Setup
        existing_session_id = "existing-session-id"
        mock_app.session_manager.get_session.return_value = {"id": existing_session_id}
        mock_app.task_manager.create_task.return_value = "test-task-id"

        # Execute
        task_id, session_id = await generation_pipeline.generate_asset_pipeline(
            description="A medieval sword", asset_type=AssetType.WEAPON, session_id=existing_session_id
        )

        # Assert
        assert session_id == existing_session_id
        mock_app.session_manager.create_session.assert_not_called()
        mock_app.session_manager.get_session.assert_called_once_with(existing_session_id)

    @pytest.mark.asyncio
    async def test_generate_asset_pipeline_invalid_session(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that generate_asset_pipeline raises error for invalid session."""
        # Setup
        mock_app.session_manager.get_session.return_value = None

        # Execute & Assert
        with pytest.raises(ValueError, match="Invalid or expired session"):
            await generation_pipeline.generate_asset_pipeline(
                description="A medieval sword", asset_type=AssetType.WEAPON, session_id="invalid-session-id"
            )

    @pytest.mark.asyncio
    async def test_execute_generation_pipeline_validation_error(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that _execute_generation_pipeline raises ValidationException for short description."""
        # Setup
        mock_app.task_manager.update_task_status = MagicMock()

        # Execute & Assert
        with pytest.raises(ValidationException, match="Description must be at least 10 characters"):
            await generation_pipeline._execute_generation_pipeline(
                description="short",
                asset_type=AssetType.WEAPON,
                style_preference=None,
                quality_level=QualityLevel.STANDARD,
                session_id="test-session",
                task_id="test-task",
            )

    @pytest.mark.asyncio
    async def test_execute_generation_pipeline_success(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test successful execution of _execute_generation_pipeline."""
        # Setup
        mock_app.task_manager.update_task_status = MagicMock()

        # Mock LLM generator to return successful result
        from src.generators.base import GenerationResult, GenerationStatus

        llm_result = GenerationResult(
            status=GenerationStatus.COMPLETED,
            data={
                "enhanced_asset": {
                    "enhanced_description": "Enhanced description of a medieval sword",
                    "asset_name": "Medieval Sword",
                    "materials": ["steel", "leather"],
                    "style_notes": ["realistic"],
                }
            },
        )
        mock_app.llm_generator.generate.return_value = llm_result

        # Mock asset generation
        mock_asset_metadata = AssetMetadata(
            name="Medieval Sword",
            original_description="A detailed medieval sword with intricate engravings",
            enhanced_description={"enhanced": "Enhanced description of a medieval sword"},
            asset_type=AssetType.WEAPON,
            style_preferences=[StylePreference.REALISTIC],
            quality_level=QualityLevel.STANDARD,
            generation_service="test_service",
            session_id="test-session",
            metadata={},
        )

        # Mock the individual pipeline steps
        async def mock_generate_3d_asset(*args: Any, **kwargs: Any) -> dict[str, str]:
            return {"model_url": "test.glb"}

        async def mock_upload_to_cloud(*args: Any, **kwargs: Any) -> dict[str, str]:
            return {"upload_url": "https://example.com"}

        generation_pipeline._generate_3d_asset = mock_generate_3d_asset  # type: ignore[method-assign]
        generation_pipeline._upload_to_cloud = mock_upload_to_cloud  # type: ignore[method-assign]
        generation_pipeline._create_asset_metadata = MagicMock(return_value=mock_asset_metadata)  # type: ignore[method-assign]
        mock_app.session_manager.add_to_session_history = MagicMock()

        # Execute
        result = await generation_pipeline._execute_generation_pipeline(
            description="A detailed medieval sword with intricate engravings",
            asset_type=AssetType.WEAPON,
            style_preference=StylePreference.REALISTIC,
            quality_level=QualityLevel.STANDARD,
            session_id="test-session",
            task_id="test-task",
        )

        # Assert
        assert result == mock_asset_metadata
        mock_app.llm_generator.generate.assert_called_once()
        generation_pipeline._create_asset_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_description_with_llm(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test _enhance_description method."""
        # Setup
        from src.generators.base import GenerationResult, GenerationStatus

        llm_result = GenerationResult(
            status=GenerationStatus.COMPLETED,
            data={
                "enhanced_asset": {
                    "enhanced_description": "Enhanced description",
                    "asset_name": "weapon asset",
                    "materials": ["metal", "wood"],
                    "style_notes": ["realistic"],
                }
            },
        )
        mock_app.llm_generator.generate.return_value = llm_result

        # Execute
        result = await generation_pipeline._enhance_description(
            description="A sword",
            asset_type=AssetType.WEAPON,
            style_preference=StylePreference.REALISTIC,
            quality_level=QualityLevel.HIGH,
        )

        # Assert
        expected_result = {
            "enhanced_description": "Enhanced description",
            "asset_name": "weapon asset",
            "materials": ["metal", "wood"],
            "style_notes": ["realistic"],
        }
        assert result == expected_result
        mock_app.llm_generator.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_description_no_llm(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test _enhance_description when no LLM is available."""
        # Setup
        mock_app.llm_generator = None
        original_description = "A sword"

        # Execute
        result = await generation_pipeline._enhance_description(
            description=original_description,
            asset_type=AssetType.WEAPON,
            style_preference=StylePreference.REALISTIC,
            quality_level=QualityLevel.HIGH,
        )

        # Assert - should return fallback dict structure
        expected_result = {
            "enhanced_description": "Enhanced version of: A sword",
            "asset_name": "weapon asset",
            "materials": ["metal", "wood"],
            "style_notes": ["realistic"],
        }
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, generation_pipeline: GenerationPipeline, mock_app: MagicMock) -> None:
        """Test that progress callback is called during pipeline execution."""
        # Setup
        progress_callback = MagicMock()
        mock_app.task_manager.update_task_status = MagicMock()
        mock_app.llm_generator.enhance_description.return_value = "Enhanced description"

        mock_asset_metadata = AssetMetadata(
            name="Test Asset",
            original_description="Enhanced description",
            enhanced_description={"enhanced": "Enhanced description"},
            asset_type=AssetType.WEAPON,
            style_preferences=[StylePreference.REALISTIC],
            quality_level=QualityLevel.STANDARD,
            generation_service="test_service",
            session_id="test-session",
            metadata={},
        )
        mock_app.asset_generator.generate_asset.return_value = mock_asset_metadata

        # Execute
        await generation_pipeline._execute_generation_pipeline(
            description="A detailed medieval sword",
            asset_type=AssetType.WEAPON,
            style_preference=None,
            quality_level=QualityLevel.STANDARD,
            session_id="test-session",
            progress_callback=progress_callback,
            task_id="test-task",
        )

        # Assert
        assert progress_callback.call_count >= 2  # At least validation and llm_enhancement steps

        # Check that progress callback was called with expected parameters
        calls = progress_callback.call_args_list
        assert any("validation" in str(call) for call in calls)
        assert any("llm_enhancement" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_task_status_updates(self, generation_pipeline: GenerationPipeline, mock_app: MagicMock) -> None:
        """Test that task status is updated during pipeline execution."""
        # Setup
        mock_app.task_manager.update_task_status = MagicMock()
        mock_app.llm_generator.enhance_description.return_value = "Enhanced description"

        mock_asset_metadata = AssetMetadata(
            name="Test Asset",
            original_description="Enhanced description",
            enhanced_description={"enhanced": "Enhanced description"},
            asset_type=AssetType.WEAPON,
            style_preferences=[StylePreference.REALISTIC],
            quality_level=QualityLevel.STANDARD,
            generation_service="test_service",
            session_id="test-session",
            metadata={},
        )
        mock_app.asset_generator.generate_asset.return_value = mock_asset_metadata

        # Execute
        await generation_pipeline._execute_generation_pipeline(
            description="A detailed medieval sword",
            asset_type=AssetType.WEAPON,
            style_preference=None,
            quality_level=QualityLevel.STANDARD,
            session_id="test-session",
            task_id="test-task",
        )

        # Assert
        assert mock_app.task_manager.update_task_status.call_count >= 2

        # Check that update_task_status was called with correct parameters
        calls = mock_app.task_manager.update_task_status.call_args_list
        assert any("test-task" in str(call) for call in calls)
        assert any("in_progress" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_all_asset_types_supported(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that all asset types are supported."""
        # Setup
        mock_app.session_manager.create_session.return_value = "test-session-id"
        mock_app.session_manager.get_session.return_value = {"id": "test-session-id"}
        mock_app.task_manager.create_task.return_value = "test-task-id"

        # Test each asset type
        for asset_type in AssetType:
            task_id, session_id = await generation_pipeline.generate_asset_pipeline(
                description="Test description for asset", asset_type=asset_type
            )
            assert task_id == "test-task-id"
            assert session_id == "test-session-id"

    @pytest.mark.asyncio
    async def test_all_quality_levels_supported(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that all quality levels are supported."""
        # Setup
        mock_app.session_manager.create_session.return_value = "test-session-id"
        mock_app.session_manager.get_session.return_value = {"id": "test-session-id"}
        mock_app.task_manager.create_task.return_value = "test-task-id"

        # Test each quality level
        for quality_level in QualityLevel:
            task_id, session_id = await generation_pipeline.generate_asset_pipeline(
                description="Test description for asset", asset_type=AssetType.WEAPON, quality_level=quality_level
            )
            assert task_id == "test-task-id"
            assert session_id == "test-session-id"

    @pytest.mark.asyncio
    async def test_all_style_preferences_supported(
        self, generation_pipeline: GenerationPipeline, mock_app: MagicMock
    ) -> None:
        """Test that all style preferences are supported."""
        # Setup
        mock_app.session_manager.create_session.return_value = "test-session-id"
        mock_app.session_manager.get_session.return_value = {"id": "test-session-id"}
        mock_app.task_manager.create_task.return_value = "test-task-id"

        # Test each style preference
        for style_preference in StylePreference:
            task_id, session_id = await generation_pipeline.generate_asset_pipeline(
                description="Test description for asset", asset_type=AssetType.WEAPON, style_preference=style_preference
            )
            assert task_id == "test-task-id"
            assert session_id == "test-session-id"

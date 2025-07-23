from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.core.app import AssetGenerationApp
from src.models.asset_model import AssetType


@pytest.mark.asyncio
async def test_app_initialization(mock_app: AssetGenerationApp, mocker: Any) -> None:
    mock_app.is_initialized = False  # Reset initialization state
    mocker.patch("src.core.task_manager.TaskManager.start_cleanup")
    await mock_app.initialize()
    assert mock_app.is_initialized, "App should be initialized"
    assert mock_app.task_manager is not None, "TaskManager should be initialized"
    assert mock_app.session_manager is not None, "SessionManager should be initialized"
    assert mock_app.cache_manager is not None, "CacheManager should be initialized"


@pytest.mark.asyncio
async def test_app_shutdown(mock_app: AssetGenerationApp, mocker: Any) -> None:
    mock_cleanup = mocker.patch("src.core.task_manager.TaskManager.shutdown", AsyncMock())
    await mock_app.shutdown()
    mock_cleanup.assert_called_once()
    assert mock_app.task_manager.tasks == {}, "Tasks should be cleared"


@pytest.mark.asyncio
async def test_generate_asset_pipeline_uninitialized(mock_app: AssetGenerationApp) -> None:
    with pytest.raises(RuntimeError, match="Application not initialized"):
        await mock_app.generate_asset_pipeline(
            description="Test asset",
            asset_type=AssetType.WEAPON,
        )

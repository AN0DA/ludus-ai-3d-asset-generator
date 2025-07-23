import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from src.core.app import AssetGenerationApp
from src.core.cache_manager import CacheManager
from src.core.session_manager import SessionManager
from src.core.task_manager import TaskManager
from src.utils.env_config import AppSettings


@pytest.fixture
def temp_dir() -> Generator[Path]:
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def cache_manager(temp_dir: Path) -> CacheManager:
    return CacheManager(temp_dir / "cache", max_size_bytes=1024 * 1024)  # 1MB for testing


@pytest.fixture
def session_manager(temp_dir: Path) -> SessionManager:
    return SessionManager(temp_dir / "sessions")


@pytest.fixture
def task_manager() -> TaskManager:
    return TaskManager()


@pytest.fixture
def mock_settings(mocker: Any) -> MagicMock:
    settings = MagicMock(spec=AppSettings)
    settings.meshy_api_key = None
    settings.llm_api_key = None
    settings.get_storage_config.return_value = {
        "access_key_id": None,
        "secret_access_key": None,
        "bucket_name": "test-bucket",
        "endpoint_url": None,
        "region": "us-east-1",
        "use_ssl": True,
        "max_file_size": 104857600,
    }
    return settings


@pytest.fixture
def mock_app(mocker: Any, temp_dir: Path, mock_settings: MagicMock) -> AssetGenerationApp:
    mocker.patch("src.core.app.get_settings", return_value=mock_settings)
    mocker.patch("src.core.app.create_llm_generator", return_value=None)
    mocker.patch("src.core.app.create_asset_generator", return_value=None)
    mocker.patch("src.core.app.create_storage", return_value=None)
    app = AssetGenerationApp()
    app.temp_dir = temp_dir / "asset_generator"
    app.temp_dir.mkdir(parents=True, exist_ok=True)
    return app


@pytest.fixture(autouse=True)
def mock_logger(mocker: Any) -> MagicMock:
    return mocker.patch.object(structlog, "get_logger", return_value=MagicMock())


@pytest.fixture
def async_mock(mocker: Any) -> type[AsyncMock]:
    return AsyncMock


# Ensure async cleanup for task manager
@pytest.fixture(autouse=True)
async def cleanup_tasks(task_manager: TaskManager) -> AsyncGenerator[None]:
    yield
    await task_manager.shutdown()

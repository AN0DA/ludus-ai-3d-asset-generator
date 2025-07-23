from unittest.mock import MagicMock, patch

from src.factories.asset_generator_factory import create_asset_generator
from src.factories.llm_factory import create_llm_generator
from src.factories.storage_factory import create_storage
from src.utils.env_config import AppSettings


class TestAssetGeneratorFactory:
    """Test suite for asset generator factory."""

    def test_create_asset_generator_with_meshy_key(self) -> None:
        """Test creating asset generator when Meshy API key is provided."""
        settings = MagicMock(spec=AppSettings)
        settings.meshy_api_key = "test-meshy-key"

        with patch("src.factories.asset_generator_factory.Asset3DGenerator") as mock_generator:
            mock_instance = MagicMock()
            mock_generator.return_value = mock_instance

            result = create_asset_generator(settings)

            assert result == mock_instance
            # Check that Asset3DGenerator was called with a configs dict
            mock_generator.assert_called_once()
            call_args = mock_generator.call_args[0][0]  # First positional argument
            assert isinstance(call_args, dict)
            # Check that the configs dict contains the expected ServiceProvider
            from src.generators.enums import ServiceProvider

            assert ServiceProvider.MESHY_AI in call_args

    def test_create_asset_generator_without_meshy_key(self) -> None:
        """Test creating asset generator when Meshy API key is not provided."""
        settings = MagicMock(spec=AppSettings)
        settings.meshy_api_key = None

        result = create_asset_generator(settings)

        assert result is None

    def test_create_asset_generator_empty_meshy_key(self) -> None:
        """Test creating asset generator when Meshy API key is empty."""
        settings = MagicMock(spec=AppSettings)
        settings.meshy_api_key = ""

        result = create_asset_generator(settings)

        assert result is None

    def test_create_asset_generator_whitespace_meshy_key(self) -> None:
        """Test creating asset generator when Meshy API key is whitespace."""
        settings = MagicMock(spec=AppSettings)
        settings.meshy_api_key = "   "

        result = create_asset_generator(settings)

        assert result is None


class TestLLMFactory:
    """Test suite for LLM generator factory."""

    def test_create_llm_generator_with_api_key(self) -> None:
        """Test creating LLM generator when API key is provided."""
        settings = MagicMock(spec=AppSettings)
        settings.get_llm_config.return_value = {
            "api_key": "test-llm-key",
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 60,
            "max_retries": 3,
        }

        with (
            patch("src.factories.llm_factory.LLMGenerator") as mock_generator,
            patch("src.factories.llm_factory.LLMConfig") as mock_config,
        ):
            mock_instance = MagicMock()
            mock_generator.return_value = mock_instance
            mock_config_instance = MagicMock()
            mock_config.return_value = mock_config_instance

            result = create_llm_generator(settings)

            assert result == mock_instance
            settings.get_llm_config.assert_called_once()
            mock_config.assert_called_once_with(
                api_key="test-llm-key",
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                max_tokens=2000,
                temperature=0.7,
                timeout=60,
                max_retries=3,
            )
            mock_generator.assert_called_once_with(mock_config_instance)

    def test_create_llm_generator_without_api_key(self) -> None:
        """Test creating LLM generator when API key is not provided."""
        settings = MagicMock(spec=AppSettings)
        settings.get_llm_config.return_value = {
            "api_key": None,
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 60,
            "max_retries": 3,
        }

        result = create_llm_generator(settings)

        assert result is None

    def test_create_llm_generator_empty_api_key(self) -> None:
        """Test creating LLM generator when API key is empty."""
        settings = MagicMock(spec=AppSettings)
        settings.get_llm_config.return_value = {
            "api_key": "",
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 60,
            "max_retries": 3,
        }

        result = create_llm_generator(settings)

        assert result is None

    def test_create_llm_generator_whitespace_api_key(self) -> None:
        """Test creating LLM generator when API key is whitespace."""
        settings = MagicMock(spec=AppSettings)
        settings.get_llm_config.return_value = {
            "api_key": "   ",
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 60,
            "max_retries": 3,
        }

        result = create_llm_generator(settings)

        assert result is None


class TestStorageFactory:
    """Test suite for storage factory."""

    def test_create_storage_with_credentials(self) -> None:
        """Test creating storage when credentials are provided."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": "test-access-key",
            "secret_access_key": "test-secret-key",
            "bucket_name": "test-bucket",
            "endpoint_url": "https://s3.amazonaws.com",
            "region": "us-west-2",
            "use_ssl": True,
            "max_file_size": 50000000,
        }

        with (
            patch("src.factories.storage_factory.S3Storage") as mock_storage,
            patch("src.factories.storage_factory.StorageConfig") as mock_config,
        ):
            mock_instance = MagicMock()
            mock_storage.return_value = mock_instance
            mock_config_instance = MagicMock()
            mock_config.return_value = mock_config_instance

            result = create_storage(settings)

            assert result == mock_instance
            settings.get_storage_config.assert_called_once()
            mock_config.assert_called_once_with(**settings.get_storage_config.return_value)
            mock_storage.assert_called_once_with(mock_config_instance)

    def test_create_storage_without_credentials(self) -> None:
        """Test creating storage when credentials are not provided."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": None,
            "secret_access_key": None,
            "bucket_name": None,
            "endpoint_url": None,
            "region": "us-east-1",
            "use_ssl": True,
            "max_file_size": 104857600,
        }

        result = create_storage(settings)

        assert result is None

    def test_create_storage_missing_access_key(self) -> None:
        """Test creating storage when access key is missing."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": None,
            "secret_access_key": "test-secret-key",
            "bucket_name": "test-bucket",
            "endpoint_url": None,
            "region": "us-east-1",
            "use_ssl": True,
            "max_file_size": 104857600,
        }

        result = create_storage(settings)

        assert result is None

    def test_create_storage_missing_secret_key(self) -> None:
        """Test creating storage when secret key is missing."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": "test-access-key",
            "secret_access_key": None,
            "bucket_name": "test-bucket",
            "endpoint_url": None,
            "region": "us-east-1",
            "use_ssl": True,
            "max_file_size": 104857600,
        }

        result = create_storage(settings)

        assert result is None

    def test_create_storage_empty_credentials(self) -> None:
        """Test creating storage when credentials are empty strings."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": "",
            "secret_access_key": "",
            "bucket_name": "",
            "endpoint_url": None,
            "region": "us-east-1",
            "use_ssl": True,
            "max_file_size": 104857600,
        }

        result = create_storage(settings)

        assert result is None

    def test_create_storage_whitespace_credentials(self) -> None:
        """Test creating storage when credentials are whitespace."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": "   ",
            "secret_access_key": "   ",
            "bucket_name": "   ",
            "endpoint_url": None,
            "region": "us-east-1",
            "use_ssl": True,
            "max_file_size": 104857600,
        }

        result = create_storage(settings)

        assert result is None

    def test_create_storage_partial_whitespace_credentials(self) -> None:
        """Test creating storage when one credential is valid and other is whitespace."""
        settings = MagicMock(spec=AppSettings)
        settings.get_storage_config.return_value = {
            "access_key_id": "test-access-key",
            "secret_access_key": "   ",
            "bucket_name": "test-bucket",
            "endpoint_url": None,
            "region": "us-east-1",
            "use_ssl": True,
            "max_file_size": 104857600,
        }

        result = create_storage(settings)

        assert result is None

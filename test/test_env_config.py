import os
from unittest.mock import patch

from src.utils.env_config import AppSettings, get_settings


class TestAppSettings:
    """Test suite for AppSettings class."""

    def test_app_settings_default_values(self) -> None:
        """Test that AppSettings has correct default values."""
        # Clear relevant environment variables to test defaults
        env_vars_to_clear = [
            "MESHY_API_KEY",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "STORAGE_ACCESS_KEY_ID",
            "STORAGE_SECRET_ACCESS_KEY",
            "STORAGE_BUCKET_NAME",
            "STORAGE_ENDPOINT_URL",
            "STORAGE_REGION",
            "STORAGE_USE_SSL",
            "STORAGE_MAX_FILE_SIZE",
            "ASSET_CACHE_TTL",
            "LOG_LEVEL",
            "LLM_MODEL",
            "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS",
        ]

        # Save original values and clear environment variables
        original_values = {}
        for var in env_vars_to_clear:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        try:
            settings = AppSettings()

            assert settings.meshy_api_key is None
            assert settings.llm_api_key is None
            assert settings.llm_base_url is None
            assert settings.storage_access_key_id is None
            assert settings.storage_secret_access_key is None
            assert settings.storage_region == "us-east-1"
            assert settings.storage_bucket_name is None
            assert settings.storage_endpoint_url is None
            assert settings.storage_use_ssl is True
            assert settings.storage_max_file_size == 104857600  # 100MB
            assert settings.asset_cache_ttl == 3600  # 1 hour
            assert settings.log_level == "INFO"
            assert settings.llm_model == "gpt-4"
            assert settings.llm_temperature == 0.7
            assert settings.llm_max_tokens == 2048
        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_app_settings_from_env_vars(self) -> None:
        """Test that AppSettings correctly reads from environment variables."""
        env_vars = {
            "MESHY_API_KEY": "test-meshy-key",
            "LLM_API_KEY": "test-llm-key",
            "STORAGE_ACCESS_KEY_ID": "test-access-key",
            "STORAGE_SECRET_ACCESS_KEY": "test-secret-key",
            "STORAGE_REGION": "eu-west-1",
            "STORAGE_BUCKET_NAME": "test-bucket",
            "STORAGE_ENDPOINT_URL": "https://s3.example.com",
            "STORAGE_USE_SSL": "false",
            "STORAGE_MAX_FILE_SIZE": "50000000",
            "ASSET_CACHE_TTL": "7200",
            "LLM_MODEL": "gpt-4",
            "LLM_TEMPERATURE": "0.8",
            "LLM_MAX_TOKENS": "2000",
            "LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars):
            settings = AppSettings()

            assert settings.meshy_api_key == "test-meshy-key"
            assert settings.llm_api_key == "test-llm-key"
            assert settings.storage_access_key_id == "test-access-key"
            assert settings.storage_secret_access_key == "test-secret-key"
            assert settings.storage_region == "eu-west-1"
            assert settings.storage_bucket_name == "test-bucket"
            assert settings.storage_endpoint_url == "https://s3.example.com"
            assert settings.storage_use_ssl is False
            assert settings.storage_max_file_size == 50000000
            assert settings.asset_cache_ttl == 7200
            assert settings.llm_model == "gpt-4"
            assert settings.llm_temperature == 0.8
            assert settings.llm_max_tokens == 2000
            assert settings.log_level == "DEBUG"

    def test_get_storage_config_with_credentials(self) -> None:
        """Test get_storage_config when storage credentials are provided."""
        env_vars = {
            "STORAGE_ACCESS_KEY_ID": "test-access-key",
            "STORAGE_SECRET_ACCESS_KEY": "test-secret-key",
            "STORAGE_REGION": "us-west-2",
            "STORAGE_BUCKET_NAME": "test-bucket",
            "STORAGE_ENDPOINT_URL": "https://s3.amazonaws.com",
            "STORAGE_USE_SSL": "true",
            "STORAGE_MAX_FILE_SIZE": "50000000",
        }

        with patch.dict(os.environ, env_vars):
            settings = AppSettings()
            config = settings.get_storage_config()

            expected_config = {
                "access_key_id": "test-access-key",
                "secret_access_key": "test-secret-key",
                "bucket_name": "test-bucket",
                "endpoint_url": "https://s3.amazonaws.com",
                "region": "us-west-2",
                "use_ssl": True,
                "max_file_size": 50000000,
            }

            assert config == expected_config

    def test_get_storage_config_without_credentials(self) -> None:
        """Test get_storage_config when storage credentials are not provided."""
        # Clear storage-related environment variables
        env_vars_to_clear = [
            "STORAGE_ACCESS_KEY_ID",
            "STORAGE_SECRET_ACCESS_KEY",
            "STORAGE_BUCKET_NAME",
            "STORAGE_ENDPOINT_URL",
            "STORAGE_REGION",
            "STORAGE_USE_SSL",
            "STORAGE_MAX_FILE_SIZE",
        ]

        # Save original values and clear environment variables
        original_values = {}
        for var in env_vars_to_clear:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        try:
            settings = AppSettings()
            config = settings.get_storage_config()

            expected_config = {
                "access_key_id": None,
                "secret_access_key": None,
                "bucket_name": None,
                "endpoint_url": None,
                "region": "us-east-1",
                "use_ssl": True,
                "max_file_size": 104857600,
            }

            assert config == expected_config
        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_get_llm_config(self) -> None:
        """Test get_llm_config returns correct LLM configuration."""
        env_vars = {
            "LLM_API_KEY": "test-llm-key",
            "LLM_MODEL": "gpt-4",
            "LLM_TEMPERATURE": "0.8",
            "LLM_MAX_TOKENS": "2000",
        }

        # Clear LLM_BASE_URL to ensure it's None for this test
        original_base_url = os.environ.get("LLM_BASE_URL")
        if "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]

        try:
            with patch.dict(os.environ, env_vars, clear=False):
                settings = AppSettings()
                config = settings.get_llm_config()

                expected_config = {
                    "api_key": "test-llm-key",
                    "model": "gpt-4",
                    "base_url": None,
                    "max_tokens": 2000,
                    "temperature": 0.8,
                    "timeout": 60,
                    "max_retries": 3,
                }

                assert config == expected_config
        finally:
            # Restore original LLM_BASE_URL if it existed
            if original_base_url is not None:
                os.environ["LLM_BASE_URL"] = original_base_url

    def test_get_threed_config(self) -> None:
        """Test get_threed_config returns correct 3D generation configuration."""
        env_vars = {
            "MESHY_API_KEY": "test-meshy-key",
            "THREED_SERVICE": "meshy",
            "THREED_GENERATION_TIMEOUT": "600",
            "THREED_QUALITY_PRESET": "high",
        }

        with patch.dict(os.environ, env_vars):
            settings = AppSettings()
            config = settings.get_threed_config()

            assert config["meshy_api_key"] == "test-meshy-key"
            assert config["service"] == "meshy"
            assert config["generation_timeout"] == 600
            assert config["quality_preset"] == "high"

    def test_get_gradio_config(self) -> None:
        """Test get_gradio_config returns correct Gradio configuration."""
        env_vars = {"GRADIO_HOST": "0.0.0.0", "GRADIO_PORT": "8080", "GRADIO_DEBUG": "true", "GRADIO_SHARE": "false"}

        with patch.dict(os.environ, env_vars):
            settings = AppSettings()
            config = settings.get_gradio_config()

            assert config["host"] == "0.0.0.0"
            assert config["port"] == 8080
            assert config["debug"] is True
            assert config["share"] is False
            assert config["title"] == "AI 3D Asset Generator"

    def test_environment_validation(self) -> None:
        """Test environment-based validation in __post_init__."""
        # Test production environment warnings
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = AppSettings()
            assert settings.environment == "production"

    def test_boolean_env_var_parsing(self) -> None:
        """Test that boolean environment variables are parsed correctly."""
        # Test various representations of true
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
        for value in true_values:
            with patch.dict(os.environ, {"STORAGE_USE_SSL": value}):
                settings = AppSettings()
                assert settings.storage_use_ssl is True, f"Failed for value: {value}"

        # Test various representations of false
        false_values = ["false", "False", "FALSE", "0", "no", "No", "NO"]
        for value in false_values:
            with patch.dict(os.environ, {"STORAGE_USE_SSL": value}):
                settings = AppSettings()
                assert settings.storage_use_ssl is False, f"Failed for value: {value}"


class TestGetSettings:
    """Test suite for get_settings function."""

    def test_get_settings_returns_singleton(self) -> None:
        """Test that get_settings returns the same instance (singleton pattern)."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_returns_app_settings(self) -> None:
        """Test that get_settings returns an AppSettings instance."""
        settings = get_settings()
        assert isinstance(settings, AppSettings)

    def test_get_settings_with_env_vars(self) -> None:
        """Test that get_settings reads from environment variables."""
        with patch.dict(os.environ, {"MESHY_API_KEY": "test-singleton-key"}):
            # Clear any existing cached settings
            import src.utils.env_config

            src.utils.env_config._settings = None

            settings = get_settings()
            assert settings.meshy_api_key == "test-singleton-key"

    def test_get_settings_caching(self) -> None:
        """Test that get_settings caches the settings instance."""
        # Clear any existing cached settings
        import src.utils.env_config

        src.utils.env_config._settings = None

        with patch.dict(os.environ, {"LLM_API_KEY": "cached-test-key"}):
            settings1 = get_settings()

        # Change environment variable after first call
        with patch.dict(os.environ, {"LLM_API_KEY": "new-test-key"}):
            settings2 = get_settings()

        # Should return cached instance with original value
        assert settings1 is settings2
        assert settings1.llm_api_key == "cached-test-key"
        assert settings2.llm_api_key == "cached-test-key"

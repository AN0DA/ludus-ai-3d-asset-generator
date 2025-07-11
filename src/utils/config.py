"""
Configuration management system for AI 3D Asset Generator.

This module provides a robust configuration management system with:
- Environment-specific settings (dev, staging, prod)
- YAML-based configuration files
- Environment variable overrides
- Secure API key management
- Pydantic validation
"""

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, root_validator, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
    logger.info(f"Loaded environment variables from: {env_file}")
else:
    logger.warning(f"No .env file found at: {env_file}")

# Configuration Models

# Note: LLM configuration is now handled directly by environment variables
# in the LLMGenerator class using a simple dataclass approach.
# This eliminates complexity and configuration conflicts.

class ObjectStorageConfig(BaseModel):
    """Simplified S3-compatible object storage configuration."""

    # Basic S3 settings
    endpoint_url: str | None = Field(default=None, description="S3 endpoint URL (for non-AWS providers)")
    region: str = Field(default="us-east-1", description="Storage region")
    
    # Credentials
    access_key_id: str | None = Field(default=None, description="S3 access key ID")
    secret_access_key: str | None = Field(default=None, description="S3 secret access key")
    
    # Bucket configuration
    bucket_name: str = Field(default="ai-3d-assets", description="Bucket name for asset storage")
    
    # Basic settings
    use_ssl: bool = Field(default=True, description="Use SSL for connections")
    max_file_size: int = Field(default=100 * 1024 * 1024, ge=1024, description="Maximum file size in bytes")

    @validator("bucket_name")
    def validate_bucket_name(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Bucket name must be at least 3 characters")
        # Convert to lowercase for S3 compatibility
        return v.lower()

    @validator("endpoint_url")
    def validate_endpoint_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        return v


class ThreeDGenerationConfig(BaseModel):
    """3D model generation service configuration."""

    meshy_api_key: str | None = Field(default=None, description="Meshy AI API key")
    kaedim_api_key: str | None = Field(default=None, description="Kaedim API key")
    service: Literal["meshy", "kaedim"] = Field(default="meshy", description="3D generation service")
    generation_timeout: int = Field(default=300, ge=60, le=1800, description="Generation timeout in seconds")
    polling_interval: int = Field(default=10, ge=5, le=60, description="Status polling interval in seconds")
    max_polling_attempts: int = Field(default=120, ge=10, le=360, description="Maximum polling attempts")
    quality_preset: Literal["draft", "standard", "high"] = Field(default="standard", description="Quality preset")
    output_formats: list[str] = Field(default=["obj", "gltf"], description="Preferred output formats")
    texture_resolution: int = Field(default=1024, ge=256, le=4096, description="Texture resolution")

    @root_validator(pre=False, skip_on_failure=True)
    def validate_service_keys(cls, values):
        service = values.get("service")
        meshy_key = values.get("meshy_api_key")
        kaedim_key = values.get("kaedim_api_key")

        if service == "meshy" and not meshy_key:
            raise ValueError("Meshy API key is required when using meshy as service")
        if service == "kaedim" and not kaedim_key:
            raise ValueError("Kaedim API key is required when using kaedim as service")

        return values


class GradioConfig(BaseModel):
    """Gradio web interface configuration."""

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=7860, ge=1024, le=65535, description="Port to bind to")
    debug: bool = Field(default=False, description="Enable debug mode")
    share: bool = Field(default=False, description="Create public share link")
    auth: tuple | None = Field(default=None, description="Authentication tuple (username, password)")
    ssl_keyfile: str | None = Field(default=None, description="SSL key file path")
    ssl_certfile: str | None = Field(default=None, description="SSL certificate file path")
    max_file_size: int = Field(default=50 * 1024 * 1024, ge=1024, description="Maximum upload file size")
    queue_max_size: int = Field(default=20, ge=1, le=100, description="Maximum queue size")
    show_error: bool = Field(default=True, description="Show error messages to users")
    theme: str = Field(default="default", description="Gradio theme")
    title: str = Field(default="AI 3D Asset Generator", description="Application title")
    description: str = Field(default="Generate 3D assets from text descriptions", description="Application description")


class DatabaseConfig(BaseModel):
    """Database configuration for asset metadata storage."""

    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=5, ge=1, le=20, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, ge=1, le=300, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=300, description="Pool recycle time in seconds")
    echo: bool = Field(default=False, description="Echo SQL statements")

    @validator("url")
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "sqlite://", "mysql://")):
            raise ValueError("Database URL must start with postgresql://, sqlite://, or mysql://")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format string")
    file_path: str | None = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10 * 1024 * 1024, ge=1024, description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, le=10, description="Number of backup files")
    json_format: bool = Field(default=False, description="Use JSON log format")


class SecurityConfig(BaseModel):
    """Security configuration."""

    secret_key: str = Field(..., description="Secret key for signing")
    allowed_hosts: list[str] = Field(default=["*"], description="Allowed host headers")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    max_request_size: int = Field(default=100 * 1024 * 1024, ge=1024, description="Maximum request size")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    @validator("secret_key")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class AppConfig(BaseSettings):
    """Main application configuration."""

    # Environment
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)

    # Component configurations
    # Note: LLM config is handled directly by environment variables in LLMGenerator
    object_storage: ObjectStorageConfig = Field(default_factory=ObjectStorageConfig)
    threed_generation: ThreeDGenerationConfig = Field(default_factory=ThreeDGenerationConfig)
    gradio: GradioConfig = Field(default_factory=GradioConfig)
    database: DatabaseConfig | None = Field(default=None)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(
        default_factory=lambda: SecurityConfig(secret_key="change-me-32-characters-minimum")
    )

    # Application settings
    app_name: str = Field(default="AI 3D Asset Generator")
    app_version: str = Field(default="0.1.0")
    asset_cache_ttl: int = Field(default=3600, ge=300, description="Asset cache TTL in seconds")
    temp_dir: str = Field(default="/tmp/ai-3d-assets", description="Temporary directory for assets")
    max_concurrent_generations: int = Field(default=5, ge=1, le=20, description="Maximum concurrent generations")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables that don't match the schema


class ConfigManager:
    """Configuration manager for loading and validating configuration."""

    def __init__(self):
        self._config: AppConfig | None = None
        self._config_dir = Path(__file__).parent.parent.parent / "config"

    def load_config(self, environment: str | None = None) -> AppConfig:
        """Load configuration for the specified environment."""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")

        logger.info(f"Loading configuration for environment: {environment}")

        # Load base configuration from YAML
        config_data = self._load_yaml_config(environment)

        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)

        # Validate and create configuration object
        self._config = AppConfig(**config_data)

        # Validate environment-specific requirements
        self._validate_environment_config(environment)

        logger.info("Configuration loaded successfully")
        return self._config

    def _load_yaml_config(self, environment: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = self._config_dir / f"{environment}.yaml"

        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}

        try:
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from: {config_file}")
            return config_data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}") from e
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise

    def _apply_env_overrides(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Environment variables with nested structure using double underscores
        env_overrides = {}

        for key, value in os.environ.items():
            if key.startswith(
                ("LLM__", "OBJECT_STORAGE__", "STORAGE_", "THREED_GENERATION__", "GRADIO__", "DATABASE__", "LOGGING__", "SECURITY__", "APP_", "ENVIRONMENT", "DEBUG")
            ):
                # Handle STORAGE_ prefix mapping to object_storage
                if key.startswith("STORAGE_"):
                    # Map STORAGE_ACCESS_KEY_ID to object_storage.access_key_id
                    storage_key = key.replace("STORAGE_", "").lower()
                    if "object_storage" not in env_overrides:
                        env_overrides["object_storage"] = {}
                    env_overrides["object_storage"][storage_key] = self._convert_env_value(value)
                else:
                    parts = key.lower().split("__")
                    current = env_overrides

                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]

                    # Convert string values to appropriate types
                    current[parts[-1]] = self._convert_env_value(value)

        # Merge environment overrides with config data
        self._deep_merge(config_data, env_overrides)

        return config_data

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string value to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # List conversion (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        return value

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Deep merge override dictionary into base dictionary."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _validate_environment_config(self, environment: str) -> None:
        """Validate environment-specific configuration requirements."""
        if not self._config:
            return

        if environment == "production":
            # Production validation
            if self._config.debug:
                logger.warning("Debug mode should be disabled in production")

            # Note: LLM configuration is now validated independently by the LLM generator

            if not self._config.object_storage.access_key_id:
                raise ValueError("Object storage credentials must be provided in production")

            if not self._config.threed_generation.meshy_api_key and not self._config.threed_generation.kaedim_api_key:
                raise ValueError("At least one 3D generation API key must be provided in production")

            if len(self._config.security.secret_key) < 32:
                raise ValueError("Secret key must be at least 32 characters in production")

        elif environment == "development":
            # Development validation
            if not self._config.debug:
                logger.info("Debug mode is recommended for development environment")

    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config

    def reload_config(self, environment: str | None = None) -> AppConfig:
        """Reload configuration."""
        return self.load_config(environment)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return config_manager.config


def load_config(environment: str | None = None) -> AppConfig:
    """Load application configuration for the specified environment."""
    return config_manager.load_config(environment)


def reload_config(environment: str | None = None) -> AppConfig:
    """Reload application configuration."""
    return config_manager.reload_config(environment)

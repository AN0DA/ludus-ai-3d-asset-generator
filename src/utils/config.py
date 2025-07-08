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

logger = logging.getLogger(__name__)

# Configuration Models


class LLMProvider(BaseModel):
    """Configuration for a single LLM provider."""

    name: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic', 'ollama')")
    api_key: str | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(default=None, description="Base URL for the API (for OpenAI-compatible APIs)")
    model: str = Field(..., description="Model name to use")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    timeout: int = Field(default=60, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")

    @validator("api_key")
    def validate_api_key(cls, v):
        if v and len(v) < 10:
            raise ValueError("API key appears to be too short")
        return v

    @validator("base_url")
    def validate_base_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v


class LLMConfig(BaseModel):
    """Large Language Model configuration settings."""

    primary_provider: str = Field(default="openai", description="Primary LLM provider name")
    fallback_provider: str | None = Field(default=None, description="Fallback LLM provider name")

    # Provider configurations
    providers: dict[str, LLMProvider] = Field(
        default_factory=lambda: {
            "openai": LLMProvider(
                name="openai",
                model="gpt-4",
                base_url=None,  # Uses default OpenAI API
            ),
            "anthropic": LLMProvider(
                name="anthropic", model="claude-3-sonnet-20240229", base_url="https://api.anthropic.com"
            ),
        },
        description="LLM provider configurations",
    )

    # Global settings
    use_fallback: bool = Field(default=True, description="Use fallback provider on primary failure")

    @root_validator(pre=False, skip_on_failure=True)
    def validate_providers(cls, values):
        primary = values.get("primary_provider")
        fallback = values.get("fallback_provider")
        providers = values.get("providers", {})

        if primary and primary not in providers:
            raise ValueError(f"Primary provider '{primary}' not found in providers configuration")

        if fallback and fallback not in providers:
            logger.warning(f"Fallback provider '{fallback}' not found in providers configuration")

        # Validate that at least the primary provider has an API key
        if primary and primary in providers:
            provider_config = providers[primary]
            if not provider_config.api_key and provider_config.name not in ["ollama", "local"]:
                logger.warning(f"Primary provider '{primary}' has no API key configured")

        return values

    def get_primary_provider(self) -> LLMProvider:
        """Get the primary provider configuration."""
        return self.providers[self.primary_provider]

    def get_fallback_provider(self) -> LLMProvider | None:
        """Get the fallback provider configuration."""
        if self.fallback_provider and self.fallback_provider in self.providers:
            return self.providers[self.fallback_provider]
        return None


class ObjectStorageConfig(BaseModel):
    """S3-compatible object storage configuration."""

    # Provider settings
    provider: Literal["aws", "minio", "r2", "wasabi", "custom"] = Field(default="aws", description="Storage provider")
    endpoint_url: str | None = Field(default=None, description="Custom endpoint URL for S3-compatible storage")
    region: str = Field(default="us-east-1", description="Storage region")

    # Credentials
    access_key_id: str | None = Field(default=None, description="Access key ID")
    secret_access_key: str | None = Field(default=None, description="Secret access key")
    session_token: str | None = Field(default=None, description="Session token (for temporary credentials)")

    # Bucket configuration
    bucket_name: str = Field(..., description="Primary bucket name for asset storage")
    bucket_public: str = Field(..., description="Public bucket name for public assets")

    # CDN and URLs
    custom_domain: str | None = Field(default=None, description="Custom domain for asset URLs")
    use_ssl: bool = Field(default=True, description="Use SSL for connections")
    public_url_template: str | None = Field(
        default=None, description="URL template for public assets: {bucket}/{key} or {domain}/{key}"
    )

    # File handling
    presigned_url_expiry: int = Field(default=3600, ge=300, le=86400, description="Pre-signed URL expiry in seconds")
    max_file_size: int = Field(default=100 * 1024 * 1024, ge=1024, description="Maximum file size in bytes")
    allowed_extensions: list[str] = Field(
        default=[".obj", ".gltf", ".fbx", ".glb", ".png", ".jpg", ".jpeg", ".mtl", ".zip"],
        description="Allowed file extensions",
    )
    cleanup_after_days: int = Field(default=30, ge=1, description="Clean up temporary files after days")

    # Performance settings
    multipart_threshold: int = Field(default=64 * 1024 * 1024, ge=1024, description="Multipart upload threshold")
    max_concurrency: int = Field(default=10, ge=1, le=100, description="Maximum concurrent uploads/downloads")

    @validator("bucket_name", "bucket_public")
    def validate_bucket_names(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Bucket name must be at least 3 characters")
        # Convert to lowercase for S3 compatibility
        return v.lower()

    @validator("endpoint_url")
    def validate_endpoint_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        return v

    @validator("custom_domain")
    def validate_custom_domain(cls, v):
        if v and v.startswith(("http://", "https://")):
            raise ValueError("Custom domain should not include protocol (http/https)")
        return v

    def get_endpoint_url(self) -> str | None:
        """Get the appropriate endpoint URL based on provider."""
        if self.endpoint_url:
            return self.endpoint_url

        # Provider-specific endpoints
        provider_endpoints = {
            "aws": None,  # Use default AWS endpoints
            "minio": "http://localhost:9000",  # Common MinIO default
            "r2": f"https://{self.region}.r2.cloudflarestorage.com",
            "wasabi": f"https://s3.{self.region}.wasabisys.com",
        }

        return provider_endpoints.get(self.provider)

    def get_public_url(self, key: str) -> str:
        """Generate public URL for an asset."""
        if self.public_url_template:
            return self.public_url_template.format(
                bucket=self.bucket_public,
                key=key,
                domain=self.custom_domain or f"{self.bucket_public}.s3.{self.region}.amazonaws.com",
            )

        if self.custom_domain:
            protocol = "https" if self.use_ssl else "http"
            return f"{protocol}://{self.custom_domain}/{key}"

        # Default S3-style URL
        if self.provider == "aws":
            return f"https://{self.bucket_public}.s3.{self.region}.amazonaws.com/{key}"
        elif self.endpoint_url:
            protocol = "https" if self.use_ssl else "http"
            endpoint = self.endpoint_url.replace("http://", "").replace("https://", "")
            return f"{protocol}://{endpoint}/{self.bucket_public}/{key}"
        else:
            return f"https://{self.bucket_public}.s3.{self.region}.amazonaws.com/{key}"


class ThreeDGenerationConfig(BaseModel):
    """3D model generation service configuration."""

    meshy_api_key: str | None = Field(default=None, description="Meshy AI API key")
    kaedim_api_key: str | None = Field(default=None, description="Kaedim API key")
    primary_service: Literal["meshy", "kaedim"] = Field(default="meshy", description="Primary 3D generation service")
    fallback_service: Literal["meshy", "kaedim"] | None = Field(default="kaedim", description="Fallback service")
    generation_timeout: int = Field(default=300, ge=60, le=1800, description="Generation timeout in seconds")
    polling_interval: int = Field(default=10, ge=5, le=60, description="Status polling interval in seconds")
    max_polling_attempts: int = Field(default=120, ge=10, le=360, description="Maximum polling attempts")
    quality_preset: Literal["draft", "standard", "high"] = Field(default="standard", description="Quality preset")
    output_formats: list[str] = Field(default=["obj", "gltf"], description="Preferred output formats")
    texture_resolution: int = Field(default=1024, ge=256, le=4096, description="Texture resolution")

    @root_validator(pre=False, skip_on_failure=True)
    def validate_service_keys(cls, values):
        primary = values.get("primary_service")
        fallback = values.get("fallback_service")
        meshy_key = values.get("meshy_api_key")
        kaedim_key = values.get("kaedim_api_key")

        if primary == "meshy" and not meshy_key:
            raise ValueError("Meshy API key is required when using meshy as primary service")
        if primary == "kaedim" and not kaedim_key:
            raise ValueError("Kaedim API key is required when using kaedim as primary service")
        if fallback == "meshy" and not meshy_key:
            logger.warning("Meshy API key not provided, fallback will be disabled")
        if fallback == "kaedim" and not kaedim_key:
            logger.warning("Kaedim API key not provided, fallback will be disabled")

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
    llm: LLMConfig = Field(default_factory=LLMConfig)
    object_storage: ObjectStorageConfig = Field(
        default_factory=lambda: ObjectStorageConfig(bucket_name="ai-3d-assets", bucket_public="ai-3d-assets-public")
    )
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
                ("LLM__", "CLOUD_STORAGE__", "THREED_GENERATION__", "GRADIO__", "DATABASE__", "LOGGING__", "SECURITY__")
            ):
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

            # Check if at least one LLM provider has an API key
            primary_provider = self._config.llm.get_primary_provider()
            fallback_provider = self._config.llm.get_fallback_provider()

            if not primary_provider.api_key and (not fallback_provider or not fallback_provider.api_key):
                raise ValueError("At least one LLM provider API key must be provided in production")

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

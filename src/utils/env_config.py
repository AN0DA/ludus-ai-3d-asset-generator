"""
Environment-based configuration system for AI 3D Asset Generator.

This module provides a simple configuration system based entirely on environment variables,
removing the complexity of YAML files and configuration management.
"""

import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
    logger.info(f"Loaded environment variables from: {env_file}")
else:
    logger.warning(f"No .env file found at: {env_file}")


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, "").lower()
    return value in ("true", "1", "yes", "on") if value else default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float value from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_list(key: str, default: Optional[list] = None, separator: str = ",") -> list:
    """Get list value from environment variable."""
    if default is None:
        default = []
    value = os.getenv(key, "")
    return [item.strip() for item in value.split(separator) if item.strip()] if value else default


@dataclass
class AppSettings:
    """Application settings from environment variables."""
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", True))
    
    # Application info (constants - not configurable via environment)
    app_name: str = "AI 3D Asset Generator"
    app_version: str = "0.1.0"
    
    # LLM Configuration
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4"))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    llm_max_tokens: int = field(default_factory=lambda: get_env_int("LLM_MAX_TOKENS", 2048))
    llm_temperature: float = field(default_factory=lambda: get_env_float("LLM_TEMPERATURE", 0.7))
    llm_timeout: int = field(default_factory=lambda: get_env_int("LLM_TIMEOUT", 60))
    llm_max_retries: int = field(default_factory=lambda: get_env_int("LLM_MAX_RETRIES", 3))
    
    # Storage Configuration
    storage_access_key_id: Optional[str] = field(default_factory=lambda: os.getenv("STORAGE_ACCESS_KEY_ID"))
    storage_secret_access_key: Optional[str] = field(default_factory=lambda: os.getenv("STORAGE_SECRET_ACCESS_KEY"))
    storage_bucket_name: Optional[str] = field(default_factory=lambda: os.getenv("STORAGE_BUCKET_NAME"))
    storage_endpoint_url: Optional[str] = field(default_factory=lambda: os.getenv("STORAGE_ENDPOINT_URL"))
    storage_region: str = field(default_factory=lambda: os.getenv("STORAGE_REGION", "us-east-1"))
    storage_use_ssl: bool = field(default_factory=lambda: get_env_bool("STORAGE_USE_SSL", True))
    storage_max_file_size: int = field(default_factory=lambda: get_env_int("STORAGE_MAX_FILE_SIZE", 104857600))  # 100MB
    
    # 3D Generation Configuration
    meshy_api_key: Optional[str] = field(default_factory=lambda: os.getenv("MESHY_API_KEY"))
    threed_service: str = field(default_factory=lambda: os.getenv("THREED_SERVICE", "meshy"))
    threed_generation_timeout: int = field(default_factory=lambda: get_env_int("THREED_GENERATION_TIMEOUT", 300))
    threed_polling_interval: int = field(default_factory=lambda: get_env_int("THREED_POLLING_INTERVAL", 10))
    threed_max_polling_attempts: int = field(default_factory=lambda: get_env_int("THREED_MAX_POLLING_ATTEMPTS", 60))
    threed_quality_preset: str = field(default_factory=lambda: os.getenv("THREED_QUALITY_PRESET", "standard"))
    threed_output_formats: list = field(default_factory=lambda: get_env_list("THREED_OUTPUT_FORMATS", ["obj", "gltf"]))
    threed_texture_resolution: int = field(default_factory=lambda: get_env_int("THREED_TEXTURE_RESOLUTION", 1024))
    
    # Gradio Configuration
    gradio_host: str = field(default_factory=lambda: os.getenv("GRADIO_HOST", "127.0.0.1"))
    gradio_port: int = field(default_factory=lambda: get_env_int("GRADIO_PORT", 7860))
    gradio_debug: bool = field(default_factory=lambda: get_env_bool("GRADIO_DEBUG"))
    gradio_share: bool = field(default_factory=lambda: get_env_bool("GRADIO_SHARE", False))
    gradio_max_file_size: int = field(default_factory=lambda: get_env_int("GRADIO_MAX_FILE_SIZE", 52428800))  # 50MB
    gradio_queue_max_size: int = field(default_factory=lambda: get_env_int("GRADIO_QUEUE_MAX_SIZE", 10))
    gradio_show_error: bool = field(default_factory=lambda: get_env_bool("GRADIO_SHOW_ERROR", True))
    gradio_theme: str = field(default_factory=lambda: os.getenv("GRADIO_THEME", "default"))
    # These are constants - not configurable via environment
    gradio_title: str = "AI 3D Asset Generator"
    gradio_description: str = "Generate 3D assets from text descriptions"
    
    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_file_path: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE_PATH"))
    log_max_file_size: int = field(default_factory=lambda: get_env_int("LOG_MAX_FILE_SIZE", 10485760))  # 10MB
    log_backup_count: int = field(default_factory=lambda: get_env_int("LOG_BACKUP_COUNT", 3))
    log_json_format: bool = field(default_factory=lambda: get_env_bool("LOG_JSON_FORMAT", False))
    
    # Security Configuration
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-32-characters-minimum"))
    allowed_hosts: list = field(default_factory=lambda: get_env_list("ALLOWED_HOSTS", ["localhost", "127.0.0.1"]))
    cors_origins: list = field(default_factory=lambda: get_env_list("CORS_ORIGINS", ["http://localhost:3000", "http://127.0.0.1:3000"]))
    max_request_size: int = field(default_factory=lambda: get_env_int("MAX_REQUEST_SIZE", 104857600))  # 100MB
    rate_limit_requests: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_REQUESTS", 1000))
    rate_limit_window: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_WINDOW", 60))
    api_key_header: str = field(default_factory=lambda: os.getenv("API_KEY_HEADER", "X-API-Key"))
    
    # Application Settings
    asset_cache_ttl: int = field(default_factory=lambda: get_env_int("ASSET_CACHE_TTL", 3600))  # 1 hour
    temp_dir: str = field(default_factory=lambda: os.getenv("TEMP_DIR", "./temp/ai-3d-assets"))
    max_concurrent_generations: int = field(default_factory=lambda: get_env_int("MAX_CONCURRENT_GENERATIONS", 5))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set debug based on environment if not explicitly set
        if self.environment == "development" and not os.getenv("DEBUG"):
            self.debug = True
        elif self.environment == "production" and not os.getenv("DEBUG"):
            self.debug = False
            
        # Validate required settings for production
        if self.environment == "production":
            if not self.storage_access_key_id or not self.storage_secret_access_key:
                logger.warning("Storage credentials not provided for production environment")
            if not self.meshy_api_key:
                logger.warning("No 3D generation API key provided for production environment")
            if len(self.secret_key) < 32:
                logger.warning("Secret key should be at least 32 characters for production")
    
    def get_storage_config(self) -> dict:
        """Get storage configuration as a dictionary."""
        return {
            "access_key_id": self.storage_access_key_id,
            "secret_access_key": self.storage_secret_access_key,
            "bucket_name": self.storage_bucket_name,
            "endpoint_url": self.storage_endpoint_url,
            "region": self.storage_region,
            "use_ssl": self.storage_use_ssl,
            "max_file_size": self.storage_max_file_size,
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration as a dictionary."""
        return {
            "api_key": self.llm_api_key,
            "model": self.llm_model,
            "base_url": self.llm_base_url,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
            "timeout": self.llm_timeout,
            "max_retries": self.llm_max_retries,
        }
    
    def get_threed_config(self) -> dict:
        """Get 3D generation configuration as a dictionary."""
        return {
            "meshy_api_key": self.meshy_api_key,
            "service": self.threed_service,
            "generation_timeout": self.threed_generation_timeout,
            "polling_interval": self.threed_polling_interval,
            "max_polling_attempts": self.threed_max_polling_attempts,
            "quality_preset": self.threed_quality_preset,
            "output_formats": self.threed_output_formats,
            "texture_resolution": self.threed_texture_resolution,
        }
    
    def get_gradio_config(self) -> dict:
        """Get Gradio configuration as a dictionary."""
        return {
            "host": self.gradio_host,
            "port": self.gradio_port,
            "debug": self.gradio_debug,
            "share": self.gradio_share,
            "max_file_size": self.gradio_max_file_size,
            "queue_max_size": self.gradio_queue_max_size,
            "show_error": self.gradio_show_error,
            "theme": self.gradio_theme,
            "title": self.gradio_title,
            "description": self.gradio_description,
        }


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get the global application settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        logger.info(f"Loaded settings for environment: {_settings.environment}")
    return _settings


def reload_settings() -> AppSettings:
    """Reload the global application settings."""
    global _settings
    # Force reload of environment variables
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    _settings = AppSettings()
    logger.info(f"Reloaded settings for environment: {_settings.environment}")
    return _settings


# Convenience functions for backward compatibility
def get_config():
    """Get configuration - backward compatibility function."""
    return get_settings()


def get_app_config():
    """Get application configuration - backward compatibility function."""
    return get_settings()

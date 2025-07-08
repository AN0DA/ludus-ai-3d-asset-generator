"""
Configuration initialization module.

This module provides utilities for initializing and setting up the configuration
system at application startup.
"""

import logging
import os
import sys
from pathlib import Path

from .config import AppConfig, load_config

logger = logging.getLogger(__name__)


def setup_logging(config: AppConfig) -> None:
    """Set up logging based on configuration."""
    log_config = config.logging

    # Create logs directory if needed
    if log_config.file_path:
        log_file = Path(log_config.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_config.level))

    # File handler (if configured)
    if log_config.file_path:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_config.file_path, maxBytes=log_config.max_file_size, backupCount=log_config.backup_count
        )
        file_handler.setLevel(getattr(logging, log_config.level))
    else:
        file_handler = None

    # Formatter
    if log_config.json_format:
        try:
            import structlog

            # Use structured logging if available
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            return
        except ImportError:
            # Fall back to standard JSON-like formatting
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
    else:
        formatter = logging.Formatter(log_config.format)

    # Apply formatter to handlers
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if file_handler:
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured: level={log_config.level}, file={log_config.file_path}")


def setup_directories(config: AppConfig) -> None:
    """Create necessary directories based on configuration."""
    directories = [
        config.temp_dir,
    ]

    # Add log directory if file logging is enabled
    if config.logging.file_path:
        log_dir = Path(config.logging.file_path).parent
        directories.append(str(log_dir))

    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


def validate_api_keys(config: AppConfig) -> None:
    """Validate that required API keys are present."""
    errors = []

    # Check LLM API keys
    primary_provider = config.llm.get_primary_provider()
    fallback_provider = config.llm.get_fallback_provider()

    if not primary_provider.api_key and (not fallback_provider or not fallback_provider.api_key):
        errors.append("At least one LLM provider API key must be provided")

    # Check 3D generation API keys
    if not config.threed_generation.meshy_api_key and not config.threed_generation.kaedim_api_key:
        errors.append("At least one 3D generation API key (Meshy or Kaedim) must be provided")

    # Check object storage credentials (only in production)
    if config.environment == "production":
        if not config.object_storage.access_key_id:
            errors.append("Object storage access key ID is required in production")
        if not config.object_storage.secret_access_key:
            errors.append("Object storage secret access key is required in production")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("API key validation passed")


def check_environment() -> str:
    """Determine the current environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()

    # Validate environment
    valid_environments = ["development", "staging", "production", "docker"]
    if environment not in valid_environments:
        logger.warning(f"Unknown environment '{environment}', defaulting to 'development'")
        environment = "development"

    logger.info(f"Detected environment: {environment}")
    return environment


def initialize_configuration(environment: str | None = None) -> AppConfig:
    """
    Initialize the complete configuration system.

    This function:
    1. Determines the environment
    2. Loads configuration from YAML and environment variables
    3. Sets up logging
    4. Creates necessary directories
    5. Validates API keys
    6. Returns the configured AppConfig instance
    """
    try:
        # Determine environment
        if environment is None:
            environment = check_environment()

        # Load configuration
        logger.info("Loading application configuration...")
        config = load_config(environment)

        # Setup logging (this will reconfigure logging with the loaded config)
        setup_logging(config)

        # Create necessary directories
        setup_directories(config)

        # Validate API keys (in development, we might not have all keys)
        if environment in ["production", "staging"]:
            validate_api_keys(config)
        else:
            logger.info("Skipping API key validation in development environment")

        logger.info(f"Configuration initialized successfully for environment: {environment}")
        return config

    except Exception as e:
        # Use basic logging if structured logging isn't set up yet
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Failed to initialize configuration: {e}")
        raise


def get_configuration_summary(config: AppConfig) -> dict:
    """Get a summary of the current configuration (without sensitive data)."""
    primary_provider = config.llm.get_primary_provider()
    fallback_provider = config.llm.get_fallback_provider()

    return {
        "environment": config.environment,
        "debug": config.debug,
        "app_name": config.app_name,
        "app_version": config.app_version,
        "llm": {
            "primary_provider": config.llm.primary_provider,
            "primary_model": primary_provider.model,
            "primary_base_url": primary_provider.base_url,
            "fallback_provider": config.llm.fallback_provider,
            "max_tokens": primary_provider.max_tokens,
            "temperature": primary_provider.temperature,
            "has_primary_key": bool(primary_provider.api_key),
            "has_fallback_key": bool(fallback_provider.api_key) if fallback_provider else False,
            "use_fallback": config.llm.use_fallback,
        },
        "object_storage": {
            "provider": config.object_storage.provider,
            "region": config.object_storage.region,
            "bucket_name": config.object_storage.bucket_name,
            "endpoint_url": config.object_storage.get_endpoint_url(),
            "has_credentials": bool(config.object_storage.access_key_id),
            "custom_domain": config.object_storage.custom_domain,
        },
        "threed_generation": {
            "primary_service": config.threed_generation.primary_service,
            "quality_preset": config.threed_generation.quality_preset,
            "has_meshy_key": bool(config.threed_generation.meshy_api_key),
            "has_kaedim_key": bool(config.threed_generation.kaedim_api_key),
        },
        "gradio": {
            "host": config.gradio.host,
            "port": config.gradio.port,
            "debug": config.gradio.debug,
        },
        "database": {
            "url": config.database.url.split("@")[-1] if config.database else None,  # Hide credentials
            "pool_size": config.database.pool_size if config.database else None,
        },
        "logging": {
            "level": config.logging.level,
            "file_path": config.logging.file_path,
            "json_format": config.logging.json_format,
        },
    }


# Convenience function for easy imports
def init_config(environment: str | None = None) -> AppConfig:
    """Initialize configuration - convenience function."""
    return initialize_configuration(environment)

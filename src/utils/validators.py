"""
Comprehensive input validation for the AI 3D Asset Generator.

This module provides validation functions, decorators, and security checks
for all user inputs, file uploads, API configurations, and potential security threats.
"""

import asyncio
import hashlib
import html
import json
import mimetypes
import re
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

from src.models.asset_model import AssetType, QualityLevel, StylePreference

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])


# Configuration and Constants


class ValidationConfig:
    """Configuration for validation parameters."""

    # Text input limits
    MIN_DESCRIPTION_LENGTH = 10
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_ASSET_NAME_LENGTH = 100
    MAX_TAG_LENGTH = 50
    MAX_TAGS_COUNT = 20

    # File upload limits (in bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_MODEL_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_TEXTURE_SIZE = 20 * 1024 * 1024  # 20MB

    # Security patterns
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"eval\s*\(",
        r"setTimeout\s*\(",
        r"setInterval\s*\(",
        r"Function\s*\(",
        r"document\.",
        r"window\.",
        r"alert\s*\(",
        r"confirm\s*\(",
        r"prompt\s*\(",
    ]

    # Allowed file formats
    ALLOWED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
    ALLOWED_MODEL_FORMATS = {".obj", ".gltf", ".glb", ".fbx", ".stl", ".ply", ".dae", ".3ds", ".blend"}
    ALLOWED_TEXTURE_FORMATS = {".jpg", ".jpeg", ".png", ".tga", ".exr", ".hdr", ".tiff"}

    # API key patterns
    LLM_KEY_PATTERN = r"^sk-[A-Za-z0-9]{48}$"  # OpenAI format, can be extended for other providers
    AWS_ACCESS_KEY_PATTERN = r"^AKIA[0-9A-Z]{16}$"
    AWS_SECRET_KEY_PATTERN = r"^[A-Za-z0-9/+=]{40}$"


# Custom Exceptions


class ValidationException(Exception):
    """Base exception for validation errors."""

    def __init__(self, message: str, field: str | None = None, code: str | None = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


class SecurityValidationException(ValidationException):
    """Exception for security-related validation failures."""

    pass


class FileValidationException(ValidationException):
    """Exception for file validation failures."""

    pass


class APIValidationException(ValidationException):
    """Exception for API-related validation failures."""

    pass


# Standardized Error Messages


class ErrorMessages:
    """Standardized error messages for consistent user experience."""

    # Text validation
    TEXT_TOO_SHORT = "Description must be at least {min_length} characters long"
    TEXT_TOO_LONG = "Description cannot exceed {max_length} characters"
    TEXT_INVALID_CHARACTERS = "Description contains invalid or unsafe characters"
    TEXT_PROFANITY_DETECTED = "Content contains inappropriate language"

    # File validation
    FILE_TOO_LARGE = "File size ({size}) exceeds maximum allowed ({max_size})"
    FILE_INVALID_FORMAT = "File format '{format}' is not supported. Allowed formats: {allowed}"
    FILE_CORRUPTED = "File appears to be corrupted or invalid"
    FILE_MISSING = "Required file is missing"

    # Security validation
    SECURITY_THREAT_DETECTED = "Potential security threat detected in input"
    INJECTION_ATTEMPT = "Input contains patterns that could be harmful"
    UNSAFE_CONTENT = "Content contains unsafe elements"

    # API validation
    API_KEY_INVALID_FORMAT = "API key format is invalid for {service}"
    API_KEY_EXPIRED = "API key appears to be expired or invalid"
    CONFIG_MISSING = "Required configuration '{key}' is missing"
    CONFIG_INVALID = "Configuration value for '{key}' is invalid"

    # Asset validation
    ASSET_TYPE_INVALID = "Asset type must be one of: {allowed_types}"
    STYLE_INVALID = "Style preference must be one of: {allowed_styles}"
    QUALITY_INVALID = "Quality level must be one of: {allowed_levels}"


# Text Input Sanitization and Validation


class TextValidator:
    """Handles text input sanitization and validation."""

    @staticmethod
    def sanitize_text(text: str, preserve_formatting: bool = False) -> str:
        """
        Sanitize text input by removing harmful content and normalizing.

        Args:
            text: Input text to sanitize
            preserve_formatting: Whether to preserve basic formatting

        Returns:
            Sanitized text

        Raises:
            SecurityValidationException: If dangerous patterns are detected
        """
        if not isinstance(text, str):
            raise ValidationException("Input must be a string", code="INVALID_TYPE")

        # Check for dangerous patterns first
        TextValidator._check_security_patterns(text)

        # HTML escape
        sanitized = html.escape(text)

        # Remove or replace potentially dangerous characters
        if not preserve_formatting:
            # Remove all HTML tags
            sanitized = re.sub(r"<[^>]+>", "", sanitized)

            # Remove control characters except newlines and tabs
            sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized.strip())

        return sanitized

    @staticmethod
    def _check_security_patterns(text: str) -> None:
        """Check for dangerous security patterns in text."""
        text_lower = text.lower()

        for pattern in ValidationConfig.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                raise SecurityValidationException(
                    ErrorMessages.SECURITY_THREAT_DETECTED, code="SECURITY_PATTERN_DETECTED"
                )

    @staticmethod
    def validate_description(description: str) -> str:
        """
        Validate and sanitize asset description.

        Args:
            description: Asset description text

        Returns:
            Validated and sanitized description

        Raises:
            ValidationException: If validation fails
        """
        if not description or not description.strip():
            raise ValidationException("Description cannot be empty", field="description")

        sanitized = TextValidator.sanitize_text(description, preserve_formatting=True)

        if len(sanitized) < ValidationConfig.MIN_DESCRIPTION_LENGTH:
            raise ValidationException(
                ErrorMessages.TEXT_TOO_SHORT.format(min_length=ValidationConfig.MIN_DESCRIPTION_LENGTH),
                field="description",
                code="TEXT_TOO_SHORT",
            )

        if len(sanitized) > ValidationConfig.MAX_DESCRIPTION_LENGTH:
            raise ValidationException(
                ErrorMessages.TEXT_TOO_LONG.format(max_length=ValidationConfig.MAX_DESCRIPTION_LENGTH),
                field="description",
                code="TEXT_TOO_LONG",
            )

        return sanitized

    @staticmethod
    def validate_asset_name(name: str) -> str:
        """Validate and sanitize asset name."""
        if not name or not name.strip():
            raise ValidationException("Asset name cannot be empty", field="name")

        sanitized = TextValidator.sanitize_text(name)

        if len(sanitized) > ValidationConfig.MAX_ASSET_NAME_LENGTH:
            raise ValidationException(
                f"Asset name cannot exceed {ValidationConfig.MAX_ASSET_NAME_LENGTH} characters",
                field="name",
                code="NAME_TOO_LONG",
            )

        # Check for valid filename characters
        if not re.match(r"^[a-zA-Z0-9\s\-_\.]+$", sanitized):
            raise ValidationException(
                "Asset name contains invalid characters. Use only letters, numbers, spaces, hyphens, and underscores.",
                field="name",
                code="INVALID_CHARACTERS",
            )

        return sanitized

    @staticmethod
    def validate_tags(tags: list[str]) -> list[str]:
        """Validate and sanitize a list of tags."""
        if not tags:
            return []

        if len(tags) > ValidationConfig.MAX_TAGS_COUNT:
            raise ValidationException(
                f"Too many tags. Maximum allowed: {ValidationConfig.MAX_TAGS_COUNT}", field="tags", code="TOO_MANY_TAGS"
            )

        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                continue

            sanitized_tag = TextValidator.sanitize_text(tag.strip())

            if not sanitized_tag:
                continue

            if len(sanitized_tag) > ValidationConfig.MAX_TAG_LENGTH:
                raise ValidationException(
                    f"Tag '{sanitized_tag}' is too long. Maximum length: {ValidationConfig.MAX_TAG_LENGTH}",
                    field="tags",
                    code="TAG_TOO_LONG",
                )

            if sanitized_tag not in validated_tags:
                validated_tags.append(sanitized_tag)

        return validated_tags


# Asset Type and Enum Validation


class AssetValidator:
    """Validates asset-specific inputs."""

    @staticmethod
    def validate_asset_type(asset_type: str | AssetType) -> AssetType:
        """Validate asset type input."""
        if isinstance(asset_type, AssetType):
            return asset_type

        if isinstance(asset_type, str):
            asset_type_lower = asset_type.lower().strip()

            for valid_type in AssetType:
                if valid_type.value == asset_type_lower:
                    return valid_type

            raise ValidationException(
                ErrorMessages.ASSET_TYPE_INVALID.format(allowed_types=", ".join([t.value for t in AssetType])),
                field="asset_type",
                code="INVALID_ASSET_TYPE",
            )

        raise ValidationException(
            "Asset type must be a string or AssetType enum", field="asset_type", code="INVALID_TYPE"
        )

    @staticmethod
    def validate_style_preference(style: str | StylePreference) -> StylePreference:
        """Validate style preference input."""
        if isinstance(style, StylePreference):
            return style

        if isinstance(style, str):
            style_lower = style.lower().strip()

            for valid_style in StylePreference:
                if valid_style.value == style_lower:
                    return valid_style

            raise ValidationException(
                ErrorMessages.STYLE_INVALID.format(allowed_styles=", ".join([s.value for s in StylePreference])),
                field="style_preference",
                code="INVALID_STYLE",
            )

        raise ValidationException(
            "Style preference must be a string or StylePreference enum", field="style_preference", code="INVALID_TYPE"
        )

    @staticmethod
    def validate_quality_level(quality: str | QualityLevel) -> QualityLevel:
        """Validate quality level input."""
        if isinstance(quality, QualityLevel):
            return quality

        if isinstance(quality, str):
            quality_lower = quality.lower().strip()

            for valid_quality in QualityLevel:
                if valid_quality.value == quality_lower:
                    return valid_quality

            raise ValidationException(
                ErrorMessages.QUALITY_INVALID.format(allowed_levels=", ".join([q.value for q in QualityLevel])),
                field="quality_level",
                code="INVALID_QUALITY",
            )

        raise ValidationException(
            "Quality level must be a string or QualityLevel enum", field="quality_level", code="INVALID_TYPE"
        )


# File Upload Validation


class FileValidator:
    """Handles file upload validation and security checks."""

    @staticmethod
    def validate_file_upload(
        file_path: str | Path, file_type: str = "general", max_size: int | None = None
    ) -> dict[str, Any]:
        """
        Validate uploaded file for size, format, and security.

        Args:
            file_path: Path to the uploaded file
            file_type: Type of file (image, model, texture)
            max_size: Custom maximum size in bytes

        Returns:
            Dict with file information and validation results

        Raises:
            FileValidationException: If validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileValidationException(ErrorMessages.FILE_MISSING, field="file", code="FILE_NOT_FOUND")

        # Get file info
        file_size = file_path.stat().st_size
        file_extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        # Validate size
        max_allowed_size = max_size or FileValidator._get_max_size_for_type(file_type)
        if file_size > max_allowed_size:
            raise FileValidationException(
                ErrorMessages.FILE_TOO_LARGE.format(
                    size=FileValidator._format_file_size(file_size),
                    max_size=FileValidator._format_file_size(max_allowed_size),
                ),
                field="file",
                code="FILE_TOO_LARGE",
            )

        # Validate format
        allowed_formats = FileValidator._get_allowed_formats_for_type(file_type)
        if file_extension not in allowed_formats:
            raise FileValidationException(
                ErrorMessages.FILE_INVALID_FORMAT.format(format=file_extension, allowed=", ".join(allowed_formats)),
                field="file",
                code="INVALID_FORMAT",
            )

        # Basic file content validation
        FileValidator._validate_file_content(file_path, file_type)

        return {
            "path": str(file_path),
            "size": file_size,
            "extension": file_extension,
            "mime_type": mime_type,
            "is_valid": True,
            "file_hash": FileValidator._calculate_file_hash(file_path),
        }

    @staticmethod
    def _get_max_size_for_type(file_type: str) -> int:
        """Get maximum allowed size for file type."""
        size_map = {
            "image": ValidationConfig.MAX_IMAGE_SIZE,
            "model": ValidationConfig.MAX_MODEL_SIZE,
            "texture": ValidationConfig.MAX_TEXTURE_SIZE,
        }
        return size_map.get(file_type, ValidationConfig.MAX_IMAGE_SIZE)

    @staticmethod
    def _get_allowed_formats_for_type(file_type: str) -> set[str]:
        """Get allowed formats for file type."""
        format_map = {
            "image": ValidationConfig.ALLOWED_IMAGE_FORMATS,
            "model": ValidationConfig.ALLOWED_MODEL_FORMATS,
            "texture": ValidationConfig.ALLOWED_TEXTURE_FORMATS,
        }
        return format_map.get(file_type, ValidationConfig.ALLOWED_IMAGE_FORMATS)

    @staticmethod
    def _validate_file_content(file_path: Path, file_type: str) -> None:
        """Perform basic file content validation."""
        try:
            # Read first few bytes to check file signature
            with open(file_path, "rb") as f:
                header = f.read(16)

            if not header:
                raise FileValidationException(ErrorMessages.FILE_CORRUPTED, code="EMPTY_FILE")

            # Basic file signature validation based on type
            if file_type == "image":
                FileValidator._validate_image_header(header, file_path.suffix)
            elif file_type == "model":
                FileValidator._validate_model_header(header, file_path.suffix)

        except OSError as e:
            raise FileValidationException(f"Cannot read file: {str(e)}", code="FILE_READ_ERROR")

    @staticmethod
    def _validate_image_header(header: bytes, extension: str) -> None:
        """Validate image file headers."""
        # Basic image format validation
        image_signatures = {
            ".jpg": [b"\xff\xd8\xff"],
            ".jpeg": [b"\xff\xd8\xff"],
            ".png": [b"\x89PNG\r\n\x1a\n"],
            ".gif": [b"GIF87a", b"GIF89a"],
            ".webp": [b"RIFF"],
            ".bmp": [b"BM"],
        }

        if extension in image_signatures:
            valid_signatures = image_signatures[extension]
            if not any(header.startswith(sig) for sig in valid_signatures):
                raise FileValidationException(
                    f"File does not appear to be a valid {extension} image", code="INVALID_IMAGE_SIGNATURE"
                )

    @staticmethod
    def _validate_model_header(header: bytes, extension: str) -> None:
        """Validate 3D model file headers."""
        # Basic model format validation
        if extension == ".gltf":
            # GLTF files are JSON, so check for JSON start
            try:
                content = header.decode("utf-8", errors="ignore")
                if not (content.strip().startswith("{") or content.strip().startswith("{")):
                    raise FileValidationException("GLTF file does not appear to be valid JSON", code="INVALID_GLTF")
            except UnicodeDecodeError:
                raise FileValidationException("GLTF file contains invalid characters", code="INVALID_GLTF_ENCODING")
        elif extension == ".glb":
            # GLB files start with 'glTF'
            if not header.startswith(b"glTF"):
                raise FileValidationException("GLB file does not have valid signature", code="INVALID_GLB_SIGNATURE")

    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


# API Key Validation


class APIKeyValidator:
    """Validates API keys and credentials."""

    @staticmethod
    def validate_llm_key(api_key: str) -> bool:
        """Validate LLM API key format."""
        if not api_key or not isinstance(api_key, str):
            raise APIValidationException(
                ErrorMessages.API_KEY_INVALID_FORMAT.format(service="LLM"), field="llm_api_key", code="INVALID_LLM_KEY"
            )

        if not re.match(ValidationConfig.LLM_KEY_PATTERN, api_key.strip()):
            raise APIValidationException(
                ErrorMessages.API_KEY_INVALID_FORMAT.format(service="LLM"),
                field="llm_api_key",
                code="INVALID_LLM_KEY_FORMAT",
            )

        return True

    @staticmethod
    def validate_aws_credentials(access_key: str, secret_key: str) -> bool:
        """Validate AWS credentials format."""
        if not access_key or not isinstance(access_key, str):
            raise APIValidationException(
                ErrorMessages.API_KEY_INVALID_FORMAT.format(service="AWS"),
                field="aws_access_key",
                code="INVALID_AWS_ACCESS_KEY",
            )

        if not secret_key or not isinstance(secret_key, str):
            raise APIValidationException(
                ErrorMessages.API_KEY_INVALID_FORMAT.format(service="AWS"),
                field="aws_secret_key",
                code="INVALID_AWS_SECRET_KEY",
            )

        if not re.match(ValidationConfig.AWS_ACCESS_KEY_PATTERN, access_key.strip()):
            raise APIValidationException(
                ErrorMessages.API_KEY_INVALID_FORMAT.format(service="AWS Access Key"),
                field="aws_access_key",
                code="INVALID_AWS_ACCESS_KEY_FORMAT",
            )

        if not re.match(ValidationConfig.AWS_SECRET_KEY_PATTERN, secret_key.strip()):
            raise APIValidationException(
                ErrorMessages.API_KEY_INVALID_FORMAT.format(service="AWS Secret Key"),
                field="aws_secret_key",
                code="INVALID_AWS_SECRET_KEY_FORMAT",
            )

        return True

    @staticmethod
    async def test_api_key_validity(service: str, api_key: str, **kwargs) -> bool:
        """
        Test if API key is valid by making a test request.

        This is a placeholder for actual API testing implementation.
        """
        # This would typically make a lightweight API call to test the key
        # Implementation would depend on specific API endpoints
        return True


# Configuration Validation


class ConfigValidator:
    """Validates configuration settings."""

    @staticmethod
    def validate_config_dict(config: dict[str, Any], required_keys: list[str]) -> dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ValidationException("Configuration must be a dictionary", code="INVALID_CONFIG_TYPE")

        # Check required keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationException(
                ErrorMessages.CONFIG_MISSING.format(key=", ".join(missing_keys)), code="MISSING_CONFIG_KEYS"
            )

        # Validate specific configuration values
        validated_config = {}

        for key, value in config.items():
            try:
                validated_config[key] = ConfigValidator._validate_config_value(key, value)
            except Exception as e:
                raise ValidationException(
                    ErrorMessages.CONFIG_INVALID.format(key=key), field=key, code="INVALID_CONFIG_VALUE"
                ) from e

        return validated_config

    @staticmethod
    def _validate_config_value(key: str, value: Any) -> Any:
        """Validate individual configuration values."""
        # URL validation
        if key.endswith("_url") or key.endswith("_endpoint"):
            if not isinstance(value, str):
                raise ValidationException(f"{key} must be a string URL")

            # Basic URL format validation
            if not re.match(r"^https?://", value):
                raise ValidationException(f"{key} must be a valid HTTP/HTTPS URL")

        # Port validation
        elif key.endswith("_port"):
            if not isinstance(value, int) or not (1 <= value <= 65535):
                raise ValidationException(f"{key} must be a valid port number (1-65535)")

        # Boolean validation
        elif key.startswith("enable_") or key.endswith("_enabled"):
            if not isinstance(value, bool):
                raise ValidationException(f"{key} must be a boolean value")

        # Timeout validation
        elif key.endswith("_timeout"):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationException(f"{key} must be a positive number")

        return value


# Custom Validation Decorators


def validate_input(**validators) -> Callable[[F], F]:
    """
    Decorator for validating function inputs.

    Usage:
        @validate_input(
            description=TextValidator.validate_description,
            asset_type=AssetValidator.validate_asset_type
        )
        def create_asset(description: str, asset_type: str):
            pass
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect

            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate specified arguments
            for param_name, validator_func in validators.items():
                if param_name in bound_args.arguments:
                    try:
                        validated_value = validator_func(bound_args.arguments[param_name])
                        bound_args.arguments[param_name] = validated_value
                    except ValidationException:
                        raise
                    except Exception as e:
                        raise ValidationException(
                            f"Validation failed for {param_name}: {str(e)}", field=param_name, code="VALIDATION_ERROR"
                        ) from e

            return func(*bound_args.args, **bound_args.kwargs)

        return cast(F, wrapper)

    return decorator


def validate_async_input(**validators) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for validating async function inputs.

    Usage:
        @validate_async_input(
            description=TextValidator.validate_description,
            asset_type=AssetValidator.validate_asset_type
        )
        async def create_asset(description: str, asset_type: str):
            pass
    """

    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import inspect

            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate specified arguments
            for param_name, validator_func in validators.items():
                if param_name in bound_args.arguments:
                    try:
                        # Handle both sync and async validators
                        if asyncio.iscoroutinefunction(validator_func):
                            validated_value = await validator_func(bound_args.arguments[param_name])
                        else:
                            validated_value = validator_func(bound_args.arguments[param_name])
                        bound_args.arguments[param_name] = validated_value
                    except ValidationException:
                        raise
                    except Exception as e:
                        raise ValidationException(
                            f"Validation failed for {param_name}: {str(e)}", field=param_name, code="VALIDATION_ERROR"
                        ) from e

            return await func(*bound_args.args, **bound_args.kwargs)

        return cast(AsyncF, wrapper)

    return decorator


def require_auth(func: F) -> F:
    """Decorator to require authentication for function access."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # This would integrate with your authentication system
        # For now, it's a placeholder
        auth_token = kwargs.get("auth_token") or (args[0] if args else None)

        if not auth_token:
            raise ValidationException("Authentication required", code="AUTH_REQUIRED")

        return func(*args, **kwargs)

    return cast(F, wrapper)


# Security Validation Functions


class SecurityValidator:
    """Security-focused validation functions."""

    @staticmethod
    def validate_safe_path(file_path: str, allowed_base_paths: list[str]) -> str:
        """
        Validate that a file path is safe and within allowed directories.

        Prevents path traversal attacks.
        """
        try:
            # Resolve the path to handle '..' and symbolic links
            resolved_path = Path(file_path).resolve()

            # Check if path is within any of the allowed base paths
            for base_path in allowed_base_paths:
                base_resolved = Path(base_path).resolve()
                try:
                    resolved_path.relative_to(base_resolved)
                    return str(resolved_path)
                except ValueError:
                    continue

            raise SecurityValidationException(
                f"File path '{file_path}' is not within allowed directories", code="PATH_TRAVERSAL_ATTEMPT"
            )

        except (OSError, ValueError) as e:
            raise SecurityValidationException(f"Invalid file path: {str(e)}", code="INVALID_PATH") from e

    @staticmethod
    def validate_json_input(json_data: str, max_depth: int = 10, max_length: int = 100000) -> dict[str, Any]:
        """
        Safely validate and parse JSON input.

        Prevents JSON bomb attacks and excessive nesting.
        """
        if not isinstance(json_data, str):
            raise ValidationException("JSON data must be a string")

        if len(json_data) > max_length:
            raise SecurityValidationException(
                f"JSON data too large (max {max_length} characters)", code="JSON_TOO_LARGE"
            )

        try:
            # Parse JSON
            parsed_data = json.loads(json_data)

            # Check nesting depth
            SecurityValidator._check_json_depth(parsed_data, max_depth)

            return parsed_data

        except json.JSONDecodeError as e:
            raise ValidationException(f"Invalid JSON format: {str(e)}", code="INVALID_JSON") from e

    @staticmethod
    def _check_json_depth(obj: Any, max_depth: int, current_depth: int = 0) -> None:
        """Recursively check JSON nesting depth."""
        if current_depth > max_depth:
            raise SecurityValidationException(f"JSON nesting too deep (max {max_depth} levels)", code="JSON_TOO_DEEP")

        if isinstance(obj, dict):
            for value in obj.values():
                SecurityValidator._check_json_depth(value, max_depth, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                SecurityValidator._check_json_depth(item, max_depth, current_depth + 1)

    @staticmethod
    def validate_sql_input(input_string: str) -> str:
        """
        Validate input to prevent SQL injection.

        This is a basic implementation - in production, use parameterized queries.
        """
        if not isinstance(input_string, str):
            raise ValidationException("SQL input must be a string")

        # Common SQL injection patterns
        dangerous_patterns = [
            r";.*(--)|(;)|(\|)|(\*)",
            r"(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE){0,1}|INSERT( +INTO){0,1}|SELECT|UNION( +ALL){0,1}|UPDATE)\b)",
            r"(\b(AND|OR)\b.*(=|>|<|<>|>=|<=))",
            r"(\'|\").*(\-\-|\#)",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise SecurityValidationException(
                    "Input contains potentially dangerous SQL patterns", code="SQL_INJECTION_ATTEMPT"
                )

        return input_string


# Utility Functions


def create_validation_report(errors: list[ValidationException]) -> dict[str, Any]:
    """Create a standardized validation error report."""
    return {
        "is_valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": [
            {
                "message": error.message,
                "field": error.field,
                "code": error.code,
                "timestamp": datetime.utcnow().isoformat(),
            }
            for error in errors
        ],
    }


def validate_batch_inputs(inputs: list[dict[str, Any]], validation_schema: dict[str, Callable]) -> dict[str, Any]:
    """
    Validate a batch of inputs using provided validation schema.

    Args:
        inputs: List of input dictionaries to validate
        validation_schema: Dictionary mapping field names to validation functions

    Returns:
        Dictionary with validation results and any errors
    """
    all_errors = []
    validated_inputs = []

    for i, input_data in enumerate(inputs):
        input_errors = []
        validated_input = {}

        for field_name, validator_func in validation_schema.items():
            if field_name in input_data:
                try:
                    validated_input[field_name] = validator_func(input_data[field_name])
                except ValidationException as e:
                    e.field = f"inputs[{i}].{field_name}"
                    input_errors.append(e)
                    all_errors.append(e)
                except Exception as e:
                    validation_error = ValidationException(
                        f"Validation failed for {field_name}: {str(e)}",
                        field=f"inputs[{i}].{field_name}",
                        code="VALIDATION_ERROR",
                    )
                    input_errors.append(validation_error)
                    all_errors.append(validation_error)

        if not input_errors:
            validated_inputs.append(validated_input)

    return {
        "is_valid": len(all_errors) == 0,
        "validated_inputs": validated_inputs,
        "validation_report": create_validation_report(all_errors),
    }


# Export main validation functions for easy import
__all__ = [
    # Classes
    "ValidationConfig",
    "TextValidator",
    "AssetValidator",
    "FileValidator",
    "APIKeyValidator",
    "ConfigValidator",
    "SecurityValidator",
    "ErrorMessages",
    # Exceptions
    "ValidationException",
    "SecurityValidationException",
    "FileValidationException",
    "APIValidationException",
    # Decorators
    "validate_input",
    "validate_async_input",
    "require_auth",
    # Utility functions
    "create_validation_report",
    "validate_batch_inputs",
]

"""
Abstract base class and interface system for AI generators.

This module provides the foundation for all AI generators in the 3D asset
generation pipeline, including configuration protocols, error handling,
retry mechanisms, rate limiting, and standardized response formats.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import structlog
from pydantic import BaseModel, Field


# Configure structured logging for generators
logger = structlog.get_logger(__name__)


# Type variables for generic typing
T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound="GeneratorConfig")


class GenerationStatus(str, Enum):
    """Status of a generation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Error Handling Classes

class GenerationError(Exception):
    """Base exception for all generation-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }


class APIError(GenerationError):
    """API-related errors (network, authentication, service unavailable)."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.details.update({
            "status_code": status_code,
            "response_data": response_data,
        })


class ValidationError(GenerationError):
    """Input validation and configuration errors."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.details.update({
            "field_name": field_name,
            "field_value": field_value,
        })


class RateLimitError(APIError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, severity=ErrorSeverity.LOW, **kwargs)
        self.retry_after = retry_after
        self.details.update({"retry_after": retry_after})


class TimeoutError(GenerationError):
    """Request timeout errors."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.timeout_duration = timeout_duration
        self.details.update({"timeout_duration": timeout_duration})


# Configuration Protocol

@runtime_checkable
class GeneratorConfig(Protocol):
    """Protocol for generator configuration classes."""
    
    @property
    def api_key(self) -> str:
        """API key for the service."""
        ...
    
    @property
    def base_url(self) -> Optional[str]:
        """Base URL for the API service."""
        ...
    
    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        ...
    
    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        ...
    
    @property
    def rate_limit_requests(self) -> int:
        """Rate limit: requests per minute."""
        ...
    
    @property
    def rate_limit_window(self) -> int:
        """Rate limit window in seconds."""
        ...
    
    def validate(self) -> None:
        """Validate the configuration."""
        ...


# Standardized Response Format

@dataclass
class GenerationResult:
    """Standardized result format for all generators."""
    
    # Core result data
    status: GenerationStatus
    data: Optional[Dict[str, Any]] = None
    
    # Metadata
    generator_name: str = ""
    generation_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Performance metrics
    processing_time_ms: Optional[int] = None
    api_calls_made: int = 0
    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
    
    # Error information
    error: Optional[GenerationError] = None
    warnings: List[str] = field(default_factory=list)
    
    # Request tracking
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if the generation was successful."""
        return self.status == GenerationStatus.COMPLETED and self.error is None
    
    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Get processing time in seconds."""
        return self.processing_time_ms / 1000.0 if self.processing_time_ms else None
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning("Generation warning", warning=warning, generation_id=self.generation_id)
    
    def set_error(self, error: GenerationError) -> None:
        """Set the error and update status."""
        self.error = error
        self.status = GenerationStatus.FAILED
        logger.error(
            "Generation failed",
            error=error.to_dict(),
            generation_id=self.generation_id,
            generator=self.generator_name,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "status": self.status.value,
            "data": self.data,
            "generator_name": self.generator_name,
            "generation_id": self.generation_id,
            "created_at": self.created_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "processing_time_seconds": self.processing_time_seconds,
            "api_calls_made": self.api_calls_made,
            "tokens_used": self.tokens_used,
            "estimated_cost": self.estimated_cost,
            "error": self.error.to_dict() if self.error else None,
            "warnings": self.warnings,
            "request_metadata": self.request_metadata,
            "is_successful": self.is_successful,
        }


# Decorators for Retry and Rate Limiting

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.window_seconds]
            
            # Check if we can make a request
            if len(self.requests) >= self.requests_per_window:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.window_seconds - (now - oldest_request)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            # Record this request
            self.requests.append(now)


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    backoff_max: float = 60.0,
    retry_on: tuple = (APIError, TimeoutError, RateLimitError),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to add retry logic to async methods.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        backoff_max: Maximum backoff time in seconds
        retry_on: Tuple of exception types to retry on
    """
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            "Max retries exceeded",
                            attempt=attempt,
                            max_retries=max_retries,
                            error=str(e),
                            func_name=func.__name__,
                        )
                        raise e
                    
                    # Calculate backoff time
                    backoff_time = min(backoff_factor * (2 ** attempt), backoff_max)
                    
                    # Handle rate limiting
                    if isinstance(e, RateLimitError) and e.retry_after:
                        backoff_time = max(backoff_time, e.retry_after)
                    
                    logger.warning(
                        "Retrying after error",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        backoff_time=backoff_time,
                        error=str(e),
                        func_name=func.__name__,
                    )
                    
                    await asyncio.sleep(backoff_time)
                
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(
                        "Non-retryable error occurred",
                        error=str(e),
                        error_type=type(e).__name__,
                        func_name=func.__name__,
                    )
                    raise e
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise GenerationError("Unexpected error in retry logic")
        
        return wrapper
    return decorator


def with_rate_limiting(
    requests_per_window: int,
    window_seconds: int = 60,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to add rate limiting to async methods.
    
    Args:
        requests_per_window: Number of requests allowed per window
        window_seconds: Window size in seconds
    """
    rate_limiter = RateLimiter(requests_per_window, window_seconds)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            await rate_limiter.acquire()
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Abstract Base Generator Class

class BaseGenerator(ABC):
    """
    Abstract base class for all AI generators.
    
    This class provides the common interface and functionality that all
    generators must implement, including configuration management, error
    handling, and standardized response formats.
    """
    
    def __init__(self, config: GeneratorConfig, name: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            config: Generator configuration
            name: Optional generator name (defaults to class name)
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = structlog.get_logger(self.__class__.__module__).bind(generator=self.name)
        
        # Validate configuration
        try:
            config.validate()
        except Exception as e:
            raise ValidationError(
                f"Invalid configuration for {self.name}",
                details={"validation_error": str(e)},
                original_exception=e,
            )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests,
            config.rate_limit_window,
        )
        
        self.logger.info("Generator initialized", config_type=type(config).__name__)
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        generation_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate content based on the input prompt.
        
        Args:
            prompt: Input prompt for generation
            generation_id: Optional unique identifier for this generation
            **kwargs: Additional generation parameters
        
        Returns:
            GenerationResult with the generated content or error information
        """
        pass
    
    @abstractmethod
    async def validate_input(self, prompt: str, **kwargs) -> None:
        """
        Validate input parameters before generation.
        
        Args:
            prompt: Input prompt to validate
            **kwargs: Additional parameters to validate
        
        Raises:
            ValidationError: If input is invalid
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Check if the generator service is healthy and accessible.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            result = await self._perform_health_check()
            self.logger.info("Health check completed", healthy=result)
            return result
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    @abstractmethod
    async def _perform_health_check(self) -> bool:
        """
        Implement the actual health check logic.
        
        Returns:
            True if service is healthy, False otherwise
        """
        pass
    
    async def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the generator service.
        
        Returns:
            Dictionary with service information
        """
        return {
            "name": self.name,
            "config_type": type(self.config).__name__,
            "base_url": getattr(self.config, "base_url", None),
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "rate_limit": {
                "requests_per_window": self.config.rate_limit_requests,
                "window_seconds": self.config.rate_limit_window,
            },
        }
    
    def _create_result(
        self,
        status: GenerationStatus,
        generation_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Create a standardized GenerationResult."""
        return GenerationResult(
            status=status,
            data=data,
            generator_name=self.name,
            generation_id=generation_id or f"{self.name}_{int(time.time() * 1000)}",
        )
    
    def _validate_common_params(self, prompt: str) -> None:
        """Validate common parameters across all generators."""
        if not prompt or not prompt.strip():
            raise ValidationError(
                "Prompt cannot be empty",
                field_name="prompt",
                field_value=prompt,
            )
        
        if len(prompt.strip()) < 10:
            raise ValidationError(
                "Prompt must be at least 10 characters long",
                field_name="prompt",
                field_value=prompt,
            )
        
        if len(prompt) > 10000:  # Reasonable upper limit
            raise ValidationError(
                "Prompt is too long (max 10000 characters)",
                field_name="prompt",
                field_value=len(prompt),
            )


# Logging Configuration

def configure_generator_logging(
    level: str = "INFO",
    format_json: bool = True,
    include_timestamp: bool = True,
) -> None:
    """
    Configure structured logging for generators.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_json: Whether to format logs as JSON
        include_timestamp: Whether to include timestamps
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="ISO"))
    
    if format_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
    )


# Export all public classes and functions
__all__ = [
    "BaseGenerator",
    "GeneratorConfig",
    "GenerationResult",
    "GenerationStatus",
    "ErrorSeverity",
    "GenerationError",
    "APIError",
    "ValidationError", 
    "RateLimitError",
    "TimeoutError",
    "RateLimiter",
    "with_retry",
    "with_rate_limiting",
    "configure_generator_logging",
]

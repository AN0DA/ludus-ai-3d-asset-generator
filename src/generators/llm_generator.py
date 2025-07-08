"""
Example LLM generator implementation using the base generator system.

This module demonstrates how to implement a concrete generator using the
abstract BaseGenerator class with proper error handling, retry logic,
and rate limiting.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field

from .base import (
    APIError,
    BaseGenerator,
    GenerationResult,
    GenerationStatus,
    GeneratorConfig,
    TimeoutError,
    ValidationError,
    with_rate_limiting,
    with_retry,
)


@dataclass
class LLMConfig:
    """Configuration for LLM generator."""
    
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_requests: int = 50
    rate_limit_window: int = 60
    max_tokens: int = 1000
    temperature: float = 0.7
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.api_key:
            raise ValidationError("API key is required")
        
        if not self.base_url:
            raise ValidationError("Base URL is required")
        
        if self.timeout <= 0:
            raise ValidationError("Timeout must be positive")
        
        if self.max_retries < 0:
            raise ValidationError("Max retries cannot be negative")
        
        if self.rate_limit_requests <= 0:
            raise ValidationError("Rate limit requests must be positive")
        
        if not (0.0 <= self.temperature <= 2.0):
            raise ValidationError("Temperature must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            raise ValidationError("Max tokens must be positive")


class LLMRequest(BaseModel):
    """Request model for LLM API calls."""
    
    model: str
    messages: list[Dict[str, str]]
    max_tokens: int = Field(ge=1, le=4000)
    temperature: float = Field(ge=0.0, le=2.0)
    stream: bool = False


class LLMResponse(BaseModel):
    """Response model from LLM API."""
    
    id: str
    object: str
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class ExampleLLMGenerator(BaseGenerator):
    """
    Example LLM generator implementation.
    
    This demonstrates how to implement a concrete generator using the
    BaseGenerator abstract class with proper async/await patterns,
    error handling, and API integration.
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config: LLMConfig = config  # Type hint for better IDE support
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )
    
    async def validate_input(self, prompt: str, **kwargs) -> None:
        """Validate input parameters before generation."""
        # Call common validation from base class
        self._validate_common_params(prompt)
        
        # Additional LLM-specific validation
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens <= 0 or max_tokens > 4000:
            raise ValidationError(
                "Max tokens must be between 1 and 4000",
                field_name="max_tokens",
                field_value=max_tokens,
            )
        
        temperature = kwargs.get("temperature", self.config.temperature)
        if not (0.0 <= temperature <= 2.0):
            raise ValidationError(
                "Temperature must be between 0.0 and 2.0",
                field_name="temperature",
                field_value=temperature,
            )
    
    @with_retry(max_retries=3, backoff_factor=1.0)
    @with_rate_limiting(requests_per_window=50, window_seconds=60)
    async def generate(
        self,
        prompt: str,
        generation_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate enhanced description using LLM.
        
        Args:
            prompt: Input prompt for generation
            generation_id: Optional unique identifier
            **kwargs: Additional generation parameters
        
        Returns:
            GenerationResult with enhanced description or error
        """
        start_time = asyncio.get_event_loop().time()
        result = self._create_result(
            status=GenerationStatus.IN_PROGRESS,
            generation_id=generation_id or str(uuid.uuid4()),
        )
        
        try:
            # Validate input
            await self.validate_input(prompt, **kwargs)
            
            # Prepare request
            request_data = self._prepare_request(prompt, **kwargs)
            
            self.logger.info(
                "Starting LLM generation",
                generation_id=result.generation_id,
                model=self.config.model,
                prompt_length=len(prompt),
            )
            
            # Make API call
            response = await self._make_api_call(request_data)
            
            # Process response
            generated_text = self._extract_generated_text(response)
            
            # Update result with success
            end_time = asyncio.get_event_loop().time()
            result.status = GenerationStatus.COMPLETED
            result.data = {
                "enhanced_description": generated_text,
                "original_prompt": prompt,
                "model_used": self.config.model,
            }
            result.processing_time_ms = int((end_time - start_time) * 1000)
            result.api_calls_made = 1
            result.tokens_used = response.usage.get("total_tokens") if response.usage else None
            
            self.logger.info(
                "LLM generation completed",
                generation_id=result.generation_id,
                processing_time_ms=result.processing_time_ms,
                tokens_used=result.tokens_used,
            )
            
            return result
        
        except ValidationError as e:
            result.set_error(e)
            return result
        
        except APIError as e:
            result.set_error(e)
            return result
        
        except Exception as e:
            error = APIError(
                message=f"Unexpected error during LLM generation: {str(e)}",
                original_exception=e,
            )
            result.set_error(error)
            return result
    
    async def _make_api_call(self, request_data: LLMRequest) -> LLMResponse:
        """Make the actual API call to the LLM service."""
        try:
            response = await self.client.post(
                "/chat/completions",
                json=request_data.model_dump(),
            )
            
            if response.status_code == 429:
                # Rate limiting
                retry_after = int(response.headers.get("retry-after", 60))
                raise APIError(
                    message="Rate limit exceeded",
                    status_code=response.status_code,
                    response_data={"retry_after": retry_after},
                )
            
            if response.status_code >= 400:
                # API error
                try:
                    error_data = response.json()
                except:
                    error_data = {"error": response.text}
                
                raise APIError(
                    message=f"LLM API error: {response.status_code}",
                    status_code=response.status_code,
                    response_data=error_data,
                )
            
            return LLMResponse(**response.json())
        
        except httpx.TimeoutException as e:
            raise TimeoutError(
                message="Request to LLM API timed out",
                timeout_duration=self.config.timeout,
                original_exception=e,
            )
        
        except httpx.RequestError as e:
            raise APIError(
                message=f"Network error calling LLM API: {str(e)}",
                original_exception=e,
            )
    
    def _prepare_request(self, prompt: str, **kwargs) -> LLMRequest:
        """Prepare the API request data."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert at creating detailed, vivid descriptions for 3D asset generation. Enhance the user's description with specific details about materials, textures, proportions, and visual characteristics."
            },
            {
                "role": "user",
                "content": f"Please enhance this description for 3D asset generation: {prompt}"
            }
        ]
        
        return LLMRequest(
            model=self.config.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )
    
    def _extract_generated_text(self, response: LLMResponse) -> str:
        """Extract the generated text from the API response."""
        if not response.choices:
            raise APIError("No choices in LLM response")
        
        choice = response.choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        
        if not content:
            raise APIError("Empty content in LLM response")
        
        return content.strip()
    
    async def _perform_health_check(self) -> bool:
        """Perform health check by making a simple API call."""
        try:
            test_request = LLMRequest(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.0,
            )
            
            response = await self.client.post(
                "/chat/completions",
                json=test_request.model_dump(),
            )
            
            return response.status_code == 200
        
        except Exception:
            return False
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.client.aclose()
        self.logger.info("LLM generator client closed")


# Export the example implementation
__all__ = [
    "LLMConfig",
    "LLMRequest", 
    "LLMResponse",
    "ExampleLLMGenerator",
]

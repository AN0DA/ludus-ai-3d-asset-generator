"""
Comprehensive LLM integration module for enhancing game asset descriptions.

This module provides a production-ready LLM generator that specializes in
creating detailed, game-specific asset descriptions with structured output,
technical specifications, and gameplay-relevant information.
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field, validator

from src.generators.base import (
    APIError,
    BaseGenerator,
    GenerationResult,
    GenerationStatus,
    TimeoutError,
    ValidationError,
    with_rate_limiting,
    with_retry,
)

# Import asset models - adjust import path as needed
try:
    from models.asset_model import AssetType, QualityLevel, StylePreference
except ImportError:
    # Fallback for different import contexts
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.asset_model import AssetType, QualityLevel, StylePreference


class OutputFormat(str, Enum):
    """Supported output formats for LLM responses."""

    JSON = "json"
    STRUCTURED_TEXT = "structured_text"
    MARKDOWN = "markdown"


@dataclass
class LLMConfig:
    """
    Configuration for game asset LLM generator.

    This uses a simplified configuration approach with direct field mapping
    to environment variables (e.g., LLM__API_KEY → api_key).
    """

    # API Configuration
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1"

    # Request Configuration
    timeout: int = 60
    max_retries: int = 3
    rate_limit_requests: int = 30
    rate_limit_window: int = 60

    # Generation Parameters
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1

    # Output Configuration
    output_format: OutputFormat = OutputFormat.JSON
    include_examples: bool = True
    validate_json: bool = True

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

        if not (0.0 <= self.top_p <= 1.0):
            raise ValidationError("Top-p must be between 0.0 and 1.0")

        if self.max_tokens <= 0 or self.max_tokens > 4000:
            raise ValidationError("Max tokens must be between 1 and 4000")


class LLMRequest(BaseModel):
    """Request model for LLM API calls."""

    model: str
    messages: list[dict[str, str]]
    max_tokens: int = Field(ge=1, le=4000)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0, default=1.0)
    frequency_penalty: float = Field(ge=-2.0, le=2.0, default=0.0)
    presence_penalty: float = Field(ge=-2.0, le=2.0, default=0.0)
    stream: bool = False
    response_format: dict[str, str] | None = None


class LLMResponse(BaseModel):
    """Response model from LLM API."""

    id: str | None = None
    object: str | None = None
    created: int | None = None
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None

    @property
    def content(self) -> str:
        """Extract the main content from the response."""
        if not self.choices:
            return ""

        choice = self.choices[0]
        if "message" in choice:
            return choice["message"].get("content", "")
        elif "text" in choice:
            return choice["text"]

        return ""

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        if not self.usage:
            return 0
        return self.usage.get("total_tokens", 0)

    @property
    def input_tokens(self) -> int:
        """Get input tokens used."""
        if not self.usage:
            return 0
        return self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output tokens used."""
        if not self.usage:
            return 0
        return self.usage.get("completion_tokens", 0)
        return self.usage.get("completion_tokens", 0) if self.usage else 0


class EnhancedAssetDescription(BaseModel):
    """Structured output model for enhanced asset descriptions."""

    # Core Description
    enhanced_description: str = Field(..., min_length=50, max_length=1000)
    asset_name: str = Field(..., min_length=1, max_length=100)
    asset_category: str = Field(..., min_length=1, max_length=50)

    # Physical Properties
    physical_properties: dict[str, Any] = Field(default_factory=dict)
    dimensions: dict[str, float] | None = None
    weight: str | None = None
    materials: list[str] = Field(default_factory=list)

    # Visual Characteristics
    visual_characteristics: dict[str, Any] = Field(default_factory=dict)
    color_palette: list[str] = Field(default_factory=list)
    textures: list[str] = Field(default_factory=list)
    lighting_effects: list[str] = Field(default_factory=list)

    # Gameplay Mechanics
    gameplay_mechanics: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, int | float] | None = None
    abilities: list[str] = Field(default_factory=list)
    rarity: str | None = None

    # Technical Requirements
    technical_requirements: dict[str, Any] = Field(default_factory=dict)
    estimated_polygon_count: int | None = Field(ge=100, le=100000)
    texture_resolution: int | None = Field(ge=256, le=4096)
    optimization_level: str | None = None

    # Additional Metadata
    style_notes: list[str] = Field(default_factory=list)
    inspiration_sources: list[str] = Field(default_factory=list)
    modeling_complexity: str | None = None

    @validator("estimated_polygon_count")
    def validate_polygon_count(cls, v):
        """Validate polygon count is reasonable."""
        if v is not None and v < 100:
            raise ValueError("Polygon count too low for game assets")
        return v

    @validator("color_palette")
    def validate_colors(cls, v):
        """Validate color palette entries."""
        if len(v) > 10:
            raise ValueError("Too many colors in palette (max 10)")
        return v


class LLMGenerator(BaseGenerator):
    """
    Comprehensive LLM generator for game asset description enhancement.

    This generator specializes in creating detailed, game-specific asset
    descriptions with structured output, technical specifications, and
    gameplay-relevant information using OpenAI-compatible APIs.
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

        # Initialize prompt templates
        self._init_prompt_templates()
        self._init_few_shot_examples()

        self.logger.info(
            "LLM Generator initialized",
            model=config.model,
            output_format=config.output_format.value,
            max_tokens=config.max_tokens,
        )

    def _init_prompt_templates(self) -> None:
        """Initialize asset-specific prompt templates."""
        self.system_prompt = """You are an expert game asset designer and technical artist with deep knowledge of:
- 3D modeling and asset creation pipelines
- Game development technical requirements
- Fantasy and sci-fi world building
- Material science and visual design
- Game balance and mechanics

Your task is to enhance basic asset descriptions into detailed, production-ready specifications for 3D asset generation. You must provide comprehensive information about physical properties, visual characteristics, gameplay mechanics, and technical requirements.

CRITICAL REQUIREMENTS:
1. Always respond with valid JSON matching the specified schema
2. Include realistic technical specifications (polygon counts, texture resolutions)
3. Provide detailed material and visual descriptions suitable for 3D artists
4. Consider gameplay balance and asset rarity
5. Specify optimization requirements for different quality levels
6. Include style-appropriate details for the asset category

Focus on creating assets that feel authentic within their respective game worlds while being technically feasible for real-time 3D rendering."""

        self.asset_prompts = {
            AssetType.WEAPON: """Enhance this weapon description with detailed specifications:

Focus on:
- Combat mechanics and damage types
- Grip ergonomics and handling characteristics  
- Material composition and craftsmanship quality
- Visual effects during use (particles, lighting)
- Appropriate polygon budget for weapon class
- Texture requirements for different components
- Gameplay stats (damage, speed, range, durability)
- Rarity and acquisition difficulty""",
            AssetType.ARMOR: """Enhance this armor description with detailed specifications:

Focus on:
- Protection values and coverage areas
- Weight distribution and mobility impact
- Material layering and construction methods
- Visual wear patterns and battle damage
- Attachment points for accessories
- Texture complexity for different materials
- Gameplay stats (defense, weight, special resistances)
- Set bonuses and upgrade paths""",
            AssetType.POTION: """Enhance this potion description with detailed specifications:

Focus on:
- Liquid properties and visual effects
- Container design and material composition
- Particle effects and magical auras
- Label design and identification markers
- Storage requirements and shelf life
- Consumption animations and effects
- Gameplay mechanics (duration, potency, side effects)
- Crafting ingredients and rarity""",
            AssetType.ENVIRONMENT: """Enhance this environment description with detailed specifications:

Focus on:
- Architectural style and structural elements
- Lighting conditions and atmospheric effects
- Interactive elements and gameplay features
- LOD (Level of Detail) requirements
- Texture streaming and optimization
- Weather and time-of-day variations
- Navigation and accessibility features
- Performance impact and draw call optimization""",
            AssetType.TOOL: """Enhance this tool description with detailed specifications:

Focus on:
- Functional mechanisms and moving parts
- Wear patterns from regular use
- Material durability and maintenance needs
- Ergonomic design for user interaction
- Animation requirements for tool usage
- Gameplay functionality and skill interactions
- Crafting requirements and upgrade potential
- Professional or magical enhancement features""",
        }

    def _init_few_shot_examples(self) -> None:
        """Initialize few-shot examples for consistent output format."""
        self.few_shot_examples = [
            {
                "input": "A simple iron sword for beginners",
                "output": {
                    "enhanced_description": "A practical iron sword designed for novice warriors, featuring a straight double-edged blade with functional cross-guard and leather-wrapped wooden grip. The blade shows subtle forge marks and a basic fuller to reduce weight while maintaining structural integrity. Simple yet reliable construction makes it ideal for training and early combat encounters.",
                    "asset_name": "Novice Iron Sword",
                    "asset_category": "Basic Weapon",
                    "physical_properties": {
                        "blade_length": "75cm",
                        "total_length": "95cm",
                        "balance_point": "12cm from guard",
                    },
                    "dimensions": {"length": 95.0, "width": 8.0, "height": 3.0},
                    "weight": "1.2kg",
                    "materials": ["iron", "leather", "wood"],
                    "visual_characteristics": {
                        "finish": "matte iron with minor oxidation",
                        "grip_style": "leather wrap with brass wire",
                    },
                    "color_palette": ["steel gray", "brown leather", "brass"],
                    "textures": ["brushed metal", "worn leather", "dark wood grain"],
                    "lighting_effects": ["subtle metallic reflection"],
                    "gameplay_mechanics": {
                        "weapon_type": "one-handed sword",
                        "attack_speed": "medium",
                        "reach": "medium",
                    },
                    "stats": {"damage": 25, "speed": 1.2, "durability": 100},
                    "abilities": ["basic slash", "thrust attack"],
                    "rarity": "common",
                    "technical_requirements": {"target_platform": "mid-range gaming", "performance_tier": "standard"},
                    "estimated_polygon_count": 1200,
                    "texture_resolution": 1024,
                    "optimization_level": "standard",
                    "style_notes": ["realistic proportions", "functional design"],
                    "inspiration_sources": ["medieval arming sword", "training weapons"],
                    "modeling_complexity": "low",
                },
            }
        ]

    async def validate_input(self, prompt: str, **kwargs) -> None:
        """Validate input parameters before generation."""
        # Call common validation from base class
        self._validate_common_params(prompt)

        # LLM-specific validation with flexible type handling
        asset_type = kwargs.get("asset_type")
        if asset_type is not None:
            if isinstance(asset_type, str):
                # Try to convert string to AssetType enum
                try:
                    asset_type_lower = asset_type.lower().strip()
                    valid_asset_type = None
                    for valid_type in AssetType:
                        if valid_type.value == asset_type_lower:
                            valid_asset_type = valid_type
                            break
                    if valid_asset_type is None:
                        raise ValueError(f"Invalid asset type: {asset_type}")
                    # Update kwargs with converted enum
                    kwargs["asset_type"] = valid_asset_type
                except (ValueError, AttributeError):
                    raise ValidationError(
                        f"Invalid asset type. Must be one of: {', '.join([t.value for t in AssetType])}",
                        field_name="asset_type",
                        field_value=asset_type,
                    )
            elif not isinstance(asset_type, AssetType):
                raise ValidationError(
                    f"Asset type must be string or AssetType enum, got {type(asset_type)}",
                    field_name="asset_type",
                    field_value=asset_type,
                )

        style_preferences = kwargs.get("style_preferences", [])
        if style_preferences:
            # Handle both single string and list of strings
            if isinstance(style_preferences, str):
                style_preferences = [style_preferences]

            validated_styles = []
            for style in style_preferences:
                if isinstance(style, str):
                    # Try to convert string to StylePreference enum
                    try:
                        style_lower = style.lower().strip()
                        valid_style = None
                        for valid_s in StylePreference:
                            if valid_s.value == style_lower:
                                valid_style = valid_s
                                break
                        if valid_style is None:
                            raise ValueError(f"Invalid style preference: {style}")
                        validated_styles.append(valid_style)
                    except (ValueError, AttributeError):
                        raise ValidationError(
                            f"Invalid style preference: {style}. Must be one of: {', '.join([s.value for s in StylePreference])}",
                            field_name="style_preferences",
                            field_value=style_preferences,
                        )
                elif isinstance(style, StylePreference):
                    validated_styles.append(style)
                else:
                    raise ValidationError(
                        f"Style preference must be string or StylePreference enum, got {type(style)}",
                        field_name="style_preferences",
                        field_value=style_preferences,
                    )
            # Update kwargs with validated styles
            kwargs["style_preferences"] = validated_styles

        quality_level = kwargs.get("quality_level")
        if quality_level is not None:
            if isinstance(quality_level, str):
                # Try to convert string to QualityLevel enum
                try:
                    quality_lower = quality_level.lower().strip()
                    valid_quality = None
                    for valid_q in QualityLevel:
                        if valid_q.value == quality_lower:
                            valid_quality = valid_q
                            break
                    if valid_quality is None:
                        raise ValueError(f"Invalid quality level: {quality_level}")
                    # Update kwargs with converted enum
                    kwargs["quality_level"] = valid_quality
                except (ValueError, AttributeError):
                    raise ValidationError(
                        f"Invalid quality level. Must be one of: {', '.join([q.value for q in QualityLevel])}",
                        field_name="quality_level",
                        field_value=quality_level,
                    )
            elif not isinstance(quality_level, QualityLevel):
                raise ValidationError(
                    f"Quality level must be string or QualityLevel enum, got {type(quality_level)}",
                    field_name="quality_level",
                    field_value=quality_level,
                )

        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens <= 0 or max_tokens > 4000:
            raise ValidationError(
                "Max tokens must be between 1 and 4000",
                field_name="max_tokens",
                field_value=max_tokens,
            )

    @with_retry(max_retries=3, backoff_factor=1.5)
    @with_rate_limiting(requests_per_window=30, window_seconds=60)
    async def generate(
        self,
        prompt: str,
        generation_id: str | None = None,
        asset_type: AssetType | None = None,
        style_preferences: list[StylePreference] | None = None,
        quality_level: QualityLevel | None = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate enhanced game asset description using LLM.

        Args:
            prompt: Input description of the asset
            generation_id: Optional unique identifier
            asset_type: Type of asset being generated
            style_preferences: Visual style preferences
            quality_level: Target quality level for the asset
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
            await self.validate_input(
                prompt,
                asset_type=asset_type,
                style_preferences=style_preferences,
                quality_level=quality_level,
                **kwargs,
            )

            # Prepare request with game-specific context
            request_data = self._prepare_game_asset_request(
                prompt=prompt,
                asset_type=asset_type,
                style_preferences=style_preferences,
                quality_level=quality_level,
                **kwargs,
            )

            self.logger.info(
                "Starting game asset enhancement",
                generation_id=result.generation_id,
                asset_type=asset_type.value if asset_type else "unknown",
                prompt_length=len(prompt),
                model=self.config.model,
            )

            # Make API call
            response = await self._make_api_call(request_data)

            # Process and validate response
            enhanced_data = await self._process_response(response, prompt)

            # Update result with success
            end_time = asyncio.get_event_loop().time()
            result.status = GenerationStatus.COMPLETED
            result.data = {
                "enhanced_asset": enhanced_data,
                "original_prompt": prompt,
                "asset_type": asset_type.value if asset_type else None,
                "style_preferences": [s.value for s in style_preferences] if style_preferences else [],
                "quality_level": quality_level.value if quality_level else None,
                "model_used": self.config.model,
                "output_format": self.config.output_format.value,
            }
            result.processing_time_ms = int((end_time - start_time) * 1000)
            result.api_calls_made = 1
            result.tokens_used = response.total_tokens

            # Add token breakdown to metadata
            result.request_metadata.update(
                {
                    "tokens_breakdown": {
                        "input": response.input_tokens,
                        "output": response.output_tokens,
                        "total": response.total_tokens,
                    },
                }
            )

            self.logger.info(
                "Game asset enhancement completed",
                generation_id=result.generation_id,
                processing_time_ms=result.processing_time_ms,
                tokens_used=result.tokens_used,
                asset_name=enhanced_data.get("asset_name") if isinstance(enhanced_data, dict) else None,
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
                message=f"Unexpected error during asset enhancement: {str(e)}",
                original_exception=e,
            )
            result.set_error(error)
            return result

    def _prepare_game_asset_request(
        self,
        prompt: str,
        asset_type: AssetType | None = None,
        style_preferences: list[StylePreference] | None = None,
        quality_level: QualityLevel | None = None,
        **kwargs,
    ) -> LLMRequest:
        """Prepare the API request with game-specific prompts."""

        # Build context information
        context_parts = []

        if asset_type:
            context_parts.append(f"Asset Type: {asset_type.value}")
            if asset_type in self.asset_prompts:
                context_parts.append(self.asset_prompts[asset_type])

        if style_preferences:
            styles = ", ".join([s.value for s in style_preferences])
            context_parts.append(f"Style Preferences: {styles}")

        if quality_level:
            context_parts.append(f"Quality Level: {quality_level.value}")

            # Add quality-specific requirements
            quality_specs = {
                QualityLevel.DRAFT: "Focus on basic shapes and simple materials. Low polygon count (500-2000). Basic textures (512px).",
                QualityLevel.STANDARD: "Balanced detail and performance. Medium polygon count (2000-8000). Standard textures (1024px).",
                QualityLevel.HIGH: "High detail with complex materials. High polygon count (8000-20000). High-res textures (2048px).",
                QualityLevel.ULTRA: "Maximum detail and visual fidelity. Very high polygon count (20000+). Ultra-high textures (4096px).",
            }
            context_parts.append(quality_specs.get(quality_level, ""))

        context = "\n\n".join(context_parts)

        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add few-shot examples if enabled
        if self.config.include_examples:
            for example in self.few_shot_examples:
                messages.extend(
                    [
                        {"role": "user", "content": f"Input: {example['input']}"},
                        {"role": "assistant", "content": json.dumps(example["output"], indent=2)},
                    ]
                )

        # Add the actual request
        user_message = f"""Please enhance the following asset description with comprehensive game development specifications.

{context}

Input Description: {prompt}

Provide a detailed JSON response following the established schema with all required fields filled out appropriately for a game asset."""

        messages.append({"role": "user", "content": user_message})

        # Prepare request object
        request_data = LLMRequest(
            model=self.config.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            frequency_penalty=kwargs.get("frequency_penalty", self.config.frequency_penalty),
            presence_penalty=kwargs.get("presence_penalty", self.config.presence_penalty),
        )

        # Add response format for JSON mode if supported
        if self.config.output_format == OutputFormat.JSON:
            request_data.response_format = {"type": "json_object"}

        return request_data

    async def _make_api_call(self, request_data: LLMRequest) -> LLMResponse:
        """Make the actual API call to the LLM service."""

        # Check for demo mode
        if self.config.api_key == "demo" or not self.config.api_key or len(self.config.api_key.strip()) < 10:
            self.logger.info("Using demo mode - generating mock response")
            return self._create_demo_response(request_data)

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

    async def _process_response(self, response: LLMResponse, original_prompt: str) -> dict[str, Any]:
        """Process and validate the LLM response."""
        content = response.content.strip()

        if not content:
            raise APIError("Empty response from LLM")

        # Parse JSON response
        if self.config.output_format == OutputFormat.JSON:
            try:
                # Try to extract JSON from response
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content

                parsed_data = json.loads(json_str)

                # Validate using Pydantic model if enabled (with better error handling)
                if self.config.validate_json:
                    try:
                        validated_data = EnhancedAssetDescription(**parsed_data)
                        return validated_data.model_dump()
                    except Exception as e:
                        self.logger.warning(
                            "JSON validation failed, using raw data",
                            error=str(e),
                            raw_data=parsed_data,
                        )
                        # For GitHub Copilot API, fallback to raw data is fine
                        return parsed_data

                return parsed_data

            except json.JSONDecodeError as e:
                raise APIError(
                    message=f"Invalid JSON in LLM response: {str(e)}",
                    response_data={"content": content[:500]},
                    original_exception=e,
                )

        # For non-JSON formats, return structured text
        return {
            "enhanced_description": content,
            "original_prompt": original_prompt,
            "format": self.config.output_format.value,
        }

    def categorize_asset(self, description: str, enhanced_data: dict[str, Any]) -> dict[str, Any]:
        """Categorize the asset based on description and enhanced data."""
        category_info = {
            "primary_category": "unknown",
            "subcategory": "generic",
            "tags": [],
            "complexity_score": 0.5,
        }

        description_lower = description.lower()

        # Primary category detection
        weapon_keywords = ["sword", "blade", "axe", "bow", "staff", "dagger", "mace", "hammer"]
        armor_keywords = ["armor", "helmet", "shield", "gauntlet", "boots", "plate", "mail"]
        potion_keywords = ["potion", "elixir", "bottle", "vial", "brew", "tonic"]
        tool_keywords = ["tool", "hammer", "pickaxe", "shovel", "wrench", "key"]

        if any(keyword in description_lower for keyword in weapon_keywords):
            category_info["primary_category"] = "weapon"
        elif any(keyword in description_lower for keyword in armor_keywords):
            category_info["primary_category"] = "armor"
        elif any(keyword in description_lower for keyword in potion_keywords):
            category_info["primary_category"] = "consumable"
        elif any(keyword in description_lower for keyword in tool_keywords):
            category_info["primary_category"] = "tool"

        # Extract tags from enhanced data
        if isinstance(enhanced_data, dict):
            if "materials" in enhanced_data:
                category_info["tags"].extend(enhanced_data["materials"])

            if "style_notes" in enhanced_data:
                category_info["tags"].extend(enhanced_data["style_notes"])

            # Calculate complexity based on polygon count and features
            poly_count = enhanced_data.get("estimated_polygon_count", 1000)
            if poly_count > 10000:
                category_info["complexity_score"] = 0.9
            elif poly_count > 5000:
                category_info["complexity_score"] = 0.7
            elif poly_count > 2000:
                category_info["complexity_score"] = 0.5
            else:
                category_info["complexity_score"] = 0.3

        return category_info

    async def _perform_health_check(self) -> bool:
        """Perform health check by making a simple API call."""
        try:
            test_request = LLMRequest(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello, are you available?"}],
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

    def _create_demo_response(self, request_data: LLMRequest) -> LLMResponse:
        """Create a mock response for demo mode."""
        import json
        import time

        # Extract the prompt from request messages
        prompt = ""
        for message in request_data.messages:
            if message.get("role") == "user":
                prompt = message.get("content", "")
                break

        # Generate a realistic demo response based on the prompt
        demo_content = {
            "asset_name": "Enhanced Fantasy Weapon",
            "enhanced_description": f"A magnificent {prompt.lower()} crafted with intricate details and magical properties. This weapon features ornate engravings, a perfectly balanced design, and emanates a subtle mystical aura that suggests both power and craftsmanship.",
            "asset_category": "weapon",
            # Physical Properties
            "physical_properties": {"weight_class": "medium", "durability": "high", "grip_type": "two_handed"},
            "dimensions": {"length": 120.0, "width": 15.0, "height": 8.0},
            "weight": "3.2 kg",
            "materials": ["enchanted_steel", "leather_wrap", "magical_crystal"],
            # Visual Characteristics
            "visual_characteristics": {
                "primary_style": "fantasy",
                "detail_level": "high",
                "surface_finish": "polished_metal",
            },
            "color_palette": ["#C0C0C0", "#8B4513", "#FFD700", "#FF4500"],
            "textures": ["metal_brushed", "leather_worn", "crystal_glow"],
            "lighting_effects": ["magical_aura", "metal_reflection"],
            # Gameplay Mechanics
            "gameplay_mechanics": {
                "weapon_class": "two_handed_sword",
                "attack_speed": "medium",
                "damage_type": "slashing",
            },
            "stats": {"damage": 85, "speed": 65, "durability": 90, "magic_power": 75},
            "abilities": ["fire_enchantment", "critical_strike", "cleave_attack"],
            "rarity": "epic",
            # Technical Requirements (Required fields)
            "technical_requirements": {"target_platform": "PC/Console", "performance_tier": "high"},
            "estimated_polygon_count": 3200,
            "texture_resolution": 1024,
            "optimization_level": "standard",
            # Additional Metadata
            "style_notes": ["fantasy_medieval", "ornate_detailing", "magical_elements"],
            "inspiration_sources": ["medieval_weaponry", "fantasy_games", "magical_artifacts"],
            "modeling_complexity": "intermediate",
        }

        # Create a realistic LLM response structure
        return LLMResponse(
            id="demo_response_" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            model=request_data.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(demo_content, indent=2)},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": len(prompt.split()) * 4,  # Rough token estimation
                "completion_tokens": 200,
                "total_tokens": len(prompt.split()) * 4 + 200,
            },
        )


# Export the production implementation
__all__ = [
    "LLMConfig",
    "LLMRequest",
    "LLMResponse",
    "EnhancedAssetDescription",
    "OutputFormat",
    "LLMGenerator",
]

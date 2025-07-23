"""
Factory for creating LLM generator instances.
"""

from src.generators.llm_generator import LLMConfig, LLMGenerator
from src.utils.env_config import AppSettings


def create_llm_generator(settings: AppSettings) -> LLMGenerator | None:
    """Create LLM generator based on configuration."""
    config_dict = settings.get_llm_config()
    api_key = config_dict["api_key"]
    if api_key and api_key.strip():  # Check for non-empty, non-whitespace string
        # Extract api_key separately since it's a required positional argument
        api_key = config_dict.pop("api_key")
        config = LLMConfig(api_key=api_key, **config_dict)
        return LLMGenerator(config)
    return None

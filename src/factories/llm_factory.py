"""
Factory for creating LLM generator instances.
"""

from src.generators.llm_generator import LLMConfig, LLMGenerator
from src.utils.env_config import AppSettings


def create_llm_generator(settings: AppSettings) -> LLMGenerator | None:
    """Create LLM generator based on configuration."""
    config_dict = settings.get_llm_config()
    if config_dict["api_key"]:
        config = LLMConfig(**config_dict)
        return LLMGenerator(config)
    return None

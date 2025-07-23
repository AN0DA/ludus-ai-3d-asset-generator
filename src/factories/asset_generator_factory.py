"""
Factory for creating asset generator instances.
"""

from src.generators.asset_generator import Asset3DGenerator, ServiceConfig, ServiceProvider
from src.models.asset_model import FileFormat, QualityLevel
from src.utils.env_config import AppSettings


def create_asset_generator(settings: AppSettings) -> Asset3DGenerator | None:
    """Create asset generator based on configuration."""
    meshy_key = settings.meshy_api_key
    if meshy_key:
        configs: dict[ServiceProvider, ServiceConfig] = {
            ServiceProvider.MESHY_AI: ServiceConfig(
                api_key=meshy_key,
                base_url="https://api.meshy.ai",
                timeout_seconds=600,
                supports_text_to_3d=True,
                supports_image_to_3d=True,
                supported_output_formats=[FileFormat.GLB, FileFormat.FBX, FileFormat.OBJ, FileFormat.USDZ],
                quality_levels=[QualityLevel.DRAFT, QualityLevel.STANDARD, QualityLevel.HIGH, QualityLevel.ULTRA],
            )
        }
        return Asset3DGenerator(configs)
    return None

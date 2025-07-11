"""
UI module for the AI 3D Asset Generator.

This module provides a clean, refactored web interface for the 3D asset
generation system with simplified components and modern styling.
"""

from .app import Asset3DGeneratorUI, create_app_interface
from .utils import UIUtils
from .components import UIComponents, CUSTOM_CSS, CUSTOM_JS

# Keep compatibility with existing imports
from .interface import AssetGeneratorInterface
from .preview import ModelPreview
from .gradio_app import GradioInterface, create_gradio_app
from .unified_interface import UnifiedAssetInterface, create_unified_interface

__all__ = [
    # New refactored interface (recommended)
    'Asset3DGeneratorUI',
    'create_app_interface',
    'UIUtils',
    
    # Simplified components (backward compatibility)
    'UIComponents',
    'CUSTOM_CSS', 
    'CUSTOM_JS',
    
    # Legacy interfaces (for compatibility)
    'AssetGeneratorInterface',
    'ModelPreview',
    'GradioInterface',
    'create_gradio_app',
    'UnifiedAssetInterface',
    'create_unified_interface'
]
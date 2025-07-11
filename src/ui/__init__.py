"""
UI module for the AI 3D Asset Generator.

This module provides a clean, refactored web interface for the 3D asset
generation system with modular components, separated concerns, and modern styling.
"""

# Main refactored interface components
from .app import Asset3DGeneratorUI, create_app_interface
from .components import UIComponents
from .handlers import UIHandlers
from .styles import MODERN_CSS
from .utils import UIUtils

__all__ = [
    # Main interface (recommended)
    'Asset3DGeneratorUI',
    'create_app_interface',
    
    # Component modules
    'UIComponents',
    'UIHandlers', 
    'UIUtils',
    
    # Styling
    'MODERN_CSS'
]
"""
UI module for the AI 3D Asset Generator.

This module provides a clean, refactored web interface for the 3D asset
generation system with simplified components and modern styling.
"""

from .app import Asset3DGeneratorUI, create_app_interface

__all__ = [
    "Asset3DGeneratorUI",
    "create_app_interface",
]

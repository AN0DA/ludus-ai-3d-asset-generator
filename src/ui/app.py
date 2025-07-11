"""
Refactored Gradio UI for AI 3D Asset Generator.

This module provides the main entry point and consolidated interface
for the 3D asset generation UI. The implementation is now split across
multiple modules for better maintainability.
"""

import gradio as gr
import structlog
from typing import Any, Dict, List, Optional, Tuple

from src.core.app import AssetGenerationApp
from .styles import MODERN_CSS
from .components import UIComponents
from .handlers import UIHandlers

logger = structlog.get_logger(__name__)


class Asset3DGeneratorUI:
    """Simplified, consolidated Gradio interface for 3D asset generation."""
    
    def __init__(self, app: AssetGenerationApp):
        """Initialize the UI with the application instance."""
        self.app = app
        self.handlers = UIHandlers(app)
        self.storage_available = self.app.cloud_storage is not None
        
        # Progress refresh timer reference
        self._progress_refresh_timer = None
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            css=MODERN_CSS,
            title="AI 3D Asset Generator"
        ) as interface:
            
            # Header
            UIComponents.create_header()
            
            # Main content in tabs
            with gr.Tabs() as tabs:
                
                # Generation Tab
                with gr.Tab("🎨 Generate", id="generate"):
                    self._create_generation_tab()
                
                # Results Tab  
                with gr.Tab("📋 Results", id="results"):
                    self._create_results_tab()
                
                # History Tab
                with gr.Tab("📚 History", id="history"):
                    self._create_history_tab()
        
        return interface
    
    def _create_generation_tab(self):
        """Create the asset generation tab."""
        
        with gr.Row():
            # Input Form
            with gr.Column(scale=2):
                description, asset_type, style, quality, format_choice, generate_btn = (
                    UIComponents.create_generation_form()
                )
            
            # Progress & Status
            with gr.Column(scale=1):
                status_display, progress_display, cancel_btn = (
                    UIComponents.create_progress_section()
                )
        
        # Error display (prominently placed)
        error_display = UIComponents.create_error_display()
        
        # Help information
        UIComponents.create_help_section()
        
        # Wire up generation logic
        generate_btn.click(
            fn=self.handlers.generate_asset_sync,
            inputs=[description, asset_type, style, quality, format_choice],
            outputs=[status_display, progress_display, error_display, cancel_btn],
            show_progress="full"
        )
        
        cancel_btn.click(
            fn=self.handlers.cancel_generation_sync,
            outputs=[status_display, progress_display, cancel_btn]
        )
        
        # Add auto-refresh for progress monitoring
        refresh_timer = UIComponents.create_timer()
        refresh_timer.tick(
            fn=self.handlers.check_progress_sync,
            outputs=[status_display, progress_display, cancel_btn, refresh_timer],
            show_progress="hidden"
        )
        
        # Store timer reference
        self._progress_refresh_timer = refresh_timer
    
    def _create_results_tab(self):
        """Create the results display tab."""
        
        (model_viewer, metadata_display, download_btn, share_btn, 
         download_file, asset_list, refresh_assets_btn) = (
            UIComponents.create_results_section(self.storage_available)
        )
        
        # Initialize asset list
        asset_list.choices = self.handlers.refresh_asset_list()
        
        # Wire up asset selection
        asset_list.change(
            fn=self.handlers.display_selected_asset,
            inputs=[asset_list],
            outputs=[model_viewer, metadata_display, download_btn, share_btn]
        )
        
        # Wire up refresh button
        refresh_assets_btn.click(
            fn=self.handlers.refresh_asset_list,
            outputs=[asset_list]
        )
        
        # Wire up download button
        download_btn.click(
            fn=lambda: self.handlers.download_selected_asset(self.handlers.current_asset_key) 
                      if self.handlers.current_asset_key else None,
            outputs=[download_file]
        )
    
    def _create_history_tab(self):
        """Create the generation history tab."""
        
        history_gallery, refresh_history_btn, clear_history_btn = (
            UIComponents.create_history_section()
        )
        
        # Wire up history functions
        refresh_history_btn.click(
            fn=self.handlers.refresh_history,
            outputs=[history_gallery]
        )
        
        clear_history_btn.click(
            fn=self.handlers.clear_history,
            outputs=[history_gallery]
        )


def create_app_interface(app: AssetGenerationApp) -> gr.Blocks:
    """Create the main application interface."""
    ui = Asset3DGeneratorUI(app)
    return ui.create_interface()

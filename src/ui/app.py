"""
Main Gradio UI application for AI 3D Asset Generator.

This module orchestrates the creation of the Gradio interface, integrating
components and handlers for a seamless user experience.
"""

import gradio as gr
import structlog

from src.core.app import AssetGenerationApp

from .components import UIComponents
from .handlers import UIHandlers
from .styles import MODERN_CSS

logger = structlog.get_logger(__name__)


class Asset3DGeneratorUI:
    """Main UI class for the 3D asset generator interface."""

    def __init__(self, app: AssetGenerationApp):
        """Initialize the UI with the application instance."""
        self.app = app
        self.handlers = UIHandlers(app)
        self.components = UIComponents()
        self.storage_available = self.app.cloud_storage is not None

    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        with gr.Blocks(css=MODERN_CSS, title="AI 3D Asset Generator", theme=gr.themes.Default()) as interface:
            self.components.create_header()
            with gr.Tabs():
                with gr.Tab("🎨 Generate"):
                    self._create_generation_tab(interface)
                with gr.Tab("📋 Results"):
                    self._create_results_tab(interface)
        return interface

    def _create_generation_tab(self, interface: gr.Blocks):
        """Create the generation tab with input form and progress monitoring."""
        with gr.Row():
            with gr.Column(scale=2):
                description, asset_type, style, quality, format_choice, generate_btn = (
                    self.components.create_generation_form()
                )
            with gr.Column(scale=1):
                status_display, progress_display, cancel_btn = self.components.create_progress_section()

        error_display = self.components.create_error_display()
        self.components.create_help_section()

        # Bind generate button
        generate_btn.click(
            fn=self.handlers.generate_asset_sync,
            inputs=[description, asset_type, style, quality, format_choice],
            outputs=[status_display, progress_display, error_display, cancel_btn],
            show_progress="full",
            _js="() => { document.querySelector('.progress-container').style.opacity = '1'; }",
        )

        # Bind cancel button
        cancel_btn.click(
            fn=self.handlers.cancel_generation_sync,
            outputs=[status_display, progress_display, cancel_btn],
        )

        # Set up periodic refresh for progress
        refresh_timer = self.components.create_timer()
        refresh_timer.tick(
            fn=self.handlers.check_progress_sync,
            outputs=[status_display, progress_display, cancel_btn, refresh_timer],
            show_progress="hidden",
            _js="() => { return { active: document.querySelector('.progress-container')?.style.opacity === '1' }; }",
        )

    def _create_results_tab(self, interface: gr.Blocks):
        """Create the results tab for displaying generated assets."""
        (
            model_viewer,
            metadata_display,
            download_btn,
            share_btn,
            download_file,
            asset_list,
            refresh_assets_btn,
        ) = self.components.create_results_section(self.storage_available)

        # Bind asset selection
        asset_list.change(
            fn=self.handlers.display_selected_asset,
            inputs=[asset_list],
            outputs=[model_viewer, metadata_display, download_btn, share_btn],
            _js="() => { document.querySelector('.model-viewer').style.opacity = '0'; setTimeout(() => { document.querySelector('.model-viewer').style.opacity = '1'; }, 100); }",
        )

        # Bind refresh button
        refresh_assets_btn.click(
            fn=self.handlers.refresh_asset_list,
            outputs=[asset_list],
            _js="() => { document.querySelector('.dropdown-assets').style.animation = 'pulse 0.3s'; }",
        )

        # Bind download button
        download_btn.click(
            fn=self.handlers.download_selected_asset,
            inputs=[asset_list],
            outputs=[download_file],
        )

        # Bind share button (placeholder for future implementation)
        share_btn.click(
            fn=lambda: "Sharing functionality coming soon!",
            outputs=[gr.HTML()],
        )


def create_app_interface(app: AssetGenerationApp) -> gr.Blocks:
    """Create the main application interface."""
    ui = Asset3DGeneratorUI(app)
    return ui.create_interface()

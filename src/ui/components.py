"""
UI Components for the AI 3D Asset Generator.

This module contains methods to create various UI components for the Gradio interface.
"""

import gradio as gr
from typing import List, Tuple

from src.models.asset_model import (
    AssetType, StylePreference, QualityLevel, FileFormat
)


class UIComponents:
    """Factory class for creating UI components."""
    
    @staticmethod
    def create_header() -> None:
        """Create the application header."""
        gr.HTML("""
            <div class="app-header">
                <h1 class="app-title">🎯 AI 3D Asset Generator</h1>
                <p class="app-subtitle">Transform your ideas into stunning 3D models</p>
            </div>
        """)
    
    @staticmethod
    def create_generation_form() -> Tuple[gr.Textbox, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Button]:
        """Create the asset generation form components."""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML('<h3 class="section-title">🎨 Asset Description</h3>')
            
            description = gr.Textbox(
                label="Describe your 3D asset",
                placeholder="e.g., A mystical sword with glowing blue runes and intricate metalwork...",
                lines=4,
                elem_classes=["form-input"],
                info="Minimum 10 characters, maximum 2000 characters"
            )
            
            with gr.Row():
                asset_type = gr.Dropdown(
                    label="Asset Type",
                    choices=[(t.value.title(), t.value) for t in AssetType],
                    value=AssetType.WEAPON.value,
                    elem_classes=["form-input"],
                    info="Select the type of 3D asset to generate"
                )
                
                style = gr.Dropdown(
                    label="Art Style",
                    choices=[(s.value.replace("_", " ").title(), s.value) for s in StylePreference],
                    value=StylePreference.REALISTIC.value,
                    elem_classes=["form-input"],
                    info="Choose the visual style"
                )
            
            with gr.Row():
                quality = gr.Dropdown(
                    label="Quality Level",
                    choices=[(q.value.title(), q.value) for q in QualityLevel],
                    value=QualityLevel.STANDARD.value,
                    elem_classes=["form-input"],
                    info="Higher quality = more detail, longer generation time"
                )
                
                format_choice = gr.Dropdown(
                    label="Output Format",
                    choices=[(f.value.upper(), f.value) for f in FileFormat],
                    value=FileFormat.OBJ.value,
                    elem_classes=["form-input"],
                    info="3D model file format"
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Asset",
                variant="primary",
                elem_classes=["btn-primary"],
                size="lg"
            )
        
        return description, asset_type, style, quality, format_choice, generate_btn
    
    @staticmethod
    def create_progress_section() -> Tuple[gr.HTML, gr.HTML, gr.Button]:
        """Create the progress monitoring section."""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML('<h3 class="section-title">📊 Progress</h3>')
            
            status_display = gr.HTML(
                '<div class="status-success">Ready to generate</div>'
            )
            
            progress_display = gr.HTML(
                '''
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                    <p>Waiting to start...</p>
                </div>
                ''',
                visible=False
            )
            
            cancel_btn = gr.Button(
                "❌ Cancel Generation",
                variant="secondary",
                elem_classes=["btn-secondary"],
                visible=False
            )
        
        return status_display, progress_display, cancel_btn
    
    @staticmethod
    def create_error_display() -> gr.HTML:
        """Create the error display component."""
        return gr.HTML(visible=False, elem_classes=["error-display"])
    
    @staticmethod
    def create_help_section() -> None:
        """Create the help/tips section."""
        gr.HTML('''
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #374151;">💡 Tips for Better Results</h4>
            <ul style="margin: 0; padding-left: 1.5rem; color: #6b7280; font-size: 0.9rem;">
                <li>Be specific about materials, colors, and style</li>
                <li>Mention size and proportions if important</li>
                <li>Include details about purpose or function</li>
                <li>Higher quality levels take longer but produce better results</li>
            </ul>
        </div>
        ''')
    
    @staticmethod
    def create_results_section(storage_available: bool) -> Tuple[gr.Model3D, gr.HTML, gr.Button, gr.Button, gr.File, gr.Dropdown, gr.Button]:
        """Create the results display section."""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML('<h3 class="section-title">✨ Generated Asset</h3>')
            
            # Storage status info
            if not storage_available:
                gr.HTML('''
                <div class="status-warning">
                    ⚠️ Cloud storage not configured. Only current session assets will be available.
                </div>
                ''')
            
            with gr.Row():
                with gr.Column(scale=2):
                    model_viewer = gr.Model3D(
                        label="3D Model Preview",
                        elem_classes=["model-viewer"]
                    )
                
                with gr.Column(scale=1):
                    metadata_display = gr.HTML(
                        '<div class="no-asset">No asset loaded. Generate an asset first or select from the list below.</div>'
                    )
            
            with gr.Row():
                download_btn = gr.Button(
                    "📥 Download Model",
                    variant="primary", 
                    elem_classes=["btn-primary"],
                    visible=False
                )
                
                share_btn = gr.Button(
                    "🔗 Share Asset",
                    variant="secondary",
                    elem_classes=["btn-secondary"],
                    visible=False
                )
            
            download_file = gr.File(
                label="Download",
                visible=False
            )
            
            # Asset selection dropdown
            with gr.Row():
                with gr.Column(scale=4):
                    asset_list = gr.Dropdown(
                        label="Available Models",
                        choices=[],
                        interactive=True,
                        elem_classes=["dropdown-assets"]
                    )
                with gr.Column(scale=1):
                    refresh_assets_btn = gr.Button(
                        "🔄 Refresh",
                        variant="secondary",
                        elem_classes=["btn-secondary"],
                        size="sm"
                    )
        
        return model_viewer, metadata_display, download_btn, share_btn, download_file, asset_list, refresh_assets_btn
    
    @staticmethod
    def create_history_section() -> Tuple[gr.Gallery, gr.Button, gr.Button]:
        """Create the generation history section."""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML('<h3 class="section-title">📚 Generation History</h3>')
            
            history_gallery = gr.Gallery(
                label="Recent Generations",
                show_label=False,
                elem_classes=["gallery-container"],
                columns=3,
                height="auto"
            )
            
            with gr.Row():
                refresh_history_btn = gr.Button(
                    "🔄 Refresh",
                    elem_classes=["btn-secondary"]
                )
                
                clear_history_btn = gr.Button(
                    "🗑️ Clear History",
                    elem_classes=["btn-secondary"]
                )
        
        return history_gallery, refresh_history_btn, clear_history_btn
    
    @staticmethod
    def create_timer() -> gr.Timer:
        """Create a timer for progress monitoring."""
        return gr.Timer(value=2, active=True)

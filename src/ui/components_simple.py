"""
Simplified components module for backward compatibility.

This module provides essential components with simplified styling,
maintaining API compatibility while reducing complexity.
"""

import gradio as gr
from typing import Any, Dict, List, Tuple
from datetime import datetime

from ..models.asset_model import AssetType, StylePreference, QualityLevel, FileFormat
from .utils import UIUtils


# Simplified CSS - much shorter and focused
SIMPLE_CSS = """
/* Simple, clean styling */
.form-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.form-section h3 {
    color: #2d3748;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3b82f6;
}

.status-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid #10b981;
    color: #10b981;
    padding: 0.75rem;
    border-radius: 6px;
    font-weight: 500;
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #ef4444;
    color: #ef4444;
    padding: 0.75rem;
    border-radius: 6px;
    font-weight: 500;
}

.progress-container {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
}

.progress-bar {
    background: #e2e8f0;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.progress-fill {
    background: #3b82f6;
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
"""

# Minimal JavaScript
SIMPLE_JS = """
// Simple progress update function
function updateProgress(percentage, message) {
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (progressBar) {
        progressBar.style.width = percentage + '%';
    }
    
    if (progressText) {
        progressText.textContent = message;
    }
}
"""


class UIComponents:
    """Simplified UI components for backward compatibility."""
    
    @staticmethod
    def create_header() -> gr.HTML:
        """Create a simple header."""
        return gr.HTML(
            value="""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; border-radius: 8px; margin-bottom: 2rem;">
                <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯 AI 3D Asset Generator</h1>
                <p style="font-size: 1.1rem; opacity: 0.9;">Transform your ideas into stunning 3D models</p>
            </div>
            """
        )
    
    @staticmethod
    def create_input_form() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create a simplified input form."""
        components = {}
        
        with gr.Group(elem_classes=["form-section"]) as form_group:
            gr.HTML("<h3>🎨 Asset Description</h3>")
            
            components["description"] = gr.Textbox(
                label="Describe your 3D asset",
                placeholder="e.g., A mystical sword with glowing runes...",
                lines=3,
                info="Provide a detailed description (minimum 10 characters)"
            )
            
            with gr.Row():
                components["asset_type"] = gr.Dropdown(
                    label="Asset Type",
                    choices=[(t.value.title(), t.value) for t in AssetType],
                    value=AssetType.WEAPON.value
                )
                
                components["style"] = gr.Dropdown(
                    label="Art Style",
                    choices=[(s.value.replace("_", " ").title(), s.value) for s in StylePreference],
                    value=StylePreference.FANTASY.value
                )
            
            with gr.Row():
                components["quality"] = gr.Dropdown(
                    label="Quality Level",
                    choices=[(q.value.title(), q.value) for q in QualityLevel],
                    value=QualityLevel.STANDARD.value
                )
                
                components["format"] = gr.Dropdown(
                    label="Output Format",
                    choices=[(f.value.upper(), f.value) for f in FileFormat],
                    value=FileFormat.OBJ.value
                )
        
        return form_group, components
    
    @staticmethod
    def create_progress_display() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create a simplified progress display."""
        components = {}
        
        with gr.Group(elem_classes=["progress-container"], visible=False) as progress_group:
            gr.HTML("<h3>📊 Progress</h3>")
            
            components["status"] = gr.HTML(
                value=UIUtils.create_status_html("info", "Ready to generate")
            )
            
            components["progress_bar"] = gr.HTML(
                value=UIUtils.create_progress_html(0, "Waiting to start...")
            )
            
            components["cancel_button"] = gr.Button(
                value="Cancel Generation",
                variant="secondary",
                visible=False
            )
        
        return progress_group, components
    
    @staticmethod
    def create_results_display() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create a simplified results display."""
        components = {}
        
        with gr.Group(elem_classes=["form-section"], visible=False) as results_group:
            gr.HTML("<h3>✨ Generated Asset</h3>")
            
            with gr.Row():
                with gr.Column(scale=2):
                    components["model_viewer"] = gr.Model3D(
                        label="3D Model Preview"
                    )
                
                with gr.Column(scale=1):
                    components["thumbnail"] = gr.Image(
                        label="Thumbnail",
                        show_label=False
                    )
                    
                    components["metadata"] = gr.HTML(
                        value="<div>No metadata available</div>"
                    )
            
            components["download_model"] = gr.File(
                label="Download Model",
                visible=False
            )
        
        return results_group, components
    
    @staticmethod
    def create_error_display() -> gr.HTML:
        """Create error message display."""
        return gr.HTML(
            value="",
            visible=False,
            elem_classes=["status-error"]
        )
    
    @staticmethod
    def create_status_badge(status: str, message: str) -> str:
        """Create a status badge HTML."""
        return UIUtils.create_status_html(status, message)
    
    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        """Format asset metadata for display."""
        return UIUtils.format_metadata(metadata)
    
    @staticmethod
    def validate_inputs(
        description: str,
        asset_type: str,
        polygon_count: int = 25000,
        priority: int = 5
    ) -> Tuple[bool, List[str]]:
        """Validate form inputs."""
        return UIUtils.validate_inputs(description, asset_type, polygon_count, priority)
    
    @staticmethod
    def create_validation_feedback(errors: List[str]) -> str:
        """Create validation error feedback."""
        return UIUtils.create_validation_feedback(errors)


# Export for compatibility
CUSTOM_CSS = SIMPLE_CSS
CUSTOM_JS = SIMPLE_JS

__all__ = [
    "UIComponents",
    "CUSTOM_CSS", 
    "CUSTOM_JS"
]

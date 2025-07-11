"""
Professional Gradio UI Components for 3D Asset Generator.

This module provides reusable, validated UI components with enhanced UX
including custom forms, progress indicators, asset galleries, and error handling.
"""

import gradio as gr
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from pathlib import Path
from datetime import datetime
import asyncio

from ..models.asset_model import (
    AssetType, StylePreference, QualityLevel, FileFormat, 
    GenerationStatus, AssetMetadata, GenerationProgress
)
from ..utils.validators import ValidationException, SecurityValidationException
from ..generators.asset_generator import GenerationRequest, ServiceProvider


# Custom CSS for professional styling
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Card styling */
.asset-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.asset-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

/* Header styling */
.main-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 32px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(45deg, #fff, #f0f0f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.main-subtitle {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.9);
    font-weight: 300;
    margin-bottom: 8px;
}

/* Form styling */
.form-section {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.form-section h3 {
    color: #2d3748;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #667eea;
}

/* Input styling */
.custom-input {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.2s ease !important;
    font-size: 14px !important;
}

.custom-input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Button styling */
.generate-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.generate-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

.secondary-button {
    background: transparent !important;
    border: 2px solid #667eea !important;
    color: #667eea !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.secondary-button:hover {
    background: #667eea !important;
    color: white !important;
}

/* Progress styling */
.progress-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.progress-bar {
    background: #e2e8f0;
    border-radius: 20px;
    height: 8px;
    overflow: hidden;
    margin: 12px 0;
}

.progress-fill {
    background: linear-gradient(90deg, #667eea, #764ba2);
    height: 100%;
    border-radius: 20px;
    transition: width 0.3s ease;
}

.progress-text {
    font-size: 14px;
    color: #4a5568;
    margin: 8px 0;
}

/* Status indicators */
.status-success {
    color: #38a169 !important;
    background: rgba(56, 161, 105, 0.1) !important;
    border: 1px solid rgba(56, 161, 105, 0.2) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

.status-error {
    color: #e53e3e !important;
    background: rgba(229, 62, 62, 0.1) !important;
    border: 1px solid rgba(229, 62, 62, 0.2) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

.status-warning {
    color: #d69e2e !important;
    background: rgba(214, 158, 46, 0.1) !important;
    border: 1px solid rgba(214, 158, 46, 0.2) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

.status-info {
    color: #3182ce !important;
    background: rgba(49, 130, 206, 0.1) !important;
    border: 1px solid rgba(49, 130, 206, 0.2) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

/* Gallery styling */
.asset-gallery {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
}

.gallery-item {
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s ease;
    cursor: pointer;
}

.gallery-item:hover {
    transform: scale(1.02);
}

/* Model viewer styling */
.model-viewer {
    border-radius: 12px;
    overflow: hidden;
    background: #f7fafc;
    border: 2px solid #e2e8f0;
    min-height: 400px;
}

/* Loading spinner */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 8px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .asset-card {
        padding: 16px;
        margin: 8px 0;
    }
    
    .form-section {
        padding: 16px;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .form-section {
        background: rgba(45, 55, 72, 0.95);
        color: #f7fafc;
    }
    
    .form-section h3 {
        color: #f7fafc;
    }
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pulse {
    animation: pulse 2s infinite;
}

/* Tooltip styling */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #2d3748;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
"""

# JavaScript helpers for enhanced interactivity
CUSTOM_JS = """
// Enhanced progress updates with smooth animations
function updateProgress(percentage, message) {
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (progressBar) {
        progressBar.style.width = percentage + '%';
    }
    
    if (progressText) {
        progressText.textContent = message;
    }
    
    // Add pulse effect for active progress
    if (percentage > 0 && percentage < 100) {
        progressBar.classList.add('pulse');
    } else {
        progressBar.classList.remove('pulse');
    }
}

// Auto-scroll to results when generation completes
function scrollToResults() {
    const resultsSection = document.querySelector('#results-section');
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Enhanced file download with progress tracking
async function downloadFile(url, filename) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Download failed');
        
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.URL.revokeObjectURL(downloadUrl);
        
        // Show success message
        showToast('File downloaded successfully!', 'success');
    } catch (error) {
        showToast('Download failed: ' + error.message, 'error');
    }
}

// Toast notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#38a169' : type === 'error' ? '#e53e3e' : '#3182ce'};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
}

// Add slide animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Form validation helpers
function validateForm(formData) {
    const errors = [];
    
    if (!formData.description || formData.description.length < 10) {
        errors.push('Description must be at least 10 characters long');
    }
    
    if (formData.description && formData.description.length > 2000) {
        errors.push('Description must be less than 2000 characters');
    }
    
    return errors;
}

// Auto-resize textareas
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Add auto-resize to textareas
    document.querySelectorAll('textarea').forEach(textarea => {
        textarea.addEventListener('input', () => autoResizeTextarea(textarea));
    });
    
    // Add fade-in animation to cards
    document.querySelectorAll('.asset-card').forEach((card, index) => {
        card.style.animationDelay = (index * 0.1) + 's';
        card.classList.add('fade-in');
    });
});
"""


class UIComponents:
    """Reusable UI components for the 3D asset generator."""
    
    @staticmethod
    def create_header() -> gr.HTML:
        """Create the main application header."""
        return gr.HTML(
            value="""
            <div class="main-header">
                <h1 class="main-title">🎯 AI 3D Asset Generator</h1>
                <p class="main-subtitle">Transform your ideas into stunning 3D models</p>
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                    Powered by multiple AI services • Professional quality • Instant generation
                </p>
            </div>
            """,
            elem_classes=["fade-in"]
        )
    
    @staticmethod
    def create_input_form() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create the main input form with validation."""
        components = {}
        
        with gr.Group(elem_classes=["form-section"]) as form_group:
            gr.HTML("<h3>🎨 Asset Description</h3>")
            
            components["description"] = gr.Textbox(
                label="Describe your 3D asset",
                placeholder="E.g., A mystical sword with glowing blue runes carved into its silver blade, ornate golden handle with sapphire gems...",
                lines=4,
                max_lines=8,
                info="Be detailed and specific for best results (10-2000 characters)",
                elem_classes=["custom-input"]
            )
            
            with gr.Row():
                components["asset_type"] = gr.Dropdown(
                    label="Asset Type",
                    choices=[(t.value.title(), t.value) for t in AssetType],
                    value=AssetType.WEAPON.value,
                    info="What type of asset do you want to create?",
                    elem_classes=["custom-input"]
                )
                
                components["style"] = gr.Dropdown(
                    label="Art Style",
                    choices=[(s.value.replace("_", " ").title(), s.value) for s in StylePreference],
                    value=StylePreference.REALISTIC.value,
                    info="Choose the visual style",
                    elem_classes=["custom-input"]
                )
            
            gr.HTML("<h3>⚙️ Technical Settings</h3>")
            
            with gr.Row():
                components["quality"] = gr.Dropdown(
                    label="Quality Level",
                    choices=[(q.value.title(), q.value) for q in QualityLevel],
                    value=QualityLevel.STANDARD.value,
                    info="Higher quality = longer generation time + higher cost",
                    elem_classes=["custom-input"]
                )
                
                components["format"] = gr.Dropdown(
                    label="Output Format",
                    choices=[(f.value.upper(), f.value) for f in FileFormat],
                    value=FileFormat.OBJ.value,
                    info="Choose your preferred 3D file format",
                    elem_classes=["custom-input"]
                )
            
            with gr.Row():
                components["polygon_count"] = gr.Slider(
                    label="Max Polygon Count",
                    minimum=1000,
                    maximum=100000,
                    value=25000,
                    step=1000,
                    info="Lower values = faster generation, higher values = more detail",
                    elem_classes=["custom-input"]
                )
                
                components["priority"] = gr.Slider(
                    label="Priority Level",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    info="Higher priority may reduce wait time",
                    elem_classes=["custom-input"]
                )
            
            gr.HTML("<h3>🎯 Advanced Options</h3>")
            
            with gr.Row():
                components["service"] = gr.Dropdown(
                    label="Preferred Service",
                    choices=[
                        ("Auto (Recommended)", "auto"),
                        ("Meshy AI", ServiceProvider.MESHY_AI.value),
                        ("Kaedim", ServiceProvider.KAEDIM.value),
                        ("Image-to-3D", ServiceProvider.OPENAI_DALLE.value)
                    ],
                    value="auto",
                    info="Choose generation service or let AI decide",
                    elem_classes=["custom-input"]
                )
                
                components["include_thumbnail"] = gr.Checkbox(
                    label="Generate Thumbnail",
                    value=True,
                    info="Create a preview image of your 3D model"
                )
            
            components["reference_images"] = gr.File(
                label="Reference Images (Optional)",
                file_count="multiple",
                file_types=["image"]
            )
        
        return form_group, components
    
    @staticmethod
    def create_progress_display() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create progress display components."""
        components = {}
        
        with gr.Group(elem_classes=["progress-container"], visible=False) as progress_group:
            gr.HTML("<h3>🚀 Generation Progress</h3>")
            
            components["status"] = gr.HTML(
                value="<div class='status-info'>Ready to generate</div>"
            )
            
            components["progress_bar"] = gr.HTML(
                value="""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                """
            )
            
            components["progress_text"] = gr.HTML(
                value="<div class='progress-text'>Waiting to start...</div>"
            )
            
            components["service_info"] = gr.HTML(
                value="<div class='status-info'>Service: Not selected</div>"
            )
            
            components["estimated_time"] = gr.HTML(
                value="<div class='progress-text'>Estimated time: Unknown</div>"
            )
            
            components["cancel_button"] = gr.Button(
                value="Cancel Generation",
                variant="secondary",
                visible=False,
                elem_classes=["secondary-button"]
            )
        
        return progress_group, components
    
    @staticmethod
    def create_results_display() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create results display components."""
        components = {}
        
        with gr.Group(elem_classes=["asset-card"], visible=False, elem_id="results-section") as results_group:
            gr.HTML("<h3>✨ Generated Asset</h3>")
            
            with gr.Row():
                with gr.Column(scale=2):
                    components["model_viewer"] = gr.Model3D(
                        label="3D Model Preview",
                        elem_classes=["model-viewer"]
                    )
                
                with gr.Column(scale=1):
                    components["thumbnail"] = gr.Image(
                        label="Thumbnail",
                        show_label=False,
                        container=False
                    )
                    
                    components["metadata"] = gr.HTML(
                        value="<div>No metadata available</div>"
                    )
            
            gr.HTML("<h3>📥 Downloads</h3>")
            
            with gr.Row():
                components["download_model"] = gr.File(
                    label="3D Model File",
                    visible=False
                )
                
                components["download_info"] = gr.HTML(
                    value="<div class='status-info'>No files available</div>"
                )
            
            with gr.Row():
                components["download_button"] = gr.Button(
                    value="📥 Download Model",
                    variant="primary",
                    visible=False,
                    elem_classes=["secondary-button"]
                )
                
                components["share_button"] = gr.Button(
                    value="🔗 Share Asset",
                    variant="secondary",
                    visible=False,
                    elem_classes=["secondary-button"]
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
    def create_gallery() -> Tuple[gr.Group, gr.Gallery]:
        """Create asset gallery for recent generations."""
        with gr.Group(elem_classes=["asset-gallery"]) as gallery_group:
            gr.HTML("<h3>🎨 Recent Generations</h3>")
            
            gallery = gr.Gallery(
                label="Asset Gallery",
                show_label=False,
                elem_classes=["gallery-item"],
                columns=3,
                rows=2,
                height="auto",
                object_fit="cover"
            )
        
        return gallery_group, gallery
    
    @staticmethod
    def create_loading_spinner(message: str = "Loading...") -> gr.HTML:
        """Create a loading spinner with message."""
        return gr.HTML(
            value=f"""
            <div style="text-align: center; padding: 20px;">
                <div class="loading-spinner"></div>
                <span>{message}</span>
            </div>
            """,
            visible=False
        )
    
    @staticmethod
    def create_status_badge(status: str, message: str) -> str:
        """Create a status badge HTML."""
        status_classes = {
            "success": "status-success",
            "error": "status-error", 
            "warning": "status-warning",
            "info": "status-info"
        }
        
        class_name = status_classes.get(status, "status-info")
        icon = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️", 
            "info": "ℹ️"
        }.get(status, "ℹ️")
        
        return f'<div class="{class_name}">{icon} {message}</div>'
    
    @staticmethod
    def create_service_status_display() -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create service status monitoring display."""
        components = {}
        
        with gr.Group(elem_classes=["form-section"]) as status_group:
            gr.HTML("<h3>🏥 Service Health Status</h3>")
            
            components["meshy_status"] = gr.HTML(
                value=UIComponents.create_status_badge("info", "Meshy AI: Checking...")
            )
            
            components["kaedim_status"] = gr.HTML(
                value=UIComponents.create_status_badge("info", "Kaedim: Checking...")
            )
            
            components["dalle_status"] = gr.HTML(
                value=UIComponents.create_status_badge("info", "DALL-E: Checking...")
            )
            
            with gr.Row():
                components["refresh_button"] = gr.Button(
                    value="🔄 Refresh Status",
                    variant="secondary",
                    elem_classes=["secondary-button"]
                )
                
                components["last_updated"] = gr.HTML(
                    value=f"<small>Last updated: {datetime.now().strftime('%H:%M:%S')}</small>"
                )
        
        return status_group, components
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        """Format asset metadata for display."""
        if not metadata:
            return "<div class='status-info'>No metadata available</div>"
        
        html_parts = ["<div class='metadata-display'>"]
        
        # Basic info
        if "polygon_count" in metadata:
            html_parts.append(f"<p><strong>Polygons:</strong> {metadata['polygon_count']:,}</p>")
        
        if "file_size" in metadata:
            size_str = UIComponents.format_file_size(metadata["file_size"])
            html_parts.append(f"<p><strong>File Size:</strong> {size_str}</p>")
        
        if "generation_time" in metadata:
            html_parts.append(f"<p><strong>Generation Time:</strong> {metadata['generation_time']:.1f}s</p>")
        
        if "cost" in metadata:
            html_parts.append(f"<p><strong>Cost:</strong> ${metadata['cost']:.2f}</p>")
        
        if "service" in metadata:
            service_name = metadata["service"].replace("_", " ").title()
            html_parts.append(f"<p><strong>Service:</strong> {service_name}</p>")
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    @staticmethod
    def validate_inputs(
        description: str,
        asset_type: str,
        polygon_count: int,
        priority: int
    ) -> Tuple[bool, List[str]]:
        """Validate form inputs and return errors if any."""
        errors = []
        
        # Description validation
        if not description or len(description.strip()) < 10:
            errors.append("Description must be at least 10 characters long")
        elif len(description) > 2000:
            errors.append("Description must be less than 2000 characters")
        
        # Check for potentially dangerous content
        dangerous_patterns = [
            '<script', 'javascript:', 'vbscript:', r'on\w+\s*=',
            'eval(', 'setTimeout(', 'setInterval('
        ]
        
        for pattern in dangerous_patterns:
            if pattern.lower() in description.lower():
                errors.append("Description contains potentially unsafe content")
                break
        
        # Technical validation
        if polygon_count < 1000 or polygon_count > 100000:
            errors.append("Polygon count must be between 1,000 and 100,000")
        
        if priority < 1 or priority > 10:
            errors.append("Priority must be between 1 and 10")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def create_validation_feedback(errors: List[str]) -> str:
        """Create HTML for validation error feedback."""
        if not errors:
            return ""
        
        html = '<div class="status-error"><strong>Please fix the following issues:</strong><ul>'
        for error in errors:
            html += f'<li>{error}</li>'
        html += '</ul></div>'
        
        return html


# Export components and utilities
__all__ = [
    "UIComponents",
    "CUSTOM_CSS",
    "CUSTOM_JS"
]

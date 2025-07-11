"""
Refactored Gradio UI for AI 3D Asset Generator.

This module provides a clean, consolidated Gradio interface with improved styling,
simplified structure, and better user experience.
"""

import gradio as gr
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import structlog

from src.core.app import AssetGenerationApp
from src.models.asset_model import (
    AssetType, StylePreference, QualityLevel, FileFormat,
    GenerationStatus, AssetMetadata
)
from src.utils.validators import ValidationException

logger = structlog.get_logger(__name__)


# Simplified, modern CSS
MODERN_CSS = """
/* Modern Variables */
:root {
    --primary-color: #2563eb;
    --primary-light: #3b82f6;
    --secondary-color: #64748b;
    --success-color: #059669;
    --warning-color: #d97706;
    --error-color: #dc2626;
    --background: #f8fafc;
    --surface: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --radius: 8px;
    --radius-lg: 12px;
}

/* Base Styling */
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--background);
    color: var(--text-primary);
}

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
    color: white;
    text-align: center;
    padding: 2rem;
    border-radius: var(--radius-lg);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
}

.app-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.app-subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 400;
}

/* Cards and Sections */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-color);
}

/* Form Elements */
.form-input {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease !important;
}

.form-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Buttons */
.btn-primary {
    background: var(--primary-color) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    background: var(--primary-light) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: transparent !important;
    border: 2px solid var(--secondary-color) !important;
    color: var(--secondary-color) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover {
    background: var(--secondary-color) !important;
    color: white !important;
}

/* Status Messages */
.status-success {
    background: rgba(5, 150, 105, 0.1) !important;
    border: 1px solid var(--success-color) !important;
    color: var(--success-color) !important;
    padding: 0.75rem !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 !important;
}

.status-error {
    background: rgba(220, 38, 38, 0.1) !important;
    border: 1px solid var(--error-color) !important;
    color: var(--error-color) !important;
    padding: 0.75rem !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 !important;
    animation: shake 0.5s ease-in-out !important;
}

.status-error strong {
    font-weight: 600 !important;
    color: #b91c1c !important;
}

.status-error small {
    font-size: 0.85em !important;
    opacity: 0.8 !important;
    display: block !important;
    margin-top: 0.25rem !important;
}

.status-warning {
    background: rgba(217, 119, 6, 0.1) !important;
    border: 1px solid var(--warning-color) !important;
    color: var(--warning-color) !important;
    padding: 0.75rem !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 !important;
}

/* Shake animation for errors */
@keyframes shake {
    0%, 20%, 50%, 80%, 100% {
        transform: translateX(0);
    }
    10%, 30%, 70%, 90% {
        transform: translateX(-3px);
    }
    40%, 60% {
        transform: translateX(3px);
    }
}

/* Progress Bar */
.progress-container {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 1rem 0;
}

.progress-bar {
    background: var(--border-color);
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.progress-fill {
    background: var(--primary-color);
    height: 100%;
    border-radius: 9999px;
    transition: width 0.3s ease;
}

/* Model Viewer */
.model-viewer {
    border: 2px solid var(--border-color);
    border-radius: var(--radius-lg);
    min-height: 400px;
    background: var(--surface);
}

/* Results Tab Styles */
.asset-metadata {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 0.5rem 0;
}

.asset-metadata h4 {
    margin: 0 0 1rem 0;
    color: var(--primary-color);
    font-size: 1.2rem;
}

.metadata-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.metadata-grid div {
    padding: 0.5rem;
    background: var(--background);
    border-radius: var(--radius-sm);
    font-size: 0.9rem;
}

/* Asset dropdown styling */
.dropdown-assets {
    margin-top: 1rem;
}

.dropdown-assets label {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.dropdown-assets select {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

.dropdown-assets select:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

.description {
    padding: 1rem;
    background: var(--background);
    border-radius: var(--radius);
    border-left: 4px solid var(--primary-color);
}

.no-asset {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    font-style: italic;
}

/* Responsive Design */
@media (max-width: 768px) {
    .app-title {
        font-size: 2rem;
    }
    
    .section-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
}

/* Loading Animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Gallery */
.gallery-container {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1rem;
}

/* Error Display */
.error-display {
    margin: 1rem 0 !important;
    border-radius: var(--radius) !important;
}

/* Form improvements */
.form-input {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.form-input:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Generation button enhancements */
.btn-primary {
    position: relative !important;
    overflow: hidden !important;
}

.btn-primary:hover {
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}

.btn-primary:disabled {
    opacity: 0.6 !important;
    cursor: not-allowed !important;
    transform: none !important;
}
"""


class Asset3DGeneratorUI:
    """Simplified, consolidated Gradio interface for 3D asset generation."""
    
    def __init__(self, app: AssetGenerationApp):
        """Initialize the UI with the application instance."""
        self.app = app
        self.generation_history = []
        self.current_task_id = None
        self._progress_refresh_timer = None
        # Track the latest generated asset for the Results tab
        self.latest_asset_info = None
        
        # Check if cloud storage is available
        self.storage_available = self.app.cloud_storage is not None
        
        # Track currently selected asset for downloads
        self.current_asset_key = None
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            css=MODERN_CSS,
            title="AI 3D Asset Generator"
        ) as interface:
            
            # Header
            self._create_header()
            
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
    
    def _create_header(self):
        """Create the application header."""
        gr.HTML("""
            <div class="app-header">
                <h1 class="app-title">🎯 AI 3D Asset Generator</h1>
                <p class="app-subtitle">Transform your ideas into stunning 3D models</p>
            </div>
        """)
    
    def _create_generation_tab(self):
        """Create the asset generation tab."""
        
        with gr.Row():
            # Input Form
            with gr.Column(scale=2):
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
            
            # Progress & Status
            with gr.Column(scale=1):
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
        
        # Error display (prominently placed)
        error_display = gr.HTML(visible=False, elem_classes=["error-display"])
        
        # Help information
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
        
        # Wire up generation logic
        generate_btn.click(
            fn=self._generate_asset,  # Use the async version directly
            inputs=[description, asset_type, style, quality, format_choice],
            outputs=[status_display, progress_display, error_display, cancel_btn],
            show_progress="full"
        )
        
        cancel_btn.click(
            fn=self._cancel_generation_async,  # Use async version directly
            outputs=[status_display, progress_display, cancel_btn]
        )
        
        # Add auto-refresh for progress monitoring
        # This will check for updates every 2 seconds when a task is running
        refresh_timer = gr.Timer(value=2, active=False)
        refresh_timer.tick(
            fn=self._check_progress_sync,
            outputs=[status_display, progress_display, cancel_btn, refresh_timer],
            show_progress="hidden"
        )
        
        # Store components for access
        self._progress_refresh_timer = refresh_timer
    
    def _create_results_tab(self):
        """Create the results display tab."""
        
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML('<h3 class="section-title">✨ Generated Asset</h3>')
            
            # Storage status info
            if not self.storage_available:
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
            # Expandable list of models in /assets, sorted by date added (newest first)
            with gr.Row():
                with gr.Column(scale=4):
                    asset_list = gr.Dropdown(
                        label="Available Models",
                        choices=self._refresh_asset_list(),
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
        

        
        # Wire up asset selection
        asset_list.change(
            fn=self._display_selected_asset,
            inputs=[asset_list],
            outputs=[model_viewer, metadata_display, download_btn, share_btn]
        )
        
        # Wire up refresh button
        refresh_assets_btn.click(
            fn=self._refresh_asset_list,
            outputs=[asset_list]
        )
        
        # Wire up download button
        download_btn.click(
            fn=lambda: self._download_selected_asset(self.current_asset_key) if self.current_asset_key else None,
            outputs=[download_file]
        )
    
    def _create_history_tab(self):
        """Create the generation history tab."""
        
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
        
        # Wire up history functions
        refresh_history_btn.click(
            fn=self._refresh_history,
            outputs=[history_gallery]
        )
        
        clear_history_btn.click(
            fn=self._clear_history,
            outputs=[history_gallery]
        )
    
    def _create_detailed_error_message(self, error: Exception) -> str:
        """Create detailed, user-friendly error messages."""
        error_str = str(error).lower()
        
        # Check for specific error types and provide helpful messages
        if "invalid asset type" in error_str or "asset type" in error_str:
            valid_types = ", ".join([t.value for t in AssetType])
            return f'''
            <div class="status-error">
                ❌ <strong>Invalid Asset Type:</strong> The selected asset type is not supported.
                <br><small>Please choose from: {valid_types}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            '''
        
        elif "invalid style" in error_str or "style preference" in error_str:
            valid_styles = ", ".join([s.value for s in StylePreference])
            return f'''
            <div class="status-error">
                ❌ <strong>Invalid Style:</strong> The selected art style is not supported.
                <br><small>Please choose from: {valid_styles}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            '''
        
        elif "invalid quality" in error_str or "quality level" in error_str:
            valid_qualities = ", ".join([q.value for q in QualityLevel])
            return f'''
            <div class="status-error">
                ❌ <strong>Invalid Quality Level:</strong> The selected quality is not supported.
                <br><small>Please choose from: {valid_qualities}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            '''
        
        elif "description" in error_str and ("short" in error_str or "length" in error_str):
            return '''
            <div class="status-error">
                ❌ <strong>Description Too Short:</strong> Please provide more detail about your asset.
                <br><small>Minimum 10 characters required. Describe the appearance, materials, and style.</small>
            </div>
            '''
        
        elif "api" in error_str or "network" in error_str or "connection" in error_str:
            return '''
            <div class="status-error">
                ❌ <strong>Service Unavailable:</strong> Unable to connect to the AI generation service.
                <br><small>This may be a temporary issue. Please try again in a few moments.</small>
                <br><small>If the problem persists, please contact support.</small>
            </div>
            '''
        
        elif "rate limit" in error_str or "quota" in error_str:
            return '''
            <div class="status-error">
                ❌ <strong>Rate Limit Reached:</strong> Too many requests in a short time.
                <br><small>Please wait a few minutes before trying again.</small>
                <br><small>Consider using a lower quality setting to reduce processing time.</small>
            </div>
            '''
        
        elif "timeout" in error_str:
            return '''
            <div class="status-error">
                ❌ <strong>Generation Timeout:</strong> The generation took too long and was cancelled.
                <br><small>Try using a lower quality setting or simpler description.</small>
                <br><small>High-quality generations can take several minutes.</small>
            </div>
            '''
        
        else:
            # Generic error message
            return f'''
            <div class="status-error">
                ❌ <strong>Generation Error:</strong> An unexpected error occurred.
                <br><small>Error details: {str(error)}</small>
                <br><small>Please try again or contact support if the problem persists.</small>
            </div>
            '''
    
    async def _generate_asset(
        self,
        description: str,
        asset_type: str,
        style: str,
        quality: str,
        format_choice: str
    ) -> Tuple[str, str, str, bool]:
        """Generate a 3D asset with the given parameters."""
        
        try:
            logger.info("_generate_asset called", description=description[:30])
            
            # Validate inputs
            if not description or len(description.strip()) < 10:
                error_msg = '''
                <div class="status-error">
                    ❌ <strong>Validation Error:</strong> Description must be at least 10 characters long.
                    <br><small>Please provide more detail about your 3D asset.</small>
                </div>
                '''
                return "", "", error_msg, False
            
            if len(description.strip()) > 2000:
                error_msg = '''
                <div class="status-error">
                    ❌ <strong>Validation Error:</strong> Description is too long (max 2000 characters).
                    <br><small>Please shorten your description.</small>
                </div>
                '''
                return "", "", error_msg, False
            
            # Validate asset type
            try:
                asset_type_enum = AssetType(asset_type)
            except ValueError:
                valid_types = ", ".join([t.value for t in AssetType])
                error_msg = f'''
                <div class="status-error">
                    ❌ <strong>Invalid Asset Type:</strong> "{asset_type}" is not valid.
                    <br><small>Valid types: {valid_types}</small>
                </div>
                '''
                return "", "", error_msg, False
            
            # Validate style preference
            style_enum = None
            if style and style.lower() != "none":
                try:
                    style_enum = StylePreference(style)
                except ValueError:
                    valid_styles = ", ".join([s.value for s in StylePreference])
                    error_msg = f'''
                    <div class="status-error">
                        ❌ <strong>Invalid Style:</strong> "{style}" is not valid.
                        <br><small>Valid styles: {valid_styles}</small>
                    </div>
                    '''
                    return "", "", error_msg, False
            
            # Validate quality level
            try:
                quality_enum = QualityLevel(quality)
            except ValueError:
                valid_qualities = ", ".join([q.value for q in QualityLevel])
                error_msg = f'''
                <div class="status-error">
                    ❌ <strong>Invalid Quality:</strong> "{quality}" is not valid.
                    <br><small>Valid qualities: {valid_qualities}</small>
                </div>
                '''
                return "", "", error_msg, False
            
            logger.info("Validation passed, creating session")
            
            # Start generation
            session_id = self.app.session_manager.create_session()
            
            logger.info("Starting asset generation", 
                       description=description[:50], 
                       asset_type=asset_type_enum.value,
                       style=style_enum.value if style_enum else None,
                       quality=quality_enum.value)
            
            logger.info("About to call generate_asset_pipeline")
            
            # Call the generate_asset_pipeline method to start the task
            task_id, session_id = await self.app.generate_asset_pipeline(
                description=description.strip(),
                asset_type=asset_type_enum,
                style_preference=style_enum,
                quality_level=quality_enum,
                session_id=session_id
            )
            self.current_task_id = task_id
            
            logger.info("Pipeline call completed", task_id=task_id)
            
            # Wait a moment to let the task start and get initial status
            await asyncio.sleep(0.1)
            
            logger.info("Checking task status")
            
            # Check the task status to see if it started successfully
            task_status = self.app.task_manager.get_task_status(task_id)
            
            logger.info("Task status retrieved", status=task_status.get("status") if task_status else "None")
            
            if task_status.get("status") == "not_found":
                error_msg = '''
                <div class="status-error">
                    ❌ <strong>Task Creation Failed:</strong> Unable to start generation task.
                    <br><small>Please try again or contact support.</small>
                </div>
                '''
                return "", "", error_msg, False
            
            status_html = '''
            <div class="status-success">
                🚀 <strong>Generation Started!</strong>
                <br><small>AI is working on your 3D asset...</small>
            </div>
            '''
            progress_html = '''
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 5%"></div>
                    </div>
                    <p>🔄 Initializing generation pipeline...</p>
                </div>
            '''
            
            logger.info("Returning success response")
            return status_html, progress_html, "", True
            
        except ValidationException as e:
            logger.warning("Validation error in asset generation", error=str(e))
            return "", "", self._create_detailed_error_message(e), False
            
        except ValueError as e:
            logger.warning("Value error in asset generation", error=str(e))
            return "", "", self._create_detailed_error_message(e), False
            
        except Exception as e:
            logger.error("Unexpected error in asset generation", error=str(e), exc_info=True)
            error_msg = self._create_detailed_error_message(e)
            return "", "", error_msg, False
    
    def _generate_asset_sync(
        self,
        description: str,
        asset_type: str,
        style: str,
        quality: str,
        format_choice: str
    ) -> Tuple[str, str, str, bool]:
        """Sync wrapper for the async _generate_asset method."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, need to run in a thread
                import concurrent.futures
                import threading
                
                def run_async_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self._generate_asset(description, asset_type, style, quality, format_choice)
                        )
                    finally:
                        new_loop.close()
                
                # Run in a thread to avoid blocking the main event loop
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_new_loop)
                    result = future.result(timeout=30)  # 30 second timeout
                    
            except RuntimeError:
                # No event loop running, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        self._generate_asset(description, asset_type, style, quality, format_choice)
                    )
                finally:
                    loop.close()
            
            # Start the progress timer if generation started successfully
            if len(result) >= 4 and result[3]:  # cancel_btn visible means task started
                if hasattr(self, '_progress_refresh_timer') and self._progress_refresh_timer:
                    self._progress_refresh_timer.active = True
            return result
                
        except Exception as e:
            logger.error("Error in sync wrapper", error=str(e), exc_info=True)
            error_msg = f'''
            <div class="status-error">
                ❌ <strong>System Error:</strong> Failed to process request.
                <br><small>Error: {str(e)}</small>
            </div>
            '''
            return "", "", error_msg, False

    async def _cancel_generation_async(self) -> Tuple[str, str, bool]:
        """Cancel the current generation."""
        
        if self.current_task_id:
            try:
                # Cancel via task manager
                success = self.app.task_manager.cancel_task(self.current_task_id)
                self.current_task_id = None
                
                if success:
                    status_html = '''
                    <div class="status-warning">
                        ⚠️ <strong>Generation Cancelled</strong>
                        <br><small>You can start a new generation anytime.</small>
                    </div>
                    '''
                    progress_html = ""
                    return status_html, progress_html, False
                else:
                    status_html = '''
                    <div class="status-error">
                        ❌ <strong>Failed to Cancel:</strong> Unable to stop the generation.
                        <br><small>The generation may have already completed or failed.</small>
                    </div>
                    '''
                    return status_html, "", False
                    
            except Exception as e:
                logger.error("Cancel failed", error=str(e))
                status_html = f'''
                <div class="status-error">
                    ❌ <strong>Cancel Error:</strong> {str(e)}
                    <br><small>The generation may still be running in the background.</small>
                </div>
                '''
                return status_html, "", False
        
        return '''
        <div class="status-success">
            Ready to generate
            <br><small>No active generation to cancel.</small>
        </div>
        ''', "", False
    
    def _cancel_generation_sync(self) -> Tuple[str, str, bool]:
        """Sync wrapper for the async _cancel_generation method."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self._cancel_generation_async())
                # Stop the progress timer
                if hasattr(self, '_progress_refresh_timer') and self._progress_refresh_timer:
                    self._progress_refresh_timer.active = False
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error("Error in cancel sync wrapper", error=str(e))
            return '''
            <div class="status-error">
                ❌ <strong>Cancel Error:</strong> Failed to cancel generation.
            </div>
            ''', "", False

    def _check_progress_sync(self) -> Tuple[str, str, bool, bool]:
        """Check the progress of the current generation task."""
        try:
            if not self.current_task_id:
                # No active task, stop the timer
                return '', '', False, False
            
            # Get task status
            task_status = self.app.task_manager.get_task_status(self.current_task_id)
            
            if not task_status or task_status.get("status") == "not_found":
                # Task not found, stop monitoring
                self.current_task_id = None
                return '''
                <div class="status-error">
                    ❌ <strong>Task Lost:</strong> Unable to find generation task.
                </div>
                ''', '', False, False
            
            status = task_status.get("status", "unknown")
            progress = task_status.get("progress", 0.0)
            message = task_status.get("message", "Processing...")
            current_step = task_status.get("current_step", "unknown")
            
            if status == "completed":
                # Task completed successfully
                self.current_task_id = None
                
                # Enhanced completion message with view results button
                completion_html = '''
                <div class="status-success">
                    ✅ <strong>Generation Complete!</strong>
                    <br><small>Your 3D asset has been generated successfully.</small>
                    <br><br>
                    <button onclick="switchToResultsTab()" class="view-results-btn" style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-weight: bold;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(102, 126, 234, 0.6)'" 
                       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(102, 126, 234, 0.4)'">
                        🎯 View Results
                    </button>
                </div>
                
                <script>
                function switchToResultsTab() {
                    // Find the Results tab and click it
                    const tabs = document.querySelectorAll('[role="tab"]');
                    for (let tab of tabs) {
                        if (tab.textContent.includes('Results') || tab.textContent.includes('📋')) {
                            tab.click();
                            // Give tab time to switch, then refresh the asset list
                            setTimeout(() => {
                                const refreshBtn = document.querySelector('button:contains("🔄 Refresh")') || 
                                                 [...document.querySelectorAll('button')].find(btn => 
                                                     btn.textContent.includes('🔄 Refresh'));
                                if (refreshBtn) {
                                    refreshBtn.click();
                                }
                            }, 500);
                            break;
                        }
                    }
                }
                </script>
                '''
                return completion_html, '', False, False
            
            elif status == "failed":
                # Task failed
                error_msg = task_status.get("error", "Unknown error occurred")
                self.current_task_id = None
                return f'''
                <div class="status-error">
                    ❌ <strong>Generation Failed:</strong> {error_msg}
                    <br><small>Please try again with different parameters.</small>
                </div>
                ''', '', False, False
            
            elif status == "cancelled":
                # Task was cancelled
                self.current_task_id = None
                return '''
                <div class="status-warning">
                    ⚠️ <strong>Generation Cancelled</strong>
                    <br><small>You can start a new generation anytime.</small>
                </div>
                ''', '', False, False
            
            else:
                # Task is still running, update progress
                progress_percent = max(0, min(100, progress * 100))
                progress_html = f'''
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress_percent:.1f}%"></div>
                    </div>
                    <p>🔄 {message} ({progress_percent:.1f}%)</p>
                    <small>Current step: {current_step}</small>
                </div>
                '''
                return '', progress_html, True, True  # Keep timer active
                
        except Exception as e:
            logger.error("Error checking progress", error=str(e))
            self.current_task_id = None
            return f'''
            <div class="status-error">
                ❌ <strong>Progress Check Failed:</strong> {str(e)}
            </div>
            ''', '', False, False
    
    def _refresh_history(self) -> List:
        """Refresh the generation history."""
        # Placeholder - would fetch from app.get_session_history()
        return []
    
    def _clear_history(self) -> List:
        """Clear the generation history."""
        self.generation_history.clear()
        return []
    
    def _get_latest_asset_info(self):
        """Get information about the latest generated asset."""
        try:
            if not self.current_task_id:
                return None, "No recent generation found", None, False, False
            
            # Get the completed task result
            task_status = self.app.task_manager.get_task_status(self.current_task_id)
            if not task_status or task_status.get("status") != "completed":
                return None, "No completed generation found", None, False, False
            
            # Get the result data from the task
            result_data = task_status.get("result")
            if not result_data:
                return None, "No result data available", None, False, False
            
            # Extract asset information
            asset_name = result_data.get("asset_name", "Generated Asset")
            model_url = result_data.get("model_url")
            thumbnail_url = result_data.get("thumbnail_url")
            
            # Create metadata HTML
            metadata_html = f'''
            <div class="asset-metadata">
                <h4>✨ {asset_name}</h4>
                <div class="metadata-grid">
                    <div><strong>Type:</strong> {result_data.get("asset_type", "Unknown")}</div>
                    <div><strong>Quality:</strong> {result_data.get("quality_level", "Standard")}</div>
                    <div><strong>Format:</strong> {result_data.get("file_format", "OBJ")}</div>
                    <div><strong>Size:</strong> {result_data.get("file_size", "Unknown")} bytes</div>
                    <div><strong>Polygons:</strong> {result_data.get("polygon_count", "Unknown")}</div>
                    <div><strong>Generated:</strong> {result_data.get("timestamp", "Recently")}</div>
                </div>
                <div class="description">
                    <strong>Description:</strong><br>
                    <em>{result_data.get("original_description", "No description available")}</em>
                </div>
            </div>
            '''
            
            return model_url, metadata_html, thumbnail_url, True, True
            
        except Exception as e:
            logger.error(f"Error getting latest asset info: {e}")
            return None, f"Error loading asset: {str(e)}", None, False, False

    def _load_latest_asset(self):
        """Load the latest generated asset into the Results tab."""
        return self._get_latest_asset_info()
    


    def _display_selected_asset(self, asset_key):
        """Display selected asset from the dropdown."""
        try:
            import json
            import asyncio
            
            # Use the app's existing storage instance
            storage = self.app.cloud_storage
            if not storage:
                error_html = '<div class="status-error">Cloud storage not configured.</div>'
                return None, error_html, False, False
            
            if not asset_key:
                return None, '<div class="no-asset">Please select an asset.</div>', False, False
            
            # Get file info
            file_info = asyncio.run(storage.get_file_info(asset_key))
            
            # Extract model name from file path
            model_name = asset_key.split("/")[-1].split(".")[0]
            metadata_key = f"metadata/{model_name}.json"
            
            # Try to load metadata
            try:
                metadata_bytes = asyncio.run(storage.download_bytes(metadata_key))
                metadata = json.loads(metadata_bytes.decode())
            except Exception as e:
                logger.warning(f"Could not load metadata for {model_name}: {e}")
                metadata = {
                    "name": model_name,
                    "file_size": file_info.size,
                    "last_modified": file_info.last_modified.strftime("%Y-%m-%d %H:%M:%S"),
                    "content_type": file_info.content_type
                }
            
            # For localhost/MinIO setups, we can't display 3D models directly in Gradio
            # due to SSRF protection. Instead, we'll focus on metadata and download options.
            model_url = None
            
            # Try to generate a public URL only if it's not localhost
            if hasattr(storage, 'generate_public_url'):
                try:
                    public_url = asyncio.run(storage.generate_public_url(asset_key))
                    logger.info(f"Generated public URL for {asset_key}: {public_url}")
                    if public_url and 'localhost' not in public_url and public_url.startswith(('http://', 'https://')):
                        model_url = public_url
                    else:
                        logger.info(f"Localhost URL detected, skipping 3D preview: {public_url}")
                except Exception as e:
                    logger.warning(f"Could not generate public URL: {e}")
            
            metadata_html = self._format_metadata_html(metadata)
            
            # If no valid model URL, add a note to metadata
            if not model_url:
                metadata_html += '''
                <div class="status-info" style="margin-top: 1rem; padding: 1rem; background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px;">
                    📁 <strong>File Information:</strong><br>
                    • Asset is stored in cloud storage<br>
                    • 3D preview not available (localhost storage)<br>
                    • Use download button to access the file
                </div>
                '''
            
            # Track the current asset for downloads
            self.current_asset_key = asset_key
            
            return model_url, metadata_html, True, True
            
        except Exception as e:
            logger.error(f"Error displaying selected asset {asset_key}: {e}")
            error_html = f'<div class="status-error">Error loading asset: {str(e)}</div>'
            return None, error_html, False, False

    def _format_metadata_html(self, metadata):
        """Format metadata for display."""
        if not metadata:
            return '<div class="no-asset">No metadata found.</div>'
            
        html = '<div class="asset-metadata">'
        
        # Format specific fields with nice labels
        field_labels = {
            'name': 'Asset Name',
            'description': 'Description',
            'asset_type': 'Type',
            'quality_level': 'Quality Level',
            'file_format': 'Format',
            'file_size': 'File Size',
            'polygon_count': 'Polygon Count',
            'content_type': 'Content Type',
            'last_modified': 'Last Modified',
            'generated_at': 'Generated At',
            'generation_time': 'Generation Time',
            'service': 'Service',
            'cost': 'Cost'
        }
        
        # Display known fields first
        for key, label in field_labels.items():
            if key in metadata:
                value = metadata[key]
                
                # Format specific values
                if key == 'file_size' and isinstance(value, (int, float)):
                    from ..ui.utils import UIUtils
                    value = UIUtils.format_file_size(int(value))
                elif key == 'polygon_count' and isinstance(value, (int, float)):
                    value = f"{int(value):,}"
                elif key == 'cost' and isinstance(value, (int, float)):
                    value = f"${float(value):.2f}"
                elif key == 'generation_time' and isinstance(value, (int, float)):
                    from ..ui.utils import UIUtils
                    value = UIUtils.format_duration(float(value))
                
                html += f'<div class="metadata-grid"><strong>{label}:</strong> {value}</div>'
        
        # Display any remaining fields
        for key, value in metadata.items():
            if key not in field_labels:
                label = key.replace('_', ' ').title()
                html += f'<div class="metadata-grid"><strong>{label}:</strong> {value}</div>'
        
        html += '</div>'
        return html

    def _refresh_asset_list(self):
        """Refresh the asset dropdown list."""
        try:
            import asyncio
            
            storage = self.app.cloud_storage
            if not storage:
                return []
            
            # List files in /assets prefix
            files = asyncio.run(storage.list_files(prefix="assets/"))
            if not files:
                return []
            
            # Sort by last modified date, newest first
            files_sorted = sorted(files, key=lambda x: x.last_modified, reverse=True)
            
            # Create dropdown choices with human-readable names
            dropdown_choices = []
            for f in files_sorted:
                file_name = f.key.split("/")[-1]
                display_name = f"{file_name} ({f.last_modified.strftime('%Y-%m-%d %H:%M')})"
                dropdown_choices.append((display_name, f.key))
            
            return dropdown_choices
            
        except Exception as e:
            logger.error(f"Error refreshing asset list: {e}")
            return []
    
    def _download_selected_asset(self, asset_key):
        """Download the selected asset file."""
        try:
            import asyncio
            import tempfile
            import os
            
            storage = self.app.cloud_storage
            if not storage or not asset_key:
                return None
            
            # Create a temporary file
            file_name = asset_key.split("/")[-1]
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, file_name)
            
            # Download the file to temporary location
            asyncio.run(storage.download_file(asset_key, temp_file_path))
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error downloading asset {asset_key}: {e}")
            return None


def create_app_interface(app: AssetGenerationApp) -> gr.Blocks:
    """Create the main application interface."""
    ui = Asset3DGeneratorUI(app)
    return ui.create_interface()

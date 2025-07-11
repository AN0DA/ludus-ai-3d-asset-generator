"""
Unified Gradio Interface for AI 3D Asset Generator.

This module provides a single, comprehensive interface that combines the best
features from both interface implementations while using AssetGenerationApp
as the unified backend.
"""

import gradio as gr
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from pathlib import Path
from datetime import datetime
import tempfile

from .components import UIComponents, CUSTOM_CSS, CUSTOM_JS
from .preview import ModelPreview, VIEWER_CSS
from ..core.app import AssetGenerationApp
from ..models.asset_model import (
    AssetType, StylePreference, QualityLevel, FileFormat,
    GenerationStatus, AssetMetadata
)
from ..utils.validators import ValidationException


class UnifiedAssetInterface:
    """
    Unified Gradio interface that combines all features from both interfaces.
    Uses AssetGenerationApp as the single backend for all operations.
    """
    
    def __init__(self, app: AssetGenerationApp):
        """Initialize the unified interface with AssetGenerationApp."""
        self.app = app
        
        # UI state management
        self.current_session_id: Optional[str] = None
        self.current_task_id: Optional[str] = None
        self.form_components: Dict[str, gr.Component] = {}
        self.progress_components: Dict[str, gr.Component] = {}
        self.results_components: Dict[str, gr.Component] = {}
        self.status_components: Dict[str, gr.Component] = {}
        
        # Initialize preview handler
        self.model_preview = ModelPreview()
    
    def create_interface(self) -> gr.Blocks:
        """Create the comprehensive Gradio interface."""
        
        # Combine all CSS for professional styling
        combined_css = CUSTOM_CSS + VIEWER_CSS
        
        with gr.Blocks(
            css=combined_css,
            js=CUSTOM_JS,
            title="AI 3D Asset Generator",
            theme="soft"
        ) as interface:
            
            # Professional Header
            UIComponents.create_header()
            
            # Session state management
            session_state = gr.State(None)
            task_state = gr.State(None)
            
            # Main application tabs
            with gr.Tabs(elem_classes=["main-tabs"]) as tabs:
                
                # Generation Tab - Enhanced version combining both interfaces
                with gr.Tab("🎨 Generate Asset", elem_id="generate-tab"):
                    self._create_generation_tab()
                
                # Results Tab - Professional preview and download
                with gr.Tab("📋 Results & Preview", elem_id="results-tab"):
                    self._create_results_tab()
                
                # History Tab - Session management
                with gr.Tab("📚 Generation History", elem_id="history-tab"):
                    self._create_history_tab()
                
                # Settings Tab - Configuration and status
                with gr.Tab("⚙️ Settings & Status", elem_id="settings-tab"):
                    self._create_settings_tab()
            
            # Wire up all event handlers
            self._setup_event_handlers(interface, session_state, task_state)
        
        return interface
    
    def _create_generation_tab(self):
        """Create the enhanced generation tab."""
        with gr.Row():
            # Input Column - Using professional components
            with gr.Column(scale=2, elem_classes=["input-column"]):
                form_group, self.form_components = UIComponents.create_input_form()
                
                # Generation controls
                with gr.Row():
                    self.generate_btn = gr.Button(
                        "🚀 Generate Asset",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"]
                    )
                    
                    self.cancel_btn = gr.Button(
                        "⏹️ Cancel",
                        variant="secondary",
                        size="lg",
                        visible=False,
                        elem_classes=["cancel-btn"]
                    )
            
            # Progress & Status Column
            with gr.Column(scale=1, elem_classes=["progress-column"]):
                # Service Status Display
                status_group, self.status_components = UIComponents.create_service_status_display()
                
                # Progress Display
                progress_group, self.progress_components = UIComponents.create_progress_display()
                
                # Validation Feedback
                self.validation_feedback = UIComponents.create_error_display()
    
    def _create_results_tab(self):
        """Create the professional results and preview tab."""
        with gr.Row():
            with gr.Column():
                # Results Display with 3D Preview
                results_group, self.results_components = UIComponents.create_results_display()
                
                # 3D Model Preview
                viewer_group, viewer_components = self.model_preview.create_model_viewer(
                    height=400,
                    show_controls=True,
                    show_info=True
                )
                
                # Merge viewer components
                self.results_components.update(viewer_components)
                
                # Download Section
                with gr.Group(elem_classes=["download-section"]):
                    gr.HTML("<h3>📥 Download Options</h3>")
                    
                    with gr.Row():
                        model_download = gr.File(
                            label="3D Model File",
                            visible=False,
                            elem_classes=["download-file"]
                        )
                        
                        metadata_download = gr.File(
                            label="Metadata File", 
                            visible=False,
                            elem_classes=["download-file"]
                        )
                    
                    share_url = gr.Textbox(
                        label="Share URL",
                        placeholder="Public URL will appear here after generation",
                        interactive=False,
                        elem_classes=["share-url"]
                    )
                    
                    # Store download components
                    self.results_components["model_download"] = model_download
                    self.results_components["metadata_download"] = metadata_download
                    self.results_components["share_url"] = share_url
    
    def _create_history_tab(self):
        """Create the session history tab."""
        with gr.Column():
            gr.HTML("<h3>📚 Generation History</h3>")
            
            with gr.Row():
                history_refresh_btn = gr.Button("🔄 Refresh History", size="sm")
                clear_history_btn = gr.Button("🗑️ Clear History", size="sm", variant="secondary")
            
            # History display with enhanced formatting
            history_display = gr.JSON(
                label="Session History",
                elem_classes=["history-display"]
            )
            
            # Gallery of generated assets
            asset_gallery = gr.Gallery(
                label="Generated Assets",
                columns=3,
                rows=2,
                elem_classes=["asset-gallery"]
            )
            
            # Store history components
            self.history_components = {
                "refresh_btn": history_refresh_btn,
                "clear_btn": clear_history_btn,
                "display": history_display,
                "gallery": asset_gallery
            }
    
    def _create_settings_tab(self):
        """Create the settings and system status tab."""
        with gr.Row():
            # Cache Management
            with gr.Column():
                gr.HTML("<h3>🗄️ Cache Management</h3>")
                
                cache_stats = gr.JSON(
                    label="Cache Statistics",
                    elem_classes=["cache-stats"]
                )
                
                with gr.Row():
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", size="sm")
                    refresh_cache_btn = gr.Button("🔄 Refresh Stats", size="sm")
            
            # Session Information
            with gr.Column():
                gr.HTML("<h3>📊 Session Information</h3>")
                
                session_info = gr.JSON(
                    label="Current Session",
                    elem_classes=["session-info"]
                )
                
                with gr.Row():
                    new_session_btn = gr.Button("🆕 New Session", size="sm")
                    export_session_btn = gr.Button("📤 Export Session", size="sm")
        
        # Store settings components
        self.settings_components = {
            "cache_stats": cache_stats,
            "clear_cache_btn": clear_cache_btn,
            "refresh_cache_btn": refresh_cache_btn,
            "session_info": session_info,
            "new_session_btn": new_session_btn,
            "export_session_btn": export_session_btn
        }
    
    def _setup_event_handlers(self, interface: gr.Blocks, session_state: gr.State, task_state: gr.State):
        """Set up all event handlers for the unified interface."""
        
        # Generation handlers
        self.generate_btn.click(
            fn=self._start_generation,
            inputs=[
                self.form_components["description"],
                self.form_components["asset_type"],
                self.form_components["style"],
                self.form_components["quality"],
                session_state
            ],
            outputs=[
                self.progress_components["status"],
                self.progress_components["progress_bar"],
                self.validation_feedback,
                self.cancel_btn,
                session_state,
                task_state
            ]
        )
        
        # Cancel generation
        self.cancel_btn.click(
            fn=self._cancel_generation,
            inputs=[task_state],
            outputs=[
                self.progress_components["status"],
                self.cancel_btn
            ]
        )
        
        # Progress polling - use a timer or manual refresh instead
        # self.progress_components["refresh_btn"].click(
        #     fn=self._check_progress,
        #     inputs=[task_state],
        #     outputs=[
        #         self.progress_components["status"],
        #         self.progress_components["progress_bar"],
        #         self.results_components["metadata_display"]
        #     ]
        # )
        
        # History management
        self.history_components["refresh_btn"].click(
            fn=self._refresh_history,
            inputs=[session_state],
            outputs=[self.history_components["display"], self.history_components["gallery"]]
        )
        
        self.history_components["clear_btn"].click(
            fn=self._clear_history,
            inputs=[session_state],
            outputs=[self.history_components["display"], self.history_components["gallery"]]
        )
        
        # Settings handlers
        self.settings_components["clear_cache_btn"].click(
            fn=self._clear_cache,
            outputs=[self.settings_components["cache_stats"]]
        )
        
        self.settings_components["refresh_cache_btn"].click(
            fn=self._get_cache_stats,
            outputs=[self.settings_components["cache_stats"]]
        )
        
        self.settings_components["new_session_btn"].click(
            fn=self._create_new_session,
            outputs=[session_state, self.settings_components["session_info"]]
        )
        
        # Initialize on load
        interface.load(
            fn=self._initialize_interface,
            outputs=[
                session_state,
                self.settings_components["session_info"],
                self.settings_components["cache_stats"]
            ]
        )
    
    # Event Handler Methods
    
    async def _start_generation(
        self,
        description: str,
        asset_type: str,
        style: str,
        quality: str,
        session_id: Optional[str]
    ) -> Tuple[str, Dict[str, Any], str, Dict[str, Any], Optional[str], Optional[str]]:
        """Start the asset generation process using AssetGenerationApp."""
        
        # Basic validation
        errors = []
        if not description or len(description.strip()) < 10:
            errors.append("Description must be at least 10 characters long")
        if len(description) > 2000:
            errors.append("Description must be less than 2000 characters")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            error_html = "<br>".join([f"❌ {error}" for error in errors])
            return (
                "❌ Validation failed",
                gr.update(visible=False),
                error_html,
                gr.update(visible=False),
                session_id,
                None
            )
        
        try:
            # Ensure app is initialized
            if not self.app.is_initialized:
                await self.app.initialize()
            
            # Create session if needed
            if not session_id:
                session_id = self.app.session_manager.create_session()
            
            # Convert parameters to proper types
            asset_type_enum = AssetType(asset_type.lower().replace(" ", "_"))
            style_enum = StylePreference(style.lower().replace(" ", "_")) if style != "None" else None
            quality_enum = QualityLevel(quality.lower())
            
            # Start generation using AssetGenerationApp
            task_id, session_id = await self.app.generate_asset_pipeline(
                description=description,
                asset_type=asset_type_enum,
                style_preference=style_enum,
                quality_level=quality_enum,
                session_id=session_id
            )
            
            return (
                "🚀 Generation started...",
                gr.update(visible=True),
                "",
                gr.update(visible=True),
                session_id,
                task_id
            )
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            return (
                f"❌ {error_msg}",
                gr.update(visible=False),
                error_msg,
                gr.update(visible=False),
                session_id,
                None
            )
    
    def _cancel_generation(self, task_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Cancel a running generation."""
        if task_id and self.app.cancel_generation(task_id):
            return (
                "⏹️ Generation cancelled",
                gr.update(visible=False)
            )
        return (
            "ℹ️ No active generation to cancel",
            gr.update(visible=False)
        )
    
    def _check_progress(self, task_id: Optional[str]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Check generation progress using AssetGenerationApp."""
        if not task_id:
            return (
                "ℹ️ No active generation",
                gr.update(visible=False),
                {}
            )
        
        status = self.app.get_generation_status(task_id)
        
        if status.get("status") == "completed":
            result = status.get("result", {})
            return (
                "✅ Generation completed!",
                gr.update(visible=False),
                result
            )
        elif status.get("status") == "failed":
            error = status.get("error", "Unknown error")
            return (
                f"❌ Generation failed: {error}",
                gr.update(visible=False),
                {}
            )
        else:
            progress = status.get("progress", 0.0)
            message = status.get("message", "Processing...")
            return (
                f"🔄 {message} ({int(progress * 100)}%)",
                gr.update(visible=True),
                {}
            )
    
    def _refresh_history(self, session_id: Optional[str]) -> Tuple[List[Dict], List[str]]:
        """Refresh session history."""
        if session_id:
            history = self.app.get_session_history(session_id)
            gallery_items = []  # Extract image URLs from history for gallery
            return history, gallery_items
        return [], []
    
    def _clear_history(self, session_id: Optional[str]) -> Tuple[List[Dict], List[str]]:
        """Clear session history."""
        if session_id:
            # For now, just return empty - implement session history clearing if needed
            pass
        return [], []
    
    def _clear_cache(self) -> Dict[str, Any]:
        """Clear application cache."""
        self.app.cache_manager.clear_all()
        return self.app.cache_manager.get_cache_stats()
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.app.cache_manager.get_cache_stats()
    
    def _create_new_session(self) -> Tuple[str, Dict[str, Any]]:
        """Create a new session."""
        session_id = self.app.session_manager.create_session()
        session_info = self.app.session_manager.get_session_info(session_id) or {}
        return session_id, session_info
    
    def _initialize_interface(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Initialize the interface on load."""
        # Create initial session
        session_id = self.app.session_manager.create_session()
        session_info = self.app.session_manager.get_session_info(session_id) or {}
        cache_stats = self.app.cache_manager.get_cache_stats()
        
        return session_id, session_info, cache_stats


def create_unified_interface(app: AssetGenerationApp) -> gr.Blocks:
    """Create the unified interface with AssetGenerationApp backend."""
    interface = UnifiedAssetInterface(app)
    return interface.create_interface()

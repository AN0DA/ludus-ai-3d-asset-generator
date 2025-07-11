"""
Gradio web interface for the AI 3D Asset Generator.

This module provides the complete web application interface using Gradio,
including the UI components, event handlers, and progress tracking.
"""

import asyncio
import uuid
from typing import Optional, Tuple

import gradio as gr
import structlog

from ..models.asset_model import AssetType, StylePreference, QualityLevel
from ..core.app import AssetGenerationApp

logger = structlog.get_logger(__name__)


class GradioInterface:
    """Gradio web interface for the asset generator."""
    
    def __init__(self, app: AssetGenerationApp):
        self.app = app
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .main-header {
            text-align: center;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .generation-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .progress-box {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 1rem 0;
        }
        """
        
        with gr.Blocks(css=css, title="AI 3D Asset Generator") as interface:
            
            # Header
            gr.HTML('''
                <div class="main-header">
                    <h1>🎮 AI 3D Asset Generator</h1>
                    <p>Create amazing 3D game assets with AI-powered generation</p>
                </div>
            ''')
            
            # State variables
            session_state = gr.State(None)
            task_state = gr.State(None)
            
            with gr.Tabs():
                
                # Generation Tab
                with gr.TabItem("🚀 Generate Asset", id="generate"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### Asset Description")
                            description_input = gr.Textbox(
                                label="Describe your asset",
                                placeholder="e.g., 'A magical healing potion with glowing blue liquid'",
                                lines=3,
                                info="Provide a detailed description (minimum 10 characters)"
                            )
                            
                            with gr.Row():
                                asset_type_dropdown = gr.Dropdown(
                                    label="Asset Type",
                                    choices=[t.value for t in AssetType],
                                    value=AssetType.PROP.value,
                                    info="Select the type of asset to generate"
                                )
                                
                                style_dropdown = gr.Dropdown(
                                    label="Style Preference",
                                    choices=["None"] + [s.value for s in StylePreference],
                                    value="None",
                                    info="Choose a visual style (optional)"
                                )
                            
                            with gr.Row():
                                quality_dropdown = gr.Dropdown(
                                    label="Quality Level",
                                    choices=[q.value for q in QualityLevel],
                                    value=QualityLevel.STANDARD.value,
                                    info="Select generation quality vs speed"
                                )
                            
                            generate_btn = gr.Button("🎨 Generate Asset", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Generation Progress")
                            
                            status_box = gr.Markdown("Ready to generate", elem_classes=["progress-box"])
                            progress_box = gr.Group(visible=False)
                            
                            with progress_box:
                                gr.Markdown("**Generation in progress...**")
                                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                            
                            cancel_box = gr.Group(visible=False)
                            with cancel_box:
                                cancel_btn = gr.Button("⏹️ Cancel Generation", variant="secondary")
                
                # Results Tab
                with gr.TabItem("📋 Results", id="results"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Generated Asset")
                            result_display = gr.JSON(label="Asset Metadata", visible=False)
                            download_section = gr.Group(visible=False)
                            
                            with download_section:
                                gr.Markdown("### Download Options")
                                model_download = gr.File(label="3D Model File", visible=False)
                                metadata_download = gr.File(label="Metadata File", visible=False)
                                
                                share_url = gr.Textbox(
                                    label="Share URL",
                                    placeholder="Model URL will appear here",
                                    interactive=False
                                )
                
                # History Tab
                with gr.TabItem("📚 History", id="history"):
                    with gr.Column():
                        gr.Markdown("### Generation History")
                        history_refresh_btn = gr.Button("🔄 Refresh History")
                        history_display = gr.JSON(label="Session History")
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings", id="settings"):
                    with gr.Column():
                        gr.Markdown("### Application Settings")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Cache Management**")
                                cache_stats = gr.JSON(label="Cache Statistics")
                                clear_cache_btn = gr.Button("🗑️ Clear Cache")
                            
                            with gr.Column():
                                gr.Markdown("**Session Info**")
                                session_info = gr.JSON(label="Session Information")
            
            # Event handlers
            def start_generation(
                description: str,
                asset_type: str,
                style: str,
                quality: str,
                session_id: Optional[str]
            ):
                """Start the asset generation process."""
                try:
                    # Validate inputs
                    if not description or len(description.strip()) < 10:
                        return (
                            "❌ Please provide a detailed description (at least 10 characters)",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            session_id,
                            None
                        )
                    
                    # Create session if needed
                    if session_id is None:
                        session_id = self.app.session_manager.create_session()
                    
                    # Convert string parameters to enum types
                    try:
                        asset_type_enum = AssetType(asset_type.lower().replace(" ", "_"))
                        style_enum = StylePreference(style.lower().replace(" ", "_")) if style != "None" else None
                        quality_enum = QualityLevel(quality.lower())
                    except ValueError as e:
                        return (
                            f"❌ Invalid parameters: {str(e)}",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            session_id,
                            None
                        )
                    
                    # Create async generation task
                    async def generation_wrapper():
                        try:
                            result = await self.app.generate_asset_pipeline(
                                description=description,
                                asset_type=asset_type_enum,
                                style_preference=style_enum,
                                quality_level=quality_enum,
                                session_id=session_id
                            )
                            return result
                        except Exception as e:
                            logger.error(f"Generation task failed: {e}")
                            raise
                    
                    # Start the background task
                    task_id = self.app.task_manager.create_task(generation_wrapper())
                    
                    return (
                        "🚀 Generation started! Please wait...",
                        gr.update(visible=True),
                        gr.update(visible=True),
                        session_id,
                        task_id
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to start generation: {e}")
                    return (
                        f"❌ Failed to start generation: {str(e)}",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        session_id,
                        None
                    )
            
            def check_progress(task_id: Optional[str]):
                """Check the progress of a generation task."""
                if not task_id:
                    return "No active generation", gr.update(visible=False)
                
                # Get real task status
                status = self.app.task_manager.get_task_status(task_id)
                
                if status.get("status") == "not_found":
                    return "Task not found", gr.update(visible=False)
                
                task_status = status.get("status", "unknown")
                progress = status.get("progress", 0.0)
                message = status.get("message", "Processing...")
                
                if task_status == "completed":
                    result = status.get("result")
                    if result:
                        return f"✅ Generation completed successfully!", gr.update(visible=False)
                    else:
                        return f"✅ Generation completed!", gr.update(visible=False)
                elif task_status == "failed":
                    error = status.get("error", "Unknown error")
                    return f"❌ Generation failed: {error}", gr.update(visible=False)
                elif task_status == "cancelled":
                    return "⏹️ Generation was cancelled", gr.update(visible=False)
                else:
                    progress_pct = int(progress * 100)
                    return f"🔄 {message} ({progress_pct}%)", gr.update(visible=True)
            
            def cancel_generation_handler(task_id: Optional[str]):
                """Cancel a running generation."""
                if task_id and self.app.cancel_generation(task_id):
                    return "⏹️ Generation cancelled", gr.update(visible=False)
                return "No active generation to cancel", gr.update(visible=False)
            
            def refresh_history(session_id: Optional[str]):
                """Refresh the session history."""
                if session_id:
                    history = self.app.get_session_history(session_id)
                    return history
                return []
            
            def get_cache_statistics():
                """Get cache statistics."""
                return self.app.cache_manager.get_cache_stats()
            
            def clear_cache():
                """Clear all cached data."""
                self.app.cache_manager.clear_all()
                return "Cache cleared successfully"
            
            def get_session_information(session_id: Optional[str]):
                """Get session information."""
                if session_id:
                    return self.app.session_manager.get_session_info(session_id)
                return None
            
            # Wire up event handlers
            generate_btn.click(
                fn=start_generation,
                inputs=[
                    description_input,
                    asset_type_dropdown,
                    style_dropdown,
                    quality_dropdown,
                    session_state
                ],
                outputs=[
                    status_box,
                    progress_box,
                    cancel_box,
                    session_state,
                    task_state
                ]
            )
            
            refresh_btn.click(
                fn=check_progress,
                inputs=[task_state],
                outputs=[status_box, progress_box]
            )
            
            cancel_btn.click(
                fn=cancel_generation_handler,
                inputs=[task_state],
                outputs=[status_box, cancel_box]
            )
            
            history_refresh_btn.click(
                fn=refresh_history,
                inputs=[session_state],
                outputs=[history_display]
            )
            
            interface.load(
                fn=get_cache_statistics,
                outputs=[cache_stats]
            )
            
            interface.load(
                fn=get_session_information,
                inputs=[session_state],
                outputs=[session_info]
            )
            
            clear_cache_btn.click(
                fn=clear_cache,
                outputs=[cache_stats]
            )
        
        return interface


def create_gradio_app(app: AssetGenerationApp) -> gr.Blocks:
    """Create and return the Gradio interface."""
    interface = GradioInterface(app)
    return interface.create_interface()

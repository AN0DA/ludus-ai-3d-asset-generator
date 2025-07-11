"""
Professional Gradio Interface for AI 3D Asset Generator.

This module creates a comprehensive, user-friendly web interface for the 3D asset
generation system with modern design, responsive layout, and enhanced UX.
"""

import gradio as gr
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import tempfile

from .components import UIComponents, CUSTOM_CSS, CUSTOM_JS
from .preview import ModelPreview, VIEWER_CSS
from ..generators.asset_generator import (
    Asset3DGenerator, GenerationRequest, ServiceProvider, 
    ServiceConfig, create_default_configs
)
from ..models.asset_model import (
    AssetType, StylePreference, QualityLevel, FileFormat,
    GenerationStatus, AssetMetadata
)
from ..utils.validators import ValidationException
from ..utils.config import ConfigManager


class AssetGeneratorInterface:
    """Main Gradio interface for the 3D asset generator."""
    
    def __init__(
        self,
        configs: Optional[Dict[ServiceProvider, ServiceConfig]] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize the interface with service configurations."""
        self.configs = configs or self._create_demo_configs()
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "asset_generator_ui"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator: Optional[Asset3DGenerator] = None
        self.current_generation_id: Optional[str] = None
        self.generation_history: List[AssetMetadata] = []
        
        # UI state
        self.is_generating = False
        self.progress_updates = []
    
    def _create_demo_configs(self) -> Dict[ServiceProvider, ServiceConfig]:
        """Create demo configurations for testing."""
        return {
            ServiceProvider.MESHY_AI: ServiceConfig(
                api_key=os.getenv("MESHY_API_KEY", "demo_key_meshy_12345678"),
                base_url="https://api.meshy.ai",
                max_requests_per_minute=10,
                max_requests_per_day=100,
                cost_per_generation=0.50,
                supports_text_to_3d=True,
                supports_image_to_3d=False,
                supported_output_formats=[FileFormat.GLB, FileFormat.FBX, FileFormat.OBJ, FileFormat.USDZ],
                quality_levels=[QualityLevel.STANDARD, QualityLevel.HIGH, QualityLevel.ULTRA]
            ),
            ServiceProvider.KAEDIM: ServiceConfig(
                api_key=os.getenv("KAEDIM_API_KEY", "demo_key_kaedim_12345678"),
                base_url="https://api.kaedim3d.com",
                max_requests_per_minute=5,
                max_requests_per_day=50,
                cost_per_generation=2.00,
                supports_text_to_3d=False,
                supports_image_to_3d=True,
                supported_output_formats=[FileFormat.OBJ, FileFormat.FBX],
                quality_levels=[QualityLevel.STANDARD, QualityLevel.HIGH]
            )
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        # Combine all CSS
        combined_css = CUSTOM_CSS + VIEWER_CSS
        
        with gr.Blocks(
            css=combined_css,
            js=CUSTOM_JS,
            title="AI 3D Asset Generator",
            theme="soft"
        ) as interface:
            
            # Header
            UIComponents.create_header()
            
            # Main application tabs
            with gr.Tabs(elem_classes=["main-tabs"]) as tabs:
                
                # Generation Tab
                with gr.Tab("🎨 Generate Asset", elem_id="generate-tab"):
                    self._create_generation_tab()
                
                # Gallery Tab
                with gr.Tab("🖼️ Asset Gallery", elem_id="gallery-tab"):
                    self._create_gallery_tab()
                
                # Settings Tab
                with gr.Tab("⚙️ Settings", elem_id="settings-tab"):
                    self._create_settings_tab()
                
                # Help Tab
                with gr.Tab("❓ Help", elem_id="help-tab"):
                    self._create_help_tab()
            
            # Footer
            self._create_footer()
        
        return interface
    
    def _create_generation_tab(self) -> None:
        """Create the main asset generation tab."""
        
        with gr.Row(elem_classes=["generation-row"]):
            # Left column - Input form
            with gr.Column(scale=1, elem_classes=["input-column"]):
                # Input form
                form_group, self.form_components = UIComponents.create_input_form()
                
                # Generate button
                self.generate_button = gr.Button(
                    value="🚀 Generate 3D Asset",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-button"]
                )
                
                # Service status
                status_group, self.status_components = UIComponents.create_service_status_display()
                
                # Validation feedback
                self.validation_feedback = UIComponents.create_error_display()
            
            # Right column - Results and preview
            with gr.Column(scale=1, elem_classes=["results-column"]):
                # Progress display
                progress_group, self.progress_components = UIComponents.create_progress_display()
                
                # 3D Model viewer
                viewer_group, self.viewer_components = ModelPreview.create_responsive_viewer()
                
                # Results display
                results_group, self.results_components = UIComponents.create_results_display()
        
        # Set up event handlers
        self._setup_generation_events()
    
    def _create_gallery_tab(self) -> None:
        """Create the asset gallery tab."""
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>🎨 Your Generated Assets</h3>")
                
                # Gallery controls
                with gr.Row():
                    self.gallery_filter = gr.Dropdown(
                        label="Filter by Type",
                        choices=[("All", "all")] + [(t.value.title(), t.value) for t in AssetType],
                        value="all",
                        elem_classes=["custom-input"]
                    )
                    
                    self.gallery_sort = gr.Dropdown(
                        label="Sort by",
                        choices=[
                            ("Newest First", "newest"),
                            ("Oldest First", "oldest"),
                            ("Name A-Z", "name_asc"),
                            ("Name Z-A", "name_desc")
                        ],
                        value="newest",
                        elem_classes=["custom-input"]
                    )
                    
                    self.refresh_gallery_btn = gr.Button(
                        value="🔄 Refresh",
                        variant="secondary",
                        elem_classes=["secondary-button"]
                    )
                
                # Gallery display
                self.asset_gallery = gr.Gallery(
                    label="Generated Assets",
                    show_label=False,
                    elem_classes=["asset-gallery-viewer"],
                    columns=3,
                    rows=3,
                    height=400
                )
                
                # Selected asset details
                with gr.Group(elem_classes=["asset-details"], visible=False) as self.asset_details_group:
                    gr.HTML("<h3>📋 Asset Details</h3>")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            self.selected_asset_preview = gr.Model3D(
                                label="Preview",
                                height=300
                            )
                        
                        with gr.Column(scale=1):
                            self.selected_asset_info = gr.HTML(
                                value="<p>Select an asset to view details</p>"
                            )
                            
                            self.download_selected_btn = gr.Button(
                                value="📥 Download Asset",
                                variant="primary",
                                elem_classes=["secondary-button"]
                            )
    
    def _create_settings_tab(self) -> None:
        """Create the settings and configuration tab."""
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>⚙️ Generator Settings</h3>")
                
                # API Configuration
                with gr.Group(elem_classes=["form-section"]):
                    gr.HTML("<h4>🔑 API Configuration</h4>")
                    
                    self.meshy_api_key = gr.Textbox(
                        label="Meshy AI API Key",
                        type="password",
                        placeholder="Enter your Meshy AI API key",
                        elem_classes=["custom-input"]
                    )
                    
                    self.kaedim_api_key = gr.Textbox(
                        label="Kaedim API Key", 
                        type="password",
                        placeholder="Enter your Kaedim API key",
                        elem_classes=["custom-input"]
                    )
                    
                    self.openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="Enter your OpenAI API key (for DALL-E)",
                        elem_classes=["custom-input"]
                    )
                
                # Generation Defaults
                with gr.Group(elem_classes=["form-section"]):
                    gr.HTML("<h4>🎯 Default Settings</h4>")
                    
                    self.default_quality = gr.Dropdown(
                        label="Default Quality Level",
                        choices=[(q.value.title(), q.value) for q in QualityLevel],
                        value=QualityLevel.STANDARD.value,
                        elem_classes=["custom-input"]
                    )
                    
                    self.default_format = gr.Dropdown(
                        label="Default Output Format",
                        choices=[(f.value.upper(), f.value) for f in FileFormat],
                        value=FileFormat.OBJ.value,
                        elem_classes=["custom-input"]
                    )
                    
                    self.auto_download = gr.Checkbox(
                        label="Auto-download generated assets",
                        value=False
                    )
                    
                    self.save_thumbnails = gr.Checkbox(
                        label="Save thumbnail images",
                        value=True
                    )
                
                # Cache Management
                with gr.Group(elem_classes=["form-section"]):
                    gr.HTML("<h4>💾 Cache Management</h4>")
                    
                    self.cache_info = gr.HTML(
                        value="<p>Loading cache information...</p>"
                    )
                    
                    with gr.Row():
                        self.clear_cache_btn = gr.Button(
                            value="🗑️ Clear Cache",
                            variant="secondary",
                            elem_classes=["secondary-button"]
                        )
                        
                        self.export_settings_btn = gr.Button(
                            value="📤 Export Settings",
                            variant="secondary",
                            elem_classes=["secondary-button"]
                        )
                
                # Save button
                self.save_settings_btn = gr.Button(
                    value="💾 Save Settings",
                    variant="primary",
                    elem_classes=["generate-button"]
                )
    
    def _create_help_tab(self) -> None:
        """Create the help and documentation tab."""
        
        help_content = """
        # 🎯 AI 3D Asset Generator - User Guide
        
        ## 🚀 Getting Started
        
        ### 1. Configure Your API Keys
        Go to the **Settings** tab and enter your API keys for the services you want to use:
        - **Meshy AI**: Best for text-to-3D generation
        - **Kaedim**: Excellent for image-to-3D conversion
        - **OpenAI**: Required for DALL-E image generation in image-to-3D pipeline
        
        ### 2. Create Your First Asset
        1. Switch to the **Generate Asset** tab
        2. Enter a detailed description of your desired 3D asset
        3. Choose the asset type and style preferences
        4. Adjust technical settings as needed
        5. Click **Generate 3D Asset**
        
        ## 📝 Writing Effective Descriptions
        
        ### Best Practices:
        - **Be specific**: "A medieval sword" vs "A ornate medieval longsword with Celtic engravings"
        - **Include materials**: "made of polished steel with leather-wrapped handle"
        - **Mention colors**: "with blue gemstones and gold accents"
        - **Describe style**: "in fantasy art style" or "realistic medieval design"
        - **Add details**: "featuring intricate patterns and weathered edges"
        
        ### Example Descriptions:
        ```
        ✅ Good: "A mystical staff topped with a glowing blue crystal orb, 
        carved from ancient oak wood with silver runes running along the shaft, 
        fantasy art style with ethereal lighting effects"
        
        ❌ Poor: "magic staff"
        ```
        
        ## ⚙️ Technical Settings
        
        ### Quality Levels:
        - **Draft**: Fast generation, lower detail (5-10 minutes)
        - **Standard**: Balanced quality and speed (10-20 minutes)
        - **High**: Detailed models, longer generation (20-45 minutes)
        - **Ultra**: Maximum detail and quality (45+ minutes)
        
        ### Polygon Count:
        - **1,000-5,000**: Simple objects, mobile/web use
        - **5,000-25,000**: Standard detail for games
        - **25,000-50,000**: High detail for renders
        - **50,000+**: Maximum detail for close-up views
        
        ### File Formats:
        - **OBJ**: Universal format, good for most 3D software
        - **GLTF/GLB**: Modern format, ideal for web and AR/VR
        - **FBX**: Industry standard, best for animation
        - **STL**: 3D printing
        - **PLY**: Scientific/research applications
        
        ## 🎨 Service Comparison
        
        | Service | Best For | Strengths | Generation Time |
        |---------|----------|-----------|-----------------|
        | Meshy AI | Text-to-3D | High quality, fast | 5-15 minutes |
        | Kaedim | Image-to-3D | Accurate conversion | 10-30 minutes |
        | Image-to-3D | Creative concepts | Unique results | 15-45 minutes |
        
        ## 🔧 Troubleshooting
        
        ### Common Issues:
        
        **Generation Failed:**
        - Check your API keys in Settings
        - Ensure description meets minimum length (10 characters)
        - Try reducing polygon count or quality level
        
        **Poor Quality Results:**
        - Use more detailed descriptions
        - Increase quality level
        - Try different service providers
        
        **Slow Generation:**
        - Lower polygon count
        - Use Draft quality for testing
        - Avoid peak usage hours
        
        **Can't Preview Model:**
        - Check if format is supported (OBJ, GLTF, GLB work best)
        - Try refreshing the page
        - Download and view in external software
        
        ## 📱 Mobile Usage
        
        The interface is optimized for mobile devices:
        - Touch-friendly controls for 3D navigation
        - Responsive layout adapts to screen size
        - Touch gestures for model manipulation
        
        ## 💡 Tips & Tricks
        
        1. **Save API costs**: Test with Draft quality first
        2. **Batch generation**: Queue multiple requests
        3. **Reference images**: Upload concept art for better results
        4. **Style consistency**: Use similar descriptions for asset sets
        5. **Quality vs speed**: Balance based on your needs
        
        ## 🔗 Useful Resources
        
        - [Meshy AI Documentation](https://docs.meshy.ai)
        - [Kaedim API Guide](https://docs.kaedim.ai)
        - [3D Model Formats Guide](https://en.wikipedia.org/wiki/List_of_file_formats#3D_graphics)
        - [Writing Prompts for AI](https://platform.openai.com/docs/guides/prompt-engineering)
        
        ## 📞 Support
        
        Having issues? Here's how to get help:
        1. Check the troubleshooting section above
        2. Review your API key configuration
        3. Try with a simple description first
        4. Check service status in the main tab
        
        ---
        
        *Made with ❤️ using Gradio and AI magic*
        """
        
        with gr.Column():
            gr.Markdown(help_content, elem_classes=["help-content"])
    
    def _create_footer(self) -> None:
        """Create the application footer."""
        footer_html = """
        <div class="app-footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h4>🎯 AI 3D Asset Generator</h4>
                    <p>Transform your imagination into stunning 3D models</p>
                </div>
                <div class="footer-section">
                    <h4>🔗 Quick Links</h4>
                    <ul>
                        <li><a href="#generate-tab">Generate Asset</a></li>
                        <li><a href="#gallery-tab">View Gallery</a></li>
                        <li><a href="#settings-tab">Settings</a></li>
                        <li><a href="#help-tab">Help</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h4>📊 Statistics</h4>
                    <p>Assets Generated: <span id="total-assets">0</span></p>
                    <p>Total Cost: $<span id="total-cost">0.00</span></p>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2025 AI 3D Asset Generator. Powered by multiple AI services.</p>
            </div>
        </div>
        
        <style>
        .app-footer {
            background: rgba(45, 55, 72, 0.95);
            color: white;
            margin-top: 40px;
            padding: 32px 24px 16px;
            border-radius: 16px 16px 0 0;
        }
        
        .footer-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 24px;
            margin-bottom: 20px;
        }
        
        .footer-section h4 {
            color: #e2e8f0;
            margin-bottom: 12px;
            font-size: 16px;
        }
        
        .footer-section p {
            color: #cbd5e0;
            font-size: 14px;
            margin: 4px 0;
        }
        
        .footer-section ul {
            list-style: none;
            padding: 0;
        }
        
        .footer-section li {
            margin: 4px 0;
        }
        
        .footer-section a {
            color: #a0aec0;
            text-decoration: none;
            font-size: 14px;
            transition: color 0.2s ease;
        }
        
        .footer-section a:hover {
            color: #e2e8f0;
        }
        
        .footer-bottom {
            text-align: center;
            padding-top: 16px;
            border-top: 1px solid rgba(226, 232, 240, 0.2);
        }
        
        .footer-bottom p {
            color: #a0aec0;
            font-size: 12px;
            margin: 0;
        }
        
        @media (max-width: 768px) {
            .footer-content {
                grid-template-columns: 1fr;
                gap: 16px;
            }
        }
        </style>
        """
        
        gr.HTML(footer_html)
    
    def _setup_generation_events(self) -> None:
        """Set up event handlers for the generation interface."""
        
        # Generate button click
        self.generate_button.click(
            fn=self._handle_generate_click,
            inputs=[
                self.form_components["description"],
                self.form_components["asset_type"],
                self.form_components["style"],
                self.form_components["quality"],
                self.form_components["format"],
                self.form_components["polygon_count"],
                self.form_components["priority"],
                self.form_components["service"],
                self.form_components["include_thumbnail"]
            ],
            outputs=[
                self.validation_feedback,
                self.progress_components["status"],
                self.generate_button
            ]
        )
        
        # Service status refresh (skip for now to avoid type issues)
        # TODO: Fix refresh button event handling in future version
    
    def _handle_generate_click(
        self,
        description: str,
        asset_type: str,
        style: str,
        quality: str,
        format_type: str,
        polygon_count: int,
        priority: int,
        service: str,
        include_thumbnail: bool
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Handle generate button click."""
        
        # Validate inputs
        is_valid, errors = UIComponents.validate_inputs(
            description, asset_type, polygon_count, priority
        )
        
        if not is_valid:
            error_html = UIComponents.create_validation_feedback(errors)
            return (
                gr.update(value=error_html, visible=True),
                UIComponents.create_status_badge("error", "Validation failed"),
                gr.update(value="🚀 Generate 3D Asset", interactive=True)
            )
        
        # Start generation
        try:
            # Create generation request
            request = GenerationRequest(
                description=description,
                asset_type=AssetType(asset_type),
                style_preference=StylePreference(style),
                quality_level=QualityLevel(quality),
                output_format=FileFormat(format_type),
                max_polygon_count=polygon_count,
                priority=priority,
                preferred_service=None if service == "auto" else ServiceProvider(service)
            )
            
            # Initialize generator if needed
            if not self.generator:
                self.generator = Asset3DGenerator(self.configs)
            
            # Start generation (this would be async in real implementation)
            self.is_generating = True
            
            return (
                gr.update(visible=False),
                UIComponents.create_status_badge("info", "Starting generation..."),
                gr.update(value="⏳ Generating...", interactive=False)
            )
            
        except Exception as e:
            return (
                gr.update(value=f'<div class="status-error">Error: {str(e)}</div>', visible=True),
                UIComponents.create_status_badge("error", f"Generation failed: {str(e)}"),
                gr.update(value="🚀 Generate 3D Asset", interactive=True)
            )
    
    def _refresh_service_status(self) -> Tuple[str, str, str, str]:
        """Refresh service health status."""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # In real implementation, this would check actual service health
        meshy_status = UIComponents.create_status_badge("success", "Meshy AI: Available")
        kaedim_status = UIComponents.create_status_badge("success", "Kaedim: Available")
        dalle_status = UIComponents.create_status_badge("success", "DALL-E: Available")
        
        return (
            meshy_status,
            kaedim_status,
            dalle_status,
            f"<small>Last updated: {current_time}</small>"
        )
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False
    ) -> None:
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )


# Factory function for easy interface creation
def create_interface(
    meshy_api_key: Optional[str] = None,
    kaedim_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> AssetGeneratorInterface:
    """Factory function to create the asset generator interface."""
    
    configs = {}
    
    if meshy_api_key:
        configs[ServiceProvider.MESHY_AI] = ServiceConfig(
            api_key=meshy_api_key,
            base_url="https://api.meshy.ai",
            supports_text_to_3d=True
        )
    
    if kaedim_api_key:
        configs[ServiceProvider.KAEDIM] = ServiceConfig(
            api_key=kaedim_api_key,
            base_url="https://api.kaedim3d.com",
            supports_image_to_3d=True
        )
    
    if openai_api_key:
        configs[ServiceProvider.OPENAI_DALLE] = ServiceConfig(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1",
            supports_text_to_3d=True
        )
    
    return AssetGeneratorInterface(configs=configs, cache_dir=cache_dir)


# Export main classes
__all__ = [
    "AssetGeneratorInterface",
    "create_interface"
]

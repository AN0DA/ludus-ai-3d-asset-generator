"""
3D Model Preview and Visualization Components.

This module provides comprehensive 3D model preview functionality including
Model3D integration, fallback image preview, interactive controls, metadata overlay,
and responsive design for the asset generator interface.
"""

import gradio as gr
import json
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import mimetypes
from PIL import Image
import io

from ..models.asset_model import FileFormat, AssetFile, AssetMetadata, TechnicalSpecs


class ModelPreview:
    """3D model preview and visualization handler."""
    
    SUPPORTED_3D_FORMATS = {
        FileFormat.OBJ, FileFormat.GLTF, FileFormat.GLB, 
        FileFormat.FBX, FileFormat.STL, FileFormat.PLY
    }
    
    FALLBACK_FORMATS = {FileFormat.DAE}
    
    @staticmethod
    def create_model_viewer(
        height: int = 400,
        width: Optional[int] = None,
        show_controls: bool = True,
        show_info: bool = True
    ) -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create comprehensive 3D model viewer with controls."""
        components = {}
        
        with gr.Group(elem_classes=["model-viewer-container"]) as viewer_group:
            # Header with controls
            if show_controls:
                with gr.Row(elem_classes=["viewer-controls"]):
                    components["reset_view"] = gr.Button(
                        value="🔄 Reset View",
                        size="sm",
                        variant="secondary"
                    )
                    
                    components["wireframe_toggle"] = gr.Button(
                        value="🔲 Wireframe",
                        size="sm",
                        variant="secondary"
                    )
                    
                    components["lighting_toggle"] = gr.Button(
                        value="💡 Lighting",
                        size="sm",
                        variant="secondary"
                    )
                    
                    components["fullscreen_button"] = gr.Button(
                        value="⛶ Fullscreen",
                        size="sm",
                        variant="secondary"
                    )
            
            # Main viewer area
            with gr.Row():
                with gr.Column(scale=3):
                    # 3D Model viewer
                    components["model_3d"] = gr.Model3D(
                        label="3D Model",
                        show_label=False,
                        height=height,
                        elem_classes=["model-viewer-3d"]
                    )
                    
                    # Fallback image viewer (hidden by default)
                    components["fallback_image"] = gr.Image(
                        label="Model Preview",
                        show_label=False,
                        height=height,
                        visible=False,
                        elem_classes=["model-viewer-fallback"]
                    )
                    
                    # Loading placeholder
                    components["loading_placeholder"] = gr.HTML(
                        value=ModelPreview._create_loading_placeholder(),
                        visible=False,
                        elem_classes=["model-viewer-loading"]
                    )
                
                if show_info:
                    with gr.Column(scale=1, elem_classes=["model-info-panel"]):
                        components["info_panel"] = ModelPreview._create_info_panel()
            
            # Control panel for advanced options
            if show_controls:
                with gr.Accordion("Advanced Viewer Options", open=False):
                    with gr.Row():
                        components["background_color"] = gr.ColorPicker(
                            label="Background Color",
                            value="#f0f0f0"
                        )
                        
                        components["model_scale"] = gr.Slider(
                            label="Model Scale",
                            minimum=0.1,
                            maximum=5.0,
                            value=1.0,
                            step=0.1
                        )
                    
                    with gr.Row():
                        components["camera_distance"] = gr.Slider(
                            label="Camera Distance",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=0.5
                        )
                        
                        components["auto_rotate"] = gr.Checkbox(
                            label="Auto Rotate",
                            value=False
                        )
        
        return viewer_group, components
    
    @staticmethod
    def _create_info_panel() -> gr.HTML:
        """Create the model information panel."""
        return gr.HTML(
            value="""
            <div class="model-info-panel">
                <h4>📊 Model Information</h4>
                <div id="model-stats">
                    <p><strong>Format:</strong> <span id="model-format">-</span></p>
                    <p><strong>Polygons:</strong> <span id="model-polygons">-</span></p>
                    <p><strong>Vertices:</strong> <span id="model-vertices">-</span></p>
                    <p><strong>File Size:</strong> <span id="model-size">-</span></p>
                    <p><strong>Materials:</strong> <span id="model-materials">-</span></p>
                </div>
                
                <h4>🎮 Controls</h4>
                <div class="control-help">
                    <p><strong>Mouse:</strong></p>
                    <ul>
                        <li>Left click + drag: Rotate</li>
                        <li>Scroll wheel: Zoom</li>
                        <li>Right click + drag: Pan</li>
                    </ul>
                    
                    <p><strong>Touch:</strong></p>
                    <ul>
                        <li>One finger: Rotate</li>
                        <li>Pinch: Zoom</li>
                        <li>Two fingers: Pan</li>
                    </ul>
                </div>
                
                <div class="viewer-stats" id="viewer-performance">
                    <h4>⚡ Performance</h4>
                    <p><strong>FPS:</strong> <span id="fps-counter">-</span></p>
                    <p><strong>Render Time:</strong> <span id="render-time">-</span>ms</p>
                </div>
            </div>
            """,
            elem_classes=["model-info-content"]
        )
    
    @staticmethod
    def _create_loading_placeholder() -> str:
        """Create loading placeholder HTML."""
        return """
        <div class="model-loading-container">
            <div class="loading-spinner-3d">
                <div class="cube">
                    <div class="face front"></div>
                    <div class="face back"></div>
                    <div class="face right"></div>
                    <div class="face left"></div>
                    <div class="face top"></div>
                    <div class="face bottom"></div>
                </div>
            </div>
            <p class="loading-text">Preparing 3D model...</p>
        </div>
        
        <style>
        .model-loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
        }
        
        .loading-spinner-3d {
            perspective: 1000px;
            margin-bottom: 20px;
        }
        
        .cube {
            position: relative;
            width: 60px;
            height: 60px;
            transform-style: preserve-3d;
            animation: rotate3d 2s infinite linear;
        }
        
        .face {
            position: absolute;
            width: 60px;
            height: 60px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            opacity: 0.8;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .front { transform: rotateY(0deg) translateZ(30px); }
        .back { transform: rotateY(180deg) translateZ(30px); }
        .right { transform: rotateY(90deg) translateZ(30px); }
        .left { transform: rotateY(-90deg) translateZ(30px); }
        .top { transform: rotateX(90deg) translateZ(30px); }
        .bottom { transform: rotateX(-90deg) translateZ(30px); }
        
        @keyframes rotate3d {
            from { transform: rotateX(0deg) rotateY(0deg); }
            to { transform: rotateX(360deg) rotateY(360deg); }
        }
        
        .loading-text {
            color: #4a5568;
            font-size: 16px;
            font-weight: 500;
            margin: 0;
        }
        </style>
        """
    
    @staticmethod
    def can_preview_3d(file_format: FileFormat) -> bool:
        """Check if format can be previewed in 3D viewer."""
        return file_format in ModelPreview.SUPPORTED_3D_FORMATS
    
    @staticmethod
    def needs_fallback(file_format: FileFormat) -> bool:
        """Check if format needs fallback image preview."""
        return file_format in ModelPreview.FALLBACK_FORMATS
    
    @staticmethod
    def load_model(
        model_path: str,
        file_format: FileFormat,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Load 3D model and return appropriate preview data.
        
        Returns:
            Tuple of (model_path_for_3d_viewer, fallback_image_path, info_html)
        """
        if not Path(model_path).exists():
            return None, None, ModelPreview._create_error_info("File not found")
        
        # Check if we can display in 3D viewer
        if ModelPreview.can_preview_3d(file_format):
            info_html = ModelPreview._create_model_info(model_path, file_format, metadata)
            return model_path, None, info_html
        
        # Use fallback for unsupported formats
        elif ModelPreview.needs_fallback(file_format):
            fallback_image = ModelPreview._generate_fallback_image(model_path, file_format)
            info_html = ModelPreview._create_model_info(model_path, file_format, metadata)
            return None, fallback_image, info_html
        
        else:
            return None, None, ModelPreview._create_error_info(f"Unsupported format: {file_format.value}")
    
    @staticmethod
    def _create_model_info(
        model_path: str,
        file_format: FileFormat,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create model information HTML."""
        file_path = Path(model_path)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        info = {
            "format": file_format.value.upper(),
            "size": ModelPreview._format_file_size(file_size),
            "polygons": "Unknown",
            "vertices": "Unknown", 
            "materials": "Unknown"
        }
        
        # Extract metadata if available
        if metadata:
            if "polygon_count" in metadata:
                info["polygons"] = f"{metadata['polygon_count']:,}"
            if "vertex_count" in metadata:
                info["vertices"] = f"{metadata['vertex_count']:,}"
            if "materials_count" in metadata:
                info["materials"] = str(metadata["materials_count"])
        
        return f"""
        <script>
        document.getElementById('model-format').textContent = '{info["format"]}';
        document.getElementById('model-polygons').textContent = '{info["polygons"]}';
        document.getElementById('model-vertices').textContent = '{info["vertices"]}';
        document.getElementById('model-size').textContent = '{info["size"]}';
        document.getElementById('model-materials').textContent = '{info["materials"]}';
        </script>
        """
    
    @staticmethod
    def _create_error_info(error_message: str) -> str:
        """Create error information HTML."""
        return f"""
        <div class="model-error">
            <h4>❌ Preview Error</h4>
            <p>{error_message}</p>
        </div>
        """
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    @staticmethod
    def _generate_fallback_image(model_path: str, file_format: FileFormat) -> Optional[str]:
        """Generate fallback image for unsupported 3D formats."""
        # Create a placeholder image for unsupported formats
        try:
            # Create a simple placeholder image
            img = Image.new('RGB', (400, 400), color=(240, 240, 240))
            
            # Add format-specific overlay (simplified implementation)
            # In a real implementation, you might use libraries like Open3D, trimesh, or Blender
            # to actually render the 3D model to an image
            
            # Save to temporary location
            temp_path = Path(model_path).parent / f"preview_{Path(model_path).stem}.png"
            img.save(temp_path)
            
            return str(temp_path)
            
        except Exception:
            return None
    
    @staticmethod
    def create_responsive_viewer(
        asset_metadata: Optional[AssetMetadata] = None
    ) -> Tuple[gr.Group, Dict[str, gr.Component]]:
        """Create responsive 3D viewer that adapts to screen size."""
        components = {}
        
        with gr.Group(elem_classes=["responsive-viewer"]) as viewer_group:
            # Mobile-friendly header
            with gr.Row(elem_classes=["viewer-header"]):
                gr.HTML(
                    value="<h3>🎯 3D Model Preview</h3>",
                    elem_classes=["viewer-title"]
                )
                
                components["fullscreen_btn"] = gr.Button(
                    value="⛶",
                    size="sm",
                    elem_classes=["fullscreen-toggle"]
                )
            
            # Responsive viewer container
            with gr.Group(elem_classes=["viewer-responsive-container"]):
                # Desktop/Tablet view
                with gr.Row(elem_classes=["desktop-view"]):
                    with gr.Column(scale=2):
                        components["model_3d"] = gr.Model3D(
                            label="3D Model",
                            show_label=False,
                            height=450,
                            elem_classes=["model-3d-desktop"]
                        )
                    
                    with gr.Column(scale=1):
                        components["info_desktop"] = gr.HTML(
                            value=ModelPreview._create_responsive_info_panel(),
                            elem_classes=["info-panel-desktop"]
                        )
                
                # Mobile view (stacked layout)
                with gr.Group(elem_classes=["mobile-view"], visible=False):
                    components["model_3d_mobile"] = gr.Model3D(
                        label="3D Model",
                        show_label=False,
                        height=300,
                        elem_classes=["model-3d-mobile"]
                    )
                    
                    with gr.Accordion("Model Info", open=False):
                        components["info_mobile"] = gr.HTML(
                            value=ModelPreview._create_responsive_info_panel(),
                            elem_classes=["info-panel-mobile"]
                        )
            
            # Touch-friendly controls
            with gr.Row(elem_classes=["mobile-controls"]):
                components["rotate_left"] = gr.Button(
                    value="↺",
                    size="sm",
                    elem_classes=["touch-control"]
                )
                
                components["rotate_right"] = gr.Button(
                    value="↻",
                    size="sm", 
                    elem_classes=["touch-control"]
                )
                
                components["zoom_in"] = gr.Button(
                    value="🔍+",
                    size="sm",
                    elem_classes=["touch-control"]
                )
                
                components["zoom_out"] = gr.Button(
                    value="🔍-",
                    size="sm",
                    elem_classes=["touch-control"]
                )
                
                components["reset_view_mobile"] = gr.Button(
                    value="🔄",
                    size="sm",
                    elem_classes=["touch-control"]
                )
        
        return viewer_group, components
    
    @staticmethod
    def _create_responsive_info_panel() -> str:
        """Create responsive info panel HTML."""
        return """
        <div class="responsive-info-panel">
            <div class="model-stats">
                <h4>📊 Model Info</h4>
                <div class="stat-grid">
                    <div class="stat-item">
                        <span class="stat-label">Format:</span>
                        <span id="resp-format" class="stat-value">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Polygons:</span>
                        <span id="resp-polygons" class="stat-value">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Size:</span>
                        <span id="resp-size" class="stat-value">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Quality:</span>
                        <span id="resp-quality" class="stat-value">-</span>
                    </div>
                </div>
            </div>
            
            <div class="viewer-help">
                <h4>🎮 How to Navigate</h4>
                <div class="help-content">
                    <p><strong>Desktop:</strong> Click & drag to rotate, scroll to zoom</p>
                    <p><strong>Mobile:</strong> Touch & drag to rotate, pinch to zoom</p>
                    <p><strong>Touch Controls:</strong> Use buttons below for precise control</p>
                </div>
            </div>
        </div>
        
        <style>
        .responsive-info-panel {
            padding: 16px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 8px;
            font-size: 14px;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 12px 0;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
        }
        
        .stat-label {
            font-weight: 500;
            color: #4a5568;
        }
        
        .stat-value {
            font-weight: 600;
            color: #2d3748;
        }
        
        .help-content p {
            margin: 4px 0;
            font-size: 12px;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .desktop-view { display: none !important; }
            .mobile-view { display: block !important; }
            .mobile-controls { display: flex !important; }
            
            .responsive-info-panel {
                font-size: 12px;
                padding: 12px;
            }
            
            .stat-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (min-width: 769px) {
            .mobile-view { display: none !important; }
            .mobile-controls { display: none !important; }
        }
        </style>
        """
    
    @staticmethod
    def create_gallery_viewer(assets: List[AssetMetadata]) -> gr.Gallery:
        """Create gallery viewer for multiple assets."""
        gallery_items = []
        
        for asset in assets:
            if asset.thumbnail_url:
                gallery_items.append((asset.thumbnail_url, asset.name))
            elif asset.files:
                # Use first file as preview
                file_path = asset.files[0].file_path
                if file_path and Path(file_path).exists():
                    gallery_items.append((file_path, asset.name))
        
        return gr.Gallery(
            value=gallery_items,
            label="Asset Gallery",
            show_label=False,
            elem_classes=["asset-gallery-viewer"],
            columns=3,
            rows=2,
            height=300,
            object_fit="cover"
        )
    
    @staticmethod
    def update_model_info(
        metadata: Dict[str, Any],
        technical_specs: Optional[TechnicalSpecs] = None
    ) -> str:
        """Update model information display with new data."""
        format_name = metadata.get("format", "Unknown").upper()
        file_size = ModelPreview._format_file_size(metadata.get("file_size", 0))
        
        polygons = "Unknown"
        vertices = "Unknown"
        materials = "Unknown"
        quality = "Unknown"
        
        if technical_specs:
            if technical_specs.polygon_count:
                polygons = f"{technical_specs.polygon_count:,}"
            if technical_specs.vertex_count:
                vertices = f"{technical_specs.vertex_count:,}"
            if technical_specs.materials_count:
                materials = str(technical_specs.materials_count)
        
        if "quality_level" in metadata:
            quality = metadata["quality_level"].title()
        
        return f"""
        <script>
        // Update desktop info
        if (document.getElementById('model-format'))
            document.getElementById('model-format').textContent = '{format_name}';
        if (document.getElementById('model-polygons'))
            document.getElementById('model-polygons').textContent = '{polygons}';
        if (document.getElementById('model-vertices'))
            document.getElementById('model-vertices').textContent = '{vertices}';
        if (document.getElementById('model-size'))
            document.getElementById('model-size').textContent = '{file_size}';
        if (document.getElementById('model-materials'))
            document.getElementById('model-materials').textContent = '{materials}';
        
        // Update responsive info
        if (document.getElementById('resp-format'))
            document.getElementById('resp-format').textContent = '{format_name}';
        if (document.getElementById('resp-polygons'))
            document.getElementById('resp-polygons').textContent = '{polygons}';
        if (document.getElementById('resp-size'))
            document.getElementById('resp-size').textContent = '{file_size}';
        if (document.getElementById('resp-quality'))
            document.getElementById('resp-quality').textContent = '{quality}';
        </script>
        """


# Additional CSS for 3D viewer components
VIEWER_CSS = """
/* 3D Model Viewer Specific Styles */
.model-viewer-container {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.viewer-controls {
    margin-bottom: 16px;
    gap: 8px;
}

.viewer-controls button {
    border-radius: 6px !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
}

.model-viewer-3d {
    border-radius: 12px;
    border: 2px solid #e2e8f0;
    background: #f7fafc;
}

.model-viewer-fallback {
    border-radius: 12px;
    border: 2px solid #e2e8f0;
}

.model-info-panel {
    background: rgba(249, 250, 251, 0.95);
    border-radius: 12px;
    padding: 16px;
    height: fit-content;
}

.model-info-content h4 {
    color: #2d3748;
    font-size: 14px;
    font-weight: 600;
    margin: 16px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
}

.model-info-content p {
    margin: 4px 0;
    font-size: 12px;
    color: #4a5568;
}

.model-info-content ul {
    margin: 8px 0;
    padding-left: 16px;
}

.model-info-content li {
    font-size: 11px;
    color: #666;
    margin: 2px 0;
}

.control-help {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
}

.viewer-stats {
    background: rgba(102, 126, 234, 0.1);
    border-radius: 6px;
    padding: 8px;
    margin: 8px 0;
}

/* Responsive Viewer Styles */
.responsive-viewer {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 16px;
    padding: 16px;
    margin: 16px 0;
}

.viewer-header {
    margin-bottom: 16px;
    align-items: center;
}

.viewer-title h3 {
    margin: 0;
    color: #2d3748;
    font-weight: 600;
}

.fullscreen-toggle {
    background: transparent !important;
    border: 1px solid #e2e8f0 !important;
    color: #4a5568 !important;
    width: 40px !important;
    height: 40px !important;
    border-radius: 8px !important;
}

.mobile-controls {
    margin-top: 12px;
    gap: 8px;
    justify-content: center;
    display: none;
}

.touch-control {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    width: 44px !important;
    height: 44px !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
}

.touch-control:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* Asset Gallery Viewer */
.asset-gallery-viewer {
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.95);
    padding: 16px;
}

.asset-gallery-viewer img {
    border-radius: 8px;
    transition: transform 0.2s ease;
}

.asset-gallery-viewer img:hover {
    transform: scale(1.05);
}

/* Loading and Error States */
.model-error {
    text-align: center;
    padding: 40px 20px;
    background: rgba(229, 62, 62, 0.1);
    border-radius: 12px;
    border: 1px solid rgba(229, 62, 62, 0.2);
}

.model-error h4 {
    color: #e53e3e;
    margin-bottom: 8px;
}

.model-error p {
    color: #c53030;
    font-size: 14px;
}

/* Performance Optimizations */
.model-viewer-3d canvas {
    border-radius: 12px;
}

/* Accessibility */
@media (prefers-reduced-motion: reduce) {
    .touch-control:hover {
        transform: none !important;
    }
    
    .asset-gallery-viewer img:hover {
        transform: none;
    }
}

/* High DPI Display Support */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    .model-viewer-3d {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
    }
}
"""


# Export classes and utilities
__all__ = [
    "ModelPreview",
    "VIEWER_CSS"
]

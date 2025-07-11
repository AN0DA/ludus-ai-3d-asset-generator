"""
Simplified UI utilities for the AI 3D Asset Generator.

This module provides essential utility functions for the Gradio interface,
extracted and simplified from the original components module.
"""

from typing import Any, Dict, List, Tuple, Optional
import gradio as gr
from datetime import datetime

from ..models.asset_model import AssetMetadata


class UIUtils:
    """Utility functions for the UI."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def create_status_html(status: str, message: str, icon: str = "") -> str:
        """Create status HTML with appropriate styling."""
        status_classes = {
            "success": "status-success",
            "error": "status-error", 
            "warning": "status-warning",
            "info": "status-info"
        }
        
        default_icons = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️", 
            "info": "ℹ️"
        }
        
        class_name = status_classes.get(status, "status-info")
        display_icon = icon or default_icons.get(status, "ℹ️")
        
        return f'<div class="{class_name}">{display_icon} {message}</div>'
    
    @staticmethod
    def create_progress_html(percentage: int, message: str) -> str:
        """Create progress bar HTML."""
        return f'''
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%"></div>
            </div>
            <p>{message}</p>
        </div>
        '''
    
    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        """Format asset metadata for display."""
        if not metadata:
            return UIUtils.create_status_html("info", "No metadata available")
        
        html_parts = ["<div class='metadata-display'>"]
        
        # Basic info
        if "polygon_count" in metadata:
            html_parts.append(f"<p><strong>Polygons:</strong> {metadata['polygon_count']:,}</p>")
        
        if "file_size" in metadata:
            size_str = UIUtils.format_file_size(metadata["file_size"])
            html_parts.append(f"<p><strong>File Size:</strong> {size_str}</p>")
        
        if "generation_time" in metadata:
            duration_str = UIUtils.format_duration(metadata["generation_time"])
            html_parts.append(f"<p><strong>Generation Time:</strong> {duration_str}</p>")
        
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
        polygon_count: Optional[int] = None,
        priority: Optional[int] = None
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
            '<script', 'javascript:', 'vbscript:', 'on\\w+\\s*=',
            'eval(', 'setTimeout(', 'setInterval('
        ]
        
        description_lower = description.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in description_lower:
                errors.append("Description contains potentially unsafe content")
                break
        
        # Technical validation
        if polygon_count is not None:
            if polygon_count < 1000 or polygon_count > 100000:
                errors.append("Polygon count must be between 1,000 and 100,000")
        
        if priority is not None:
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


# Export utilities
__all__ = ["UIUtils"]

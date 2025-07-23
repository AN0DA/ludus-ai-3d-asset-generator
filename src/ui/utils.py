"""
Utility functions for the UI module.

This module contains helper functions for formatting and processing UI data.
"""

import structlog

logger = structlog.get_logger(__name__)


class UIUtils:
    """Utility functions for UI operations."""

    @staticmethod
    def format_metadata_html(metadata: dict) -> str:
        """Format metadata as HTML for display."""
        if not metadata:
            return '<div class="no-asset">No metadata found.</div>'

        html = '<div class="asset-metadata"><h4>Asset Details</h4>'
        field_labels = {
            "name": "Asset Name",
            "description": "Description",
            "asset_type": "Type",
            "quality_level": "Quality Level",
            "file_format": "Format",
            "file_size": "File Size",
            "polygon_count": "Polygon Count",
            "content_type": "Content Type",
            "last_modified": "Last Modified",
            "generated_at": "Generated At",
            "generation_time": "Generation Time",
            "service": "Service",
            "cost": "Cost",
        }

        for key, label in field_labels.items():
            if key in metadata:
                value = metadata[key]
                if key == "file_size" and isinstance(value, int | float):
                    value = (
                        f"{int(value) / 1024:.2f} KB" if value < 1024 * 1024 else f"{int(value) / (1024 * 1024):.2f} MB"
                    )
                elif key == "polygon_count" and isinstance(value, int | float):
                    value = f"{int(value):,}"
                elif key == "cost" and isinstance(value, int | float):
                    value = f"${float(value):.2f}"
                elif key == "generation_time" and isinstance(value, int | float):
                    value = f"{float(value):.1f} seconds"
                html += f'<div class="metadata-grid"><strong>{label}:</strong> {value}</div>'

        for key, value in metadata.items():
            if key not in field_labels:
                label = key.replace("_", " ").title()
                html += f'<div class="metadata-grid"><strong>{label}:</strong> {value}</div>'

        html += "</div>"
        return html

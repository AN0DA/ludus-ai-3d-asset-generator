"""
Event Handlers for the AI 3D Asset Generator UI.

This module contains all the event handling logic and business operations
for the Gradio interface.
"""

import asyncio
import json
import os
import tempfile
from typing import Any

import structlog

from src.core.app import AssetGenerationApp
from src.models.asset_model import AssetType, FileFormat, QualityLevel, StylePreference
from src.utils.validators import ValidationException

from .utils import UIUtils

logger = structlog.get_logger(__name__)


class UIHandlers:
    """Event handlers for UI interactions."""

    def __init__(self, app: AssetGenerationApp):
        """Initialize handlers with the application instance."""
        self.app = app
        self.generation_history: list[dict[str, Any]] = []
        self.current_task_id: str | None = None
        self.current_asset_key: str | None = None

    async def generate_asset_async(
        self, description: str, asset_type: str, style: str, quality: str, format_choice: str
    ) -> tuple[str, str, str, bool]:
        """Generate a 3D asset with the given parameters."""
        try:
            logger.info("generate_asset_async called", description=description[:30])

            # Validate inputs
            validation_error = self._validate_inputs(description, asset_type, style, quality, format_choice)
            if validation_error:
                return "", "", validation_error, False

            # Convert string values to enums
            asset_type_enum = AssetType(asset_type)
            style_enum = StylePreference(style) if style and style.lower() != "none" else None
            quality_enum = QualityLevel(quality)
            # format_enum = FileFormat(format_choice)

            logger.info("Validation passed, creating session")

            # Start generation
            session_id = self.app.session_manager.create_session()
            task_id, session_id = await self.app.generate_asset_pipeline(
                description=description.strip(),
                asset_type=asset_type_enum,
                style_preference=style_enum,
                quality_level=quality_enum,
                session_id=session_id,
            )
            self.current_task_id = task_id

            # Wait briefly to allow task initialization
            await asyncio.sleep(0.1)

            # Check initial task status
            task_status = self.app.task_manager.get_task_status(task_id)
            if task_status.get("status") == "not_found":
                error_msg = """
                <div class="status-error">
                    ❌ <strong>Task Creation Failed:</strong> Unable to start generation task.
                    <br><small>Please try again or contact support.</small>
                </div>
                """
                return "", "", error_msg, False

            status_html = """
            <div class="status-success">
                🚀 <strong>Generation Started!</strong>
                <br><small>AI is working on your 3D asset...</small>
            </div>
            """
            progress_html = """
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 5%"></div>
                </div>
                <p>🔄 Initializing generation pipeline...</p>
            </div>
            """

            return status_html, progress_html, "", True

        except ValidationException as e:
            logger.warning("Validation error in asset generation", error=str(e))
            return "", "", self._create_detailed_error_message(e), False

        except ValueError as e:
            logger.warning("Value error in asset generation", error=str(e))
            return "", "", self._create_detailed_error_message(e), False

        except Exception as e:
            logger.error("Unexpected error in asset generation", error=str(e), exc_info=True)
            return "", "", self._create_detailed_error_message(e), False

    def generate_asset_sync(
        self, description: str, asset_type: str, style: str, quality: str, format_choice: str
    ) -> tuple[str, str, str, bool]:
        """Sync wrapper for the async generate_asset method."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(
                            self.generate_asset_async(description, asset_type, style, quality, format_choice)
                        )
                    )
                    return future.result(timeout=30)
            else:
                return asyncio.run(self.generate_asset_async(description, asset_type, style, quality, format_choice))
        except Exception as e:
            logger.error("Error in sync wrapper", error=str(e), exc_info=True)
            error_msg = f"""
            <div class="status-error">
                ❌ <strong>System Error:</strong> Failed to process request.
                <br><small>Error: {str(e)}</small>
            </div>
            """
            return "", "", error_msg, False

    async def cancel_generation_async(self) -> tuple[str, str, bool]:
        """Cancel the current generation."""
        if not self.current_task_id:
            return (
                """
                <div class="status-success">
                    Ready to generate
                    <br><small>No active generation to cancel.</small>
                </div>
                """,
                "",
                False,
            )

        try:
            success = self.app.task_manager.cancel_task(self.current_task_id)
            self.current_task_id = None
            if success:
                return (
                    """
                    <div class="status-warning">
                        ⚠️ <strong>Generation Cancelled</strong>
                        <br><small>You can start a new generation anytime.</small>
                    </div>
                    """,
                    "",
                    False,
                )
            else:
                return (
                    """
                    <div class="status-error">
                        ❌ <strong>Failed to Cancel:</strong> Unable to stop the generation.
                        <br><small>The generation may have already completed or failed.</small>
                    </div>
                    """,
                    "",
                    False,
                )
        except Exception as e:
            logger.error("Cancel failed", error=str(e))
            return (
                f"""
                <div class="status-error">
                    ❌ <strong>Cancel Error:</strong> {str(e)}
                    <br><small>The generation may still be running in the background.</small>
                </div>
                """,
                "",
                False,
            )

    def cancel_generation_sync(self) -> tuple[str, str, bool]:
        """Sync wrapper for the async cancel_generation method."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.cancel_generation_async()))
                    return future.result(timeout=10)
            else:
                return asyncio.run(self.cancel_generation_async())
        except Exception as e:
            logger.error("Error in cancel sync wrapper", error=str(e))
            return (
                """
                <div class="status-error">
                    ❌ <strong>Cancel Error:</strong> Failed to cancel generation.
                </div>
                """,
                "",
                False,
            )

    def check_progress_sync(self) -> tuple[str, str, bool, bool]:
        """Check the progress of the current generation task."""
        if not self.current_task_id:
            return "", "", False, False

        try:
            task_status = self.app.task_manager.get_task_status(self.current_task_id)
            if not task_status or task_status.get("status") == "not_found":
                self.current_task_id = None
                return (
                    """
                    <div class="status-error">
                        ❌ <strong>Task Lost:</strong> Unable to find generation task.
                    </div>
                    """,
                    "",
                    False,
                    False,
                )

            status = task_status.get("status", "unknown")
            progress = task_status.get("progress", 0.0)
            message = task_status.get("message", "Processing...")
            current_step = task_status.get("current_step", "unknown")

            if status == "completed":
                self.current_task_id = None
                completion_html = """
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
                    const tabs = document.querySelectorAll('[role="tab"]');
                    for (let tab of tabs) {
                        if (tab.textContent.includes('Results') || tab.textContent.includes('📋')) {
                            tab.click();
                            setTimeout(() => {
                                const refreshBtn = document.querySelector('button:contains("🔄 Refresh")') ||
                                                 [...document.querySelectorAll('button')].find(btn =>
                                                     btn.textContent.includes('🔄 Refresh'));
                                if (refreshBtn) {
                                    refreshBtn.click();
                                }
                            }, 100);
                            break;
                        }
                    }
                }
                </script>
                """
                return completion_html, "", False, False

            elif status == "failed":
                self.current_task_id = None
                error_msg = task_status.get("error", "Unknown error occurred")
                return (
                    f"""
                    <div class="status-error">
                        ❌ <strong>Generation Failed:</strong> {error_msg}
                        <br><small>Please try again with different parameters.</small>
                    </div>
                    """,
                    "",
                    False,
                    False,
                )

            elif status == "cancelled":
                self.current_task_id = None
                return (
                    """
                    <div class="status-warning">
                        ⚠️ <strong>Generation Cancelled</strong>
                        <br><small>You can start a new generation anytime.</small>
                    </div>
                    """,
                    "",
                    False,
                    False,
                )

            else:
                progress_percent = max(0, min(100, progress * 100))
                progress_html = f"""
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress_percent:.1f}%"></div>
                    </div>
                    <p>🔄 {message} ({progress_percent:.1f}%)</p>
                    <small>Current step: {current_step}</small>
                </div>
                """
                return "", progress_html, True, True

        except Exception as e:
            logger.error("Error checking progress", error=str(e))
            self.current_task_id = None
            return (
                f"""
                <div class="status-error">
                    ❌ <strong>Progress Check Failed:</strong> {str(e)}
                </div>
                """,
                "",
                False,
                False,
            )

    def refresh_asset_list(self) -> list[tuple[str, str]]:
        """Refresh the asset dropdown list."""
        try:
            storage = self.app.cloud_storage
            if not storage:
                return []

            files = asyncio.run(storage.list_files(prefix="assets/"))
            files_sorted = sorted(files, key=lambda x: x.last_modified, reverse=True)
            dropdown_choices = [
                (f"{f.key.split('/')[-1]} ({f.last_modified.strftime('%Y-%m-%d %H:%M')})", f.key) for f in files_sorted
            ]
            return dropdown_choices

        except Exception as e:
            logger.error(f"Error refreshing asset list: {e}")
            return []

    def display_selected_asset(self, asset_key: str) -> tuple[str | None, str, bool, bool]:
        """Display selected asset from the dropdown."""
        try:
            storage = self.app.cloud_storage
            if not storage or not asset_key:
                return None, '<div class="no-asset">Please select an asset.</div>', False, False

            file_info = asyncio.run(storage.get_file_info(asset_key))
            model_name = asset_key.split("/")[-1].split(".")[0]
            metadata_key = f"metadata/{model_name}.json"

            try:
                metadata_bytes = asyncio.run(storage.download_bytes(metadata_key))
                metadata = json.loads(metadata_bytes.decode())
            except Exception as e:
                logger.warning(f"Could not load metadata for {model_name}: {e}")
                metadata = {
                    "name": model_name,
                    "file_size": file_info.size,
                    "last_modified": file_info.last_modified.strftime("%Y-%m-%d %H:%M:%S"),
                    "content_type": file_info.content_type,
                }

            model_url = None
            if hasattr(storage, "generate_public_url"):
                try:
                    public_url = asyncio.run(storage.generate_public_url(asset_key))
                    if public_url and "localhost" not in public_url and public_url.startswith(("http://", "https://")):
                        model_url = public_url
                except Exception as e:
                    logger.warning(f"Could not generate public URL: {e}")

            metadata_html = UIUtils.format_metadata_html(metadata)
            if not model_url:
                metadata_html += """
                <div class="status-info" style="margin-top: 1rem; padding: 1rem; background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px;">
                    📁 <strong>File Information:</strong><br>
                    • Asset is stored in cloud storage<br>
                    • 3D preview not available (localhost storage)<br>
                    • Use download button to access the file
                </div>
                """

            self.current_asset_key = asset_key
            return model_url, metadata_html, True, True

        except Exception as e:
            logger.error(f"Error displaying selected asset {asset_key}: {e}")
            error_html = f'<div class="status-error">Error loading asset: {str(e)}</div>'
            return None, error_html, False, False

    def download_selected_asset(self, asset_key: str | None) -> str | None:
        """Download the selected asset file."""
        try:
            storage = self.app.cloud_storage
            if not storage or not asset_key:
                return None

            file_name = asset_key.split("/")[-1]
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, file_name)
            asyncio.run(storage.download_file(asset_key, temp_file_path))
            return temp_file_path

        except Exception as e:
            logger.error(f"Error downloading asset {asset_key}: {e}")
            return None

    def _validate_inputs(
        self, description: str, asset_type: str, style: str, quality: str, format_choice: str
    ) -> str | None:
        """Validate user inputs and return error message if invalid."""
        if not description or len(description.strip()) < 10:
            return """
            <div class="status-error">
                ❌ <strong>Validation Error:</strong> Description must be at least 10 characters long.
                <br><small>Please provide more detail about your 3D asset.</small>
            </div>
            """

        if len(description.strip()) > 2000:
            return """
            <div class="status-error">
                ❌ <strong>Validation Error:</strong> Description is too long (max 2000 characters).
                <br><small>Please shorten your description.</small>
            </div>
            """

        try:
            AssetType(asset_type)
        except ValueError:
            valid_types = ", ".join([t.value for t in AssetType])
            return f'''
            <div class="status-error">
                ❌ <strong>Invalid Asset Type:</strong> "{asset_type}" is not valid.
                <br><small>Valid types: {valid_types}</small>
            </div>
            '''

        if style and style.lower() != "none":
            try:
                StylePreference(style)
            except ValueError:
                valid_styles = ", ".join([s.value for s in StylePreference])
                return f'''
                <div class="status-error">
                    ❌ <strong>Invalid Style:</strong> "{style}" is not valid.
                    <br><small>Valid styles: {valid_styles}</small>
                </div>
                '''

        try:
            QualityLevel(quality)
        except ValueError:
            valid_qualities = ", ".join([q.value for q in QualityLevel])
            return f'''
            <div class="status-error">
                ❌ <strong>Invalid Quality:</strong> "{quality}" is not valid.
                <br><small>Valid qualities: {valid_qualities}</small>
            </div>
            '''

        try:
            FileFormat(format_choice)
        except ValueError:
            valid_formats = ", ".join([f.value for f in FileFormat])
            return f'''
            <div class="status-error">
                ❌ <strong>Invalid Format:</strong> "{format_choice}" is not valid.
                <br><small>Valid formats: {valid_formats}</small>
            </div>
            '''

        return None

    def _create_detailed_error_message(self, error: Exception) -> str:
        """Create detailed, user-friendly error messages."""
        error_str = str(error).lower()
        if "invalid asset type" in error_str or "asset type" in error_str:
            valid_types = ", ".join([t.value for t in AssetType])
            return f"""
            <div class="status-error">
                ❌ <strong>Invalid Asset Type:</strong> The selected asset type is not supported.
                <br><small>Please choose from: {valid_types}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            """
        elif "invalid style" in error_str or "style preference" in error_str:
            valid_styles = ", ".join([s.value for s in StylePreference])
            return f"""
            <div class="status-error">
                ❌ <strong>Invalid Style:</strong> The selected art style is not supported.
                <br><small>Please choose from: {valid_styles}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            """
        elif "invalid quality" in error_str or "quality level" in error_str:
            valid_qualities = ", ".join([q.value for q in QualityLevel])
            return f"""
            <div class="status-error">
                ❌ <strong>Invalid Quality Level:</strong> The selected quality is not supported.
                <br><small>Please choose from: {valid_qualities}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            """
        elif "invalid format" in error_str or "file format" in error_str:
            valid_formats = ", ".join([f.value for f in FileFormat])
            return f"""
            <div class="status-error">
                ❌ <strong>Invalid Format:</strong> The selected file format is not supported.
                <br><small>Please choose from: {valid_formats}</small>
                <br><small>If you're seeing this error, please refresh the page and try again.</small>
            </div>
            """
        elif "description" in error_str and ("short" in error_str or "length" in error_str):
            return """
            <div class="status-error">
                ❌ <strong>Description Too Short:</strong> Please provide more detail about your asset.
                <br><small>Minimum 10 characters required. Describe the appearance, materials, and style.</small>
            </div>
            """
        elif "api" in error_str or "network" in error_str or "connection" in error_str:
            return """
            <div class="status-error">
                ❌ <strong>Service Unavailable:</strong> Unable to connect to the AI generation service.
                <br><small>This may be a temporary issue. Please try again in a few moments.</small>
                <br><small>If the problem persists, please contact support.</small>
            </div>
            """
        elif "rate limit" in error_str or "quota" in error_str:
            return """
            <div class="status-error">
                ❌ <strong>Rate Limit Reached:</strong> Too many requests in a short time.
                <br><small>Please wait a few minutes before trying again.</small>
                <br><small>Consider using a lower quality setting to reduce processing time.</small>
            </div>
            """
        elif "timeout" in error_str:
            return """
            <div class="status-error">
                ❌ <strong>Generation Timeout:</strong> The generation took too long and was cancelled.
                <br><small>Try using a lower quality setting or simpler description.</small>
                <br><small>High-quality generations can take several minutes.</small>
            </div>
            """
        else:
            return f"""
            <div class="status-error">
                ❌ <strong>Generation Error:</strong> An unexpected error occurred.
                <br><small>Error details: {str(error)}</small>
                <br><small>Please try again or contact support if the problem persists.</small>
            </div>
            """

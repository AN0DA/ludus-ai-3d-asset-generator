"""
Main Gradio application for AI 3D Asset Generator.

This module provides the complete web application with async pipeline,
background task management, state management, and comprehensive error handling.
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import asynccontextmanager
import weakref
import gc

import gradio as gr
import structlog

from src.generators.llm_generator import LLMGenerator, LLMConfig, EnhancedAssetDescription
from src.generators.asset_generator import (
    Asset3DGenerator, GenerationRequest, ServiceProvider, 
    ServiceConfig, GenerationResult
)
from src.storage.cloud_storage import CloudStorage, StorageConfig, StorageProvider
from src.storage.s3_storage import S3Storage
from src.models.asset_model import (
    AssetType, StylePreference, QualityLevel, FileFormat,
    GenerationStatus, AssetRequest, AssetMetadata, EnhancedDescription,
    TechnicalSpecs, CloudStorageInfo
)
from src.ui.interface import AssetGeneratorInterface
from src.ui.components import UIComponents
from src.ui.preview import ModelPreview
from src.utils.config import ConfigManager, get_config
from src.utils.validators import ValidationException


# Configure logging
logger = structlog.get_logger(__name__)


class TaskManager:
    """Manages background tasks and their lifecycle."""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_status: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = 300  # 5 minutes
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def start_cleanup(self) -> None:
        """Start the cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up completed tasks."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_completed_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in task cleanup", error=str(e))
    
    async def cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from memory."""
        completed_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.done():
                completed_tasks.append(task_id)
                # Clean up any exceptions
                try:
                    if not task.cancelled():
                        task.result()
                except Exception as e:
                    logger.error("Task completed with error", task_id=task_id, error=str(e))
        
        for task_id in completed_tasks:
            self.tasks.pop(task_id, None)
            # Keep status for UI updates but mark as cleaned
            if task_id in self.task_status:
                self.task_status[task_id]["cleaned"] = True
        
        logger.info(f"Cleaned up {len(completed_tasks)} completed tasks")
    
    def create_task(self, coro, task_id: Optional[str] = None) -> str:
        """Create and track a new background task."""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        self.task_status[task_id] = {
            "created_at": datetime.utcnow(),
            "status": "started",
            "progress": 0.0,
            "message": "Task started",
            "result": None,
            "error": None
        }
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task."""
        return self.task_status.get(task_id, {"status": "not_found"})
    
    def update_task_status(self, task_id: str, **kwargs) -> None:
        """Update the status of a task."""
        if task_id in self.task_status:
            self.task_status[task_id].update(kwargs)
            self.task_status[task_id]["updated_at"] = datetime.utcnow()
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if not task.done():
                task.cancel()
                self.update_task_status(task_id, status="cancelled")
                return True
        return False
    
    async def shutdown(self) -> None:
        """Shutdown the task manager and clean up all tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task_id, task in self.tasks.items():
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        self.tasks.clear()
        self.task_status.clear()


class SessionManager:
    """Manages user sessions and temporary data."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session_dir = self.temp_dir / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions[session_id] = {
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "temp_dir": session_dir,
            "generation_history": [],
            "cached_results": {},
            "user_preferences": {}
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data if it exists and is valid."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if datetime.utcnow() - session["last_activity"] > self.session_timeout:
            self.cleanup_session(session_id)
            return None
        
        session["last_activity"] = datetime.utcnow()
        return session
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and its temporary files."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            temp_dir = session.get("temp_dir")
            if temp_dir and temp_dir.exists():
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup session dir: {e}")
            
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up all expired sessions."""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if datetime.utcnow() - session["last_activity"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)


class CacheManager:
    """Manages caching for LLM responses and generation results."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 500):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_cache_index()
    
    def _load_cache_index(self) -> None:
        """Load the cache index from disk."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate a cache key for the given data."""
        import hashlib
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, key: str, data: Any) -> Optional[Any]:
        """Get cached data if it exists."""
        cache_key = self._get_cache_key(data)
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Update access time
                    cache_info["last_accessed"] = datetime.utcnow().isoformat()
                    self._save_cache_index()
                    
                    return cached_data
                except Exception as e:
                    logger.warning(f"Failed to load cached data: {e}")
                    # Remove invalid cache entry
                    self.invalidate(key, data)
        
        return None
    
    def set(self, key: str, data: Any, result: Any) -> None:
        """Cache the result for the given data."""
        cache_key = self._get_cache_key(data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            self.cache_index[cache_key] = {
                "key": key,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "file_size": cache_file.stat().st_size
            }
            
            self._save_cache_index()
            self._cleanup_if_needed()
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def invalidate(self, key: str, data: Any) -> None:
        """Remove cached data."""
        cache_key = self._get_cache_key(data)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limit."""
        total_size = sum(info["file_size"] for info in self.cache_index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time and remove oldest
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            for cache_key, info in sorted_entries:
                cache_file = self.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[cache_key]
                total_size -= info["file_size"]
                
                if total_size <= self.max_size_bytes * 0.8:  # Leave some headroom
                    break
            
            self._save_cache_index()


class AssetGenerationApp:
    """Main application class for the AI 3D Asset Generator."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the application."""
        try:
            self.config_manager = ConfigManager()
            self.config = get_config()
        except Exception as e:
            logger.warning(f"Config loading failed, using defaults: {e}")
            self.config = None
        
        # Initialize directories
        self.temp_dir = Path(tempfile.gettempdir()) / "asset_generator"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.task_manager = TaskManager()
        self.session_manager = SessionManager(self.temp_dir)
        self.cache_manager = CacheManager(self.temp_dir / "cache")
        
        # Initialize generators and storage
        self.llm_generator: Optional[LLMGenerator] = None
        self.asset_generator: Optional[Asset3DGenerator] = None
        self.cloud_storage: Optional[CloudStorage] = None
        
        # Initialize UI components
        self.ui_components = UIComponents()
        self.model_preview = ModelPreview()
        
        # Application state
        self.is_initialized = False
        self.generation_queue: List[str] = []
        self.active_generations: Dict[str, str] = {}  # task_id -> session_id
    
    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing Asset Generation App")
            
            # Initialize LLM generator
            llm_config = self._create_llm_config()
            self.llm_generator = LLMGenerator(llm_config)
            
            # Initialize 3D asset generator
            service_configs = self._create_service_configs()
            if service_configs:
                self.asset_generator = Asset3DGenerator(service_configs)
                await self.asset_generator.initialize()
            else:
                logger.warning("No 3D generation services configured")
            
            # Initialize cloud storage
            storage_config = self._create_storage_config()
            self.cloud_storage = S3Storage(storage_config)
            
            # Start background tasks
            self.task_manager.start_cleanup()
            
            self.is_initialized = True
            logger.info("App initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize app: {e}")
            raise
    
    def _create_llm_config(self) -> LLMConfig:
        """Create LLM configuration from app config."""
        if self.config:
            # Get primary LLM provider config
            primary_provider = self.config.llm.get_primary_provider()
            return LLMConfig(
                api_key=primary_provider.api_key or os.getenv("OPENAI_API_KEY", ""),
                base_url=primary_provider.base_url or "https://api.openai.com/v1",
                model=primary_provider.model,
                timeout=primary_provider.timeout,
                max_retries=primary_provider.max_retries,
                max_tokens=primary_provider.max_tokens,
                temperature=primary_provider.temperature
            )
        else:
            # Use defaults
            return LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url="https://api.openai.com/v1",
                model="gpt-4-turbo",
                timeout=60,
                max_retries=3,
                max_tokens=2000,
                temperature=0.7
            )
    
    def _create_service_configs(self) -> Dict[ServiceProvider, ServiceConfig]:
        """Create 3D service configurations from app config."""
        configs = {}
        
        if self.config:
            # Use configuration from app config
            threed_config = self.config.threed_generation
            
            # Meshy AI
            if threed_config.meshy_api_key:
                configs[ServiceProvider.MESHY_AI] = ServiceConfig(
                    api_key=threed_config.meshy_api_key,
                    base_url="https://api.meshy.ai",
                    max_requests_per_minute=10,
                    timeout_seconds=300,
                    cost_per_generation=0.10
                )
        else:
            # Use environment variables as fallback
            meshy_key = os.getenv("MESHY_API_KEY")
            if meshy_key:
                configs[ServiceProvider.MESHY_AI] = ServiceConfig(
                    api_key=meshy_key,
                    base_url="https://api.meshy.ai",
                    max_requests_per_minute=10,
                    timeout_seconds=300,
                    cost_per_generation=0.10
                )
        
        return configs
    
    def _create_storage_config(self) -> StorageConfig:
        """Create storage configuration from app config."""
        if self.config:
            storage_config = self.config.object_storage
            return StorageConfig(
                provider=StorageProvider.S3_COMPATIBLE,
                bucket_name=storage_config.bucket_name,
                region=storage_config.region,
                access_key_id=storage_config.access_key_id,
                secret_access_key=storage_config.secret_access_key,
                endpoint_url=storage_config.endpoint_url
            )
        else:
            # Use environment variables as fallback
            return StorageConfig(
                provider=StorageProvider.S3_COMPATIBLE,
                bucket_name=os.getenv("AWS_BUCKET_NAME", "ai-3d-assets"),
                region=os.getenv("AWS_REGION", "us-east-1"),
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
                endpoint_url=os.getenv("AWS_ENDPOINT_URL")
            )
    
    async def generate_asset_pipeline(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: Optional[StylePreference] = None,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, str]:  # Returns (task_id, session_id)
        """
        Main asset generation pipeline.
        
        Returns task_id and session_id for tracking progress.
        """
        # Create or get session
        if session_id is None:
            session_id = self.session_manager.create_session()
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError("Invalid or expired session")
        
        # Create generation task with a pre-allocated task_id
        task_id = str(uuid.uuid4())
        
        async def task_wrapper():
            return await self._execute_generation_pipeline(
                description=description,
                asset_type=asset_type,
                style_preference=style_preference,
                quality_level=quality_level,
                session_id=session_id,
                progress_callback=progress_callback,
                task_id=task_id
            )
        
        # Create and start the task
        actual_task_id = self.task_manager.create_task(task_wrapper(), task_id)
        
        self.active_generations[actual_task_id] = session_id
        return actual_task_id, session_id
    
    async def _execute_generation_pipeline(
        self,
        description: str,
        asset_type: AssetType,
        style_preference: Optional[StylePreference],
        quality_level: QualityLevel,
        session_id: str,
        progress_callback: Optional[Callable] = None,
        task_id: Optional[str] = None
    ) -> AssetMetadata:
        """Execute the complete asset generation pipeline."""
        
        def update_progress(step: str, progress: float, message: str):
            if progress_callback:
                progress_callback(step, progress, message)
            # Update task manager status
            if task_id:
                self.task_manager.update_task_status(
                    task_id,
                    status="in_progress",
                    progress=progress,
                    message=message,
                    current_step=step
                )
        
        try:
            update_progress("validation", 0.05, "Validating input parameters")
            
            # Validate input
            if len(description.strip()) < 10:
                raise ValidationException("Description must be at least 10 characters")
            
            # Step 1: Enhance description with LLM
            update_progress("llm_enhancement", 0.15, "Enhancing description with AI")
            
            if self.llm_generator:
                try:
                    llm_result = await self.llm_generator.generate(
                        prompt=description,
                        asset_type=asset_type,
                        style_preferences=[style_preference] if style_preference else None,
                        quality_level=quality_level
                    )
                    
                    if llm_result.status == GenerationStatus.COMPLETED and llm_result.data:
                        enhanced_asset_data = llm_result.data.get("enhanced_asset", {})
                        enhanced_description = {
                            "enhanced_description": enhanced_asset_data.get("enhanced_description", f"Enhanced version of: {description}"),
                            "asset_name": enhanced_asset_data.get("asset_name", f"{asset_type.value} asset"),
                            "materials": enhanced_asset_data.get("materials", ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"]),
                            "style_notes": enhanced_asset_data.get("style_notes", [style_preference.value] if style_preference else ["generic"])
                        }
                    else:
                        # Fallback if LLM fails
                        enhanced_description = {
                            "enhanced_description": f"Enhanced version of: {description}",
                            "asset_name": f"{asset_type.value} asset",
                            "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
                            "style_notes": [style_preference.value] if style_preference else ["generic"]
                        }
                        error_msg = llm_result.error.message if llm_result.error else "Unknown error"
                        logger.warning("LLM generation failed, using fallback description", error=error_msg)
                        
                except Exception as e:
                    logger.error(f"LLM enhancement failed: {e}")
                    # Use fallback description
                    enhanced_description = {
                        "enhanced_description": f"Enhanced version of: {description}",
                        "asset_name": f"{asset_type.value} asset",
                        "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
                        "style_notes": [style_preference.value] if style_preference else ["generic"]
                    }
            else:
                # No LLM generator available, use basic enhancement
                enhanced_description = {
                    "enhanced_description": f"Enhanced version of: {description}",
                    "asset_name": f"{asset_type.value} asset",
                    "materials": ["metal", "wood"] if asset_type == AssetType.WEAPON else ["fabric"],
                    "style_notes": [style_preference.value] if style_preference else ["generic"]
                }
            
            update_progress("llm_enhancement", 0.25, "Description enhanced successfully")
            
            # Step 2: Generate 3D asset
            update_progress("asset_generation", 0.35, "Generating 3D model")
            
            model_file_path = None
            if self.asset_generator:
                try:
                    # Create progress callback for 3D generation
                    def asset_progress_callback(progress_update):
                        # Convert 3D generation progress to overall progress (35% to 65% range)
                        overall_progress = 0.35 + (progress_update.progress_percentage * 0.30 / 100.0)
                        update_progress("asset_generation", overall_progress, progress_update.message or progress_update.current_step)
                    
                    # Create generation request
                    generation_request = GenerationRequest(
                        description=enhanced_description["enhanced_description"],
                        asset_type=asset_type,
                        style_preference=style_preference,
                        quality_level=quality_level,
                        session_id=session_id,
                        output_format=FileFormat.OBJ,  # Default to OBJ format
                        max_polygon_count=None,  # Use service default
                        priority=1  # Standard priority
                    )
                    
                    # Generate the 3D asset
                    asset_result = await self.asset_generator.generate_asset(
                        request=generation_request,
                        progress_callback=asset_progress_callback
                    )
                    
                    if asset_result.status == GenerationStatus.COMPLETED:
                        generation_result = {
                            "status": GenerationStatus.COMPLETED,
                            "file_format": asset_result.file_format or FileFormat.OBJ,
                            "file_size": asset_result.file_size_bytes or 0,
                            "polygon_count": asset_result.polygon_count or 0,
                            "generation_time": asset_result.generation_time_seconds or 0.0
                        }
                        
                        # Get the generated file path if available
                        model_file_path = asset_result.model_file_path
                        
                    else:
                        # Fallback if 3D generation fails
                        generation_result = {
                            "status": GenerationStatus.FAILED,
                            "file_format": FileFormat.OBJ,
                            "file_size": 0,
                            "polygon_count": 0,
                            "generation_time": 0.0,
                            "error": asset_result.error_message or "Unknown error"
                        }
                        logger.warning("3D asset generation failed", error=asset_result.error_message)
                        
                except Exception as e:
                    logger.error(f"3D asset generation failed: {e}")
                    generation_result = {
                        "status": GenerationStatus.FAILED,
                        "file_format": FileFormat.OBJ,
                        "file_size": 0,
                        "polygon_count": 0,
                        "generation_time": 0.0,
                        "error": str(e)
                    }
            else:
                # No 3D generator available, create placeholder result
                generation_result = {
                    "status": GenerationStatus.FAILED,
                    "file_format": FileFormat.OBJ,
                    "file_size": 0,
                    "polygon_count": 0,
                    "generation_time": 0.0,
                    "error": "No 3D generator configured"
                }
                logger.warning("No 3D asset generator available")
            
            update_progress("asset_generation", 0.65, "3D model generation completed")
            
            # Step 3: Upload to cloud storage
            update_progress("cloud_upload", 0.75, "Uploading to cloud storage")
            
            # Generate unique filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{asset_type.value}_{timestamp}_{str(uuid.uuid4())[:8]}"
            
            model_url = None
            metadata_url = None
            
            if self.cloud_storage and generation_result["status"] == GenerationStatus.COMPLETED and model_file_path:
                try:
                    # Upload the generated model file
                    model_key = f"assets/{filename}.{generation_result['file_format'].value.lower()}"
                    
                    def upload_progress_callback(bytes_transferred: int, total_bytes: int):
                        progress_pct = bytes_transferred / total_bytes if total_bytes > 0 else 0
                        # Map upload progress to 75% to 85% overall progress
                        overall_progress = 0.75 + (progress_pct * 0.10)
                        update_progress("cloud_upload", overall_progress, f"Uploading model file ({int(progress_pct * 100)}%)")
                    
                    model_info = await self.cloud_storage.upload_file(
                        file_path=model_file_path,
                        key=model_key,
                        content_type=self._get_content_type_for_format(generation_result['file_format']),
                        metadata={
                            "asset_type": asset_type.value,
                            "session_id": session_id,
                            "generation_time": str(generation_result['generation_time']),
                            "polygon_count": str(generation_result['polygon_count'])
                        },
                        progress_callback=upload_progress_callback
                    )
                    model_url = model_info.public_url
                    
                    # Upload metadata
                    metadata_content = {
                        "original_description": description,
                        "enhanced_description": enhanced_description,
                        "generation_request": {
                            "description": description,
                            "asset_type": asset_type.value,
                            "style_preference": style_preference.value if style_preference else None,
                            "quality_level": quality_level.value
                        },
                        "generation_result": generation_result,
                        "generated_at": datetime.utcnow().isoformat(),
                        "session_id": session_id,
                        "model_url": model_url
                    }
                    
                    # Create temporary metadata file
                    metadata_key = f"metadata/{filename}.json"
                    temp_metadata_path = self.temp_dir / f"metadata_{filename}.json"
                    
                    with open(temp_metadata_path, 'w') as f:
                        json.dump(metadata_content, f, indent=2, default=str)
                    
                    try:
                        metadata_info = await self.cloud_storage.upload_file(
                            file_path=temp_metadata_path,
                            key=metadata_key,
                            content_type="application/json",
                            metadata={
                                "asset_type": asset_type.value,
                                "session_id": session_id
                            }
                        )
                        metadata_url = metadata_info.public_url
                    finally:
                        # Clean up temporary file
                        if temp_metadata_path.exists():
                            temp_metadata_path.unlink()
                    
                    update_progress("cloud_upload", 0.85, "Files uploaded successfully")
                    
                except Exception as e:
                    logger.error(f"Cloud storage upload failed: {e}")
                    # Create fallback URLs for local development
                    model_url = f"file://{model_file_path}" if model_file_path else None
                    metadata_url = None
                    update_progress("cloud_upload", 0.85, "Upload failed, using local files")
            
            else:
                # No cloud storage or no file to upload
                if not self.cloud_storage:
                    logger.warning("No cloud storage configured")
                elif generation_result["status"] != GenerationStatus.COMPLETED:
                    logger.warning("Skipping upload due to failed generation")
                elif not model_file_path:
                    logger.warning("No model file to upload")
                
                # Create fallback URLs
                model_url = f"file://{model_file_path}" if model_file_path else None
                metadata_url = None
                update_progress("cloud_upload", 0.85, "Skipped cloud upload")
            
            # Step 4: Create asset metadata
            service_used = "local" if not self.cloud_storage else "integrated"
            if generation_result["status"] == GenerationStatus.COMPLETED:
                service_used = f"llm+3d+storage"
            elif self.llm_generator and not self.asset_generator:
                service_used = "llm_only"
            elif self.asset_generator and not self.llm_generator:
                service_used = "3d_only"
            
            asset_metadata = AssetMetadata(
                asset_id=str(uuid.uuid4()),
                name=enhanced_description.get("asset_name", f"Generated {asset_type.value}"),
                original_description=description,
                asset_type=asset_type,
                style_preferences=[style_preference] if style_preference else [],
                quality_level=quality_level,
                generation_service=service_used,
                session_id=session_id,
                metadata={
                    "model_url": model_url, 
                    "metadata_url": metadata_url,
                    "enhanced_description": enhanced_description,
                    "generation_result": generation_result,
                    "model_file_path": model_file_path
                }
            )
            
            # Add to session history
            session = self.session_manager.get_session(session_id)
            if session:
                session["generation_history"].append(asset_metadata.dict())
            
            update_progress("completed", 1.0, "Asset generation completed successfully")
            
            # Mark task as completed
            if task_id:
                self.task_manager.update_task_status(
                    task_id,
                    status="completed",
                    progress=1.0,
                    message="Asset generation completed successfully",
                    result=asset_metadata
                )
            
            logger.info(f"Asset generation completed successfully: {asset_metadata.asset_id}")
            return asset_metadata
            
        except Exception as e:
            error_message = f"Generation failed: {str(e)}"
            logger.error(error_message, error=str(e))
            update_progress("error", 0.0, error_message)
            
            # Mark task as failed
            if task_id:
                self.task_manager.update_task_status(
                    task_id,
                    status="failed",
                    progress=0.0,
                    message=error_message,
                    error=error_message
                )
            raise
    
    def get_generation_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a generation task."""
        return self.task_manager.get_task_status(task_id)
    
    def cancel_generation(self, task_id: str) -> bool:
        """Cancel a running generation task."""
        success = self.task_manager.cancel_task(task_id)
        if success and task_id in self.active_generations:
            del self.active_generations[task_id]
        return success
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the generation history for a session."""
        session = self.session_manager.get_session(session_id)
        if session:
            return session.get("generation_history", [])
        return []
    
    async def shutdown(self) -> None:
        """Shutdown the application and clean up resources."""
        logger.info("Shutting down Asset Generation App")
        
        # Shutdown task manager
        await self.task_manager.shutdown()
        
        # Clear active generations
        self.active_generations.clear()
        
        # Clean up generators
        self.llm_generator = None
        self.asset_generator = None
        self.cloud_storage = None
        
        logger.info("Application shutdown completed")

    def _get_content_type_for_format(self, file_format: FileFormat) -> str:
        """Get MIME content type for a file format."""
        content_type_map = {
            FileFormat.OBJ: "model/obj",
            FileFormat.FBX: "application/octet-stream",
            FileFormat.GLB: "model/gltf-binary",
            FileFormat.GLTF: "model/gltf+json",
            FileFormat.PLY: "application/octet-stream",
            FileFormat.STL: "model/stl",
            FileFormat.DAE: "model/vnd.collada+xml"
        }
        return content_type_map.get(file_format, "application/octet-stream")

"""
Task management system for background operations.

This module provides comprehensive task lifecycle management including
creation, monitoring, cancellation, and cleanup of background tasks.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

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

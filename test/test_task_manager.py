import asyncio

import pytest

from src.core.task_manager import TaskManager


@pytest.mark.asyncio
async def test_create_task(task_manager: TaskManager) -> None:
    async def test_task() -> int:
        await asyncio.sleep(0.1)
        return 42

    task_id = task_manager.create_task(test_task())
    assert task_id in task_manager.tasks, "Task should be created"
    assert task_manager.get_task_status(task_id)["status"] == "started", "Task status should be started"


@pytest.mark.asyncio
async def test_task_cleanup(task_manager: TaskManager) -> None:
    async def test_task() -> int:
        await asyncio.sleep(0.1)
        return 42

    task_id = task_manager.create_task(test_task())
    await asyncio.sleep(0.2)  # Wait for task completion
    await task_manager.cleanup_completed_tasks()
    status = task_manager.get_task_status(task_id)
    assert "cleaned" in status, "Completed task should be marked as cleaned"
    assert task_id not in task_manager.tasks, "Completed task should be removed"


@pytest.mark.asyncio
async def test_cancel_task(task_manager: TaskManager) -> None:
    async def test_task() -> int:
        await asyncio.sleep(1)
        return 42

    task_id = task_manager.create_task(test_task())
    success = task_manager.cancel_task(task_id)
    assert success, "Task cancellation should succeed"
    assert task_manager.get_task_status(task_id)["status"] == "cancelled", "Task status should be cancelled"

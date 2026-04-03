"""Async task queue with backpressure for the Lopen orchestrator."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    intent: str
    payload: str
    priority: int = 5          # 1 (high) … 10 (low)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Make Tasks comparable by priority for use in a priority queue
    def __lt__(self, other: "Task") -> bool:
        return self.priority < other.priority


class TaskQueue:
    """Asyncio-based priority task queue with configurable backpressure."""

    def __init__(self, max_size: int = 100) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, Task]] = asyncio.PriorityQueue(maxsize=max_size)
        self._task_index: dict[str, Task] = {}
        self._max_size = max_size
        logger.info("TaskQueue initialised (max_size=%d)", max_size)

    # ------------------------------------------------------------------
    # Enqueue / Dequeue
    # ------------------------------------------------------------------

    async def enqueue(self, task: Task, timeout: float = 5.0) -> bool:
        """Add a task. Returns False if the queue is full (backpressure)."""
        if task.id in self._task_index:
            logger.warning("Duplicate task id=%s ignored", task.id)
            return False
        try:
            await asyncio.wait_for(
                self._queue.put((task.priority, task)),
                timeout=timeout,
            )
            self._task_index[task.id] = task
            logger.info("Enqueued task id=%s intent=%s priority=%d", task.id, task.intent, task.priority)
            return True
        except asyncio.TimeoutError:
            logger.error("TaskQueue full — task id=%s dropped (backpressure)", task.id)
            return False

    async def dequeue(self) -> Task:
        """Block until a task is available; returns the highest-priority task."""
        _, task = await self._queue.get()
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        logger.debug("Dequeued task id=%s", task.id)
        return task

    def task_done(self) -> None:
        """Signal the underlying queue that the last dequeued item is processed."""
        self._queue.task_done()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def complete_task(self, task: Task, result: str) -> None:
        task.status = TaskStatus.DONE
        task.result = result
        task.finished_at = datetime.now(timezone.utc)
        self.task_done()

    def fail_task(self, task: Task, error: str) -> None:
        task.status = TaskStatus.FAILED
        task.error = error
        task.finished_at = datetime.now(timezone.utc)
        self.task_done()

    def get_status(self, task_id: str) -> Optional[Task]:
        return self._task_index.get(task_id)

    def list_tasks(self) -> list[Task]:
        return list(self._task_index.values())

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

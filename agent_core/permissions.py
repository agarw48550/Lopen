"""Permission model for sensitive Lopen operations."""

from __future__ import annotations

import functools
import logging
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    LOW = 0        # read-only, no side effects
    MEDIUM = 1     # file reads, web browsing
    HIGH = 2       # file writes, system queries
    CRITICAL = 3   # shell execution, WhatsApp, desktop control


# Global allow threshold — operations at or below this level are permitted.
_ALLOWED_LEVEL: PermissionLevel = PermissionLevel.HIGH


def set_permission_threshold(level: PermissionLevel) -> None:
    """Set the global permission threshold at runtime."""
    global _ALLOWED_LEVEL
    _ALLOWED_LEVEL = level
    logger.info("Permission threshold set to %s", level.name)


def check_permission(operation: str, level: PermissionLevel) -> bool:
    """Return True if the operation is permitted at the given level."""
    allowed = level <= _ALLOWED_LEVEL
    if not allowed:
        logger.warning(
            "Permission denied: operation=%r requires level=%s but threshold=%s",
            operation,
            level.name,
            _ALLOWED_LEVEL.name,
        )
    return allowed


def permission_required(level: PermissionLevel, operation: str = "") -> Callable[..., Any]:
    """Decorator that gates a function call behind a permission check."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        op_name = operation or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not check_permission(op_name, level):
                raise PermissionError(
                    f"Operation '{op_name}' requires permission level {level.name}."
                )
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not check_permission(op_name, level):
                raise PermissionError(
                    f"Operation '{op_name}' requires permission level {level.name}."
                )
            return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator

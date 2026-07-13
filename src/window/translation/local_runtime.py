"""Explicit lifecycle states for lazy local translation models."""

from enum import Enum


class LocalModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


__all__ = ["LocalModelState"]

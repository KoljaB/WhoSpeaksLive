"""Immutable configuration snapshots for the window diarization runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class DiarizationConfig:
    """A detached, immutable view of parsed CLI configuration."""

    values: Mapping[str, Any]

    @classmethod
    def from_namespace(cls, value: argparse.Namespace | Mapping[str, Any] | "DiarizationConfig") -> "DiarizationConfig":
        if isinstance(value, cls):
            return value
        raw = vars(value) if isinstance(value, argparse.Namespace) else dict(value)
        return cls(MappingProxyType(dict(raw)))

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def with_updates(self, **updates: Any) -> "DiarizationConfig":
        values = dict(self.values)
        values.update(updates)
        return replace(self, values=MappingProxyType(values))

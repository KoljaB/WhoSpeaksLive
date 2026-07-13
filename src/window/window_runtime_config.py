"""Immutable runtime configuration for the live window application."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class WindowConfig(Mapping[str, Any]):
    """Detached read-only values produced immediately after CLI parsing."""

    _values: Mapping[str, Any]

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "WindowConfig":
        return cls.from_mapping(vars(namespace))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WindowConfig":
        return cls(_freeze(values))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def with_updates(self, **updates: Any) -> "WindowConfig":
        values = dict(self._values)
        values.update(updates)
        return type(self)(_freeze(values))

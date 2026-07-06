"""Python client for the public window diarization event stream."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import threading
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


EventHandler = Callable[[dict[str, Any]], None]


def _event_url(base_url: str, endpoint: str, query: dict[str, str] | None = None) -> str:
    base = base_url if base_url.endswith("/") else f"{base_url}/"
    path = endpoint.lstrip("/")
    url = urljoin(base, path)
    if query:
        return f"{url}?{urlencode(query)}"
    return url


def iter_sse(url: str, *, timeout: float | None = None) -> Iterator[tuple[str, str]]:
    """Yield ``(event_name, data)`` pairs from a Server-Sent Events URL."""
    request = Request(url, headers={"Accept": "text/event-stream"})
    with urlopen(request, timeout=timeout) as response:
        event_name = "message"
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())


class DiarizationEventClient:
    """Subscribe to normalized diarization events from ``/api/events``.

    Example:

    .. code-block:: python

        client = DiarizationEventClient("http://localhost:8796")

        @client.on("transcript.final_unknown")
        def handle_unknown(event):
            print(event["payload"]["text"])

        client.run_forever()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8796",
        *,
        endpoint: str = "/api/events",
        include_snapshot: bool = True,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url
        self.endpoint = endpoint
        self.include_snapshot = include_snapshot
        self.timeout = timeout
        self._handlers: dict[str, list[EventHandler]] = {}

    def on(self, event_type: str, handler: EventHandler | None = None) -> EventHandler | Callable[[EventHandler], EventHandler]:
        """Register a callback for one event type.

        Use ``"*"`` as the event type to receive every event.
        """
        event_type = str(event_type or "*")

        def register(callback: EventHandler) -> EventHandler:
            self._handlers.setdefault(event_type, []).append(callback)
            return callback

        if handler is not None:
            return register(handler)
        return register

    def events(self) -> Iterator[dict[str, Any]]:
        query = {"snapshot": "1" if self.include_snapshot else "0"}
        url = _event_url(self.base_url, self.endpoint, query)
        for _event_name, data in iter_sse(url, timeout=self.timeout):
            event = json.loads(data)
            if isinstance(event, dict):
                yield event

    def dispatch(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        handlers = list(self._handlers.get(event_type, [])) + list(self._handlers.get("*", []))
        for handler in handlers:
            handler(event)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        for event in self.events():
            if stop_event is not None and stop_event.is_set():
                return
            self.dispatch(event)

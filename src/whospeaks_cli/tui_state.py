"""Explicit state owners for the Textual setup application.

The UI renders immutable snapshots from these classes.  Process handles stay
inside :class:`ServerSupervisor`; they are never inferred from listening ports.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from collections.abc import Callable
from typing import Any


SERVER_KINDS = ("live", "reports", "translation", "macos_asr", "macos_embeddings")


class PendingAction(str, enum.Enum):
    NONE = "none"
    LAUNCH_LIVE_AFTER_TRANSLATION = "launch_live_after_translation"
    START_MACOS_EMBEDDINGS = "start_macos_embeddings"
    LAUNCH_AFTER_MACOS_SERVICES = "launch_after_macos_services"


@dataclasses.dataclass(frozen=True)
class OperationState:
    name: str = ""
    status: str = "idle"
    title: str = "Setup is idle"
    step: str = ""
    latest: str = "Choose a target, review the plan, then select Install."
    started_at: float | None = None
    spinner_index: int = 0
    cancel_requested: bool = False


@dataclasses.dataclass(frozen=True)
class PendingInstall:
    command: tuple[str, ...]
    title: str


@dataclasses.dataclass(frozen=True)
class SetupState:
    operation: OperationState = dataclasses.field(default_factory=OperationState)
    pending_install: PendingInstall | None = None
    pending_action: PendingAction = PendingAction.NONE


class SetupCoordinator:
    """The sole writer of operation, confirmation, and deferred-action state."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state = SetupState()

    @property
    def snapshot(self) -> SetupState:
        return self._state

    def start_operation(self, name: str, title: str, step: str) -> SetupState:
        self._state = dataclasses.replace(
            self._state,
            operation=OperationState(
                name=name,
                status="running",
                title=title,
                step=step,
                latest="The operation has started.",
                started_at=self._clock(),
            ),
        )
        return self._state

    def finish_operation(self, status: str, title: str, detail: str) -> SetupState:
        self._state = dataclasses.replace(
            self._state,
            operation=OperationState(status=status, title=title, latest=detail),
        )
        return self._state

    def set_feedback(self, status: str, title: str, detail: str) -> bool:
        if self._state.operation.name:
            return False
        self._state = dataclasses.replace(
            self._state,
            operation=OperationState(status=status, title=title, latest=detail),
        )
        return True

    def tick(self) -> SetupState:
        operation = self._state.operation
        if operation.name:
            self._state = dataclasses.replace(
                self._state,
                operation=dataclasses.replace(operation, spinner_index=operation.spinner_index + 1),
            )
        return self._state

    def update_progress(self, *, step: str | None = None, latest: str | None = None) -> SetupState:
        operation = self._state.operation
        self._state = dataclasses.replace(
            self._state,
            operation=dataclasses.replace(
                operation,
                step=operation.step if step is None else step,
                latest=operation.latest if latest is None else latest,
            ),
        )
        return self._state

    def request_cancel(self) -> SetupState:
        operation = self._state.operation
        self._state = dataclasses.replace(
            self._state,
            operation=dataclasses.replace(operation, cancel_requested=True),
        )
        return self._state

    def set_pending_install(self, command: list[str], title: str) -> SetupState:
        self._state = dataclasses.replace(
            self._state,
            pending_install=PendingInstall(tuple(command), title),
        )
        return self._state

    def take_pending_install(self) -> PendingInstall | None:
        pending = self._state.pending_install
        self._state = dataclasses.replace(self._state, pending_install=None)
        return pending

    def clear_pending_install(self) -> SetupState:
        self._state = dataclasses.replace(self._state, pending_install=None)
        return self._state

    def set_pending_action(self, action: PendingAction) -> SetupState:
        self._state = dataclasses.replace(self._state, pending_action=action)
        return self._state

    def clear_pending_action(self) -> SetupState:
        self._state = dataclasses.replace(self._state, pending_action=PendingAction.NONE)
        return self._state

    def take_pending_action(self, expected: PendingAction) -> bool:
        if self._state.pending_action is not expected:
            return False
        self._state = dataclasses.replace(self._state, pending_action=PendingAction.NONE)
        return True


@dataclasses.dataclass(frozen=True)
class ServerState:
    status: str = "stopped"
    ownership: str = "none"

    @property
    def display_status(self) -> str:
        if self.status == "running" and self.ownership == "external":
            return "external"
        return self.status


@dataclasses.dataclass(frozen=True)
class ServerTransition:
    kind: str
    previous: ServerState
    current: ServerState
    became_app_ready: bool = False
    app_failed: bool = False
    exit_code: int | None = None


class ServerSupervisor:
    """Own server processes and distinguish them from unrelated listeners."""

    def __init__(self) -> None:
        self._processes: dict[str, object | None] = {kind: None for kind in SERVER_KINDS}
        self._states: dict[str, ServerState] = {kind: ServerState() for kind in SERVER_KINDS}

    def process(self, kind: str) -> object | None:
        return self._processes[kind]

    def state(self, kind: str) -> ServerState:
        return self._states[kind]

    def begin(self, kind: str, process: object) -> ServerState:
        self._processes[kind] = process
        state = ServerState("starting", "app")
        self._states[kind] = state
        return state

    def fail_start(self, kind: str) -> ServerState:
        self._processes[kind] = None
        state = ServerState("failed", "app")
        self._states[kind] = state
        return state

    def clear(self, kind: str) -> ServerState:
        """Forget one owned process after its tree has been synchronously terminated."""

        self._processes[kind] = None
        state = ServerState()
        self._states[kind] = state
        return state

    @staticmethod
    def return_code(process: object | None) -> int | None:
        if process is None:
            return None
        poll = getattr(process, "poll", None)
        return poll() if callable(poll) else None

    def process_is_running(self, kind: str) -> bool:
        process = self._processes[kind]
        return process is not None and self.return_code(process) is None

    def observe_backend(self, kind: str, *, available: bool) -> ServerTransition:
        """Record current health for a required ASR or embeddings backend."""

        previous = self._states[kind]
        process = self._processes[kind]
        return_code = self.return_code(process)
        exit_code: int | None = None
        if process is not None and return_code is None:
            current = ServerState("running" if available else "unavailable", "app")
        elif process is not None:
            exit_code = return_code
            self._processes[kind] = None
            current = ServerState("failed", "app")
        else:
            current = ServerState("running" if available else "unavailable", "external")
        self._states[kind] = current
        return ServerTransition(
            kind=kind,
            previous=previous,
            current=current,
            became_app_ready=(
                previous.ownership == "app"
                and previous.status != "running"
                and current.ownership == "app"
                and current.status == "running"
            ),
            app_failed=(
                current.ownership == "app"
                and current.status == "failed"
                and previous.status != "failed"
            ),
            exit_code=exit_code,
        )

    def mirror_component(self, kind: str, source: ServerState) -> ServerTransition:
        """Mirror a component that lives inside another supervised process."""

        previous = self._states[kind]
        self._processes[kind] = None
        current = ServerState(source.status, source.ownership)
        self._states[kind] = current
        return ServerTransition(
            kind=kind,
            previous=previous,
            current=current,
            became_app_ready=(
                previous.ownership == "app"
                and previous.status != "running"
                and current.ownership == "app"
                and current.status == "running"
            ),
            app_failed=(
                current.ownership == "app"
                and current.status == "failed"
                and previous.status != "failed"
            ),
        )

    def observe(self, kind: str, *, listening: bool, probe_due: bool) -> ServerTransition:
        previous = self._states[kind]
        process = self._processes[kind]
        return_code = self.return_code(process)
        exit_code: int | None = None
        if process is not None and return_code is None:
            current = ServerState("running" if probe_due and listening else previous.status, "app")
            if current.status not in {"running", "starting"}:
                current = ServerState("starting", "app")
        elif process is not None:
            exit_code = return_code
            self._processes[kind] = None
            current = ServerState("stopped" if return_code == 0 else "failed", "app")
        elif not probe_due:
            current = previous
        elif listening:
            current = ServerState("running", "external")
        elif previous.status == "failed" and previous.ownership == "app":
            current = previous
        else:
            current = ServerState()
        self._states[kind] = current
        return ServerTransition(
            kind=kind,
            previous=previous,
            current=current,
            became_app_ready=(
                previous.ownership == "app"
                and previous.status != "running"
                and current.ownership == "app"
                and current.status == "running"
            ),
            app_failed=(
                current.ownership == "app"
                and current.status == "failed"
                and previous.status != "failed"
            ),
            exit_code=exit_code,
        )

"""Diagnostics and installer workers for the WhoSpeaks setup TUI."""

from __future__ import annotations

import os
import signal
import subprocess

from textual import work
from textual.widgets import Button, TabbedContent

from . import main as backend


class SetupWorkersMixin:
    def _show_activity(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "activity-tab"
        self.call_after_refresh(self.query_one("#clear-log", Button).focus)

    def run_doctor_worker(self, deep: bool) -> None:
        if self.active_operation:
            self.notify("Another operation is already running", severity="warning")
            return
        title = "Running complete diagnostics" if deep else "Checking system readiness"
        self._start_operation("doctor", title, "Inspecting installed components")
        self._append_log("Starting complete diagnostics..." if deep else "Starting readiness check...")
        self._run_doctor_worker(deep)

    @work(thread=True, exclusive=True, group="doctor")
    def _run_doctor_worker(self, deep: bool) -> None:
        try:
            report = self.doctor_runner(self.profile, self.profile.mode, deep=deep)
        except Exception as exc:
            self.call_from_thread(self._append_log, f"Diagnostics failed: {type(exc).__name__}: {exc}")
            self.call_from_thread(self.notify, str(exc), title="Diagnostics failed", severity="error")
            self.call_from_thread(
                self._finish_operation,
                "error",
                "Diagnostics failed",
                f"{type(exc).__name__}: {exc}",
            )
        else:
            self.call_from_thread(self._render_report, report)
            readiness = backend.report_readiness_line(report)
            self.call_from_thread(self._append_log, readiness)
            statuses = {check.status for check in report.checks}
            if "fail" in statuses:
                status, result_title = "error", "Readiness check found required fixes"
            elif "warn" in statuses:
                status, result_title = "warning", "Readiness check found warnings"
            else:
                status, result_title = "success", "Readiness check completed"
            self.call_from_thread(self._finish_operation, status, result_title, readiness)

    def start_install_worker(self, command: list[str], *, title: str = "Selected setup") -> None:
        if self.active_operation:
            self.notify("Another operation is already running", severity="warning")
            return
        self._start_operation("install", f"Install: {title}", "Starting installer")
        self._run_install_worker(command)

    @work(thread=True, exclusive=True, group="install")
    def _run_install_worker(self, command: list[str]) -> None:
        self.call_from_thread(self._append_log, "")
        self.call_from_thread(self._append_log, f"> {backend.format_command(command)}")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            process = self.popen_factory(command, **kwargs)
            self.install_process = process
            if self.install_cancelled and process.poll() is None:
                self._terminate_process_tree(process)
            if process.stdout is not None:
                for line in process.stdout:
                    self.call_from_thread(self._append_log, line.rstrip())
            return_code = int(process.wait())
        except Exception as exc:
            return_code = 1
            self.call_from_thread(self._append_log, f"Installer failed: {type(exc).__name__}: {exc}")
        finally:
            self.install_process = None

        if self.install_cancelled:
            self.call_from_thread(self._append_log, "Installation cancelled.")
            self.call_from_thread(self.notify, "Installation cancelled", severity="warning")
            self.call_from_thread(
                self._finish_operation,
                "warning",
                "Installation cancelled",
                "The running installer was stopped.",
            )
        elif return_code == 0:
            self.call_from_thread(self._append_log, "Installation completed.")
            self.call_from_thread(self.notify, "Installation completed", title="WhoSpeaks")
            self.profile = backend.load_profile()
            self.call_from_thread(
                self._finish_operation,
                "success",
                "Installation completed",
                "Packages were installed. Verifying system readiness next.",
            )
        else:
            self.call_from_thread(self._append_log, f"Installation stopped with exit code {return_code}.")
            self.call_from_thread(self.notify, f"Installer exit code {return_code}", title="Installation failed", severity="error")
            self.call_from_thread(
                self._finish_operation,
                "error",
                "Installation failed",
                f"Installer stopped with exit code {return_code}. Open Activity for details.",
            )
        if not self.install_cancelled and return_code == 0:
            self.call_from_thread(self.run_doctor_worker, False)

    def _request_cancel(self) -> None:
        if self.active_operation != "install":
            return
        self._coordinator.request_cancel()
        self._coordinator.update_progress(
            step="Cancelling installation",
            latest="Stopping the installer and its child processes...",
        )
        self._render_operation()
        self._append_log("Cancelling installation...")
        self.cancel_install_worker()

    @work(thread=True, exclusive=True, group="cancel")
    def cancel_install_worker(self) -> None:
        process = self.install_process
        if process is None or process.poll() is not None:
            return
        self._terminate_process_tree(process)

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as exc:
            self.call_from_thread(self._append_log, f"Cancellation failed: {exc}")
            process.terminate()

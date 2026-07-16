"""Profile persistence and service-start actions for the setup TUI."""

from __future__ import annotations

import sys
from typing import Any

from textual.widgets import Checkbox, Input, Select

from . import main as backend


class ProfileActionsMixin:
    def _selected_installer_backend(self) -> str:
        return backend.normalize_installer_backend(
            str(self.query_one("#installer-select", Select).value or "pip")
        )

    def _install_command(self, plan: backend.InstallPlan) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "whospeaks_cli",
            "install",
            "--target",
            plan.target,
            "--installer",
            self._selected_installer_backend(),
            "--yes",
        ]
        if plan.target != "server":
            command.extend(["--realtime-preview-engine", plan.realtime_preview_engine])
            if plan.realtime_preview_model_preset:
                command.extend(["--realtime-preview-model-preset", plan.realtime_preview_model_preset])
            if plan.realtime_preview_engine == "sherpa_onnx":
                model_dir = self.query_one("#realtime-model-dir-input", Input).value.strip()
                if model_dir:
                    command.extend(["--realtime-preview-model-dir", model_dir])
        command.extend(["--translation-model-profile", plan.translation_model_profile])
        return command

    def _persist_profile_updates(
        self,
        updates: list[tuple[str, Any]],
        *,
        log_name: str,
        invalid_title: str,
        failed_title: str,
        success_title: str,
        notify: bool,
    ) -> bool:
        """Validate, persist, and atomically replace the current profile snapshot."""

        try:
            updated = backend.apply_profile_updates(self.profile, updates)
        except SystemExit as exc:
            self._set_feedback("error", failed_title, str(exc))
            self.notify(str(exc), title=invalid_title, severity="error")
            return False
        try:
            path = backend.save_profile(updated)
        except OSError as exc:
            self._append_log(f"Could not save {log_name}: {exc}")
            self._set_feedback("error", failed_title, str(exc))
            self.notify(str(exc), title=f"Could not save {log_name}", severity="error")
            return False
        self.profile = updated
        self._append_log(f"Saved {log_name}: {path}")
        if notify:
            self._set_feedback("success", success_title, str(path))
            self.notify(success_title, title="WhoSpeaks")
        return True

    def _save_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("language", self.query_one("#language-select", Select).value),
            ("provider_preset", self.query_one("#provider-select", Select).value),
            ("live_speaker_assignment", self.query_one("#live-speakers-checkbox", Checkbox).value),
            ("model", self.query_one("#model-input", Input).value),
            ("device", self.query_one("#device-select", Select).value),
            ("compute_type", self.query_one("#compute-input", Input).value),
            ("host", self.query_one("#host-input", Input).value),
            ("port", self.query_one("#port-input", Input).value),
            ("remote_asr_url", self.query_one("#asr-url-input", Input).value),
            ("remote_embeddings_url", self.query_one("#embeddings-url-input", Input).value),
            ("realtime_preview_engine", self.query_one("#realtime-engine-select", Select).value),
            ("realtime_preview_model_preset", self.query_one("#realtime-preset-select", Select).value),
            ("realtime_preview_model_dir", self.query_one("#realtime-model-dir-input", Input).value),
        ]
        return self._persist_profile_updates(
            updates,
            log_name="settings",
            invalid_title="Invalid settings",
            failed_title="Settings were not saved",
            success_title="Settings saved",
            notify=notify,
        )

    def _save_reports_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("reports_enabled", self.query_one("#reports-enabled-checkbox", Checkbox).value),
            ("reports_port", self.query_one("#reports-port-input", Input).value),
            ("report_language", self.query_one("#report-language-select", Select).value),
            ("report_llm_provider", self.query_one("#report-llm-provider-select", Select).value),
            ("report_llm_base_url", self.query_one("#report-llm-base-url-input", Input).value),
            ("report_llm_model", self.query_one("#report-llm-model-input", Input).value),
            ("report_auto_generate", self.query_one("#report-auto-generate-checkbox", Checkbox).value),
        ]
        saved = self._persist_profile_updates(
            updates,
            log_name="report settings",
            invalid_title="Invalid report settings",
            failed_title="Report settings were not saved",
            success_title="Report settings saved",
            notify=notify,
        )
        if not saved:
            return False
        self._sync_action_buttons()
        return True

    def _save_translation_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("translation_enabled", self.query_one("#translation-enabled-checkbox", Checkbox).value),
            ("translation_browser_preferred", self.query_one("#translation-browser-preferred-checkbox", Checkbox).value),
            ("translation_provider", self.query_one("#translation-provider-select", Select).value),
            ("translation_target_languages", self.query_one("#translation-targets-input", Input).value),
            ("translation_max_targets", self.query_one("#translation-max-targets-input", Input).value),
            ("translation_model_profile", self.query_one("#translation-model-profile-select", Select).value),
            ("translation_model", self.query_one("#translation-model-input", Input).value),
            ("translation_base_url", self.query_one("#translation-base-url-input", Input).value),
            ("translation_api_key_env", self.query_one("#translation-api-key-env-input", Input).value),
            ("translation_region", self.query_one("#translation-region-input", Input).value),
            ("translation_python", self.query_one("#translation-python-input", Input).value),
            ("translation_port", self.query_one("#translation-port-input", Input).value),
            ("translation_device", self.query_one("#translation-device-select", Select).value),
        ]
        saved = self._persist_profile_updates(
            updates,
            log_name="translation settings",
            invalid_title="Invalid translation settings",
            failed_title="Translation settings were not saved",
            success_title="Translation settings saved",
            notify=notify,
        )
        if not saved:
            return False
        self.query_one("#translation-targets-input", Input).value = self.profile.translation_target_languages
        self.query_one("#translation-max-targets-input", Input).value = str(self.profile.translation_max_targets)
        self._sync_translation_settings()
        self._sync_action_buttons()
        return True

    def _start_reports_server(self, *, save_settings: bool = True) -> None:
        if self.active_operation:
            self.notify("Wait for the current operation or cancel it", severity="warning")
            return
        if save_settings and not self._save_reports_settings(notify=False):
            return
        command = backend.build_reports_command(
            self.profile,
            port=self.profile.reports_port,
            report_language=self.profile.report_language,
            llm_provider=self.profile.report_llm_provider,
            llm_base_url=self.profile.report_llm_base_url,
            llm_model=self.profile.report_llm_model,
            auto_generate=self.profile.report_auto_generate,
        )
        if self._start_server_process("reports", command):
            self._set_feedback(
                "success",
                "Reports server starting in another window",
                f"Open http://{self.profile.host}:{self.profile.reports_port}/",
            )

    def _start_translation_server(self, *, save_settings: bool = True) -> bool:
        if self.active_operation:
            self.notify("Wait for the current operation or cancel it", severity="warning")
            return False
        if save_settings and not self._save_translation_settings(notify=False):
            return False
        if not self.profile.translation_enabled:
            self.notify("Enable translation before starting its local server", severity="warning")
            return False
        if self.profile.translation_provider != "sidecar":
            self.notify("The selected provider runs through the live server and has no sidecar", severity="warning")
            return False
        command = backend.build_translation_command(self.profile)
        if not self._start_server_process("translation", command):
            return False
        self._set_feedback(
            "success",
            "Translation model warming in another window",
            "The API URL will become available after the model reports ready.",
        )
        return True

    def _start_live_server(self) -> bool:
        command = backend.build_launch_command(self.profile)
        if not self._start_server_process("live", command):
            return False
        self._set_feedback(
            "success",
            "Live server starting in another window",
            f"Open http://{self.profile.host}:{self.profile.port}/",
        )
        return True

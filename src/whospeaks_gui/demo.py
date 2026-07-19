"""Deterministic, side-effect-free controller data for screenshots and tests."""

from __future__ import annotations

from pathlib import Path

from whospeaks_cli.cli_diagnostics import CheckResult, DoctorReport
from whospeaks_cli.launcher_controller import EventKind, LauncherController
from whospeaks_cli.profiles import Profile
from whospeaks_cli.launcher_state import ServerSupervisor


DEMO_STATES = (
    "first_run",
    "ready",
    "starting",
    "stopping",
    "running",
    "partial_failure",
    "failed",
    "disconnected",
    "cancelled",
    "success",
    "diagnostics",
    "settings",
    "invalid_configuration",
    "activity",
    "about",
    "stop_confirmation",
)


class _DemoProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code
        self.pid = 4242

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return int(self.return_code or 0)


def canonical_profile() -> Profile:
    return Profile.from_mapping(
        {
            "mode": "remote",
            "language": "en",
            "host": "127.0.0.1",
            "port": 8796,
            "asr_backend": "remote",
            "embeddings_backend": "remote",
            "remote_asr_url": "http://127.0.0.1:8650",
            "remote_embeddings_url": "http://127.0.0.1:8660",
            "realtime_preview_engine": "sherpa_onnx",
            "realtime_preview_model_preset": "nemotron-3.5-160ms-int8",
            "live_speaker_assignment": True,
            "reports_enabled": True,
            "reports_port": 8798,
            "report_language": "en",
            "report_llm_provider": "llama_cpp",
            "report_llm_model": "local-instruct-model.gguf",
            "translation_enabled": True,
            "translation_provider": "sidecar",
            "translation_port": 8799,
            "translation_model_profile": "nllb-200-600m",
            "translation_device": "auto",
        }
    )


def canonical_report(*, first_run: bool = False) -> DoctorReport:
    if first_run:
        return DoctorReport(
            "remote",
            [
                CheckResult("Python", "ok", "CPython 3.12.4"),
                CheckResult("WhoSpeaks package", "ok", "WhoSpeaks 0.0.4"),
                CheckResult("ffmpeg", "ok", "ffmpeg 7.1"),
                CheckResult("Browser UI port", "ok", "127.0.0.1:8796 is available"),
                CheckResult("Launch profile", "ok", "remote controller"),
                CheckResult("Controller runtime", "fail", "Required controller modules are not installed", "Run install."),
                CheckResult("Nemotron model files", "fail", "Model files are missing", "Run install."),
                CheckResult("Translation model files", "fail", "NLLB-200 600M is not installed", "Run install."),
            ],
        )
    return DoctorReport(
        "remote",
        [
            CheckResult("Python", "ok", "CPython 3.12.4"),
            CheckResult("WhoSpeaks package", "ok", "WhoSpeaks 0.0.4"),
            CheckResult("ffmpeg", "ok", "ffmpeg 7.1"),
            CheckResult("Remote ASR health", "ok", "responding at 127.0.0.1:8650"),
            CheckResult("Remote embeddings health", "ok", "responding at 127.0.0.1:8660"),
            CheckResult(
                "Remote embedding cache",
                "warn",
                "Remote embedding cache not verified",
                "Run complete check.",
            ),
            CheckResult(
                "Translation model",
                "warn",
                "Translation model warms on first launch",
                "Start translation.",
            ),
            CheckResult("Browser UI port", "ok", "127.0.0.1:8796 is available"),
            CheckResult("Controller Python modules", "ok", "Required modules are importable"),
            CheckResult("Launch profile", "ok", "remote controller"),
        ],
    )


def canonical_logs() -> tuple[str, ...]:
    return (
        "14:31:58  WhoSpeaks 0.0.4",
        "14:31:58  Profile loaded: Remote ASR + embeddings (127.0.0.1:8796)",
        "14:31:58  Language: English · Nemotron 3.5 (160 ms) · Live speaker labels: On",
        "14:31:59  Starting readiness check…",
        "14:32:00  Remote ASR health responded at 127.0.0.1:8650",
        "14:32:00  Remote embeddings health responded at 127.0.0.1:8660",
        "14:32:00  Browser UI port 127.0.0.1:8796 is available",
        "14:32:00  8 passed, 2 warnings, 0 failed, 0 skipped",
        "14:34:11  > whospeaks install --target core --realtime-preview-engine sherpa_onnx --yes",
        "14:34:11  Resolving package dependencies…",
        "14:34:13  All required Python packages are already installed",
        "14:34:13  Checking Nemotron model availability…",
        "14:34:13  Downloading sherpa-onnx model archive nemotron-3.5-160ms-int8.zip",
        "14:35:42  Checksum verified: 7f3c0b9d2a8e4c6f1b7d9e2a3c4f5e6b7d8a9c0e1f2d3c4b5a6e7d8f9c0a1b2c",
        r"14:35:43  Extracting to C:\Users\Alex\AppData\Local\WhoSpeaks\models\sherpa-onnx\nemotron-3.5-160ms-int8",
        r"14:35:45  Model file C:\Users\Alex\AppData\Local\WhoSpeaks\models\sherpa-onnx\nemotron-3.5-160ms-int8\model.int8.onnx",
        r"14:35:45  Token file C:\Users\Alex\AppData\Local\WhoSpeaks\models\sherpa-onnx\nemotron-3.5-160ms-int8\tokens.txt",
        r"14:35:45  Config file C:\Users\Alex\AppData\Local\WhoSpeaks\models\sherpa-onnx\nemotron-3.5-160ms-int8\config.json",
        "14:35:45  Preparing online recognizer files…",
        "14:35:46  Validating feature configuration",
        "14:35:46  Sample rate: 16000 Hz",
        "14:35:46  Feature dimension: 80",
        "14:35:46  Initializing token decoder",
        "14:35:47  Loading speaker embedding metadata",
        "14:35:47  Speaker embedding cache is available",
        "14:35:47  Verifying browser controller package",
        "14:35:47  Browser controller entry point is available",
        "14:35:48  Verifying Meeting Intelligence dependencies",
        "14:35:48  Meeting Intelligence dependencies are available",
        "14:35:48  Verifying translation sidecar dependencies",
        "14:35:48  Translation sidecar dependencies are available",
        "14:35:49  Writing installation receipt",
        "14:35:49  Finalizing component setup",
        "14:35:49  Requested components are ready",
    )


class DemoLauncherController(LauncherController):
    """A controller that never launches processes, writes config, or contacts services."""

    def __init__(self, state: str = "ready") -> None:
        if state not in DEMO_STATES:
            raise ValueError(f"Unknown demo state: {state}")
        super().__init__(
            canonical_profile(),
            doctor_runner=lambda *_args, **_kwargs: canonical_report(),
            profile_saver=lambda _profile: Path("demo-config.json"),
            popen_factory=lambda *_args, **_kwargs: _DemoProcess(),
        )
        self.demo_state = state
        self.report = canonical_report(first_run=state == "first_run")
        self._logs.extend(canonical_logs())
        self._apply_services(state)

    def _apply_services(self, state: str) -> None:
        self.servers = ServerSupervisor()
        if state not in {"first_run", "disconnected"}:
            self.servers.observe_backend("macos_asr", available=True)
            self.servers.observe_backend("macos_embeddings", available=True)
        elif state == "disconnected":
            self.servers.observe_backend("macos_asr", available=False)
            self.servers.observe_backend("macos_embeddings", available=False)
        if state in {"running", "partial_failure"}:
            for kind in ("live", "reports"):
                self.servers.begin(kind, _DemoProcess())
                self.servers.observe(kind, listening=True, probe_due=True)
            if state == "running":
                self.servers.begin("translation", _DemoProcess())
                self.servers.observe("translation", listening=True, probe_due=True)
            else:
                self.servers.begin("translation", _DemoProcess(1))
                self.servers.observe("translation", listening=False, probe_due=True)
        elif state == "starting":
            self.servers.begin("live", _DemoProcess())
            self.servers.begin("reports", _DemoProcess())
            self.servers.observe("reports", listening=True, probe_due=True)
            self.servers.begin("translation", _DemoProcess())
        elif state == "failed":
            for kind in ("live", "reports", "translation"):
                self.servers.begin(kind, _DemoProcess(1))
                self.servers.observe(kind, listening=False, probe_due=True)

    def set_demo_state(self, state: str) -> None:
        if state not in DEMO_STATES:
            raise ValueError(f"Unknown demo state: {state}")
        self.demo_state = state
        self.report = canonical_report(first_run=state == "first_run")
        self._apply_services(state)
        self._emit(EventKind.REPORT, payload=self.report)
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)

    def update_profile(self, updates: dict[str, object], *, persist: bool = True) -> Profile:
        del persist
        return super().update_profile(updates, persist=False)

    def run_diagnostics(self, *, deep: bool = False) -> DoctorReport:
        del deep
        self._emit(EventKind.REPORT, payload=self.report)
        return self.report

    def launch(self) -> object:
        self.set_demo_state("starting")
        return object()

    def stop_owned_services(self) -> None:
        self.set_demo_state("ready")

    def refresh_services(self, *, force: bool = False) -> tuple[object, ...]:
        del force
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
        return tuple(self.servers.state(kind) for kind in self.SERVICE_KINDS)

    def shutdown(self) -> None:
        return None

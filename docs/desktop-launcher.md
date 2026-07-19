# Desktop Launcher

The desktop launcher gives WhoSpeaks one native, production-oriented place to check, configure, install, launch, inspect, and safely stop its services.

## Install and open it

The GUI is an optional dependency, so a terminal-only installation stays small and does not import Qt.

```powershell
pip install "whospeaks[gui]"
whospeaks
```

When PySide6 is installed and a desktop session is available, plain `whospeaks` opens the desktop launcher. Routing is explicit and backward compatible:

- `whospeaks --gui` requests the desktop launcher and explains how to install the extra if it is unavailable.
- `whospeaks --tui` always opens the Textual full-screen terminal interface.
- `whospeaks --classic` always opens the numbered terminal interface.
- `whospeaks --no-interactive` and all existing subcommands remain non-GUI automation paths.
- A base `pip install whospeaks` has no PySide6 dependency and falls back to Textual in an interactive terminal.

The interface flags are mutually exclusive, preventing an ambiguous request such as `--gui --tui`.

## How the workflow is organized

A launch profile is the saved collection of deployment, language, endpoint, preview, reporting, and translation settings. Overview shows that profile beside the services it will start. Diagnostics answers whether the selected configuration is ready. Activity keeps command and progress output selectable. Settings exposes every persisted profile field, including dependent values even while their feature is disabled.

The main states are deliberately explicit:

- First run identifies missing required components and offers installation.
- Ready keeps all services stopped until Launch is chosen.
- Starting reports each service independently; live capture can become available while an optional model continues warming.
- Running exposes the live window and a stop action.
- Partial failure leaves healthy services usable and shows the failing service, original error, remediation, and activity link.

WhoSpeaks only terminates process trees that this launcher started. An already-running service discovered on an optional port is treated as external and is not claimed or killed. Closing with owned services or an installer still active opens a confirmation dialog; cancel is the safe default. Worker exceptions are surfaced in the page operation region or an error dialog rather than silently discarded.

## Navigation, resizing, and accessibility

Overview, Diagnostics, Settings, Activity, and About share a stable left rail. At narrow widths the rail becomes icon-only, while tooltips and accessible names preserve meaning. Below the workspace breakpoint, Overview stacks the profile beneath the service list in one vertical scroll area. The minimum verified client size is 960 by 640 logical pixels.

All controls use native Qt focus and keyboard behavior, with visible focus treatment and logical tab order. Useful shortcuts are `Ctrl+R` for readiness, `Ctrl+L` for launch when allowed, `Ctrl+,` for Settings, and `Ctrl+Q` for safe shutdown. Status uses shape and text as well as color. Activity is read-only, selectable, monospace, and retains horizontal scrolling for long paths.

## Architecture and safety boundary

`LauncherController` is a typed, UI-independent adapter around the existing profile, diagnostics, planning, installation, and server-supervision modules. An immutable snapshot is a read-only copy of current profile, diagnostics, services, and operation state. The controller publishes snapshots and structured events; the Qt bridge moves slow work to worker threads and delivers results back to the UI thread.

This boundary lets the desktop UI and existing terminal surfaces reuse the same profile and command-planning rules. It also makes cancellation and ownership testable without starting real services.

## Deterministic demo and screenshots

Demo mode is fake, fixed data. It never writes configuration, probes the network, installs packages, or starts a process.

```powershell
whospeaks-gui --demo-state ready
whospeaks-gui --demo-state partial_failure
whospeaks-gui --demo-state settings
```

Capture the canonical 1586 by 992 client image used by the exact-size visual audit:

```powershell
whospeaks-gui --demo-state ready --screenshot ready.png --width 1586 --height 992
```

Supported demo states are `first_run`, `ready`, `starting`, `stopping`, `running`, `partial_failure`, `failed`, `disconnected`, `cancelled`, `success`, `diagnostics`, `settings`, `invalid_configuration`, `activity`, `about`, and `stop_confirmation`. Screenshot mode requires a demo state and uses settled animation timing. For design-only motion review, `--motion-phase 0` through `--motion-phase 11` selects a reproducible activity-arc phase.

## Source-tree development

Install the project and GUI extra in the active environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
```

Then run the focused verification:

```powershell
$env:PYTHONPATH = "src"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest tests.test_whospeaks_launcher_controller tests.test_whospeaks_gui
```

The design references, versioned contract, actual captures, visual overlays, heatmaps, and metric files live in `design-artifacts/desktop-launcher-v1/`.

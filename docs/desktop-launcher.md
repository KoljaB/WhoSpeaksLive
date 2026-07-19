# Desktop Launcher

The desktop launcher gives WhoSpeaks one native, production-oriented place to check, configure, install, launch, inspect, and safely stop its services.

## Install and open it

The native PySide6 launcher is the standard WhoSpeaks interface.

```powershell
pip install whospeaks
whospeaks
```

In a graphical desktop session, plain `whospeaks` opens the desktop launcher:

- `whospeaks --gui` explicitly requests the desktop launcher, including from a non-interactive shell.
- `whospeaks --no-interactive` prints readiness once and exits.
- Subcommands such as `doctor`, `install`, `config`, and `launch` remain non-GUI automation paths.
- On a machine without a graphical desktop session, plain `whospeaks` prints readiness and points to those subcommands instead of starting a terminal user interface.

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

All controls use native Qt focus and keyboard behavior, with visible focus treatment and logical tab order. Interactive pages share one fixed-height action footer: the primary action comes first, secondary actions use a clearly bordered surface, and status text remains separate. Every Settings control has a concise mouseover tooltip and an accessible description. The context-help panel at the bottom of Settings follows keyboard focus and mouse hover, shows the current value, and explains the control's immediate effect. `F1` opens a bounded, scrollable explanation with operational consequences, compatibility and privacy cautions, provider-specific defaults, and current-preset details where they matter; this content is intentionally separate from the short tooltip. Other useful shortcuts are `Ctrl+R` for readiness, `Ctrl+L` for launch when allowed, `Ctrl+,` for Settings, and `Ctrl+Q` for safe shutdown. Status uses shape and text as well as color. Activity is read-only, selectable, monospace, and retains horizontal scrolling for long paths.

## Architecture and safety boundary

`LauncherController` is a typed, UI-independent adapter around the existing profile, diagnostics, planning, installation, and server-supervision modules. An immutable snapshot is a read-only copy of current profile, diagnostics, services, and operation state. The controller publishes snapshots and structured events; the Qt bridge moves slow work to worker threads and delivers results back to the UI thread.

This boundary lets the desktop UI and scriptable CLI subcommands reuse the same profile and command-planning rules. It also makes cancellation and ownership testable without starting real services.

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

Install the project in the active environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Then run the focused verification:

```powershell
$env:PYTHONPATH = "src"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest tests.test_whospeaks_launcher_controller tests.test_whospeaks_gui
```

The design references, versioned contract, actual captures, visual overlays, heatmaps, and metric files live in `design-artifacts/desktop-launcher-v1/`.

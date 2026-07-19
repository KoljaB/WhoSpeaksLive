# Installation

## Optional local translation server

Choose a local translation model during the guided installer, similarly to the Nemotron/Kroko live-text choice, or install providers independently:

```text
whospeaks install-translation --model-profile nllb-200-600m --torch auto --yes
whospeaks install-translation --model-profile translate-gemma-4b --torch auto --yes
whospeaks install-translation --model-profile madlad-400-3b --torch auto --yes
```

Each command creates a separate virtual environment and model directory on Windows or Linux. NLLB is the recommended broad, lower-memory default. TranslateGemma requires accepting the Gemma terms on Hugging Face. MADLAD uses considerably more memory than NLLB. See [Live translation](translation.md) for paths, licensing, launch behavior, and offline verification.

Install the `whospeaks` command with its native desktop interface, then let it guide the full local, controller, or server setup.

## Recommended First Command

From a Python 3.11 environment:

```powershell
pip install whospeaks
whospeaks
```

### uv alternative

Use uv's tool installer for the same globally available `whospeaks` command:

```powershell
uv tool install --python 3.11 whospeaks
uv tool update-shell
```

`uv tool install` keeps WhoSpeaks in an isolated, persistent environment and exposes its console commands through uv's tool executable directory. `uv tool update-shell` adds that directory to `PATH`; it is only needed once per machine. Open a new terminal after it completes, then start the launcher from any directory:

```powershell
whospeaks
```

After the launcher starts, it checks `PATH` for **uv** and selects it by default when available; otherwise it selects **pip**. The Overview setup form still lets you switch between the available installers. The selected backend is used for the larger target-specific dependency sets, PyTorch, managed macOS services, translation sidecars, and the Python-package portion of a Kroko sidecar. The scriptable equivalent is:

```powershell
whospeaks install --target local --installer uv --yes
```

For non-interactive CLI commands, pip remains the default because it is bundled with normal Python installations and has the broadest compatibility; the desktop launcher performs the automatic uv detection described above. Pass `--installer pip` / `--installer uv` explicitly for scripted installs. WhoSpeaks always gives uv an explicit target interpreter, so the second-stage packages are added to the same environment that runs the launcher, including an environment created by `uv tool install`. When pip is selected, the guided installer first attempts a best-effort pip update and continues with the existing version if that update fails.

Do not use `uvx whospeaks` for guided setup. `uvx` uses an ephemeral environment, while WhoSpeaks needs a persistent environment for the heavier runtime selected in the launcher.

To test a development build from TestPyPI, replace `<version>` with the published development version:

```powershell
uv tool install --python 3.11 --index https://pypi.org/simple/ --index https://test.pypi.org/simple/ --index-strategy unsafe-first-match --refresh-package whospeaks "whospeaks==<version>"
uv tool update-shell
```

Open a new terminal and run `whospeaks`. The explicit index strategy is limited to this TestPyPI workflow. PyPI is deliberately listed first so ordinary dependencies use its complete wheel set; the exact WhoSpeaks development version then falls through to TestPyPI. This avoids both TestPyPI placeholder packages and incomplete TestPyPI file sets, such as a release that lacks a CPython 3.11 Windows wheel. Normal PyPI installations keep uv's safer default strategy. Once installed, the launcher recognizes the development version and applies the same PyPI-first TestPyPI fallback when uv expands the selected WhoSpeaks extras.

The base install includes PySide6, the native desktop launcher, doctor checks, profile storage, and the launch-command generator. GPU frameworks, speech models, media libraries, and translation runtimes remain optional and are installed only for the deployment you select. On SSH/headless machines, use the scriptable CLI subcommands rather than an interactive terminal launcher. Overview asks what you want on this machine:

- Full local installation: browser controller, final ASR, and speaker embeddings on one machine.
- Core/controller: browser UI on this machine, with ASR and embeddings served by remote HTTP services.
- ASR and embeddings server packages: service-side dependencies for a remote GPU/server machine.

For local or core/controller installs, use the realtime text selection to include or exclude optional preview support. Kroko setup is still a native post-install step, while Nemotron 3.5 preview can be enabled manually with the `sherpa_onnx` runtime option while the installer flow is being integrated. Leave preview off when you want the cleanest install first; final ASR and speaker diarization can still run without live preview text.

The interface has five operational views:

- Overview: first-run installation choices, component readiness, launch, and live service state.
- Diagnostics: quick and complete doctor reports with remediation details.
- Settings: language, speaker provider, ASR runtime, browser address, and remote service URLs.
- Activity: live installer output and cancellation for a running operation.
- About: version, project links, interfaces, and keyboard shortcuts.

The `whospeaks install` subcommand exposes the same installer for automation and advanced use:

```powershell
whospeaks install --target local --without-kroko --yes
whospeaks install --target local --with-kroko --yes
whospeaks install --target core --without-kroko --yes
whospeaks install --target server --yes
whospeaks install --target server --installer uv --yes
```

Internally, the installer uses PyPI optional dependency sets to install the right Python packages through the selected pip or uv backend, but users should treat those as implementation details. For a full local install without Kroko, it installs the local Python stack. With Kroko selected, it also installs lightweight preview support and then checks whether realtime preview still lacks the native `kroko_onnx` runtime. If it is missing, the CLI offers a separate Kroko install/build step using the vendored RealtimeSTT installer:

```powershell
whospeaks install-kroko
```

The native Kroko runtime is handled as a post-install setup step instead of a PyPI dependency because it may need an upstream source build, Docker, or a Python 3.12 sidecar on Windows. With uv selected, the launcher can ask uv for Python 3.12 while creating that seeded sidecar; Kroko's own native build then continues to use its upstream installer. System prerequisites such as GPU drivers, `ffmpeg`, model downloads, Hugging Face access, and native build tooling are still verified by `whospeaks doctor` because no Python package installer can safely guarantee them. Use `--deep` when you want checks that may touch provider parsing or load endpoints; plain doctor avoids expensive model startup.

## How The Short `whospeaks` Command Works

The short command opens the native desktop launcher in a graphical desktop session. On a headless machine it prints readiness once and points to the scriptable subcommands. The browser app itself is started by the longer `whospeaks-window` command built from your saved profile. `--gui` explicitly requests the desktop surface; scriptable subcommands open none.

`pip install whospeaks` and `uv tool install whospeaks` install console scripts from the package metadata. A console script is a command-line wrapper that imports a Python function. In this package:

- `whospeaks` runs `whospeaks_cli.main:main`.
- `whospeaks-gui` runs `whospeaks_gui.main:main` directly.
- `whospeaks-window` runs `window.youtube_window_diarize_gui:main`.

With a normal pip environment on `PATH`, or after running `uv tool update-shell`, you can type this from any directory:

```powershell
whospeaks
```

Developers who deliberately install into a project-local virtual environment can activate it or use its environment-local executable:

```powershell
.\.venv\Scripts\whospeaks.exe
```

Command lookup is controlled by the operating system `PATH`. For a persistent end-user UV installation, prefer `uv tool install` and `uv tool update-shell` over adding a project-specific `.venv` directory to the user `PATH`.

The launcher stores a profile in:

- Windows: `%APPDATA%\WhoSpeaks\config.json`
- Linux/macOS: `$XDG_CONFIG_HOME/whospeaks/config.json` or `~/.config/whospeaks/config.json`

Set `WHOSPEAKS_CONFIG` to use a specific config file. If the selected location is not writable, the launcher reports the error and leaves the existing profile unchanged.

The normal interactive flow is:

1. Run `whospeaks` and select the installation target in Overview.
2. Select whether to include Kroko realtime text.
3. Select pip or uv as the Python package installer.
4. Review the plan and start installation; live output appears on the Activity tab.
5. Use Diagnostics to verify the resulting component state.
6. Launch the browser UI from Overview.

The equivalent automation interfaces remain `whospeaks install`, `whospeaks doctor`, `whospeaks config`, and `whospeaks launch`.

For local embeddings and realtime preview, the launcher passes explicit helper interpreters with `--embedding-python` and `--realtime-preview-python`. This prevents helper subprocesses from accidentally using the wrong Python environment.

## Setup Modes

The desktop launcher supports these profiles:

- Full local installation: ASR, embeddings, browser controller, and Kroko realtime preview on one machine.
- Controller with remote GPU services: browser UI on this machine, ASR and embeddings behind HTTP endpoints.
- GPU server machine: installs service-side dependencies for ASR and embeddings endpoints.
- Repair / doctor: inspect an existing setup and print exact remediation commands.

## Recommended Topology

The easiest reliable setup is:

- Windows machine: browser UI, media download, orchestration, People library, and saved sessions.
- Linux GPU server: faster-whisper ASR on port `8650`.
- Linux GPU server: voice embeddings on port `8660`.

This keeps the interactive UI responsive and avoids loading several large ML models on the Windows controller. Local all-in-one operation is possible, but it is harder to install and needs enough VRAM for ASR plus embeddings.

## Install Order

1. Install Windows prerequisites.
2. Install the lightweight `whospeaks` command.
3. Run `whospeaks` and choose a setup mode.
4. Let `whospeaks doctor` verify local packages, `ffmpeg`, ports, services, providers, and model/cache state.
5. Launch from the desktop launcher or with the command printed by `whospeaks launch --print`.

## Windows Prerequisites

Install these on the Windows controller:

- Git.
- Python 3.11, 64-bit.
- `ffmpeg` on `PATH`.
- A modern browser.

Check Python:

```powershell
py -0p
py -3.11 --version
```

Install `ffmpeg` with one of these package managers:

```powershell
winget install Gyan.FFmpeg
```

or:

```powershell
choco install ffmpeg
```

Open a new PowerShell window after installing `ffmpeg`, then verify:

```powershell
ffmpeg -version
```

## Get The Source

Clone the repository or open an existing checkout. The examples below assume:

```powershell
cd C:\Projects
git clone https://github.com/KoljaB/WhoSpeaksLive.git
cd WhoSpeaksLive
```

If your checkout lives somewhere else, run the commands from that repository root.

## Create The Controller Venv

Create and update the virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Install the lightweight starter:

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\whospeaks.exe
```

For the older manual remote-controller setup, install the minimal controller dependency set:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-controller.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Check that the console entry points are installed:

```powershell
.\.venv\Scripts\whospeaks.exe --help
.\.venv\Scripts\whospeaks-window.exe --help
.\.venv\Scripts\whospeaks-realtime.exe --help
.\.venv\Scripts\whospeaks-filefeed-replay.exe --help
.\.venv\Scripts\whospeaks-embedding-benchmark.exe --help
.\.venv\Scripts\whospeaks-browser-live-eval.exe --help
```

## First Controller Smoke Check

This only checks that the controller package imports and the browser server can parse options. It does not prove the remote GPU services are running:

```powershell
.\.venv\Scripts\whospeaks.exe doctor --mode remote
.\.venv\Scripts\whospeaks-window.exe --help
```

After the remote servers are running, verify them from Windows:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
.\.venv\Scripts\whospeaks.exe doctor --mode remote --deep --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660
```

Replace `YOUR_GPU_SERVER_IP` with the IP address of your Linux GPU server, for example `192.168.1.50`.

## First End-To-End Launch

Use a conservative first launch that avoids optional local preview/VAD dependencies:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

When this works, move to the tuned provider commands in [Quickstart](quickstart.md).

## Docker Server Install

For a reproducible Linux server container, use the root Dockerfile:

```bash
docker build -t whospeaks:local .
docker volume create whospeaks-data
docker volume create whospeaks-models
docker run --rm --name whospeaks -p 127.0.0.1:8796:8796 -v whospeaks-data:/data -v whospeaks-models:/models whospeaks:local
```

Open `http://127.0.0.1:8796/` on the Docker host. Remote access requires a protected private path such as a VPN, SSH tunnel, or authenticated TLS reverse proxy; the built-in server has no authentication. See [Docker](docker.md) and [Security And Data Privacy](security-and-data-privacy.md).

## Runtime Data

Mutable files are kept under `runtime/` by default and ignored by Git.

Default paths:

- `runtime/cache/`: library and model caches.
- `runtime/models/hub/`: ESPnet and S3PRL hub checkpoints.
- `runtime/models/kroko-onnx/`: Kroko preview models.
- `runtime/media/local-filefeed/`: downloaded replay media.
- `runtime/outputs/window-diarize/`: generated window diarization audio.
- `runtime/outputs/window-diarize-validation/`: validation output.
- `runtime/sessions/`: saved transcripts, Speaker/Person links, sentence embeddings, audio or audio references, corrections, translations, and reports.
- `runtime/speakers/people.json`: People, persistent recognition settings, sample provenance, and embedding vectors.
- `runtime/speakers/voice-samples/`: retained original audio for manual Voice samples.
- `runtime/speakers/`: legacy Speaker-group files and People migration backups.

You can redirect local state with environment variables:

```powershell
$env:WHOSPEAKS_RUNTIME_DIR = "C:\whospeaks-runtime"
$env:WHOSPEAKS_CACHE_DIR = "C:\whospeaks-runtime\cache"
$env:WHOSPEAKS_MODEL_DIR = "C:\whospeaks-runtime\models"
$env:WHOSPEAKS_SPEAKER_LIBRARY_DIR = "C:\whospeaks-runtime\speakers"
```

Treat the speaker-library and session directories as one sensitive backup unit. See [Security And Data Privacy](security-and-data-privacy.md) before exposing the browser server or moving voice data between machines.

## Local All-In-One Setup

The guided path is:

```powershell
.\.venv\Scripts\whospeaks.exe
.\.venv\Scripts\whospeaks.exe doctor --mode local
.\.venv\Scripts\whospeaks.exe doctor --mode local --deep
```

On the first-run Overview, select Full local and then Install components. For an existing installation, run `whospeaks install --target local` to add or repair runtime components. The launcher uses `--device auto` for this path. If pip installed a CPU-only PyTorch build, speaker embeddings fall back to CPU instead of crashing; install a CUDA-enabled PyTorch build when you need GPU embeddings. The Live text and Live model controls determine which preview runtime is configured.

The older manual path installs the larger historical environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

This path is slower to install and more likely to hit CUDA, PyTorch, model-cache, or VRAM issues. Use the remote-server setup first unless you specifically need everything on one Windows machine.

## Realtime Preview

Realtime preview text is optional. Final transcript diarization and live speaker probing still work if preview is disabled or if a preview backend fails to install. Kroko preview depends on the `kroko_onnx` native Python extension and a Kroko model file. Nemotron preview depends on `sherpa-onnx`, `sherpa-onnx-bin`, and a verified Nemotron model directory.

### Nemotron 3.5 (experimental)

Nemotron 3.5 is an opt-in CPU-only preview backend. It uses `sherpa-onnx` in a separate process and does not replace final ASR. The 560 ms int8 preset is the initial balance of text quality and update latency; the 160 ms preset is available for comparison.

```powershell
.\.venv\Scripts\python.exe -m pip install "sherpa-onnx>=1.13.4,<1.14" "sherpa-onnx-bin>=1.13.4,<1.14"
.\.venv\Scripts\whospeaks-window.exe --language de --realtime-preview-engine sherpa_onnx --realtime-preview-model-preset nemotron-3.5-560ms-int8
```

On first start, WhoSpeaks downloads the selected 453 MB archive from the upstream `k2-fsa/sherpa-onnx` release, verifies its pinned SHA-256 checksum, and installs it under `runtime/models/sherpa-onnx/`. Use `--no-realtime-preview-auto-download` to require a preinstalled model or `--realtime-preview-model-dir` to select one manually.

Nemotron 3.5 exposes `en`, `de`, `es`, `fr`, `it`, `nl`, `pt`, `tr`, and `sv` for realtime preview in WhoSpeaksLive. English, German, Spanish, French, Italian, Dutch, Portuguese, and Turkish are the main supported languages; Swedish is broad-coverage. Hebrew should remain on Kroko or have preview disabled. The underlying model may produce text for more languages, but they are not treated as supported until validated in the app's realtime path.

Nemotron model weights use NVIDIA Open Model Development and Weight License 1.1, not the project MIT license. It remains experimental and Kroko remains available. See [Third-Party Model Licenses](third-party-model-licenses.md).

The `--language` flag selects the matching community model name for supported realtime languages, for example `--language de` selects `Kroko-DE-Community-64-L-Streaming-001.data`. Missing public Community model files are downloaded automatically to `runtime/models/kroko-onnx/`; use `--no-realtime-preview-auto-download` to require preinstalled files. If doctor reports `Kroko ONNX runtime` as missing, install a `kroko_onnx` wheel matching the active Python or run the Kroko setup wrapper:

```powershell
whospeaks install-kroko
```

That wrapper runs `python -m RealtimeSTT.install_kroko --build` where the active environment can build Kroko directly.

On Windows, Kroko builds may be available only for Python 3.12 while the main WhoSpeaks install uses Python 3.11. If `py -3.12` is available, `whospeaks install-kroko` can create a Python 3.12 realtime-preview sidecar, install the preview package there, build/install `kroko_onnx`, and save that interpreter in the launcher. You can also create or reuse a Python 3.12 realtime-preview venv yourself and save it explicitly:

```powershell
whospeaks config --realtime-preview-python C:\path\to\kroko-preview\Scripts\python.exe
```

The launcher passes the installed WhoSpeaks package path to that subprocess, so the Python 3.12 environment only needs the preview runtime packages and native `kroko_onnx` wheel. During first-run setup, choose Kroko / Banafo under Live text before installing components. On an existing installation, use `whospeaks install-kroko` to add or repair that runtime. The `install` and `install-kroko` subcommands remain available for automation and troubleshooting.

## Next Step

Set up the Linux GPU services in [External Servers](external-servers.md), then launch the app with [Quickstart](quickstart.md).

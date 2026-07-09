# Installation

Install the lightweight `whospeaks` command first, then let it guide the full local, controller, or server setup.

## Recommended First Command

From a Python 3.11 environment:

```powershell
pip install whospeaks
whospeaks
```

The base install is intentionally small. It installs the starter CLI, doctor checks, profile storage, and launch-command generator. On first run, choose `Full local setup` for a one-machine install. The CLI checks what is missing and offers the matching installer. For a full local setup, it can run the internal command:

```powershell
python -m pip install "whospeaks[complete,preview]"
```

You can also run the same path non-interactively:

```powershell
whospeaks setup --mode local --install
whospeaks doctor --mode local
whospeaks doctor --mode local --deep
whospeaks launch --print
```

The guided full local setup installs the `complete` and `preview` extras together. `complete` covers the Python-side controller, local ASR, and local embeddings dependency set; `preview` adds lightweight realtime preview support without pulling the older RealtimeSTT dependency chain. System prerequisites such as GPU drivers, `ffmpeg`, native Kroko runtime installation, model downloads, and Hugging Face access are still verified by `whospeaks doctor` because they cannot be safely guaranteed by pip alone. Use `--deep` when you want checks that may touch provider parsing or load endpoints; plain doctor avoids expensive model startup.

## How The Short `whospeaks` Command Works

The short command is a launcher and setup assistant; the browser app itself is started by the longer `whospeaks-window` command that the launcher builds from your saved profile.

`pip install whospeaks` installs console scripts from the package metadata. A console script is a command-line wrapper that imports a Python function. In this package:

- `whospeaks` runs `whospeaks_cli.main:main`.
- `whospeaks-window` runs `window.youtube_window_diarize_gui:main`.

When the active environment is on `PATH`, you can type:

```powershell
whospeaks
```

Without activation, use the environment-local executable:

```powershell
.\.venv\Scripts\whospeaks.exe
```

You can keep WhoSpeaks installed in a venv and still type the short command. Command lookup is controlled by the operating system `PATH`, not by whether the package is installed globally. On Windows, adding `.venv\Scripts` to the user `PATH` makes `whospeaks` resolve to that venv's console script; on Linux/macOS, adding `.venv/bin` does the same. Installer scripts should add that directory only if it is not already present.

The launcher stores a profile in:

- Windows: `%APPDATA%\WhoSpeaks\config.json`
- Linux/macOS: `$XDG_CONFIG_HOME/whospeaks/config.json` or `~/.config/whospeaks/config.json`

Set `WHOSPEAKS_CONFIG` to use a specific config file. If the user config location is not writable, the launcher falls back to `.whospeaks/config.json` in the current directory.

The normal flow is:

1. `whospeaks setup --mode local --install` saves a local profile and installs the recommended extras.
2. `whospeaks doctor --mode local` checks packages, ffmpeg, ports, helper Python paths, and model/cache state.
3. `whospeaks config ...` changes saved fields such as language, provider preset, backend URLs, helper Python paths, and port.
4. `whospeaks launch --print` prints the exact `whospeaks-window ...` command.
5. `whospeaks launch` runs that command.

For local embeddings and realtime preview, the launcher passes explicit helper interpreters with `--embedding-python` and `--realtime-preview-python`. This prevents helper subprocesses from accidentally using the wrong Python environment.

## Setup Modes

The starter CLI supports these profiles:

- Full local installation: ASR, embeddings, browser controller, and Kroko realtime preview on one machine.
- Controller with remote GPU services: browser UI on this machine, ASR and embeddings behind HTTP endpoints.
- GPU server machine: installs service-side dependencies for ASR and embeddings endpoints.
- Repair / doctor: inspect an existing setup and print exact remediation commands.

## Recommended Topology

The easiest reliable setup is:

- Windows machine: browser UI, media download, orchestration, speaker library files.
- Linux GPU server: faster-whisper ASR on port `8650`.
- Linux GPU server: voice embeddings on port `8660`.

This keeps the interactive UI responsive and avoids loading several large ML models on the Windows controller. Local all-in-one operation is possible, but it is harder to install and needs enough VRAM for ASR plus embeddings.

## Install Order

1. Install Windows prerequisites.
2. Install the lightweight `whospeaks` command.
3. Run `whospeaks` and choose a setup mode.
4. Let `whospeaks doctor` verify local packages, `ffmpeg`, ports, services, providers, and model/cache state.
5. Launch from the starter CLI or with the command printed by `whospeaks launch --print`.

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
docker run --rm --name whospeaks -p 8796:8796 -v whospeaks-data:/data -v whospeaks-models:/models whospeaks:local
```

Open `http://127.0.0.1:8796/` on the Docker host, or use the host IP from another machine. See [Docker](docker.md) for build arguments, media mounts, validation commands, and volume notes.

## Runtime Data

Mutable files are kept under `runtime/` by default and ignored by Git.

Default paths:

- `runtime/cache/`: library and model caches.
- `runtime/models/hub/`: ESPnet and S3PRL hub checkpoints.
- `runtime/models/kroko-onnx/`: Kroko preview models.
- `runtime/media/local-filefeed/`: downloaded replay media.
- `runtime/outputs/window-diarize/`: generated window diarization audio.
- `runtime/outputs/window-diarize-validation/`: validation output.
- `runtime/speakers/`: saved speaker groups and references.

You can redirect local state with environment variables:

```powershell
$env:WHOSPEAKS_RUNTIME_DIR = "C:\whospeaks-runtime"
$env:WHOSPEAKS_CACHE_DIR = "C:\whospeaks-runtime\cache"
$env:WHOSPEAKS_MODEL_DIR = "C:\whospeaks-runtime\models"
$env:WHOSPEAKS_SPEAKER_LIBRARY_DIR = "C:\whospeaks-runtime\speakers"
```

## Local All-In-One Setup

The guided path is:

```powershell
.\.venv\Scripts\whospeaks.exe setup --mode local --install
.\.venv\Scripts\whospeaks.exe doctor --mode local
.\.venv\Scripts\whospeaks.exe doctor --mode local --deep
```

The launcher uses `--device auto` for this path. If pip installed a CPU-only PyTorch build, speaker embeddings fall back to CPU instead of crashing; install a CUDA-enabled PyTorch build when you need GPU embeddings. It also enables `--realtime-preview-engine kroko_onnx` so live text is available after the preview dependencies and public Kroko model are installed.

The older manual path installs the larger historical environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

This path is slower to install and more likely to hit CUDA, PyTorch, model-cache, or VRAM issues. Use the remote-server setup first unless you specifically need everything on one Windows machine.

## Realtime Preview

The guided full local setup enables local realtime preview with `--realtime-preview-engine kroko_onnx`. Final transcript diarization and live speaker probing still work if preview fails, but the live text row depends on the `kroko_onnx` native Python extension and a Kroko model file.

The `--language` flag selects the matching community model name for supported realtime languages, for example `--language de` selects `Kroko-DE-Community-64-L-Streaming-001.data`. Missing public Community model files are downloaded automatically to `runtime/models/kroko-onnx/`; use `--no-realtime-preview-auto-download` to require preinstalled files. If doctor reports `Kroko ONNX runtime` as missing, install a `kroko_onnx` wheel matching the active Python or run `python -m RealtimeSTT.install_kroko --build` on a supported build environment.

On Windows, Kroko builds may be available only for Python 3.12 while the main WhoSpeaks install uses Python 3.11. In that case, create or reuse a Python 3.12 realtime-preview venv with `kroko_onnx` installed and save that interpreter in the launcher:

```powershell
whospeaks config --realtime-preview-python D:\Projekte\SpeakerDiarization\.venvs\kroko-install-test\Scripts\python.exe
```

The launcher passes the installed WhoSpeaks package path to that subprocess, so the Python 3.12 environment only needs the preview runtime packages and native `kroko_onnx` wheel.

The standalone package extra for this path is:

```powershell
python -m pip install "whospeaks[preview]"
```

## Next Step

Set up the Linux GPU services in [External Servers](external-servers.md), then launch the app with [Quickstart](quickstart.md).

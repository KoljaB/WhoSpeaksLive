# Installation

Install the Windows controller first, then connect it to ASR and embeddings either on a Linux GPU server or on the same machine.

## Recommended Topology

The easiest reliable setup is:

- Windows machine: browser UI, media download, orchestration, speaker library files.
- Linux GPU server: faster-whisper ASR on port `8650`.
- Linux GPU server: voice embeddings on port `8660`.

This keeps the interactive UI responsive and avoids loading several large ML models on the Windows controller. Local all-in-one operation is possible, but it is harder to install and needs enough VRAM for ASR plus embeddings.

## Install Order

1. Install Windows prerequisites.
2. Create the Windows controller virtual environment.
3. Install the controller package and controller dependencies.
4. Set up the Linux GPU servers from [External Servers](external-servers.md).
5. Verify both remote `/health` endpoints.
6. Launch the browser app from [Quickstart](quickstart.md).

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

Install the minimal controller dependency set for the recommended remote-server setup:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-controller.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Check that the console entry points are installed:

```powershell
.\.venv\Scripts\whospeaks-window.exe --help
.\.venv\Scripts\whospeaks-realtime.exe --help
.\.venv\Scripts\whospeaks-filefeed-replay.exe --help
.\.venv\Scripts\whospeaks-embedding-benchmark.exe --help
.\.venv\Scripts\whospeaks-browser-live-eval.exe --help
```

## First Controller Smoke Check

This only checks that the controller package imports and the browser server can parse options. It does not prove the remote GPU services are running:

```powershell
.\.venv\Scripts\whospeaks-window.exe --help
```

After the remote servers are running, verify them from Windows:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Replace `YOUR_GPU_SERVER_IP` with the IP address of your Linux GPU server, for example `192.168.1.50`.

## First End-To-End Launch

Use a conservative first launch that avoids optional local preview/VAD dependencies:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

When this works, move to the tuned provider commands in [Quickstart](quickstart.md).

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

Local ASR and local embeddings require the larger historical environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

This path is slower to install and more likely to hit CUDA, PyTorch, model-cache, or VRAM issues. Use the remote-server setup first unless you specifically need everything on one Windows machine.

## Optional Realtime Preview

The first launch above disables the local realtime preview engine with `--realtime-preview-engine off`. Final transcript diarization and live speaker probing still work.

Kroko/RealtimeSTT preview is optional and currently depends on a separate local RealtimeSTT/Kroko environment. If that environment exists, remove `--realtime-preview-engine off` or point `--realtime-preview-python` at the working preview environment. The `--language` flag selects the matching community model name for supported realtime languages, for example `--language de` selects `Kroko-DE-Community-64-L-Streaming-001.data`. Missing public Community model files are downloaded automatically to `runtime/models/kroko-onnx/`; use `--no-realtime-preview-auto-download` to require preinstalled files.

## Next Step

Set up the Linux GPU services in [External Servers](external-servers.md), then launch the app with [Quickstart](quickstart.md).

# Installation

Install WhoSpeaksLive from a checkout, then choose whether heavy ASR and embeddings run locally or on a remote GPU server.

## Requirements

- Windows is the primary local development environment used by this checkout.
- Python 3.11 or newer.
- A virtual environment for the app.
- `ffmpeg` or compatible media tooling available to the audio stack.
- Optional: a CUDA-capable GPU for local ASR and embeddings.
- Optional: a Linux GPU server for remote ASR and embeddings.

The project metadata intentionally keeps Python dependencies light. The historical `requirements.txt` is present for the full runtime environment, but large ML packages are not installed automatically by package metadata.

## Local Editable Install

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Check the command entry points:

```powershell
whospeaks-window --help
whospeaks-realtime --help
whospeaks-filefeed-replay --help
whospeaks-embedding-benchmark --help
```

The compatibility wrappers are also available:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help
.\.venv\Scripts\python.exe tools\realtime_speakerdiarize.py --help
.\.venv\Scripts\python.exe tools\youtube_local_filefeed_replay.py --help
.\.venv\Scripts\python.exe tools\benchmark_voice_embeddings.py --help
```

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
$env:WHOSPEAKS_RUNTIME_DIR = "D:\whospeaks-runtime"
$env:WHOSPEAKS_CACHE_DIR = "D:\whospeaks-runtime\cache"
$env:WHOSPEAKS_MODEL_DIR = "D:\whospeaks-runtime\models"
$env:WHOSPEAKS_SPEAKER_LIBRARY_DIR = "D:\whospeaks-runtime\speakers"
```

## First Startup Cost

The default high-quality embedding stack can download and load large models on a clean machine. If the app starts but waits a long time before releasing the browser URL, either let model loading finish or increase the helper timeout:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --embedding-helper-response-timeout-seconds 900
```

For faster iteration, use remote ASR and remote embeddings as described in [External Servers](external-servers.md).

# WhoSpeaks

WhoSpeaks is a local speaker diarization toolkit for building and validating
speaker-labeled voice datasets.

## Setup

Install the project in editable mode from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

The historical `requirements.txt` is still present for the full runtime
environment. The package metadata itself intentionally does not install those
heavy audio and ML dependencies.

## Commands

Preferred console scripts:

```powershell
whospeaks-window --help
whospeaks-realtime --help
whospeaks-filefeed-replay --help
whospeaks-embedding-benchmark --help
```

Old script paths remain available as compatibility wrappers:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help
.\.venv\Scripts\python.exe tools\realtime_speakerdiarize.py --help
.\.venv\Scripts\python.exe tools\youtube_local_filefeed_replay.py --help
.\.venv\Scripts\python.exe tools\benchmark_voice_embeddings.py --help
```

When using remote ASR, speaker embeddings still run locally. The first startup
of the default high-quality stacked embedding provider may download and load
large models before the browser URL is released. If a clean machine needs more
time, use:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --embedding-helper-response-timeout-seconds 900
```

To run both final ASR and speaker embeddings on the Linux GPU box:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --port 8796 --asr-backend remote --remote-asr-url http://192.168.178.22:8650 --embeddings-backend remote --remote-embeddings-url http://192.168.178.22:8660
```

Fast live-speaker highlighting probes the latest `1.25` seconds of audio every
`0.4` seconds by default. Tune it with
`--live-speaker-probe-window-seconds` and
`--live-speaker-probe-interval-seconds`.

## Layout

Source code lives under `src/whospeaks/`. Copied third-party sources live under
`vendor/`. Mutable local data lives under `runtime/` and is ignored by Git.

See `docs/STRUCTURE.md` for the filesystem policy and runtime path overrides.

## Runtime Data

Default mutable paths:

- `runtime/cache/` for library and model caches
- `runtime/models/hub/` for ESPnet/S3PRL hub checkpoints
- `runtime/models/kroko-onnx/` for Kroko preview models
- `runtime/media/local-filefeed/` for downloaded replay media
- `runtime/outputs/window-diarize/` for generated window diarization audio
- `runtime/speakers/` for speaker libraries and uploaded references

You can redirect local state with `WHOSPEAKS_RUNTIME_DIR`,
`WHOSPEAKS_CACHE_DIR`, `WHOSPEAKS_MODEL_DIR`, and
`WHOSPEAKS_SPEAKER_LIBRARY_DIR`.

Small deterministic fixtures, such as the Cunk canonical validation JSON, live
under `tests/fixtures/`.

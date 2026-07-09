# Quickstart

Start with a small smoke run, then move to the tuned provider stacks after the selected local or remote services are healthy.

## Before You Start

Complete:

1. [Installation](installation.md) on the Windows controller.
2. [External Servers](external-servers.md) on the Linux GPU server if you are using remote ASR or remote embeddings.

Verify from Windows:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Replace `YOUR_GPU_SERVER_IP` with your Linux GPU server address.

Skip the remote health checks for a completely local Windows run.

## First End-To-End Run

The shortest guided path is:

```powershell
whospeaks
```

Choose the setup/profile you want, then choose `Launch browser UI`. To see the exact command before running it:

```powershell
whospeaks launch --print
```

The launcher reads the saved profile, expands it into a full `whospeaks-window ...` command, and injects helper Python paths for local embeddings and realtime preview when those features are enabled.

Use a smoke provider first. This proves the UI, media loading, ASR route, embedding route, and speaker assignment pipeline work:

In the starter CLI, choose `Speaker provider quality` -> `First start`, or run:

```powershell
.\.venv\Scripts\whospeaks.exe config --set provider_preset=smoke
```

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

The command prints a browser URL. Open it, load or replay media, then press Start.

## Docker Quickstart

For a Linux container smoke server:

```bash
docker build -t whospeaks:local .
docker volume create whospeaks-data
docker volume create whospeaks-models
docker run --rm --name whospeaks -p 8796:8796 -v whospeaks-data:/data -v whospeaks-models:/models whospeaks:local
```

Open `http://127.0.0.1:8796/`. The container path uses CPU defaults and is meant as a reproducible server install first; see [Docker](docker.md) for local media mounts, build args, and validation commands.

## Completely Local Windows Run

Use this when ASR, embeddings, realtime preview, and the browser app all run on the same Windows machine. It keeps one high-quality final embedding provider and uses the safer live-speaker timing defaults so final ASR can stay responsive on a single GPU:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend local --model large-v2 --device cuda --compute-type float16 --embeddings-backend local --embedding-provider espnet_ecapa_wavlm_joint --live-speaker-embedding-provider espnet_ecapa_wavlm_joint --embedding-device cuda --vad-backend rms --realtime-preview-engine kroko_onnx --beam-size 5 --interval-seconds 2.5 --min-playback-advance-seconds 2.5 --unstable-tail-seconds 1.1
```

If final transcript rows still fall behind playback, isolate live speaker scoring first:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend local --model large-v2 --device cuda --compute-type float16 --embeddings-backend local --embedding-provider espnet_ecapa_wavlm_joint --live-speaker-embedding-provider espnet_ecapa_wavlm_joint --embedding-device cuda --vad-backend rms --realtime-preview-engine kroko_onnx --beam-size 5 --interval-seconds 2.5 --min-playback-advance-seconds 2.5 --unstable-tail-seconds 1.1 --no-live-speaker-assignment
```

`--no-live-speaker-assignment` keeps realtime text preview enabled, but disables live speaker scoring and highlighting. Final speaker labels still run on committed transcript rows.

## Public High-Quality Run

After the smoke provider works, use this public provider stack:

In the starter CLI, choose `Speaker provider quality` -> `Public high quality`, or run:

```powershell
.\.venv\Scripts\whospeaks.exe config --set provider_preset=public_quality
```

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms --realtime-preview-engine off
```

The live-speaker timing defaults are tuned to keep final ASR responsive while live feedback runs. The command does not need the older long list of `--live-speaker-*` timing flags.

## Optional Local Preview

The commands above disable local realtime preview with `--realtime-preview-engine off`. Remove that flag only after the local RealtimeSTT/Kroko preview environment is installed and working.

For a German realtime session with Kroko preview enabled, add `--language de` and keep the default `community-64l` preview preset:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --language de --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms
```

This selects `Kroko-DE-Community-64-L-Streaming-001.data` for preview, sends `de` to the final faster-whisper server, and uses German stream2sentence sentence splitting.
If the German Kroko model is not present locally, the app downloads it from Hugging Face before starting realtime preview.

## First Session

1. Start the command.
2. Open the browser URL printed by the process.
3. Load or replay the target media.
4. Press Start in the browser UI.
5. Watch the live speaker tag in the speaker panel for fast feedback.
6. Watch the live transcript for the current sentence view.
7. Let the clip finish if you want a complete speaker library.
8. Save or export the speaker group if you want to reuse speakers later.

## What To Expect

The app has two speaker assignment layers:

- The live layer updates quickly from recent audio windows.
- The final layer assigns speaker IDs to completed sentences.

The live layer may change faster than the final transcript. The final transcript is intentionally more conservative because it can use sentence boundaries, longer audio, and speaker-memory updates.

## Next Steps

- Learn the full browser workflow in [Live Window Workflow](live-window-workflow.md).
- Reuse speakers with [Speaker Libraries](speaker-libraries.md).
- Tune behavior with [Configuration](configuration.md).
- Validate changes with [Validation And Scoring](validation-and-scoring.md).

# Quickstart

Start with a small smoke run, then move to the tuned provider stacks after the remote services are healthy.

## Before You Start

Complete:

1. [Installation](installation.md) on the Windows controller.
2. [External Servers](external-servers.md) on the Linux GPU server.

Verify from Windows:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Replace `YOUR_GPU_SERVER_IP` with your Linux GPU server address.

## First End-To-End Run

Use a smoke provider first. This proves the UI, media loading, ASR route, embedding route, and speaker assignment pipeline work:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

The command prints a browser URL. Open it, load or replay media, then press Start.

## Public High-Quality Run

After the smoke provider works, use this public provider stack:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms --realtime-preview-engine off
```

This avoids `jungjee_rawnet3`, which needs an extra RawNet3 artifact that is not included in the public source snapshot.

## Tuned Best Run

Use this when the embeddings server has the `jungjee_rawnet3` artifact provisioned:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms --realtime-preview-engine off
```

The live-speaker timing defaults are tuned for fast feedback, so the command does not need the older long list of `--live-speaker-*` timing flags.

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

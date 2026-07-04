# Quickstart

Start the browser app, open the displayed URL, load media, and watch the right-hand speaker panel plus the live transcript while the final transcript is built.

## Recommended GPU Server Launch

Use this when ASR and embeddings are already running on a remote Linux GPU server:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --port 8796 --asr-backend remote --remote-asr-url http://192.168.178.22:8650 --embeddings-backend remote --remote-embeddings-url http://192.168.178.22:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50"
```

The live-speaker timing defaults are tuned for fast feedback, so the command does not need the older long list of `--live-speaker-*` timing flags.

## Minimal Local Launch

Use this when the local machine has the required ASR and embedding dependencies:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --port 8796
```

If the port is already in use, choose another port:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --port 8797
```

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
- Run GPU services with [External Servers](external-servers.md).
- Tune behavior with [Configuration](configuration.md).

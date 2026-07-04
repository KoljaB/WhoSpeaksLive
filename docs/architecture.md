# Architecture

WhoSpeaksLive is a browser-controlled Python pipeline that keeps fast live feedback separate from more stable final sentence assignment.

## High-Level Flow

```text
media playback
  -> audio capture / replay
  -> realtime preview and live speaker probes
  -> ASR growing-window transcription
  -> sentence boundary decisions
  -> speaker embeddings
  -> speaker memory and refinement
  -> browser events and transcript UI
```

## Main Components

- `src/window/youtube_window_diarize_gui.py`: CLI, HTTP server, API routes, and application wiring.
- `src/window/window_diarizer.py`: main diarization controller and growing-window loop.
- `src/window/window_gui_html.py`: browser UI HTML, CSS, and JavaScript.
- `src/window/window_remote_asr.py`: remote ASR client.
- `src/embeddings/embedding_providers.py`: local and remote embedding clients.
- `src/speakers/`: speaker memory and clustering logic.
- `src/window/*_scoring.py`: validation scoring helpers.
- `vendor/remote_servers/`: copied FastAPI server snapshots for remote ASR and embeddings.

## Final ASR Loop

The final loop advances through media with a growing window. Each pass sends the current window to ASR and accepts complete sentences when boundaries are stable enough.

After a successful sentence split, the loop emits all accepted sentences, advances the left edge, updates the realtime preview, and waits for the configured `--interval-seconds` cooldown before trying another final split.

## Live Speaker Path

The live path uses shorter audio windows and embedding probes. Its job is to update the active speaker indicator quickly.

Live speaker events feed:

- The Live tag in the speaker panel.
- The active sentence styling in the live transcript.
- Optional browser-observed validation samples.

The live transcript does not blindly use the last observed live speaker. It chooses the dominant speaker over the stable part of the active sentence so short pauses or late changes do not create noisy color flips.

## Speaker Memory

Speaker memory stores known profiles and decides whether a new embedding belongs to an existing speaker, an unknown speaker, or a new speaker.

There can be two compatible memories:

- Final memory, built from `--embedding-provider`.
- Live memory, built from `--live-speaker-embedding-provider` when that provider differs.

This separation is necessary because embeddings from different model families can live in different vector spaces.

## Browser Event Model

The Python process serves a browser UI and streams events. The browser updates transcript rows, speaker cards, live indicators, and validation observation state from those events.

Most user actions in the browser call JSON API routes, such as speaker rename, save, load, import, export, clear, and reference upload.

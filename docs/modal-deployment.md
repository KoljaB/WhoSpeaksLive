# Modal Deployment

Modal deployment is intended for running supported remote components away from the local UI machine.

## Current Scope

The repository contains Modal-related code for the window diarization flow:

- `src/window/modal_youtube_window_diarize_gui.py`
- `src/window/modal_asr_server.py`

This public document keeps only stable operational guidance.

## When Modal Helps

Use Modal when:

- You need GPU-backed ASR without managing a long-lived Linux desktop.
- You want reproducible remote startup for demos or experiments.
- You want to separate the local browser UI from compute-heavy model loading.

Use a persistent Linux GPU server instead when:

- You need very low latency every day.
- You already have warmed ASR and embedding services.
- You need direct control over model caches and GPU memory.

## Practical Workflow

1. Install and authenticate the Modal CLI according to Modal's own documentation.
2. Keep secrets and tokens in Modal secrets, not in source files.
3. Start from the Modal entry points under `src/window/`.
4. Validate health and latency with a small replay before running a full session.
5. Record the exact provider strings used for any scoring run.

## Local Fallback

If Modal startup is slow or a deployed function is cold, use the persistent remote server workflow in [External Servers](external-servers.md) for live demos.

# Overview

WhoSpeaksLive helps you see who is speaking while media is still playing, then produces a more stable speaker-labeled transcript after each sentence is finalized.

## What It Does

WhoSpeaksLive combines automatic speech recognition, speaker embeddings, speaker clustering, and a browser UI.

- Automatic speech recognition, or ASR, converts audio into words and timestamps.
- A speaker embedding is a numeric vector that represents the sound of a voice.
- Diarization is the process of assigning speech segments to speakers.
- A meeting **Speaker** is a local voice cluster. A persistent **Person** owns reusable Voice samples and may be suggested in later meetings after enough evidence accumulates.

The app is built for replaying videos, browser audio, and local media where you want both fast live feedback and a more reliable final transcript.

## Main Use Cases

- Watch a YouTube or media clip and see the likely active speaker in real time.
- Generate a transcript where each final sentence has a speaker label.
- Link recurring participants to People, keep a deliberate automatic-recognition roster, and confirm or reject cross-meeting suggestions.
- Import legacy Speaker-group files when compatibility with an older workflow is required.
- Compare diarization settings against a canonical transcript.
- Run expensive ASR and embedding models on a separate GPU server while keeping the UI local.

## What Makes This Project Different

The app separates two timing needs:

- Live speaker feedback should update quickly, even before a sentence is complete.
- Final speaker assignment should wait for enough audio and sentence context to be more stable.

Because those two jobs can use different embedding providers, the app can maintain separate live-speaker memory when needed. This avoids comparing live embeddings against profiles produced by a different model stack.

## Core Tools

- `whospeaks-window`: browser-based growing-window diarization for replayed media.
- `whospeaks-realtime`: realtime speaker diarization entry point.
- `whospeaks-filefeed-replay`: local filefeed replay support.
- `whospeaks-embedding-benchmark`: compare speaker embedding providers.
- `whospeaks-browser-live-eval`: drive the browser UI and score rendered live speaker output.

Command entry points are declared in `pyproject.toml` and call package modules directly.

## Recommended Setup

The practical install path is a Windows controller plus Linux GPU services:

- Install the Windows controller from [Installation](installation.md).
- Start ASR and embeddings from [External Servers](external-servers.md).
- Launch and verify the app from [Quickstart](quickstart.md).

This setup keeps the UI local while expensive model loading happens on the GPU server.

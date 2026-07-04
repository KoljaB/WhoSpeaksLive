# Overview

WhoSpeaksLive helps you see who is speaking while media is still playing, then produces a more stable speaker-labeled transcript after each sentence is finalized.

## What It Does

WhoSpeaksLive combines automatic speech recognition, speaker embeddings, speaker clustering, and a browser UI.

- Automatic speech recognition, or ASR, converts audio into words and timestamps.
- A speaker embedding is a numeric vector that represents the sound of a voice.
- Diarization is the process of assigning speech segments to speakers.
- A speaker library stores known speaker profiles so future sessions can identify them immediately.

The app is built for replaying videos, browser audio, and local media where you want both fast live feedback and a more reliable final transcript.

## Main Use Cases

- Watch a YouTube or media clip and see the likely active speaker in real time.
- Generate a transcript where each final sentence has a speaker label.
- Build reusable speaker groups from a complete run, then load them in later sessions.
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

Compatibility wrappers remain in `tools/` for older command lines.

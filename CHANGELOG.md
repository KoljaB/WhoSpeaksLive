# Changelog

## Unreleased

## 0.1.6 - 2026-07-23

- Prevented high ASR no-speech probabilities from silently deleting ordinary spoken text; conflicting evidence now keeps the transcript and adds a visible review warning.
- Restricted suppression to independently unconfirmed standalone high-risk hallucination phrases and persisted unresolved suppression notices for review.
- Isolated and prewarmed stateful Silero VAD instances for main transcription, realtime preview, and live speaker probing to prevent cross-thread state corruption and first-call latency spikes.
- Preserved ASR review warnings through transcript persistence, saved-session reloads, sentence merging, and speaker corrections.

## 0.1.5 - 2026-07-23

- Preserved word-level source and media timestamps through realtime transcript splitting.
- Added a configurable local embedding-helper timeout for cold model startup.
- Hardened authentic GUI evidence with final transcript DOM snapshots, media identity checks, and authoritative final speaker-profile state.
- Improved replay pacing so validation follows the requested wall-clock schedule without accumulating processing drift.
- Prevented weak Unknown sentences from being retroactively assigned to the sole existing speaker profile.
- Clarified media loading status so the browser reports cache progress once and a distinct player-ready result.

## 0.1.4 - 2026-07-22

- Prevented long startup and model-path status messages from expanding the speaker sidebar beyond the browser viewport.
- Added a focused layout regression contract for the sidebar's shrinkable grid columns and wrapped status output.

## 0.1.3 - 2026-07-22

- Kept live speaker cards, controls, and the active-speaker indicator inside the speaker-panel width when transcription starts.
- Prevented long executable paths, help text, and validation messages from widening or clipping the desktop launcher settings view.

## 0.1.0 - 2026-07-20

- Added a production CPU-only profile using Kroko/Nemotron fixed transcripts, guarded faster-whisper Base forced alignment, SpeechBrain CPU embeddings, bounded worker pools, native-timestamp fallback, launcher controls, diagnostics, and dedicated installation/docs.
- Made the Windows Kroko community-wheel build explicitly CPU-only and staged its downloaded ONNX Runtime DLL for reliable `delvewheel` packaging.
- Kept the base `whospeaks` launcher importable before NumPy and the CPU runtime extras are installed.
- Preserved unsaved launcher choices across background status checks, skipped redundant idle renders, and clarified which deployment profiles use local CPU, local GPU, or remote model servers.
- Added persistent People with Person-owned manual and meeting Voice samples, suggestion-first cross-meeting recognition, a deliberate recognition roster, and saved-session identity linking.
- Added evidence-gated enrollment, persistent recognition opt-out, truthful linked-state UI, durable suppression of deleted meeting samples, interruption recovery for saved link/unlink, and cascading Person/sample cleanup.
- Added a complete People workflow, legacy Speaker-group migration guide, and security/data-privacy documentation covering trusted deployment, storage, backup, retention, and deletion limits.
- Persisted the People library under `/data/speakers` in Docker and changed examples to publish the browser UI on host loopback by default.

## 0.0.3 - 2026-07-14

- Added grounded Ask sessions for live, saved, and multi-session transcripts, with full-transcript answers for short sessions and hybrid retrieval for longer scopes.
- Added managed Apple Silicon setup for local MLX ASR and MPS speaker-embedding services.
- Added a dedicated lightweight PyPI package-scope policy and tightened wheel and source-archive discovery to exclude repository-only tests, tools, documentation, and local artifacts.

## 0.0.2 - 2026-07-13

- Refactored the live application into explicit runtime, persistence, translation, reporting, and web-asset components.
- Added the full-screen setup application, expanded diagnostics, and reproducible launch planning.
- Added multilingual live translation, custom meeting reports, evidence links, and report templates.
- Added Nemotron and Kroko realtime-preview integration and delayed multi-row speaker clustering.
- Added a tracked RealtimeSTT warm-up asset contract and bundled third-party license notices.
- Expanded the Python regression suite and added JavaScript ownership/bootstrap tests.

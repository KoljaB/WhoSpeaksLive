# Changelog

## Unreleased

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

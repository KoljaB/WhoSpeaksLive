# Remote Server Copies

This folder contains the source snapshots used by the recommended WhoSpeaksLive setup.

- `faster-whisper-asr/`: FastAPI server for faster-whisper ASR on port `8650`.
- `voice-embeddings-server/`: FastAPI server for speaker embeddings on port `8660`.

These folders intentionally exclude virtual environments, logs, Python bytecode caches, test audio, model caches, and large model artifacts.

For full setup instructions, read:

- `../../docs/external-servers.md` from the repository root.
- `faster-whisper-asr/README.md`.
- `voice-embeddings-server/README.md`.

Typical commands after each server has a venv and dependencies:

```bash
cd faster-whisper-asr
.venv/bin/python -m uvicorn asr_server:app --host 0.0.0.0 --port 8650 --log-level info
```

```bash
cd voice-embeddings-server
.venv/bin/python -m uvicorn embeddings_server:app --host 0.0.0.0 --port 8660 --log-level info
```

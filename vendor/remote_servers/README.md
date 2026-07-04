# Remote Server Copies

Copied from the Linux worker at `192.168.178.22` on 2026-07-04.

- `faster-whisper-asr/` comes from `/home/lon/Dev/faster-whisper-asr`.
- `voice-embeddings-server/` comes from `/home/lon/Dev/voice-embeddings-server`.

The copy intentionally excludes runtime artifacts: virtual environments, logs,
Python bytecode caches, test audio, model caches, and the embeddings `hub/`
model artifact.

The ASR service was running as:

```bash
/home/lon/Dev/faster-whisper-asr/.venv/bin/python -m uvicorn asr_server:app --host 0.0.0.0 --port 8650 --log-level info
```

The embeddings service was running as:

```bash
/home/lon/Dev/voice-embeddings-server/.venv/bin/python -m uvicorn embeddings_server:app --host 0.0.0.0 --port 8660 --log-level info
```

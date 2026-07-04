# Session: copy remote servers

Date: 2026-07-04
Agent: Codex
Branch: master
Status: completed

## Goal

Copy the ASR and embeddings server code from the remote Linux server into this repository in separate folders.

## What changed

Added vendored copies of the remote ASR and embeddings service source trees under `vendor/remote_servers/`.
Runtime artifacts were excluded.

## Files changed

- `vendor/remote_servers/faster-whisper-asr/`: copied ASR service Python source, README, desktop control scripts, and existing local backup files from `/home/lon/Dev/faster-whisper-asr`.
- `vendor/remote_servers/voice-embeddings-server/`: copied embeddings service Python source, desktop control scripts, and benchmark tool from `/home/lon/Dev/voice-embeddings-server`.
- `vendor/remote_servers/README.md`: documents remote source paths, running commands, and excluded artifacts.
- `sessions/2026-07-04_1534_copy-remote-servers.md`: this handoff note.

## Commands run

- `python D:\Projekte\remote-codex_cli\remote-codex_cli\tools\codex_remote_cli.py --hello`: worker reachable at `192.168.178.22:8765`.
- Remote process inspection showed ASR on port `8650` from `/home/lon/Dev/faster-whisper-asr` and embeddings on port `8660` from `/home/lon/Dev/voice-embeddings-server`.
- Remote `find`/`ls` inspected source layouts and runtime artifacts.
- Remote `tar` with explicit excludes plus local extraction copied the source trees.
- Local `py_compile` succeeded for the copied Python files.
- Remote `sha256sum` matched local `Get-FileHash` for the main copied files.

## Findings

The ASR service is running as:

```bash
/home/lon/Dev/faster-whisper-asr/.venv/bin/python -m uvicorn asr_server:app --host 0.0.0.0 --port 8650 --log-level info
```

The embeddings service is running as:

```bash
/home/lon/Dev/voice-embeddings-server/.venv/bin/python -m uvicorn embeddings_server:app --host 0.0.0.0 --port 8660 --log-level info
```

The embeddings project contains a large `hub/` model artifact, which was intentionally not copied.

## Decisions made

Use `vendor/remote_servers/faster-whisper-asr/` and `vendor/remote_servers/voice-embeddings-server/` as the local separate folders for the copied server code.

## Problems / risks

The copied server code was syntax-checked but not executed locally.
The repository already had unrelated uncommitted changes before this copy.

## Next recommended step

Review whether these vendored server copies should be wired into the local app, documented further, or committed as reference snapshots only.

## Git status summary

Working tree is dirty. New files from this task are under `vendor/remote_servers/` and `sessions/`.
Existing modified files include docs and `src/window/*` changes from prior work.

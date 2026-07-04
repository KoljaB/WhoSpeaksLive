# External ASR And Embeddings Servers

External servers move expensive ASR and embedding work to a GPU machine while the browser UI and orchestration stay local.

## When To Use Remote Servers

Use remote ASR and embeddings when:

- Your local machine does not have enough GPU memory.
- You want the browser UI to stay responsive.
- You want to run larger models such as faster-whisper `large-v2`.
- You want several embedding providers loaded on one GPU host.

## Current Server Snapshot

Reference copies of the remote server code live in:

- `vendor/remote_servers/faster-whisper-asr/`
- `vendor/remote_servers/voice-embeddings-server/`

Those folders are source snapshots. They intentionally exclude virtual environments, logs, model caches, test audio, and large model artifacts.

## ASR Server

The ASR server exposes faster-whisper through FastAPI.

Common routes:

- `GET /health`
- `POST /transcribe`
- `POST /transcribe-memory`
- `POST /transcribe-pcm16`
- `POST /transcribe-window`
- `POST /transcribe-file`
- `POST /v1/audio/transcriptions`

The copied server defaults to port `8650` and model `large-v2`.

Example run command on the Linux GPU host:

```bash
cd /home/lon/Dev/faster-whisper-asr
.venv/bin/python -m uvicorn asr_server:app --host 0.0.0.0 --port 8650 --log-level info
```

Health check:

```powershell
curl.exe http://192.168.178.22:8650/health
```

## Embeddings Server

The embeddings server exposes voice embedding providers through FastAPI.

Common routes:

- `GET /health`
- `GET /providers`
- `POST /load`
- `POST /unload`
- `POST /embed`
- `POST /embed-pcm16`
- `POST /embed-window`

The copied server defaults to port `8660`.

Example run command on the Linux GPU host:

```bash
cd /home/lon/Dev/voice-embeddings-server
.venv/bin/python -m uvicorn embeddings_server:app --host 0.0.0.0 --port 8660 --log-level info
```

Health check:

```powershell
curl.exe http://192.168.178.22:8660/health
```

List providers:

```powershell
curl.exe http://192.168.178.22:8660/providers
```

## Local App Command

Point the local app at both remote services:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --port 8796 --asr-backend remote --remote-asr-url http://192.168.178.22:8650 --embeddings-backend remote --remote-embeddings-url http://192.168.178.22:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50"
```

## Provider Stacks

The embeddings server accepts a provider stack string:

```text
provider_a=weight+provider_b=weight
```

The server embeds the same audio with each provider, scales each normalized vector by its weight, concatenates the components, and normalizes the result.

Example:

```text
espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

Use the same provider string for comparable final speaker profiles. Use a separate live provider only when you want a faster live-speaker path and understand that it needs separate live memory.

# Voice Embeddings Server

This service exposes speaker embedding providers through FastAPI for WhoSpeaksLive final and live speaker assignment.

## Setup

```bash
cd vendor/remote_servers/voice-embeddings-server
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch and torchaudio for your CUDA runtime. For CUDA 12.1 wheels:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio
```

Install the server and provider dependencies:

```bash
python -m pip install -r requirements.txt
```

Set a Hugging Face token if you use gated pyannote providers:

```bash
export HF_TOKEN=hf_your_token_here
```

## Run

```bash
export EMBEDDINGS_DEVICE=auto
python -m uvicorn embeddings_server:app --host 0.0.0.0 --port 8660 --log-level info
```

The server coordinates individual providers across all incoming requests. It
coalesces identical provider/audio work that is already in flight and keeps a
small result cache for immediately repeated live/final requests. The relevant
settings are:

```bash
export EMBEDDINGS_COMPONENT_CONCURRENCY=2
export EMBEDDINGS_RESULT_CACHE_TTL_SECONDS=3
export EMBEDDINGS_RESULT_CACHE_MAX_ENTRIES=128
```

Two component workers are the measured RTX 4090 setting used by the Linux
WhoSpeaks server. Higher values caused latency spikes with the current provider
mix. A matching systemd drop-in is included as
`voice-embeddings-server-concurrency.conf`.

## Health Checks

From the Windows controller:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Load a simple smoke provider first:

```powershell
curl.exe -X POST "http://YOUR_GPU_SERVER_IP:8660/load?provider=speechbrain_ecapa&device=auto"
```

## Provider Stacks

Smoke:

```text
speechbrain_ecapa
```

Public high quality:

```text
espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

Tuned best:

```text
espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

Fast live:

```text
pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50
```

`jungjee_rawnet3` requires RawNet3-compatible source/model files under `.cache/source/RawNet/python/RawNet3`. That artifact is not included in the public source snapshot. Use the public high-quality stack until it is provisioned.

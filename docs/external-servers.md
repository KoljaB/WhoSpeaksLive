# External ASR And Embeddings Servers

External servers run the expensive ASR and embedding models on a Linux GPU machine while the Windows controller runs the browser UI.

## What You Need

Use a Linux machine with:

- NVIDIA GPU and working NVIDIA driver.
- Python 3.11.
- Git.
- Enough disk space for model caches.
- Network access from the Windows controller.

The examples use two HTTP services:

- ASR server: `http://YOUR_GPU_SERVER_IP:8650`
- Embeddings server: `http://YOUR_GPU_SERVER_IP:8660`

These plain-HTTP examples assume a trusted private network. Speech audio sent to them is not encrypted and the services do not become safe merely because they are self-hosted. Use firewall rules plus a VPN or authenticated TLS proxy outside that threat model, and never expose these ports directly to the public internet. See [Security And Data Privacy](security-and-data-privacy.md#data-flow).

Find the Linux server IP:

```bash
hostname -I
```

On Windows, replace `YOUR_GPU_SERVER_IP` with that address.

On Ubuntu, the base packages are usually:

```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv python3.11-dev ffmpeg
```

If `ufw` is enabled, allow the two service ports:

```bash
sudo ufw allow 8650/tcp
sudo ufw allow 8660/tcp
```

## Server Source Folders

The repository contains source snapshots for both services:

- `vendor/remote_servers/faster-whisper-asr/`
- `vendor/remote_servers/voice-embeddings-server/`

You can clone the full repository on the Linux server, or copy only `vendor/remote_servers/` to the Linux server.

## ASR Server Setup

From the Linux checkout:

```bash
cd vendor/remote_servers/faster-whisper-asr
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The ASR server uses faster-whisper. By default it expects `large-v2`, CUDA, and `float16`.
The Windows controller sends the selected `--language` with each request, so one ASR server can serve all supported faster-whisper languages. Set `ASR_LANGUAGE=de` only if you want the server's own default language to be German when a request omits `language`.

For a first run that can download the model if it is not already cached:

```bash
export ASR_MODEL=large-v2
export ASR_DEVICE=cuda
export ASR_COMPUTE_TYPE=float16
export ASR_LOCAL_FILES_ONLY=0
python -m uvicorn asr_server:app --host 0.0.0.0 --port 8650 --log-level info
```

After the model is cached, you can use `ASR_LOCAL_FILES_ONLY=1` to avoid network use during startup.

From Windows, verify:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
```

Expected result: JSON with `"ok":true`, `"model":"large-v2"`, and route names such as `/transcribe-window`.

## Embeddings Server Setup

From the Linux checkout:

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

Then install the server/provider dependencies:

```bash
python -m pip install -r requirements.txt
```

Some providers download gated Hugging Face models. If you use pyannote-based providers, set a token that has accepted the relevant model terms:

```bash
export HF_TOKEN=hf_your_token_here
```

Start the server:

```bash
export EMBEDDINGS_DEVICE=auto
python -m uvicorn embeddings_server:app --host 0.0.0.0 --port 8660 --log-level info
```

From Windows, verify:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Expected result: `/health` returns JSON with `"ok":true`; `/providers` returns supported provider IDs.

## Provider Readiness Checks

Start with a simple public provider:

```powershell
curl.exe -X POST "http://YOUR_GPU_SERVER_IP:8660/load?provider=speechbrain_ecapa&device=auto"
```

Then try the fast live-provider stack:

```powershell
curl.exe -X POST "http://YOUR_GPU_SERVER_IP:8660/load?provider=pyannote_wespeaker_resnet34_lm%3D1.0%2Bwespeaker_resnet34_lm_onnx%3D0.50&device=auto"
```

The `%3D` and `%2B` are URL-encoded `=` and `+`.

## Provider Stack Levels

Use these stacks in order while bringing up a new server:

| Level | Provider string | Notes |
| --- | --- | --- |
| Smoke | `speechbrain_ecapa` | Good first test. Smaller and easier to load. |
| Public high quality | `espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12` | Recommended public quality stack. |
| Fast live | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Recommended for live speaker assignment. |

## Windows Launch With Remote Services

After both health checks pass:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

Then switch to the public high-quality command in [Quickstart](quickstart.md).

## Keeping Servers Running

For a temporary terminal session, keep the `uvicorn` commands open.

For daily use, run each service under `systemd`, `tmux`, `screen`, or your desktop autostart system. The copied `desktop-controls/` folders contain optional `systemctl --user` start/stop helpers. They assume service names `faster-whisper-asr.service` and `voice-embeddings-server.service`; override them with `WHOSPEAKS_ASR_SERVICE` or `WHOSPEAKS_EMBEDDINGS_SERVICE` if your unit names differ.

## Common Failures

- `model_not_loaded`: the server process started but model loading failed or has not finished.
- Connection refused: the service is not running, the port is blocked, or `--host` is not `0.0.0.0`.
- CUDA out of memory: unload other models, reduce providers, or use the smoke provider first.
- Hugging Face authorization errors: set `HF_TOKEN` and accept gated model terms.
- Provider not found: use the smoke provider first, then switch to the public high-quality stack after dependencies load successfully.

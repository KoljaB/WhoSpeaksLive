# faster-whisper ASR Server

This service exposes faster-whisper through FastAPI for WhoSpeaksLive final-window transcription.

## Setup

```bash
cd vendor/remote_servers/faster-whisper-asr
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## First Run

Allow model download on first startup:

```bash
export ASR_MODEL=large-v2
export ASR_DEVICE=cuda
export ASR_COMPUTE_TYPE=float16
export ASR_LOCAL_FILES_ONLY=0
python -m uvicorn asr_server:app --host 0.0.0.0 --port 8650 --log-level info
```

After the model is cached, set `ASR_LOCAL_FILES_ONLY=1` if you want startup to fail instead of downloading.

## Health Check

From the Windows controller:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
```

Expected result: JSON with `"ok":true`, `"model":"large-v2"`, and route names.

## Endpoints

- `GET /health`
- `POST /transcribe`: encoded-audio route, decoded in memory.
- `POST /transcribe-memory`: explicit encoded-audio memory route.
- `POST /transcribe-pcm16`: raw mono 16 kHz PCM16 or float32.
- `POST /transcribe-window`: WhoSpeaksLive raw mono 16 kHz float32 windows.
- `POST /transcribe-file`: temp-file route for comparison.
- `POST /v1/audio/transcriptions`: OpenAI-style transcription route.

## WhoSpeaksLive Route

`/transcribe-window` defaults match the local faster-whisper call used by `whospeaks-window`:

```text
language=en
task=transcribe
beam_size=5
word_timestamps=true
vad_filter=false
condition_on_previous_text=false
```

Example:

```bash
curl --data-binary @window.f32le \
  -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8650/transcribe-window?sample_rate=16000&encoding=float32'
```

The response includes `segments` and a top-level `words` list with `word`, `text`, `start`, `end`, and `probability`.

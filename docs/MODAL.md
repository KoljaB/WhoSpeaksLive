# Modal Deployment Notes

WhoSpeaks currently uses Modal for the remote ASR backend: the browser UI runs
normally, but final transcription windows are sent to a GPU-backed Modal HTTP
service.

## Current Endpoint

- App name: `whospeaks-live-asr`
- Web endpoint: `https://lonligrin--whospeaks-live-asr.modal.run`
- Source file: `src/whospeaks/window/modal_asr_server.py`
- Model: faster-whisper `large-v2`
- Runtime: CUDA 12.4, Python 3.11, Modal T4 GPU
- Cache volume: `whospeaks-faster-whisper-cache`

The service exposes:

- `GET /health`
- `POST /transcribe-window?sample_rate=16000&encoding=float32`

`POST /transcribe-window` accepts raw little-endian `float32` mono PCM audio and
returns faster-whisper segments, word timestamps, and timing metadata.

## Verified Command

Deploy or redeploy the ASR service:

```powershell
.\.venv\Scripts\python.exe -m modal deploy --strategy recreate --name whospeaks-live-asr src\whospeaks\window\modal_asr_server.py
```

Run the current UI against the Modal ASR endpoint:

```powershell
.\.venv\Scripts\python.exe -m whospeaks.window.youtube_gui `
  --host 127.0.0.1 `
  --port 8796 `
  --no-browser `
  --asr-backend remote `
  --remote-asr-url https://lonligrin--whospeaks-live-asr.modal.run `
  --remote-asr-timeout-seconds 240 `
  --skip-download `
  --work-dir runtime\media\local-filefeed `
  --no-startup-warmup-before-url
```

## Browser Verification

Verified on 2026-07-02 with the in-app Browser at `http://127.0.0.1:8796/`.

The deployed health endpoint also returned:

```json
{"ok":true,"service":"whospeaks-live-asr","model":"large-v2","device":"cuda","compute_type":"float16","model_loaded":true}
```

Observed results:

- Start transcription launched the YouTube sample run.
- The UI entered the active `Diarizing` state.
- The transcript showed final accepted sentences with speaker names, time ranges,
  speech/audio rates, and probability bars.
- Live transcript rows appeared while the run was still in progress.
- Speaker assignment worked: the UI showed 5 detected speakers.
- Speaker totals updated, for example `Speaker 1` with 23 sentences and `1:12`
  total speaking time.
- New speaker detection worked: `Speaker 5` was created during the run.
- Reassignment worked in logs, including unknown sentences reassigned to known
  speakers after stronger speaker references were available.

Representative local verification log lines:

```text
Remote faster-whisper large-v2 ASR ready
Transcribed 8.50s window in 2.03s; segments=2 words=33 accepted=1
Embedded sentence 15 in 0.05s; speaker=S3 new=0 unk=0.0669 top=0.7729 margin=0.5348
Embedded sentence 20 in 0.03s; speaker=S4 new=1 unk=0.0 top=0.0592 margin=0.0501
Reassigned unknown sentence 17 to S1 (sim=0.3338, margin=0.2005)
Embedded sentence 28 in 0.06s; speaker=S5 new=1 unk=0.0 top=0.3187 margin=0.1368
```

## Problems And Fixes

| Problem | Symptom | Fix |
| --- | --- | --- |
| FastAPI treated the request body parameter as a missing query parameter. | `/transcribe-window` returned HTTP 422 with `loc=["query","request"]`. | Removed postponed annotations from the Modal ASR module so `request: Request` resolves to the actual FastAPI `Request` class at route registration time. |
| Modal kept serving an older function object during route testing. | The 422 persisted after the local source was changed. | Redeployed with `--strategy recreate` so existing containers were terminated and rebuilt. |
| The first image did not include CUDA runtime libraries. | `/transcribe-window` returned HTTP 500 with `RuntimeError: Library libcublas.so.12 is not found`. | Switched from Debian slim to `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with Python 3.11. |
| Starting the local verification server with PowerShell `Start-Process` hit duplicate environment keys. | Windows reported duplicate `PATH`/`Path` entries. | Started the verification server with Python `subprocess.Popen` and a sanitized environment. |
| Browser media playback initially needed a user gesture. | The status log showed `audio playback blocked: NotAllowedError` at startup. | Browser interaction still started the run; backend warmup and transcript processing proceeded after the user-driven start action. |
| The historical full Modal GUI app was not immediately schedulable. | The old app `whospeaks-youtube-window-diarize` waited for GPU capacity. | Verified the current repo through the local UI against the new Modal ASR endpoint. The old full-GUI wrapper remains useful for a future port. |

## Old Full-GUI Modal App

The previous repo at `D:\Projekte\WhoSpeaks` contains
`tools\modal_youtube_window_diarize_gui.py`, a full Modal web-server wrapper for
the older GUI. It deployed app `whospeaks-youtube-window-diarize` and ran the
whole UI inside Modal.

Important details from that wrapper:

- Base image: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`
- Runtime packages: ffmpeg, git, curl, build tools, PortAudio, libsndfile
- Python packages: torch/torchaudio CUDA wheels, faster-whisper, RealtimeSTT,
  stream2sentence, yt-dlp, onnxruntime-gpu, ESPnet and speaker-model packages
- Persistent volume: `whospeaks-youtube-window-diarize-cache`
- Modal primitive: `@modal.web_server(PORT, startup_timeout=20 * 60)`
- GPU: `gpu="any"`

That wrapper has not yet been ported to the cleaned `src/whospeaks` layout. If
we need the entire current UI hosted on Modal, port this file next instead of
starting from scratch.

## Security And Repo Hygiene

- Do not upload `.env` to Modal images.
- Use Modal secrets for private tokens if future deployment steps need them.
- Keep local media, generated transcripts, model caches, and verification logs
  under `runtime/` so they remain untracked.

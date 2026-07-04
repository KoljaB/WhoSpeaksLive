# Modal Deployment Notes

WhoSpeaks now has two Modal deployments: the public full browser UI and the
remote ASR service used for smaller split deployments.

## Public Full UI

- App name: `whospeaks-youtube-window-diarize`
- Public URL: `https://lonligrin--whospeaks-youtube-window-diarize-youtube-wind-4fd2fa.modal.run`
- Source file: `src/whospeaks/window/modal_youtube_window_diarize_gui.py`
- Runtime: CUDA 12.6, Python 3.11, Modal T4 GPU by default
- Cache volume: `whospeaks-youtube-window-diarize-cache`
- Entrypoint inside Modal: `python -m whospeaks.window.youtube_gui`
- Default embedding stack:
  `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50`
- Default live speaker tuning matches the best measured browser-live run:
  0.20s probe cadence, EMA count 1, raw-change snapping, and sentence hints.

The full UI image bakes the cleaned `src/`, `vendor/`, `tools/`, test fixtures,
and the local Philomena Cunk media cache into the image. Runtime models, speaker
libraries, generated outputs, and mutable media are stored under `/cache`.

Deploy:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m modal deploy --strategy recreate --name whospeaks-youtube-window-diarize src\whospeaks\window\modal_youtube_window_diarize_gui.py
```

Cost controls:

- The full UI wrapper defaults to `WHOSPEAKS_MODAL_GPU=T4`.
- Override only when needed, for example
  `$env:WHOSPEAKS_MODAL_GPU='L4'` before deploy.
- Idle containers scale down after `60` seconds by default. Override with
  `WHOSPEAKS_MODAL_SCALEDOWN_WINDOW_SECONDS` if a longer warm container window
  is worth the extra spend.
- Stop a running public UI container immediately with:

```powershell
.\.venv\Scripts\python.exe -m modal app stop --yes whospeaks-youtube-window-diarize
```

## Remote ASR Endpoint

- App name: `whospeaks-live-asr`
- Web endpoint: `https://lonligrin--whospeaks-live-asr.modal.run`
- Source file: `src/whospeaks/window/modal_asr_server.py`
- Model: faster-whisper `large-v2`
- Runtime: CUDA 12.4, Python 3.11, Modal T4 GPU
- Cache volume: `whospeaks-faster-whisper-cache`

The service exposes:

- `GET /health`
- `POST /transcribe-window?sample_rate=16000&encoding=float32`

## Public Browser Verification

Verified on 2026-07-02 with the in-app Browser against the public URL:

```text
https://lonligrin--whospeaks-youtube-window-diarize-youtube-wind-4fd2fa.modal.run
```

Visible browser state from the public page:

- Header showed `Transcribing`.
- Playback was active at `01:47 / 05:03`.
- Live transcript area showed final sentence rows with speaker labels, time
  ranges, speech/audio rates, and probability summaries.
- A live row was visible: `Speaker 3 Live`, with realtime text still updating.
- Speaker panel showed `Detected speakers (4)`.
- Speaker assignment was visible for `Speaker 1`, `Speaker 2`, `Speaker 3`, and
  `Speaker 4`.
- Speaker totals were visible, for example `Speaker 1 15 sentences · 51.4s`.
- Status log showed model warmup, realtime preview startup, ASR window
  transcription, and accepted sentence counts.

Representative visible transcript rows:

```text
Speaker 1: Which was more culturally significant, the Renaissance or Single Ladies by Beyoncé?
Speaker 2: They both have their period.
Speaker 3: I don't think they had many homeless people in ancient Egypt.
Speaker 3 Live: You got your dead body, and you light out on a table...
```

Representative Modal log lines from the same public run:

```text
Realtime preview ready in 3.82s (kroko_onnx).
Growing-window transcription started (continuous, min playback advance 0.75s).
Transcribed 7.78s window in 0.35s; segments=1 words=12 accepted=1.
Embedded sentence 0 in 0.02s; speaker=S1 new=1 unk=0.0 top=1.0 margin=1.0.
Transcribed 6.73s window in 0.49s; segments=2 words=16 accepted=2.
Embedded sentence 1 in 0.03s; speaker=S2 new=1 unk=0.0 top=0.0792 margin=1.0.
```

## Earlier ASR-Only Verification

The remote ASR endpoint was also verified from the local UI on 2026-07-02. Its
health endpoint returned:

```json
{"ok":true,"service":"whospeaks-live-asr","model":"large-v2","device":"cuda","compute_type":"float16","model_loaded":true}
```

## Problems And Fixes

| Problem | Symptom | Fix |
| --- | --- | --- |
| FastAPI treated the ASR request body parameter as a missing query parameter. | `/transcribe-window` returned HTTP 422 with `loc=["query","request"]`. | Removed postponed annotations from the Modal ASR module so `request: Request` resolves to the actual FastAPI `Request` class at route registration time. |
| Modal kept serving an older ASR function object during route testing. | The 422 persisted after the local source was changed. | Redeployed with `--strategy recreate` so existing containers were terminated and rebuilt. |
| The first ASR image did not include CUDA runtime libraries. | `/transcribe-window` returned HTTP 500 with `RuntimeError: Library libcublas.so.12 is not found`. | Switched from Debian slim to `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with Python 3.11. |
| The old full-UI wrapper used the pre-restructure `tools/` layout. | It could not represent the cleaned `src/whospeaks` project. | Added `modal_youtube_window_diarize_gui.py`, which bakes `src/`, `vendor/`, `tools/`, fixtures, and cached demo media, then launches `python -m whospeaks.window.youtube_gui`. |
| Modal imports the deployment file from `/root/modal_youtube_window_diarize_gui.py` inside the container. | The first full-UI container crashed with `IndexError: 3` from `Path(__file__).parents[3]`. | Added `_local_root()` so the wrapper resolves either the local checkout during deploy or `/root/WhoSpeaksLive` inside Modal. |
| Windows could not print Modal build progress characters. | The deploy command failed locally with a `charmap` encoding error. | Forced UTF-8 for deploys with `$env:PYTHONIOENCODING='utf-8'`. |
| The first fixed full-UI redeploy hit the local command timeout while building Kroko. | Modal built the expensive `kroko-onnx` wheel but the local deploy process exited before creating the app objects. | Reran the deploy after the image layer was cached; the second deploy completed in about 10 seconds. |
| Browser automation click calls timed out while the public page was busy. | Playwright and coordinate click calls reset the browser-control session even though the click reached the app. | Confirmed delivery via Modal logs, recovered the live public tab, and verified visible DOM state from the same running page. |
| Browser media playback initially needed a user gesture. | Status log showed `audio playback blocked: NotAllowedError`. | The explicit browser Start interaction still triggered backend warmup and synchronized playback; the public page then advanced and produced live/final transcript rows. |
| The full public UI could keep a GPU web-server container alive while the page was open or shortly after testing. | Modal usage dropped to only a few dollars of remaining workspace credit. | Stopped the active container and app, changed the full-UI default from `gpu="any"` to T4, and reduced the default idle scaledown window from 10 minutes to 60 seconds. |

## Security And Repo Hygiene

- Do not upload `.env` to Modal images.
- Use Modal secrets for private tokens if future deployment steps need them.
- Keep local media, generated transcripts, model caches, and verification logs
  under `runtime/` so they remain untracked.

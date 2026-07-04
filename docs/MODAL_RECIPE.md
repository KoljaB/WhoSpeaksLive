# Modal Recipe

This recipe deploys the public WhoSpeaks browser UI to Modal and verifies that
the public URL works end to end.

## 1. Install The Project

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Make sure Modal is installed and authenticated:

```powershell
.\.venv\Scripts\python.exe -m modal token set
```

## 2. Deploy The Public UI

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m modal deploy --strategy recreate --name whospeaks-youtube-window-diarize src\whospeaks\window\modal_youtube_window_diarize_gui.py
```

The deployment defaults to `WHOSPEAKS_MODAL_GPU=T4` and a 60-second idle
scaledown window. It also defaults to the best measured live speaker stack:

```text
pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50
```

To test a stronger GPU only when needed:

```powershell
$env:WHOSPEAKS_MODAL_GPU='L4'
```

Open:

```text
https://lonligrin--whospeaks-youtube-window-diarize-youtube-wind-4fd2fa.modal.run
```

## 3. Verify The Public URL

1. Open the public URL in a browser.
2. Click `Start transcription`.
3. Wait for model warmup on first start.
4. Confirm the header changes to `Transcribing`.
5. Confirm realtime text appears as a `Live` transcript row.
6. Confirm final transcript rows appear with speaker names, timestamps, and
   probability summaries.
7. Confirm the speaker panel shows detected speakers and sentence/time totals.
8. Confirm at least one speaker is marked `Live` while audio is playing.

## 4. Optional Remote ASR Service

For a split deployment where the UI runs elsewhere and only final ASR windows
run on Modal:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m modal deploy --strategy recreate --name whospeaks-live-asr src\whospeaks\window\modal_asr_server.py
```

Endpoint:

```text
https://lonligrin--whospeaks-live-asr.modal.run
```

Health check:

```powershell
curl.exe -sS https://lonligrin--whospeaks-live-asr.modal.run/health
```

## 5. Troubleshooting

- Use `$env:PYTHONIOENCODING='utf-8'` before deploy on Windows.
- Stop the full public UI app when done testing so no GPU web-server container
  remains active:

```powershell
.\.venv\Scripts\python.exe -m modal app stop --yes whospeaks-youtube-window-diarize
```

- Use `--strategy recreate` after changing route definitions, image packages,
  or web-server startup behavior.
- First full-UI deploys are slow because Kroko is built into the image.
- First public page loads may wait for Modal GPU scheduling and model warmup.
- `libcublas.so.12 is not found` means the image does not include CUDA runtime
  libraries.
- HTTP 422 on `/transcribe-window` usually means FastAPI did not resolve the
  `Request` annotation correctly.

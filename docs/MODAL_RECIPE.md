# Modal Recipe

This recipe deploys the WhoSpeaks remote ASR backend to Modal and points the
local UI at that GPU service.

## 1. Install The Project

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Make sure Modal is installed and authenticated:

```powershell
.\.venv\Scripts\python.exe -m modal token set
```

## 2. Deploy Remote ASR

```powershell
.\.venv\Scripts\python.exe -m modal deploy --strategy recreate --name whospeaks-live-asr src\whospeaks\window\modal_asr_server.py
```

The endpoint should be:

```text
https://lonligrin--whospeaks-live-asr.modal.run
```

Check health:

```powershell
curl.exe -sS https://lonligrin--whospeaks-live-asr.modal.run/health
```

## 3. Start The UI Against Modal

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

Open:

```text
http://127.0.0.1:8796/
```

## 4. Verify End To End

1. Click `Start transcription`.
2. Wait for Modal warmup on first start.
3. Confirm the header changes to active transcription or diarization.
4. Confirm final transcript rows appear with speaker labels and timestamps.
5. Confirm the speaker panel creates and updates detected speakers.
6. Confirm at least one new speaker or reassignment appears in the status log
   on the Philomena Cunk sample video.

## 5. Troubleshooting

- HTTP 422 on `/transcribe-window` usually means FastAPI did not resolve the
  `Request` annotation correctly.
- `libcublas.so.12 is not found` means the image does not include CUDA runtime
  libraries.
- First Modal starts can take around two minutes while the image starts and the
  model cache is warmed.
- Use `--strategy recreate` after changing route definitions or image packages.

## 6. Future Full-UI Modal Hosting

For hosting the entire UI on Modal, port the old wrapper from:

```text
D:\Projekte\WhoSpeaks\tools\modal_youtube_window_diarize_gui.py
```

That file already contains the full Modal web-server shape, CUDA image setup,
cache volume policy, and long-running GPU worker settings.

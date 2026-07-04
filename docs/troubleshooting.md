# Troubleshooting

Most setup failures are caused by missing local dependencies, remote services not listening, model downloads not finished, or provider artifacts that are not installed.

## Start With The Health Checklist

From the Windows controller:

```powershell
.\.venv\Scripts\whospeaks-window.exe --help
ffmpeg -version
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

If one of these fails, fix it before launching the browser app.

## PowerShell Cannot Find The Command

If `.\.venv\Scripts\whospeaks-window.exe` does not exist, reinstall the controller package:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-controller.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

If `.venv` does not exist, follow [Installation](installation.md) from the beginning.

## ffmpeg Is Missing

Media download and decoding can fail if `ffmpeg` is not on `PATH`.

Install it, open a new PowerShell window, then verify:

```powershell
ffmpeg -version
```

## The Browser URL Does Not Appear

Check whether model loading is still in progress. Large embedding stacks can take a while on first startup.

For the first run, prefer the smoke command from [Quickstart](quickstart.md):

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

If using local embeddings, increase the helper timeout:

```powershell
.\.venv\Scripts\whospeaks-window.exe --embedding-helper-response-timeout-seconds 900
```

## Remote ASR Fails

Confirm the local command includes:

```text
--asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650
```

Check the ASR route:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
```

If health fails:

- Make sure the Linux command used `--host 0.0.0.0 --port 8650`.
- Check that the Linux firewall allows port `8650`.
- Check that `ASR_LOCAL_FILES_ONLY=0` is set for the first model download, or that the model is already cached.
- Check the server terminal for faster-whisper or CUDA errors.

## Remote Embeddings Fail

Confirm the local command includes:

```text
--embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660
```

Check health and provider list:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Load the smoke provider:

```powershell
curl.exe -X POST "http://YOUR_GPU_SERVER_IP:8660/load?provider=speechbrain_ecapa&device=auto"
```

If provider loading fails:

- Check the Linux server terminal for the real Python exception.
- Use `speechbrain_ecapa` before trying provider stacks.
- Set `HF_TOKEN` for pyannote providers.
- Use the public high-quality stack if `jungjee_rawnet3` is missing.
- Restart the embeddings process after changing Python packages.

## CUDA Out Of Memory

Start smaller:

1. Load only `speechbrain_ecapa`.
2. Stop other GPU processes.
3. Start ASR and embeddings in separate terminals so you can see which process uses VRAM.
4. Move to provider stacks only after the smoke provider works.

## Loaded Speakers Do Not Assign Immediately

Check whether final and live providers changed.

If `--embedding-provider` differs from `--live-speaker-embedding-provider`, the app needs live profiles made with the live provider. A speaker group saved before that live provider existed may still work for final sentences but may not provide immediate live assignment.

Save a fresh group after a complete run with the current provider settings.

## Speaker Colors Flicker In The Live Transcript

The transcript uses dominant live speaker evidence over the stable part of the sentence. If it still flickers:

- Check whether ASR is producing very short sentence fragments.
- Check whether the live provider is unstable on short windows.
- Increase the live probe window before changing clustering thresholds.

## Wrong Speaker Is Created Too Often

Start with the browser sensitivity control. If that is not enough, validate a controlled run before changing low-level thresholds.

Common causes:

- Short sentences with too little voice information.
- Background audio or imitation voices.
- A speaker library saved with different provider settings.
- A live provider used for final comparison by mistake.

## Tests Fail After Documentation Changes

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core_regressions
```

Documentation-only changes should not affect runtime tests. If tests fail after docs-only edits, inspect uncommitted code changes before assuming the docs caused it.

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

## Final Transcriptions Fall Behind Playback

If final transcript rows get slower after several sentences, first decide whether the delay comes from ASR itself or from other work competing with ASR. ASR means automatic speech recognition: the final model that turns audio into committed transcript text. Live speaker scoring embeds recent audio windows while playback is running, so it can compete with final ASR when both use the same local GPU.

Run the same command with live speaker scoring disabled:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend local --model large-v2 --device cuda --compute-type float16 --embeddings-backend local --embedding-provider espnet_ecapa_wavlm_joint --live-speaker-embedding-provider espnet_ecapa_wavlm_joint --embedding-device cuda --vad-backend rms --realtime-preview-engine kroko_onnx --beam-size 5 --interval-seconds 2.5 --min-playback-advance-seconds 2.5 --unstable-tail-seconds 1.1 --no-live-speaker-assignment
```

If this catches up, the bottleneck is live speaker embedding contention, not large-v2 alone. Keep `--no-live-speaker-assignment` for transcript-quality runs, or re-enable live speaker scoring with conservative timing:

```powershell
--live-speaker-embedding-min-interval-seconds 0.75 --live-speaker-embedding-target-utilization 0.25 --live-speaker-probe-interval-seconds 0.75 --live-speaker-probe-min-advance-seconds 0.75
```

If final ASR is still slow with live speaker scoring disabled:

- Check `nvidia-smi` while the run is active and confirm the ASR process is using the GPU.
- Confirm the command uses `--device cuda --compute-type float16`.
- Try `--beam-size 1` only as a speed comparison. Use `--beam-size 5` when transcript quality matters.
- Stop other GPU-heavy processes before testing.
- Prefer remote ASR and remote embeddings if the local GPU must also run realtime preview and browser work.

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

## Meeting Intelligence OpenAI Report Fails With HTTP 400

Open the progress panel detail first. Current builds include the provider's HTTP error body when available, so the message should usually name the rejected field, schema, model, or parameter.

For OpenAI and OpenRouter, the meeting intelligence server uses strict structured output schemas. Strict structured output means every returned JSON object must match a closed schema: no undeclared properties, and every declared property is required. The server normalizes report schemas before sending OpenAI `response_format`, so failures mentioning `additionalProperties`, `required`, or nested object schemas should be treated as bugs in the report schema path.

Check these items:

- Confirm the selected model supports chat completions and structured JSON output.
- Click `Models` in the provider controls and choose a model returned by the account-visible `/models` list.
- Confirm the server process can see `OPENAI_API_KEY`; restart the server if you added the global environment variable after it started.
- Regenerate after changing provider or model, because cached reports are provider-aware and may be stale.

For a cheap OpenAI smoke test, select `openai` and a returned `nano` or `mini` model, then generate a short demo report. If the progress reaches evidence extraction but fails on the first section, the likely issue is structured-output schema compatibility.

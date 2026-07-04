# Troubleshooting

Most problems fall into four groups: models are not loaded, the wrong backend is selected, speaker profiles are incompatible, or browser-visible live state differs from backend events.

## The Browser URL Does Not Appear

Check whether model loading is still in progress. Large embedding stacks can take a while on first startup.

Try:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --embedding-helper-response-timeout-seconds 900
```

If using remote services, check:

```powershell
curl.exe http://192.168.178.22:8650/health
curl.exe http://192.168.178.22:8660/health
```

## Remote ASR Fails

Confirm the local command includes:

```text
--asr-backend remote --remote-asr-url http://192.168.178.22:8650
```

Then check the ASR server route:

```powershell
curl.exe http://192.168.178.22:8650/health
```

If health fails, restart the ASR server on the GPU host.

## Remote Embeddings Fail

Confirm the local command includes:

```text
--embeddings-backend remote --remote-embeddings-url http://192.168.178.22:8660
```

Check health and provider list:

```powershell
curl.exe http://192.168.178.22:8660/health
curl.exe http://192.168.178.22:8660/providers
```

If a provider needs Hugging Face access, make sure the token is configured as an environment variable on the server, not committed to the repo.

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

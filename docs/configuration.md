# Configuration

Most users only need backend URLs and provider choices; deeper timing and clustering flags are available when validating a specific workflow.

## Backends

Use local backends when all dependencies and models are installed locally:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --port 8796
```

Use remote backends when ASR and embeddings run on a GPU server:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --asr-backend remote --remote-asr-url http://192.168.178.22:8650 --embeddings-backend remote --remote-embeddings-url http://192.168.178.22:8660
```

## Embedding Providers

Final speaker assignment uses `--embedding-provider`.

Live speaker feedback can use `--live-speaker-embedding-provider`. If omitted, the app can use the final provider. If specified differently, the app keeps live profiles compatible with that live provider.

Recommended high-quality final stack:

```text
espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

Recommended fast live stack:

```text
pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50
```

## Timing Defaults

Recent defaults are tuned for faster live speaker feedback:

- `--interval-seconds 0.7`: final ASR loop delay and post-split cooldown.
- `--live-speaker-probe-interval-seconds 0.2`: fallback live-speaker probe cadence.
- `--live-speaker-probe-min-advance-seconds 0.2`: minimum media advance before another probe.
- `--live-speaker-embedding-min-interval-seconds 0.2`: minimum wall-clock spacing between live embedding requests.
- `--live-speaker-embedding-target-utilization 1.0`: disables latency backoff from utilization.
- `--live-speaker-ema-count 1`: uses the latest live probability snapshot.
- `--live-speaker-raw-change-snap`: enabled by default.
- `--live-speaker-raw-change-min-probability 0.70`: raw probability needed for a snap.
- `--live-speaker-raw-change-min-margin 0.25`: raw lead over the active speaker needed for a snap.
- `--live-speaker-sentence-hint`: enabled by default.
- `--live-speaker-sentence-hint-override`: enabled by default.
- `--live-speaker-sentence-hint-hold-seconds 0.30`: browser hold time for final sentence hints.

## Speaker Sensitivity

Speaker detection balances two errors:

- Merging two people into one speaker.
- Splitting one person into multiple speakers.

The app exposes a new-speaker sensitivity preset in the browser UI and lower-level command flags for experiments. Prefer the UI preset for normal use, then use validation before changing low-level thresholds.

## Runtime Directories

Use these environment variables to move mutable files:

- `WHOSPEAKS_RUNTIME_DIR`
- `WHOSPEAKS_CACHE_DIR`
- `WHOSPEAKS_MODEL_DIR`
- `WHOSPEAKS_SPEAKER_LIBRARY_DIR`

See [Installation](installation.md) for the default path layout.

## Discovering More Flags

The CLI is the authoritative option list:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help
```

Document only the flags you actually rely on in shared workflows. Keep one-off optimization experiments out of public docs until they become repeatable defaults.

# Configuration

Most users only need backend URLs and provider choices; deeper timing and clustering flags are available when validating a specific workflow. For the complete list, see the [CLI reference](cli-reference.md).

## Starter UI

Run the starter without arguments for the interactive launcher:

```powershell
whospeaks
```

Choose `Configuration` to edit the main launch parameters directly. The launcher keeps short, high-contrast summaries on the first line and prints the exact model/provider strings in dimmer detail lines for validation work:

- Language and realtime text
- Speaker provider quality
- Backends and remote URLs
- ASR model, device, and compute type
- Browser host and port
- Advanced launch arguments for lower-level flags

The first launcher screen exposes those same controls directly, with launch, doctor, install recommendation, and exact launch-command printing beside them. The `All saved profile fields` editor is still available from the configuration menu for less common saved fields.

The `Language and realtime text` section also exposes the realtime preview Python path. Use it when the main WhoSpeaks environment is Python 3.11 but the installed Kroko native runtime is in a separate Python 3.12 environment:

```powershell
whospeaks config --realtime-preview-python D:\Projekte\SpeakerDiarization\.venvs\kroko-install-test\Scripts\python.exe
whospeaks launch --print
```

## How The Launcher Builds Commands

The saved profile is a small JSON configuration. The launcher converts it into the longer browser-server command when you run:

```powershell
whospeaks launch --print
```

or:

```powershell
whospeaks launch
```

For example, a local smoke profile becomes a command like:

```powershell
whospeaks-window --host 127.0.0.1 --port 8796 --language en --model large-v2 --device auto --compute-type float16 --asr-backend local --embeddings-backend local --embedding-provider speechbrain_ecapa --live-speaker-embedding-provider speechbrain_ecapa --vad-backend rms --realtime-preview-engine kroko_onnx --embedding-python C:\Path\To\Python.exe --realtime-preview-python C:\Path\To\Python.exe
```

This indirection keeps the user command short while preserving a fully inspectable launch command. It also makes repeated runs stable because the profile is saved once and reused until you change it.

Profile locations:

- Windows: `%APPDATA%\WhoSpeaks\config.json`
- Linux/macOS: `$XDG_CONFIG_HOME/whospeaks/config.json` or `~/.config/whospeaks/config.json`
- Override: `WHOSPEAKS_CONFIG`
- Fallback when the user config cannot be written: `.whospeaks/config.json`

Use these commands for non-interactive changes:

```powershell
whospeaks config --language de
whospeaks config --provider-preset public_quality
whospeaks config --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660
whospeaks launch --print
```

## Backends

Use local backends when all dependencies and models are installed locally:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796
```

Use remote backends when ASR and embeddings run on a GPU server:

```powershell
.\.venv\Scripts\whospeaks-window.exe --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660
```

## Language

Use `--language` to keep final ASR, realtime Kroko/Banafo preview, and stream2sentence sentence splitting on the same language:

```powershell
.\.venv\Scripts\whospeaks-window.exe --language de --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660
```

Supported realtime language codes are `de`, `en`, `es`, `fr`, `it`, `he`/`iw`, `nl`, `pt`, `sv`, and `tr`. The app maps these to Kroko model files such as `Kroko-DE-Community-64-L-Streaming-001.data`; English can still explicitly select the legacy `pro-16l` preset if that model is installed locally.

| Language | CLI code | Kroko Community model code |
| --- | --- | --- |
| German | `de` | `DE` |
| English | `en` | `EN` |
| Spanish | `es` | `ES` |
| French | `fr` | `FR` |
| Italian | `it` | `IT` |
| Hebrew | `he` or `iw` | `IW` |
| Dutch | `nl` | `NL` |
| Portuguese | `pt` | `PT` |
| Swedish | `sv` | `SV` |
| Turkish | `tr` | `TR` |

Missing public Kroko Community models are downloaded automatically from `Banafo/Kroko-ASR` into `runtime/models/kroko-onnx/` when realtime preview starts. Disable this with `--no-realtime-preview-auto-download` if you need strictly offline startup. Pro/private models are not auto-downloaded; pass an existing `.data` file with `--realtime-preview-model-path`.

By default stream2sentence uses `nltk+rule-based` for the Latin-script supported languages and `rule-based` for Hebrew. Override this only for a validated setup:

```powershell
.\.venv\Scripts\whospeaks-window.exe --language de --sentence-tokenizer nltk+rule-based
```

Additional Whisper languages can be used for final ASR and sentence splitting when realtime preview text is disabled:

```powershell
.\.venv\Scripts\whospeaks-window.exe --language pl --realtime-preview-engine off
```

Automatic sentence-tokenizer selection prefers NLTK whenever both NLTK and Stanza support the language. Languages supported only by Stanza, such as Chinese, use `stanza`; Hebrew keeps the lightweight rule-based splitter used by the realtime Kroko path.

## Embedding Providers

Final speaker assignment uses `--embedding-provider`.

Live speaker feedback can use `--live-speaker-embedding-provider`. If omitted, the app can use the final provider. If specified differently, the app keeps live profiles compatible with that live provider.

The `whospeaks` starter can manage these as named provider presets:

| Preset ID | Simple label | Final provider | Live provider | Notes |
| --- | --- | --- | --- | --- |
| `smoke` | First start | `speechbrain_ecapa` | `speechbrain_ecapa` | Fast setup check, not the highest-accuracy setting. |
| `single_espnet` | Single ESPnet | `espnet_ecapa_wavlm_joint` | `espnet_ecapa_wavlm_joint` | Validates one provider in isolation. |
| `smoke_fast_live` | Smoke final + fast live | `speechbrain_ecapa` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Simple final provider with the fast live stack. |
| `public_quality` | Public high quality | `espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Public multi-provider stack, no RawNet3 artifact. |
| `promoted_public` | Promoted public stack | `espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Matches the current `whospeaks-window` default final provider stack. |
| `tuned_private` | Private tuned | `espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Requires the `jungjee_rawnet3` artifact on the embeddings server. |

Use the interactive starter menu, or set a preset explicitly:

```powershell
whospeaks config --set provider_preset=public_quality
whospeaks launch --print
```

Smoke-test stack:

```text
speechbrain_ecapa
```

Public high-quality final stack:

```text
espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

Tuned best final stack:

```text
espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

The tuned best stack requires `jungjee_rawnet3`, which is not fully provisioned by the public source snapshot. Use the public high-quality stack until the RawNet3 artifact is installed on the embeddings server.

Recommended fast live stack:

```text
pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50
```

## Timing Defaults

Recent defaults are tuned to keep final ASR responsive while live speaker feedback runs on the same GPU:

- `--interval-seconds 0.7`: final ASR loop delay and post-split cooldown.
- `--live-speaker-probe-interval-seconds 0.75`: fallback live-speaker probe cadence.
- `--live-speaker-probe-min-advance-seconds 0.75`: minimum media advance before another probe.
- `--live-speaker-embedding-min-interval-seconds 0.75`: minimum wall-clock spacing between live embedding requests.
- `--live-speaker-embedding-target-utilization 0.25`: backs off live embeddings when they start taking too much wall time.
- `--live-speaker-ema-count 1`: uses the latest live probability snapshot.
- `--live-speaker-raw-change-snap`: enabled by default.
- `--live-speaker-raw-change-min-probability 0.70`: raw probability needed for a snap.
- `--live-speaker-raw-change-min-margin 0.25`: raw lead over the active speaker needed for a snap.
- `--live-speaker-sentence-hint`: enabled by default.
- `--live-speaker-sentence-hint-override`: enabled by default.
- `--live-speaker-sentence-hint-hold-seconds 0.30`: browser hold time for final sentence hints.

## ASR No-Speech Filtering

The final ASR path can reject segments that the ASR model itself marks as likely non-speech. This is useful for music beds, long pauses, or other non-speech audio that can otherwise make Whisper-like models produce plausible but invented text.

This filter does not compare transcript text against known phrases. It uses faster-whisper segment metadata, especially `no_speech_prob`.

The filter is enabled by default:

```powershell
.\.venv\Scripts\whospeaks-window.exe --asr-no-speech-filter
```

Disable it for comparison runs:

```powershell
.\.venv\Scripts\whospeaks-window.exe --no-asr-no-speech-filter
```

Tuning flags:

- `--asr-no-speech-prob-threshold 0.65`: discard ASR segments at or above this `no_speech_prob`.
- `--asr-no-speech-hard-threshold 0.85`: discard even very short ASR segments at or above this value.
- `--asr-no-speech-keep-short-max-words 2`: keep short interjections below the hard threshold when they have at most this many words.
- `--asr-no-speech-keep-short-max-seconds 0.45`: keep short interjections below the hard threshold when they are at most this long.

Use a lower `--asr-no-speech-prob-threshold` when music-only sections still produce text. Use a higher value, or disable the filter, only when validation shows real speech is being dropped. The short-segment exception is meant to preserve real utterances such as "yes", "no", "ok", or "ja" while still rejecting high-confidence non-speech segments.

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

## Complete Parameter Reference

The CLI help shows accepted syntax:

```powershell
.\.venv\Scripts\whospeaks-window.exe --help
```

The [CLI reference](cli-reference.md) documents every parser parameter found in the repo, including installed commands, helper modules, and environment variables. Update that reference whenever a new parameter is added.

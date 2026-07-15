# Quickstart

Start with a small smoke run, then move to the tuned provider stacks after the selected local or remote services are healthy.

## Before You Start

Complete:

1. On Apple Silicon, follow [macOS Setup](macos-setup.md).
2. On Windows, follow [Installation](installation.md) on the controller.
3. Use [External Servers](external-servers.md) on a Linux GPU server if the controller uses remote services.

## Apple Silicon Quickstart

After creating `.venv`, install the managed macOS target once:

```bash
.venv/bin/whospeaks install --target macos --yes
.venv/bin/whospeaks doctor
```

Then start MLX ASR, MPS embeddings, and the browser controller in health-checked order with one command:

```bash
.venv/bin/whospeaks launch
```

Realtime preview is off for the reliable first run. The manual service commands in [macOS Setup](macos-setup.md) remain available for troubleshooting.

Verify from Windows:

```powershell
curl.exe http://YOUR_GPU_SERVER_IP:8650/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/health
curl.exe http://YOUR_GPU_SERVER_IP:8660/providers
```

Replace `YOUR_GPU_SERVER_IP` with your Linux GPU server address.

The examples use plain HTTP for a trusted private network. Plain HTTP does not encrypt speech audio or authenticate the remote service; use a VPN or authenticated TLS proxy outside that threat model.

Skip the remote health checks for a completely local Windows run.

## First End-To-End Run

The shortest guided path is:

```powershell
whospeaks
```

Choose the setup/profile you want, then choose `Launch browser UI`. To see the exact command before running it:

```powershell
whospeaks launch --print
```

The launcher reads the saved profile, expands it into a full `whospeaks-window ...` command, and injects helper Python paths for local embeddings and realtime preview when those features are enabled.

Use a smoke provider first. This proves the UI, media loading, ASR route, embedding route, and speaker assignment pipeline work:

Do not build a durable People library during this smoke step. Voice samples are tied to the embedding-provider contract and are not automatically re-embedded when you switch to the production provider.

In the full-screen starter, open **Settings** and choose **Speaker model preset** -> **Low VRAM - SpeechBrain ECAPA**, or run:

```powershell
.\.venv\Scripts\whospeaks.exe config --set provider_preset=smoke
```

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "speechbrain_ecapa" --live-speaker-embedding-provider "speechbrain_ecapa" --vad-backend rms --realtime-preview-engine off
```

The command prints a browser URL. Open it, load or replay media, then press Start.

## Docker Quickstart

For a Linux container smoke server:

```bash
docker build -t whospeaks:local .
docker volume create whospeaks-data
docker volume create whospeaks-models
docker run --rm --name whospeaks -p 127.0.0.1:8796:8796 -v whospeaks-data:/data -v whospeaks-models:/models whospeaks:local
```

Open `http://127.0.0.1:8796/`. The container path uses CPU defaults and is meant as a reproducible server install first; see [Docker](docker.md) for local media mounts, build args, and validation commands.

## Completely Local Windows Run

Use this when ASR, embeddings, realtime preview, and the browser app all run on the same Windows machine. It keeps one high-quality final embedding provider and uses the safer live-speaker timing defaults so final ASR can stay responsive on a single GPU:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend local --model large-v2 --device cuda --compute-type float16 --embeddings-backend local --embedding-provider espnet_ecapa_wavlm_joint --live-speaker-embedding-provider espnet_ecapa_wavlm_joint --embedding-device cuda --vad-backend rms --realtime-preview-engine kroko_onnx --beam-size 5 --interval-seconds 2.5 --min-playback-advance-seconds 2.5 --unstable-tail-seconds 1.1
```

If final transcript rows still fall behind playback, isolate live speaker scoring first:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend local --model large-v2 --device cuda --compute-type float16 --embeddings-backend local --embedding-provider espnet_ecapa_wavlm_joint --live-speaker-embedding-provider espnet_ecapa_wavlm_joint --embedding-device cuda --vad-backend rms --realtime-preview-engine kroko_onnx --beam-size 5 --interval-seconds 2.5 --min-playback-advance-seconds 2.5 --unstable-tail-seconds 1.1 --no-live-speaker-assignment
```

`--no-live-speaker-assignment` keeps realtime text preview enabled, but disables live speaker scoring and highlighting. Final speaker labels still run on committed transcript rows.

## Public High-Quality Run

After the smoke provider works, use this public provider stack:

In the full-screen starter, open **Settings** and choose **Speaker model preset** -> **High quality - public ensemble**, or run:

```powershell
.\.venv\Scripts\whospeaks.exe config --set provider_preset=public_quality
```

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms --realtime-preview-engine off
```

The live-speaker timing defaults are tuned to keep final ASR responsive while live feedback runs. The command does not need the older long list of `--live-speaker-*` timing flags.

## Optional Local Preview

The commands above disable local realtime preview with `--realtime-preview-engine off`. Remove that flag only after the chosen local preview backend is installed and working. Kroko uses RealtimeSTT/`kroko_onnx`; Nemotron 3.5 uses CPU `sherpa-onnx`.

For a German realtime session with Kroko preview enabled, add `--language de` and keep the default `community-64l` preview preset:

```powershell
.\.venv\Scripts\whospeaks-window.exe --port 8796 --language de --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms
```

This selects `Kroko-DE-Community-64-L-Streaming-001.data` for preview, sends `de` to the final faster-whisper server, and uses German stream2sentence sentence splitting.
If the German Kroko model is not present locally, the app downloads it from Hugging Face before starting realtime preview.

For a German realtime session with Nemotron 3.5 preview, install the preview runtime once and select the `sherpa_onnx` engine:

```powershell
.\.venv\Scripts\python.exe -m pip install "sherpa-onnx>=1.13.4,<1.14" "sherpa-onnx-bin>=1.13.4,<1.14"
.\.venv\Scripts\whospeaks-window.exe --port 8796 --language de --asr-backend remote --remote-asr-url http://YOUR_GPU_SERVER_IP:8650 --embeddings-backend remote --remote-embeddings-url http://YOUR_GPU_SERVER_IP:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50" --vad-backend rms --realtime-preview-engine sherpa_onnx --realtime-preview-model-preset nemotron-3.5-560ms-int8
```

On first start, the app downloads and verifies the selected Nemotron archive under `runtime/models/sherpa-onnx/`. Use `nemotron-3.5-160ms-int8` only when you specifically want to compare lower latency against the default 560 ms model.

## First Session

1. Start the command.
2. Open the browser URL printed by the process.
3. Load or replay the target media.
4. Press Start in the browser UI.
5. Watch the live speaker tag in the speaker panel for fast feedback.
6. Watch the live transcript for the current sentence view.
7. After selecting the intended production embedding provider, let a recurring participant speak at least three clean sentences or about six seconds.
8. In **Speakers**, choose **Link to Person…** and create or select that Person.
9. In **Settings → People**, turn on **Include in automatic recognition** only for plausible upcoming attendees.
10. In a later session, review **Likely Person** with **Confirm** or **Not Person** instead of assuming the suggestion is correct.

## What To Expect

The app has two speaker assignment layers:

- The live layer updates quickly from recent audio windows.
- The final layer assigns speaker IDs to completed sentences.

The live layer may change faster than the final transcript. The final transcript is intentionally more conservative because it can use sentence boundaries, longer audio, and speaker-memory updates.

## Next Steps

- Learn the full browser workflow in [Live Window Workflow](live-window-workflow.md).
- Remember recurring participants with [People And Voice Recognition](people-and-recognition.md).
- Review the deployment and voice-data threat model in [Security And Data Privacy](security-and-data-privacy.md).
- Use [Legacy Speaker-Group Files](speaker-libraries.md) only for compatibility with the older portable workflow.
- Tune behavior with [Configuration](configuration.md).
- Validate changes with [Validation And Scoring](validation-and-scoring.md).

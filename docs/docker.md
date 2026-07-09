# Docker

Use Docker when you want a reproducible Linux server image for the WhoSpeaks browser UI.

The Docker image runs `whospeaks-window` directly. It does not run the interactive `whospeaks` launcher, because containers should start one predictable server process.

## What The Dockerfile Contains

The root [Dockerfile](../Dockerfile) is intentionally small as a build context: `.dockerignore` sends only the Dockerfile and ignore file to Docker. The final image is still large because local ASR, speaker embeddings, PyTorch, ffmpeg, and Kroko are ML-heavy dependencies.

The image:

- starts from `python:3.11-slim`
- installs ffmpeg and runtime audio libraries
- installs CPU PyTorch from the PyTorch CPU wheel index
- installs `whospeaks[all]`
- builds and installs `kroko-onnx`
- runs as a non-root `whospeaks` user
- serves the browser UI on container port `8796`
- uses `/data` for sessions/output/work files
- uses `/models` for Hugging Face and model caches

The default build currently pins the first PyPI release:

```text
whospeaks 0.0.1
```

Override the build arguments when testing a newer release.

## Build

From the repository root:

```bash
docker build -t whospeaks:local .
```

To build from a normal package index version instead of the pinned wheel URL:

```bash
docker build \
  --build-arg WHOSPEAKS_WHEEL_URL= \
  --build-arg WHOSPEAKS_VERSION=0.0.1 \
  -t whospeaks:local .
```

Use a newer `WHOSPEAKS_VERSION` after it is available on the selected package index.

## Run The Server

Use Docker-managed volumes for `/data` and `/models`. They avoid host-directory ownership problems with the non-root container user and keep downloaded models between restarts.

```bash
docker volume create whospeaks-data
docker volume create whospeaks-models

docker run --rm \
  --name whospeaks \
  -p 8796:8796 \
  -v whospeaks-data:/data \
  -v whospeaks-models:/models \
  whospeaks:local
```

Open:

```text
http://127.0.0.1:8796/
```

For another machine on the LAN, use the Docker host IP instead of `127.0.0.1`.

## Run With A Local Media File

Mount media read-only and override the command with explicit file paths:

```bash
docker run --rm \
  --name whospeaks \
  -p 8796:8796 \
  -v whospeaks-data:/data \
  -v whospeaks-models:/models \
  -v /path/to/media:/media:ro \
  whospeaks:local \
  --host 0.0.0.0 \
  --port 8796 \
  --no-browser \
  --no-startup-warmup-before-url \
  --work-dir /data/work \
  --output-dir /data/output \
  --session-dir /data/sessions \
  --audio-file /media/example.wav \
  --video-file /media/example.mp4 \
  --skip-download \
  --language en \
  --model tiny.en \
  --device cpu \
  --compute-type int8 \
  --asr-backend local \
  --embeddings-backend local \
  --embedding-provider speechbrain_ecapa \
  --embedding-device cpu \
  --live-speaker-embedding-provider speechbrain_ecapa \
  --embedding-python /usr/local/bin/python \
  --vad-backend rms \
  --realtime-preview-engine kroko_onnx \
  --realtime-preview-python /usr/local/bin/python
```

The explicit Python helper paths are important. They make the embedding helper and Kroko realtime preview worker use the same Python environment that contains the installed package and dependencies.

## Check The Container

Basic checks:

```bash
docker run --rm --entrypoint python whospeaks:local -c "import importlib.metadata, kroko_onnx, torch; print(importlib.metadata.version('whospeaks')); print(torch.__version__); print(torch.cuda.is_available()); print(kroko_onnx.__name__)"
docker run --rm --entrypoint python whospeaks:local -m pip check
```

Server health check after `docker run`:

```bash
curl http://127.0.0.1:8796/
docker ps --filter name=whospeaks
docker logs whospeaks --tail 100
```

The image includes a Docker `HEALTHCHECK` that fetches the root page inside the container.

## Validation Notes

The Docker path was validated on Linux with:

- `whospeaks 0.0.1`
- `torch 2.12.1+cpu`
- `kroko_onnx 1.12.9`
- `pip check` with no broken requirements
- a healthy container serving the browser UI
- HTML containing `Growing Window Speaker Diarization`, `WhoSpeaks Live`, and `Start transcription`

The first run may download ASR or preview models into `/models`. Keep `/models` on a persistent Docker volume unless you want every container start to behave like a fresh install.

## Common Docker Problems

### Permission denied under `/models`

Problem: Binding a host directory to `/models` can fail if the directory is owned by a different host user.

Fix: Use a Docker-managed named volume:

```bash
docker volume create whospeaks-models
docker run -v whospeaks-models:/models ...
```

### Image is still large

Problem: The build context is tiny, but the final image is several GB.

Reason: Local ASR and embedding dependencies include PyTorch, transformers, ffmpeg, audio libraries, and Kroko. That is expected for the all-local image.

### CPU performance is slow

Problem: The default Docker image uses CPU Torch and `--device cpu`.

Fix: Use it as a reproducible server smoke image first. GPU Docker support needs a CUDA base image, NVIDIA container runtime, and a matching PyTorch CUDA wheel strategy.

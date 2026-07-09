# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG WHOSPEAKS_WHEEL_URL=https://test-files.pythonhosted.org/packages/8a/b9/891577c7ba3b0179c6c7872212b7566d32f2c31c06404ab5cebe55c9bb03/whospeaks-0.1.0.dev16-py3-none-any.whl#sha256=c62481febff48d02fe8a7aa5e2d75e0c03cc3989b6da7181a0a197b22198a7ac
ARG WHOSPEAKS_VERSION=0.1.0.dev16
ARG INSTALL_KROKO=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache \
    WHOSPEAKS_WORK_DIR=/data/work \
    WHOSPEAKS_OUTPUT_DIR=/data/output \
    WHOSPEAKS_SESSION_DIR=/data/sessions \
    KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
        pkg-config \
    && python -m pip install --upgrade pip \
    && python -m pip install --index-url "$TORCH_CPU_INDEX_URL" torch torchaudio \
    && if [ -n "$WHOSPEAKS_WHEEL_URL" ]; then \
        python -m pip install --index-url "$PIP_INDEX_URL" --extra-index-url "$TORCH_CPU_INDEX_URL" "whospeaks[all] @ $WHOSPEAKS_WHEEL_URL"; \
       else \
        python -m pip install --index-url "$PIP_INDEX_URL" --extra-index-url "$TORCH_CPU_INDEX_URL" "whospeaks[all]==$WHOSPEAKS_VERSION"; \
       fi \
    && if [ "$INSTALL_KROKO" = "1" ]; then \
        python -m RealtimeSTT.install_kroko --build --work-dir /tmp/kroko-build; \
       fi \
    && apt-get purge -y --auto-remove cmake build-essential git pkg-config \
    && rm -rf /var/lib/apt/lists/* /tmp/kroko-build /root/.cache

RUN useradd --create-home --uid 10001 whospeaks \
    && mkdir -p /data/work /data/output /data/sessions /models \
    && chown -R whospeaks:whospeaks /data /models

USER whospeaks
WORKDIR /data
VOLUME ["/data", "/models"]
EXPOSE 8796

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8796/', timeout=3).read(1)"

ENTRYPOINT ["whospeaks-window"]
CMD ["--host", "0.0.0.0", "--port", "8796", "--no-browser", "--no-startup-warmup-before-url", "--work-dir", "/data/work", "--output-dir", "/data/output", "--session-dir", "/data/sessions", "--language", "en", "--model", "tiny.en", "--device", "cpu", "--compute-type", "int8", "--asr-backend", "local", "--embeddings-backend", "local", "--embedding-provider", "speechbrain_ecapa", "--embedding-device", "cpu", "--live-speaker-embedding-provider", "speechbrain_ecapa", "--embedding-python", "/usr/local/bin/python", "--vad-backend", "rms", "--realtime-preview-engine", "kroko_onnx", "--realtime-preview-python", "/usr/local/bin/python"]

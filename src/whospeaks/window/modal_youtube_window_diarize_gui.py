"""Modal deployment wrapper for the full WhoSpeaks YouTube window GUI."""

from __future__ import annotations

import importlib.util
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "whospeaks-youtube-window-diarize"
PORT = 8000
MODAL_GPU = os.environ.get("WHOSPEAKS_MODAL_GPU", "T4")
MODAL_SCALEDOWN_WINDOW_SECONDS = int(os.environ.get("WHOSPEAKS_MODAL_SCALEDOWN_WINDOW_SECONDS", "60"))
REMOTE_ROOT = PurePosixPath("/root/WhoSpeaksLive")
REMOTE_BAKED_MEDIA = REMOTE_ROOT / "runtime" / "media" / "local-filefeed"
REMOTE_CACHE = PurePosixPath("/cache")
REMOTE_MEDIA = REMOTE_CACHE / "media"
KROKO_MODEL_NAME = "Kroko-EN-Community-64-L-Streaming-001.data"
KROKO_MODEL_REPO = "Banafo/Kroko-ASR"
KROKO_MODEL_PATH = REMOTE_CACHE / "kroko" / KROKO_MODEL_NAME

def _local_root() -> Path:
    current = Path(__file__).resolve()
    candidates: list[Path] = []
    candidates.extend(current.parents)
    candidates.append(Path(os.environ.get("WHOSPEAKS_PROJECT_ROOT", str(REMOTE_ROOT))))
    for candidate in candidates:
        if (candidate / "src" / "whospeaks").is_dir() and (candidate / "vendor").is_dir():
            return candidate
    return Path(str(REMOTE_ROOT))


LOCAL_ROOT = _local_root()
LOCAL_MEDIA = LOCAL_ROOT / "runtime" / "media" / "local-filefeed"

IGNORE_PYTHON_CACHE = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
]

CACHE_ENV = {
    "WHOSPEAKS_PROJECT_ROOT": str(REMOTE_ROOT),
    "WHOSPEAKS_RUNTIME_DIR": str(REMOTE_CACHE),
    "WHOSPEAKS_CACHE_DIR": str(REMOTE_CACHE / "cache"),
    "WHOSPEAKS_MODEL_DIR": str(REMOTE_CACHE / "models"),
    "WHOSPEAKS_SPEAKER_LIBRARY_DIR": str(REMOTE_CACHE / "speakers"),
    "HF_HOME": str(REMOTE_CACHE / "huggingface"),
    "TRANSFORMERS_CACHE": str(REMOTE_CACHE / "huggingface" / "transformers"),
    "HF_HUB_CACHE": str(REMOTE_CACHE / "huggingface" / "hub"),
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
    "TORCH_HOME": str(REMOTE_CACHE / "torch"),
    "MPLCONFIGDIR": str(REMOTE_CACHE / "matplotlib"),
    "NUMBA_CACHE_DIR": str(REMOTE_CACHE / "numba"),
    "XDG_CACHE_HOME": str(REMOTE_CACHE),
    "WESPEAKER_HOME": str(REMOTE_CACHE / "wespeaker"),
    "MODELSCOPE_CACHE": str(REMOTE_CACHE / "modelscope"),
    "ESPNET_MODEL_ZOO_CACHE": str(REMOTE_CACHE / "espnet_model_zoo"),
    "NLTK_DATA": "/usr/share/nltk_data:/cache/nltk",
    "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'vendor'}",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT": "1",
}

PYPI_PACKAGES = [
    "faster-whisper==1.2.1",
    "RealtimeSTT[kroko-builder]==1.0.2",
    "stream2sentence==1.0.0",
    "speechbrain==1.0.3",
    "yt-dlp>=2025.6.9",
    "numpy==1.26.4",
    "scipy==1.17.1",
    "soundfile==0.13.1",
    "librosa==0.10.2.post1",
    "nltk==3.9.4",
    "scikit-learn>=1.3.2",
    "onnxruntime-gpu==1.20.1",
    "silero-vad==6.2.1",
    "huggingface-hub>=0.34.0,<1.0",
    "transformers==4.57.6",
]


def _base_image() -> modal.Image:
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "ffmpeg",
            "git",
            "curl",
            "libsndfile1",
            "libssl-dev",
            "build-essential",
            "clang",
            "cmake",
            "ninja-build",
            "pkg-config",
            "portaudio19-dev",
        )
        .run_commands("python -m pip install --upgrade pip setuptools wheel")
        .run_commands(
            "python -m pip install --index-url https://download.pytorch.org/whl/cu126 "
            "torch==2.6.0 torchaudio==2.6.0"
        )
        .pip_install(*PYPI_PACKAGES)
        .run_commands("python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab")
        .env(CACHE_ENV)
    )

    for directory_name in ("src", "vendor", "tools", "tests"):
        image = image.add_local_dir(
            LOCAL_ROOT / directory_name,
            str(REMOTE_ROOT / directory_name),
            copy=True,
            ignore=IGNORE_PYTHON_CACHE,
        )
    for file_name in ("pyproject.toml", "README.md"):
        image = image.add_local_file(
            LOCAL_ROOT / file_name,
            str(REMOTE_ROOT / file_name),
            copy=True,
        )
    if LOCAL_MEDIA.is_dir():
        image = image.add_local_dir(
            LOCAL_MEDIA,
            str(REMOTE_BAKED_MEDIA),
            copy=True,
            ignore=IGNORE_PYTHON_CACHE,
        )

    return (
        image
        .run_commands("python -m pip install -e /root/WhoSpeaksLive --no-deps")
        .run_commands(
            "python -m RealtimeSTT.install_kroko --build --variant free "
            "--work-dir /tmp/realtimestt-kroko-builder"
        )
    )


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("whospeaks-youtube-window-diarize-cache", create_if_missing=True)
image = _base_image()


def _ensure_kroko_model() -> None:
    target = Path(str(KROKO_MODEL_PATH))
    if target.is_file():
        print(f"Kroko preview model already cached: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kroko preview model {KROKO_MODEL_REPO}/{KROKO_MODEL_NAME} to {target}.", flush=True)
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(
        repo_id=KROKO_MODEL_REPO,
        filename=KROKO_MODEL_NAME,
        local_dir=str(target.parent),
    ))
    if downloaded != target and downloaded.is_file():
        downloaded.replace(target)
    if not target.is_file():
        raise RuntimeError(f"Kroko model download did not produce {target}")
    print(f"Kroko preview model ready: {target}", flush=True)


def _seed_media_cache() -> None:
    source = Path(str(REMOTE_BAKED_MEDIA))
    target = Path(str(REMOTE_MEDIA))
    target.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        print("No baked media directory found; URL loads will download on demand.", flush=True)
        return

    copied = 0
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = target / item.name
        if destination.exists():
            continue
        shutil.copy2(item, destination)
        copied += 1
    print(f"Seeded Modal media cache with {copied} baked file(s); runtime media cache is {target}.", flush=True)


def _silero_model_path() -> Path | None:
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        return None
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    for filename in ("silero_vad_op18_ifless.onnx", "silero_vad.onnx"):
        candidate = package_dir / "data" / filename
        if candidate.is_file():
            return candidate
    return None


def _command() -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "whospeaks.window.youtube_gui",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--no-browser",
        "--no-startup-warmup-before-url",
        "--skip-download",
        "--work-dir",
        str(REMOTE_MEDIA),
        "--output-dir",
        str(REMOTE_CACHE / "window_diarize"),
        "--download-root",
        str(REMOTE_CACHE / "faster-whisper"),
        "--speaker-library-dir",
        str(REMOTE_CACHE / "speakers"),
        "--embedding-python",
        sys.executable,
        "--embedding-provider",
        os.environ.get("WHOSPEAKS_MODAL_EMBEDDING_PROVIDER", "speechbrain_ecapa"),
        "--embedding-device",
        "cuda",
        "--device",
        "cuda",
        "--compute-type",
        "float16",
        "--realtime-preview-engine",
        "kroko_onnx",
        "--realtime-preview-model",
        KROKO_MODEL_NAME,
        "--realtime-preview-model-path",
        str(KROKO_MODEL_PATH),
        "--realtime-preview-download-root",
        str(REMOTE_CACHE / "kroko"),
        "--realtime-preview-python",
        sys.executable,
        "--realtime-preview-realtimestt-root",
        str(REMOTE_ROOT / "vendor"),
        "--realtime-preview-provider",
        "cpu",
        "--realtime-preview-num-threads",
        "2",
        "--realtime-preview-startup-timeout-seconds",
        "45",
    ]
    silero_path = _silero_model_path()
    if silero_path is not None:
        command.extend(["--vad-silero-onnx-model-path", str(silero_path)])
    extra_args = os.environ.get("WHOSPEAKS_MODAL_EXTRA_ARGS", "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={str(REMOTE_CACHE): cache_volume},
    timeout=60 * 60 * 24,
    max_containers=1,
    scaledown_window=MODAL_SCALEDOWN_WINDOW_SECONDS,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=20 * 60)
def youtube_window_diarize_gui() -> None:
    for directory in (
        REMOTE_CACHE,
        REMOTE_CACHE / "media",
        REMOTE_CACHE / "window_diarize",
        REMOTE_CACHE / "faster-whisper",
        REMOTE_CACHE / "kroko",
        REMOTE_CACHE / "speakers",
        REMOTE_CACHE / "cache",
        REMOTE_CACHE / "models",
    ):
        Path(str(directory)).mkdir(parents=True, exist_ok=True)

    _ensure_kroko_model()
    _seed_media_cache()
    command = _command()
    print("Starting WhoSpeaks GUI on Modal:", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.Popen(command, cwd=str(REMOTE_ROOT))

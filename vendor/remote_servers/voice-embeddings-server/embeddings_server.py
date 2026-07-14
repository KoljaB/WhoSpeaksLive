#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

HOST = os.environ.get("EMBEDDINGS_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBEDDINGS_PORT", "8660"))
SAMPLE_RATE = 16000
DEFAULT_PROVIDER = "speechbrain_ecapa"
DEFAULT_DEVICE = os.environ.get("EMBEDDINGS_DEVICE", "auto")
WARMUP_SECONDS = float(os.environ.get("EMBEDDINGS_WARMUP_SECONDS", "2.0"))

BENCHMARK_PROVIDER_ALIASES = {
    "espnet_voxcelebs12_ecapa_wavlm_joint": "espnet_ecapa_wavlm_joint",
    "3d_speaker_cam": "speaker3d_campplus",
    "3d_speaker_campplus": "speaker3d_campplus",
    "wespeaker_cam": "wespeaker_campplus",
    "wespeaker_campplus": "wespeaker_campplus",
    "wespeaker_resnet34_lm": "wespeaker_resnet34_lm_onnx",
    "pyannote_wespeaker_voxceleb_resnet34_lm": "pyannote_wespeaker_resnet34_lm",
    "espnet_voxcelebs12_rawnet3": "espnet_rawnet3",
    "3d_speaker_eres2netv2": "speaker3d_eres2netv2",
    "speechbrain_spkrec_ecapa_voxceleb": "speechbrain_ecapa",
    "speechbrain_spkrec_resnet_voxceleb": "speechbrain_resnet",
    "microsoft_wavlm_base_sv": "wavlm_base_sv",
    "jungjee_rawnet3": "jungjee_rawnet3",
    "nvidia_speakerverification_en_titanet_large": "nemo_titanet_large",
    "speechbrain_spkrec_xvect_voxceleb": "speechbrain_xvector",
}

DISPLAY_NAMES = {
    "speaker3d_campplus": "3D-Speaker CAM++",
    "wespeaker_campplus": "Wespeaker CAM++",
    "speechbrain_resnet": "speechbrain/spkrec-resnet-voxceleb",
    "wespeaker_resnet34_lm_onnx": "Wespeaker ResNet34-LM",
    "pyannote_wespeaker_resnet34_lm": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "espnet_ecapa_wavlm_joint": "espnet/voxcelebs12_ecapa_wavlm_joint",
    "pyannote_embedding": "pyannote/embedding",
    "speechbrain_ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "espnet_rawnet3": "espnet/voxcelebs12_rawnet3",
    "wavlm_base_sv": "microsoft/wavlm-base-sv",
    "speaker3d_eres2netv2": "3D-Speaker ERes2NetV2",
    "jungjee_rawnet3": "jungjee/RawNet3",
    "nemo_titanet_large": "nvidia/speakerverification_en_titanet_large",
    "resemblyzer": "Resemblyzer",
    "speechbrain_xvector": "speechbrain/spkrec-xvect-voxceleb",
}

SUPPORTED_PROVIDERS = sorted(DISPLAY_NAMES)

app = FastAPI(title="Voice Embeddings Server", version="1.0.0")
_provider_cache: dict[tuple[str, str], Any] = {}
_provider_locks: dict[tuple[str, str], threading.Lock] = {}
_cache_lock = threading.Lock()


def start_parent_watchdog() -> None:
    # These servers are started by hand (docs/macos-setup.md) and hold multi-GB
    # models; if the launching shell dies they would otherwise run forever as
    # orphans. Exit once reparented. Opt out (nohup-style daemonizing) with
    # WHOSPEAKS_EXIT_WITH_PARENT=0.
    if os.environ.get("WHOSPEAKS_EXIT_WITH_PARENT", "1") in {"0", "false", "False"}:
        return
    parent = os.getppid()
    if parent <= 1:
        return

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(5)
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()


@app.on_event("startup")
def on_startup() -> None:
    start_parent_watchdog()


def configure_env() -> None:
    cache = ROOT / ".cache"
    env_defaults = {
        "HF_HOME": cache / "huggingface",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "MPLCONFIGDIR": cache / "matplotlib",
        "NUMBA_CACHE_DIR": cache / "numba",
        "XDG_CACHE_HOME": cache,
        "WESPEAKER_HOME": cache / "wespeaker",
        "MODELSCOPE_CACHE": cache / "modelscope",
        "NLTK_DATA": cache / "nltk",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, str(value))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def canonical_provider_name(value: str) -> str:
    provider = str(value or DEFAULT_PROVIDER).strip().lower()
    provider = provider.replace("/", "_").replace("-", "_")
    provider = re.sub(r"[^a-z0-9_]+", "_", provider).strip("_")
    provider = re.sub(r"_+", "_", provider)
    return BENCHMARK_PROVIDER_ALIASES.get(provider, provider)


def parse_provider_stack_specs(provider: str) -> list[tuple[str, float]]:
    value = (provider or DEFAULT_PROVIDER).strip()
    if value.lower().startswith("stack:"):
        value = value[6:]
    separator = "+" if "+" in value else ","
    specs: list[tuple[str, float]] = []
    for raw_item in value.split(separator):
        item = raw_item.strip()
        if not item:
            continue
        raw_provider = item
        raw_weight = "1.0"
        if "=" in item:
            raw_provider, raw_weight = item.rsplit("=", 1)
        elif ":" in item:
            possible_provider, possible_weight = item.rsplit(":", 1)
            try:
                float(possible_weight)
            except ValueError:
                pass
            else:
                raw_provider = possible_provider
                raw_weight = possible_weight
        weight = float(raw_weight)
        if weight < 0.0:
            raise HTTPException(status_code=400, detail=f"provider weight must be non-negative: {item!r}")
        name = canonical_provider_name(raw_provider)
        if name not in DISPLAY_NAMES:
            raise HTTPException(status_code=400, detail=f"unsupported provider: {raw_provider!r}")
        specs.append((name, weight))
    if not specs:
        specs.append((DEFAULT_PROVIDER, 1.0))
    return specs


def resolve_runtime_device(device: str) -> str:
    normalized = str(device or DEFAULT_DEVICE).strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized != "auto":
        return normalized
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def is_cuda_device(device: str) -> bool:
    normalized = resolve_runtime_device(device)
    return normalized == "cuda" or normalized.startswith("cuda:")


def choose_torch_device(device: str) -> str:
    return resolve_runtime_device(device)


def normalize_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("embedding provider returned an empty vector")
    return (vector / norm).astype(np.float32)


def warmup_audio() -> np.ndarray:
    sample_count = max(int(SAMPLE_RATE * max(WARMUP_SECONDS, 0.25)), SAMPLE_RATE // 4)
    t = np.arange(sample_count, dtype=np.float32) / float(SAMPLE_RATE)
    envelope = np.minimum(t / 0.08, 1.0) * np.minimum((float(sample_count) / SAMPLE_RATE - t) / 0.08, 1.0)
    envelope = np.clip(envelope, 0.0, 1.0)
    audio = (
        0.055 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.030 * np.sin(2.0 * np.pi * 360.0 * t + 0.4)
        + 0.015 * np.sin(2.0 * np.pi * 720.0 * t + 0.8)
    )
    return np.nan_to_num(audio * envelope, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def patch_torchaudio_compat() -> None:
    try:
        import torchaudio
    except Exception:
        return
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    if "torchaudio.sox_effects" not in sys.modules:
        sox_effects = types.ModuleType("torchaudio.sox_effects")

        def apply_effects_tensor(waveform: Any, sample_rate: int, _effects: Any) -> tuple[Any, int]:
            return waveform, sample_rate

        sox_effects.apply_effects_tensor = apply_effects_tensor  # type: ignore[attr-defined]
        sys.modules["torchaudio.sox_effects"] = sox_effects


class SpeechBrainProvider:
    def __init__(self, device: str, model_id: str) -> None:
        configure_env()
        import torch
        from speechbrain_compat import load_speechbrain_encoder

        self.torch = torch
        self.device = choose_torch_device(device)
        self.lock = threading.Lock()
        savedir = ROOT / ".cache" / "speechbrain" / sanitize(model_id)
        self.model = load_speechbrain_encoder(model_id, str(savedir), self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        waveform = self.torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.lock:
            with self.torch.inference_mode():
                embedding = self.model.encode_batch(waveform, normalize=False)
        return normalize_vector(embedding)


class ResemblyzerProvider:
    def __init__(self, device: str) -> None:
        configure_env()
        from resemblyzer import VoiceEncoder

        self.device = choose_torch_device(device)
        self.encoder = VoiceEncoder(device=self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        return normalize_vector(self.encoder.embed_utterance(np.asarray(audio, dtype=np.float32).reshape(-1)))


class PyannoteModelProvider:
    def __init__(self, device: str, model_id: str) -> None:
        configure_env()
        import torch
        from pyannote.audio import Inference, Model

        token = os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.torch = torch
        self.device = torch.device(choose_torch_device(device))
        load_kwargs: dict[str, Any] = {"cache_dir": str(ROOT / ".cache" / "pyannote")}
        if token:
            load_kwargs["use_auth_token"] = token
        if model_id == "pyannote/embedding":
            load_kwargs["strict"] = False
        original_torch_load = torch.load

        def trusted_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = trusted_torch_load
        try:
            model = Model.from_pretrained(model_id, **load_kwargs)
        finally:
            torch.load = original_torch_load
        if model is None:
            raise RuntimeError(f"{model_id} could not be loaded")
        self.inference = Inference(model, window="whole", device=self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        waveform = self.torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
        return normalize_vector(self.inference({"waveform": waveform, "sample_rate": sample_rate}))


class BenchmarkAdapterProvider:
    def __init__(self, device: str, engine_id: str) -> None:
        configure_env()
        import torch
        from benchmark_voice_embeddings import ADAPTERS, ENGINES, configure_env as configure_benchmark_env

        configure_benchmark_env()
        if engine_id not in ENGINES:
            raise ValueError(f"unknown benchmark embedding engine {engine_id!r}")
        self.device = choose_torch_device(device)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = ENGINES[engine_id]
        patch_torchaudio_compat()
        self.adapter = ADAPTERS[engine["kind"]](engine["model"], self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        return normalize_vector(self.adapter.infer(np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate))


def create_single_provider(provider: str, device: str) -> Any:
    name = canonical_provider_name(provider)
    if name == "speechbrain_ecapa":
        return SpeechBrainProvider(device=device, model_id="speechbrain/spkrec-ecapa-voxceleb")
    if name == "speechbrain_resnet":
        return SpeechBrainProvider(device=device, model_id="speechbrain/spkrec-resnet-voxceleb")
    if name == "speechbrain_xvector":
        return SpeechBrainProvider(device=device, model_id="speechbrain/spkrec-xvect-voxceleb")
    if name == "resemblyzer":
        return ResemblyzerProvider(device=device)
    if name == "pyannote_embedding":
        return PyannoteModelProvider(device=device, model_id="pyannote/embedding")
    if name == "pyannote_wespeaker_resnet34_lm":
        return PyannoteModelProvider(device=device, model_id="pyannote/wespeaker-voxceleb-resnet34-LM")
    if name in {
        "wespeaker_campplus",
        "wespeaker_resnet34_lm_onnx",
        "speaker3d_campplus",
        "speaker3d_eres2netv2",
        "nemo_titanet_large",
        "espnet_rawnet3",
        "espnet_ecapa_wavlm_joint",
        "jungjee_rawnet3",
        "wavlm_base_sv",
    }:
        return BenchmarkAdapterProvider(device=device, engine_id=name)
    raise ValueError(f"unsupported provider {provider!r}")


def provider_key(provider: str, device: str) -> tuple[str, str]:
    return canonical_provider_name(provider), resolve_runtime_device(device)


def get_provider(provider: str, device: str) -> Any:
    key = provider_key(provider, device)
    with _cache_lock:
        if key in _provider_cache:
            return _provider_cache[key]
        lock = _provider_locks.setdefault(key, threading.Lock())
    with lock:
        with _cache_lock:
            if key in _provider_cache:
                return _provider_cache[key]
        loaded = create_single_provider(key[0], key[1])
        with _cache_lock:
            _provider_cache[key] = loaded
        return loaded


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    try:
        container = av.open(io.BytesIO(audio_bytes), mode="r")
    except av.FFmpegError as exc:
        raise HTTPException(status_code=400, detail=f"audio_decode_failed: {exc}") from exc
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise HTTPException(status_code=400, detail="no_audio_stream")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                array = resampled.to_ndarray()
                chunks.append(array.reshape(-1).astype(np.float32, copy=False) / 32768.0)
    finally:
        container.close()
    if not chunks:
        raise HTTPException(status_code=400, detail="no_audio_samples")
    return np.concatenate(chunks)


def raw_bytes_to_float32(audio_bytes: bytes, sample_rate: int, encoding: str) -> tuple[np.ndarray, int]:
    normalized = encoding.lower().replace("-", "").replace("_", "")
    if normalized in {"pcm16", "s16le", "int16"}:
        if len(audio_bytes) % 2:
            raise HTTPException(status_code=400, detail="pcm16_payload_length_must_be_even")
        return np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0, sample_rate
    if normalized in {"float32", "f32le"}:
        if len(audio_bytes) % 4:
            raise HTTPException(status_code=400, detail="float32_payload_length_must_be_multiple_of_4")
        return np.frombuffer(audio_bytes, dtype="<f4").astype(np.float32, copy=False), sample_rate
    raise HTTPException(status_code=400, detail="unsupported_audio_encoding")


def embed_one(name: str, weight: float, device: str, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        provider = get_provider(name, device)
        vector = normalize_vector(provider.embed(audio, sample_rate))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{name}_provider_failed: {type(exc).__name__}: {exc}",
        ) from exc
    elapsed = time.perf_counter() - started
    actual_device = getattr(provider, "device", device)
    return {
        "provider": name,
        "weight": weight,
        "device": str(actual_device),
        "elapsed_seconds": elapsed,
        "embedding": vector,
    }


def warmup_one(name: str, weight: float, device: str) -> dict[str, Any]:
    result = embed_one(name, weight, device, warmup_audio(), SAMPLE_RATE)
    result["dim"] = int(result["embedding"].shape[0])
    result["warmup_seconds"] = WARMUP_SECONDS
    result.pop("embedding", None)
    return result


def embed_stack(provider_spec: str, device: str, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    specs = [(name, weight) for name, weight in parse_provider_stack_specs(provider_spec) if weight > 0.0]
    if not specs:
        raise HTTPException(status_code=400, detail="provider stack has no positive weights")

    runtime_device = resolve_runtime_device(device)
    if is_cuda_device(runtime_device):
        cpu_specs: list[tuple[str, float]] = []
        gpu_specs = specs
    else:
        cpu_specs = specs
        gpu_specs: list[tuple[str, float]] = []
    results: list[dict[str, Any]] = []

    if cpu_specs and len(cpu_specs) > 1:
        with ThreadPoolExecutor(max_workers=len(cpu_specs)) as pool:
            futures = [pool.submit(embed_one, name, weight, runtime_device, audio, sample_rate) for name, weight in cpu_specs]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for name, weight in cpu_specs:
            results.append(embed_one(name, weight, runtime_device, audio, sample_rate))

    for name, weight in gpu_specs:
        results.append(embed_one(name, weight, runtime_device, audio, sample_rate))

    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_name.setdefault(str(item["provider"]), []).append(item)
    ordered = [by_name[name].pop(0) for name, _weight in specs]
    vectors = [normalize_vector(item["embedding"]) * float(item["weight"]) for item in ordered]
    stacked = normalize_vector(np.concatenate(vectors))
    for item in ordered:
        item["dim"] = int(item["embedding"].shape[0])
        item.pop("embedding", None)
    return stacked, ordered, runtime_device


@app.get("/health")
def health() -> dict[str, Any]:
    with _cache_lock:
        loaded = [{"provider": key[0], "device": key[1]} for key in _provider_cache]
    return {
        "ok": True,
        "service": "voice-embeddings-server",
        "port": PORT,
        "default_provider": DEFAULT_PROVIDER,
        "default_device": DEFAULT_DEVICE,
        "loaded": loaded,
    }


@app.get("/providers")
def providers() -> dict[str, Any]:
    return {
        "providers": [
            {"id": provider, "name": DISPLAY_NAMES[provider]}
            for provider in SUPPORTED_PROVIDERS
        ],
        "aliases": BENCHMARK_PROVIDER_ALIASES,
        "stack_syntax": "provider_a=0.7+provider_b=1.0+provider_c=0.35",
    }


@app.post("/load")
def load(provider: str = Query(DEFAULT_PROVIDER), device: str = Query(DEFAULT_DEVICE)) -> dict[str, Any]:
    started = time.perf_counter()
    specs = parse_provider_stack_specs(provider)
    runtime_device = resolve_runtime_device(device)
    unique_specs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for name, weight in specs:
        if weight <= 0.0:
            continue
        if name in seen:
            continue
        seen.add(name)
        unique_specs.append((name, weight))

    warmups: list[dict[str, Any]] = []
    if unique_specs and is_cuda_device(runtime_device):
        for name, weight in unique_specs:
            warmups.append(warmup_one(name, weight, runtime_device))
    elif len(unique_specs) > 1:
        with ThreadPoolExecutor(max_workers=len(unique_specs)) as pool:
            futures = [pool.submit(warmup_one, name, weight, runtime_device) for name, weight in unique_specs]
            by_provider = {}
            for future in as_completed(futures):
                item = future.result()
                by_provider[item["provider"]] = item
        warmups = [by_provider[name] for name, _weight in unique_specs]
    else:
        for name, weight in unique_specs:
            warmups.append(warmup_one(name, weight, runtime_device))

    loaded = [
        {
            "provider": item["provider"],
            "weight": item["weight"],
            "device": item["device"],
            "dim": item["dim"],
        }
        for item in warmups
    ]
    return {
        "ok": True,
        "requested_device": device,
        "resolved_device": runtime_device,
        "warmup_seconds": WARMUP_SECONDS,
        "loaded": loaded,
        "warmups": warmups,
        "elapsed_seconds": time.perf_counter() - started,
    }


@app.post("/unload")
def unload(provider: str | None = Query(None), device: str = Query(DEFAULT_DEVICE)) -> dict[str, Any]:
    with _cache_lock:
        if provider is None or provider == "all":
            count = len(_provider_cache)
            _provider_cache.clear()
            return {"ok": True, "unloaded": count}
        specs = parse_provider_stack_specs(provider)
        count = 0
        for name, _weight in specs:
            key = provider_key(name, device)
            if key in _provider_cache:
                del _provider_cache[key]
                count += 1
        return {"ok": True, "unloaded": count}


def embedding_response(provider: str, device: str, audio: np.ndarray, sample_rate: int, input_mode: str, decode_seconds: float) -> JSONResponse:
    started = time.perf_counter()
    embedding, components, runtime_device = embed_stack(provider, device, audio, sample_rate)
    elapsed = time.perf_counter() - started
    return JSONResponse({
        "provider": provider,
        "canonical_stack": [
            {"provider": name, "weight": weight}
            for name, weight in parse_provider_stack_specs(provider)
        ],
        "requested_device": device,
        "device": runtime_device,
        "input_mode": input_mode,
        "sample_rate": sample_rate,
        "duration": float(len(audio)) / float(sample_rate or SAMPLE_RATE),
        "decode_seconds": decode_seconds,
        "elapsed_seconds": elapsed,
        "dim": int(embedding.shape[0]),
        "embedding": embedding.astype(float).tolist(),
        "components": components,
    })


@app.post("/embed-pcm16")
@app.post("/embed-window")
async def embed_raw(
    request: Request,
    provider: str = Query(DEFAULT_PROVIDER),
    device: str = Query(DEFAULT_DEVICE),
    sample_rate: int = Query(SAMPLE_RATE),
    encoding: str = Query("pcm16"),
) -> JSONResponse:
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty_audio_payload")
    started = time.perf_counter()
    audio, sr = raw_bytes_to_float32(payload, sample_rate, encoding)
    decode_seconds = time.perf_counter() - started
    return embedding_response(provider, device, audio, sr, encoding.lower(), decode_seconds)


@app.post("/embed")
async def embed_encoded(
    request: Request,
    provider: str = Query(DEFAULT_PROVIDER),
    device: str = Query(DEFAULT_DEVICE),
) -> JSONResponse:
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty_audio_payload")
    started = time.perf_counter()
    audio = decode_audio_bytes(payload)
    decode_seconds = time.perf_counter() - started
    return embedding_response(provider, device, audio, SAMPLE_RATE, "encoded_memory", decode_seconds)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("embeddings_server:app", host=HOST, port=PORT, log_level="info")

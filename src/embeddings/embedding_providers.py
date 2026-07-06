"""Embedding provider factory and helper subprocess protocol."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from common.audio_utils import (
    SAMPLE_RATE,
    json_dumps,
    load_audio_file,
    normalize_vector,
    pad_audio,
    trim_silence,
)
from paths import CACHE_DIR, EMBEDDING_VENV, PROJECT_ROOT

DEFAULT_HELPER_MODULE = "realtime.realtime_speakerdiarize"

DEFAULT_EMBEDDING_PROVIDER = "speechbrain_ecapa"
DEFAULT_SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SPEECHBRAIN_RESNET_MODEL = "speechbrain/spkrec-resnet-voxceleb"

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


def _load_project_env() -> None:
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _normalize_hf_token_env() -> bool:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    if not token:
        return False
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HF_ACCESS_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    return True


def configure_embedding_env() -> None:
    _load_project_env()
    cache = CACHE_DIR
    env_defaults = {
        "HF_HOME": cache / "huggingface",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "MPLCONFIGDIR": cache / "matplotlib",
        "XDG_CACHE_HOME": cache,
        "NUMBA_CACHE_DIR": cache / "numba",
        "WESPEAKER_HOME": cache / "wespeaker",
        "NLTK_DATA": cache / "nltk",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, str(value))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    s3prl_download_dir = cache / "s3prl" / "download"
    s3prl_download_dir.mkdir(parents=True, exist_ok=True)
    try:
        from importlib import import_module

        s3prl_download = import_module("s3prl.util.download")
        s3prl_download.set_dir(s3prl_download_dir)
    except Exception:
        pass

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not _normalize_hf_token_env():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def canonical_embedding_provider_name(value: str) -> str:
    provider = str(value or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    provider = provider.replace("/", "_").replace("-", "_")
    provider = re.sub(r"[^a-z0-9_]+", "_", provider).strip("_")
    provider = re.sub(r"_+", "_", provider)
    return BENCHMARK_PROVIDER_ALIASES.get(provider, provider)


def default_embedding_python() -> Path:
    candidate = EMBEDDING_VENV / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def choose_torch_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpeechBrainProvider:
    def __init__(self, device: str, model_id: str) -> None:
        configure_embedding_env()
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.device = choose_torch_device(device)
        savedir = CACHE_DIR / "speechbrain" / sanitize(model_id)
        self.model = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=str(savedir),
            run_opts={"device": self.device},
        )

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        waveform = self.torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            embedding = self.model.encode_batch(waveform, normalize=False)
        return normalize_vector(embedding)


class ResemblyzerProvider:
    def __init__(self, device: str) -> None:
        configure_embedding_env()
        from resemblyzer import VoiceEncoder

        self.device = choose_torch_device(device)
        self.encoder = VoiceEncoder(device=self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        return normalize_vector(self.encoder.embed_utterance(audio))


class PyannoteModelProvider:
    def __init__(self, device: str, model_id: str) -> None:
        configure_embedding_env()
        import torch
        from pyannote.audio import Inference, Model

        token = (
            os.getenv("HF_ACCESS_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.torch = torch
        self.device = torch.device(choose_torch_device(device))
        original_torch_load = torch.load

        def trusted_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
            load_kwargs["weights_only"] = False
            return original_torch_load(*load_args, **load_kwargs)

        load_kwargs: dict[str, Any] = {"cache_dir": str(CACHE_DIR / "pyannote")}
        if token:
            load_kwargs["use_auth_token"] = token
        if model_id == "pyannote/embedding":
            load_kwargs["strict"] = False

        torch.load = trusted_torch_load
        try:
            model = Model.from_pretrained(model_id, **load_kwargs)
        finally:
            torch.load = original_torch_load
        if model is None:
            raise RuntimeError(f"{model_id} could not be loaded from the local pyannote cache.")
        inference_kwargs: dict[str, Any] = {"window": "whole", "device": self.device}
        if token:
            inference_kwargs["use_auth_token"] = token
        self.inference = Inference(model, **inference_kwargs)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        waveform = self.torch.from_numpy(audio).unsqueeze(0)
        embedding = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        return normalize_vector(embedding)


class BenchmarkAdapterProvider:
    def __init__(self, device: str, engine_id: str) -> None:
        configure_embedding_env()
        import torch

        from embeddings.benchmark_voice_embeddings import ADAPTERS, ENGINES, configure_env

        configure_env()
        if engine_id not in ENGINES:
            raise ValueError(f"Unknown benchmark embedding engine {engine_id!r}.")
        engine = ENGINES[engine_id]
        self.device = choose_torch_device(device)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adapter = ADAPTERS[engine["kind"]](engine["model"], self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        return normalize_vector(self.adapter.infer(audio, sample_rate))


class StackedEmbeddingProvider:
    def __init__(self, providers: list[Any], weights: list[float] | None = None) -> None:
        if len(providers) < 2:
            raise ValueError("A stacked embedding provider needs at least two providers.")
        if weights is None:
            weights = [1.0] * len(providers)
        if len(weights) != len(providers):
            raise ValueError("Stacked embedding provider weights must match providers.")
        if not any(float(weight) > 0.0 for weight in weights):
            raise ValueError("At least one stacked embedding provider weight must be positive.")
        self.providers = providers
        self.weights = [float(weight) for weight in weights]

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        vectors = [
            normalize_vector(provider.embed(audio, sample_rate)) * weight
            for provider, weight in zip(self.providers, self.weights)
            if weight > 0.0
        ]
        return normalize_vector(np.concatenate(vectors))


def parse_embedding_provider_stack_specs(provider: str) -> list[tuple[str, float]]:
    value = (provider or DEFAULT_EMBEDDING_PROVIDER).strip()
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
            raise ValueError(f"Embedding provider weights must be non-negative: {item!r}")
        specs.append((canonical_embedding_provider_name(raw_provider), weight))
    return specs


def parse_embedding_provider_stack(provider: str) -> list[str]:
    return [name for name, _weight in parse_embedding_provider_stack_specs(provider)]


def create_single_embedding_provider(provider: str, device: str) -> Any:
    provider = canonical_embedding_provider_name(provider)
    if provider in {"speechbrain_ecapa", "ecapa"}:
        return SpeechBrainProvider(device=device, model_id=DEFAULT_SPEECHBRAIN_MODEL)
    if provider in {"speechbrain_resnet", "resnet", "speechbrain_resnet_voxceleb"}:
        return SpeechBrainProvider(device=device, model_id=DEFAULT_SPEECHBRAIN_RESNET_MODEL)
    if provider == "resemblyzer":
        return ResemblyzerProvider(device=device)
    if provider in {"pyannote_embedding", "pyannote"}:
        return PyannoteModelProvider(device=device, model_id="pyannote/embedding")
    if provider in {
        "pyannote_wespeaker",
        "pyannote_wespeaker_resnet34_lm",
        "pyannote_wespeaker_voxceleb_resnet34_lm",
    }:
        return PyannoteModelProvider(
            device=device,
            model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        )
    benchmark_provider_ids = {
        "wespeaker_campplus",
        "wespeaker_resnet34_lm_onnx",
        "speaker3d_campplus",
        "speaker3d_eres2netv2",
        "nemo_titanet_large",
        "speechbrain_xvector",
        "espnet_rawnet3",
        "espnet_ecapa_wavlm_joint",
        "jungjee_rawnet3",
        "wavlm_base_sv",
    }
    if provider in benchmark_provider_ids:
        return BenchmarkAdapterProvider(device=device, engine_id=provider)
    raise ValueError(
        f"Unsupported embedding provider {provider!r}. Supported providers: "
        "speechbrain_ecapa, speechbrain_resnet, resemblyzer, pyannote_embedding, "
        "pyannote_wespeaker_resnet34_lm, benchmark provider IDs from "
        "whospeaks-embedding-benchmark, or a '+' stack of them."
    )


def create_embedding_provider(provider: str, device: str) -> Any:
    specs = parse_embedding_provider_stack_specs(provider)
    if not specs:
        specs = [(DEFAULT_EMBEDDING_PROVIDER, 1.0)]
    if len(specs) == 1:
        return create_single_embedding_provider(specs[0][0], device)
    return StackedEmbeddingProvider(
        [create_single_embedding_provider(name, device) for name, _weight in specs],
        [weight for _name, weight in specs],
    )


class EmbeddingSubprocessClient:
    def __init__(
        self,
        python: Path,
        provider: str,
        device: str,
        helper_script: Path | None = None,
        response_timeout_seconds: float = 120.0,
    ) -> None:
        self.python = Path(python)
        self.provider = provider
        self.device = device
        self.helper_script = Path(helper_script) if helper_script is not None else None
        self.response_timeout_seconds = max(0.1, float(response_timeout_seconds))
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._noise_lines: list[str] = []
        self._stdout_messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stdout_closed = threading.Event()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._stdout_messages = queue.Queue()
        self._stdout_closed = threading.Event()
        self._stderr_lines = []
        self._noise_lines = []
        command = [
            str(self.python),
            *(["-m", DEFAULT_HELPER_MODULE] if self.helper_script is None else [str(self.helper_script)]),
            "--embedding-helper",
            "--embedding-provider",
            self.provider,
            "--embedding-device",
            self.device,
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        process = self._process
        stdout_closed = self._stdout_closed
        threading.Thread(
            target=self._collect_stderr,
            args=(process,),
            name="EmbeddingHelperStderr",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._collect_stdout,
            args=(process, stdout_closed),
            name="EmbeddingHelperStdout",
            daemon=True,
        ).start()

    def embed_wav(self, path: Path) -> np.ndarray:
        with self._lock:
            self.start()
            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Embedding helper did not start.")
            if self._process.poll() is not None:
                raise RuntimeError(
                    "Embedding helper exited before processing audio. "
                    + self._stderr_tail()
                )
            request = {"cmd": "embed", "path": str(path)}
            self._process.stdin.write(json_dumps(request) + "\n")
            self._process.stdin.flush()
            try:
                response = self._read_json_response()
            except TimeoutError:
                self._kill_process_locked()
                raise
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "Embedding failed"))
            return normalize_vector(response["embedding"])

    def shutdown(self, lock_timeout_seconds: float = 5.0) -> None:
        acquired = self._lock.acquire(timeout=max(0.0, float(lock_timeout_seconds)))
        if not acquired:
            self._kill_process_locked()
            return
        try:
            process = self._process
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write(json_dumps({"cmd": "shutdown"}) + "\n")
                    process.stdin.flush()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                self._kill_process_locked(process)
            self._process = None
        finally:
            self._lock.release()

    def _read_json_response(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.response_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"Embedding helper did not respond within {self.response_timeout_seconds:.1f}s. "
                    + self._stderr_tail()
                )
            try:
                return self._stdout_messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(
                        "Embedding helper exited before returning a response. "
                        + self._stderr_tail()
                    )
                if self._stdout_closed.is_set():
                    raise RuntimeError(
                        "Embedding helper closed stdout. "
                        + self._stderr_tail()
                    )

    def _collect_stdout(
        self,
        process: subprocess.Popen[str] | None,
        stdout_closed: threading.Event,
    ) -> None:
        if process is None or process.stdout is None:
            stdout_closed.set()
            return
        try:
            for line in process.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    self._stdout_messages.put(json.loads(stripped))
                except json.JSONDecodeError:
                    self._noise_lines.append(stripped)
                    self._noise_lines = self._noise_lines[-20:]
        finally:
            stdout_closed.set()

    def _kill_process_locked(self, process: subprocess.Popen[str] | None = None) -> None:
        process = process or self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
        except Exception:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        if process is self._process:
            self._process = None

    def _collect_stderr(self, process: subprocess.Popen[str] | None) -> None:
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.rstrip()
            if text:
                self._stderr_lines.append(text)
                self._stderr_lines = self._stderr_lines[-30:]

    def _stderr_tail(self) -> str:
        lines = self._stderr_lines[-8:] + self._noise_lines[-4:]
        if not lines:
            return ""
        return "Helper output: " + " | ".join(lines)


class RemoteEmbeddingClient:
    """HTTP client for the Linux GPU embeddings server."""

    def __init__(
        self,
        base_url: str,
        provider: str,
        device: str = "auto",
        timeout_seconds: float = 600.0,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("Remote embeddings base URL must not be empty.")
        self.provider = str(provider or DEFAULT_EMBEDDING_PROVIDER)
        self.device = str(device or "auto")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._loaded = False
        self._lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        raw = self._read_url(f"{self.base_url}/health", timeout=min(self.timeout_seconds, 10.0))
        return self._json_response(raw, "health")

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_locked()

    def embed_audio(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        with self._lock:
            if not self._loaded:
                self._load_locked()
            prepared = pad_audio(trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate), 0.5, sample_rate)
            pcm16 = (np.clip(prepared, -1.0, 1.0) * 32767.0).astype(np.int16)
            query = urlencode({
                "provider": self.provider,
                "device": self.device,
                "sample_rate": int(sample_rate),
                "encoding": "pcm16",
            })
            request = Request(
                f"{self.base_url}/embed-pcm16?{query}",
                data=np.ascontiguousarray(pcm16).tobytes(),
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            result = self._json_response(
                self._open_request(request, timeout=self.timeout_seconds),
                "embed-pcm16",
            )
        if result.get("error"):
            raise RuntimeError(f"Remote embeddings error: {result['error']}")
        embedding = self._embedding_from_result(result)
        return normalize_vector(embedding)

    def embed_wav(self, path: Path) -> np.ndarray:
        audio, sample_rate = load_audio_file(path)
        return self.embed_audio(audio, sample_rate)

    def shutdown(self, lock_timeout_seconds: float = 5.0) -> None:
        return None

    def _load_locked(self) -> dict[str, Any]:
        if self._loaded:
            return {"ok": True, "cached": True}
        query = urlencode({"provider": self.provider, "device": self.device})
        request = Request(f"{self.base_url}/load?{query}", data=b"", method="POST")
        result = self._json_response(
            self._open_request(request, timeout=self.timeout_seconds),
            "load",
        )
        if result.get("error"):
            raise RuntimeError(f"Remote embeddings load error: {result['error']}")
        if result.get("ok") is False:
            raise RuntimeError(f"Remote embeddings load failed: {result}")
        self._loaded = True
        return result

    def _embedding_from_result(self, result: dict[str, Any]) -> Any:
        for key in ("embedding", "vector"):
            if key in result:
                return result[key]
        embeddings = result.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return embeddings[0]
        raise RuntimeError("Remote embeddings response did not include an embedding vector.")

    def _json_response(self, raw: bytes, action: str) -> dict[str, Any]:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Remote embeddings returned non-JSON {action} response.") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Remote embeddings returned an unexpected {action} response.")
        return data

    def _read_url(self, url: str, timeout: float) -> bytes:
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Remote embeddings HTTP {exc.code}: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Remote embeddings connection failed: {exc.reason}") from exc

    def _open_request(self, request: Request, timeout: float) -> bytes:
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Remote embeddings HTTP {exc.code}: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Remote embeddings connection failed: {exc.reason}") from exc


def run_embedding_helper(args: argparse.Namespace) -> int:
    provider = create_embedding_provider(args.embedding_provider, args.embedding_device)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                print(json_dumps({"ok": True}), flush=True)
                return 0
            if command != "embed":
                raise ValueError(f"Unknown helper command: {command!r}")
            path = Path(str(request["path"]))
            audio, sample_rate = load_audio_file(path)
            audio = trim_silence(audio, sample_rate)
            audio = pad_audio(audio, 0.5, sample_rate)
            embedding = provider.embed(audio, sample_rate)
            print(
                json_dumps({
                    "ok": True,
                    "embedding": embedding.astype(float).tolist(),
                    "seconds": len(audio) / float(sample_rate),
                }),
                flush=True,
            )
        except Exception as exc:
            print(
                json_dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }),
                flush=True,
            )
    return 0

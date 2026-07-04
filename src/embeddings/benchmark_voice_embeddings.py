from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from paths import CACHE_DIR, OUTPUTS_DIR, PROJECT_ROOT

OUT_DIR = OUTPUTS_DIR / "benchmarks" / "voice_embeddings"
DEFAULT_AUDIO = OUT_DIR / "five_second_voice.wav"
RESULTS_JSONL = OUT_DIR / "results.jsonl"
SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"
SAMPLE_RATE = 16000


ENGINES: dict[str, dict[str, str]] = {
    "speechbrain_ecapa": {
        "name": "speechbrain/spkrec-ecapa-voxceleb",
        "kind": "speechbrain",
        "model": "speechbrain/spkrec-ecapa-voxceleb",
    },
    "pyannote_embedding": {
        "name": "pyannote/embedding",
        "kind": "pyannote_model",
        "model": "pyannote/embedding",
    },
    "pyannote_wespeaker_resnet34_lm": {
        "name": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "kind": "pyannote_model",
        "model": "pyannote/wespeaker-voxceleb-resnet34-LM",
    },
    "wespeaker_campplus": {
        "name": "Wespeaker CAM++",
        "kind": "wespeaker_native",
        "model": "campplus",
    },
    "wespeaker_resnet34_lm_onnx": {
        "name": "Wespeaker ResNet34-LM",
        "kind": "pyannote_pretrained_embedding",
        "model": "hbredin/wespeaker-voxceleb-resnet34-LM",
    },
    "speaker3d_campplus": {
        "name": "3D-Speaker CAM++",
        "kind": "modelscope_sv",
        "model": "iic/speech_campplus_sv_zh-cn_16k-common",
    },
    "speaker3d_eres2netv2": {
        "name": "3D-Speaker ERes2NetV2",
        "kind": "modelscope_sv",
        "model": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    },
    "nemo_titanet_large": {
        "name": "nvidia/speakerverification_en_titanet_large",
        "kind": "nemo_titanet",
        "model": "nvidia/speakerverification_en_titanet_large",
    },
    "speechbrain_resnet": {
        "name": "speechbrain/spkrec-resnet-voxceleb",
        "kind": "speechbrain",
        "model": "speechbrain/spkrec-resnet-voxceleb",
    },
    "speechbrain_xvector": {
        "name": "speechbrain/spkrec-xvect-voxceleb",
        "kind": "speechbrain",
        "model": "speechbrain/spkrec-xvect-voxceleb",
    },
    "espnet_rawnet3": {
        "name": "espnet/voxcelebs12_rawnet3",
        "kind": "espnet",
        "model": "espnet/voxcelebs12_rawnet3",
    },
    "espnet_ecapa_wavlm_joint": {
        "name": "espnet/voxcelebs12_ecapa_wavlm_joint",
        "kind": "espnet",
        "model": "espnet/voxcelebs12_ecapa_wavlm_joint",
    },
    "jungjee_rawnet3": {
        "name": "jungjee/RawNet3",
        "kind": "rawnet3_local",
        "model": str(CACHE_DIR / "source" / "RawNet" / "python" / "RawNet3"),
    },
    "wavlm_base_sv": {
        "name": "microsoft/wavlm-base-sv",
        "kind": "wavlm_xvector",
        "model": "microsoft/wavlm-base-sv",
    },
    "resemblyzer": {
        "name": "Resemblyzer",
        "kind": "resemblyzer",
        "model": "resemblyzer",
    },
}


def configure_env() -> None:
    cache = CACHE_DIR
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

    s3prl_download_dir = cache / "s3prl" / "download"
    s3prl_download_dir.mkdir(parents=True, exist_ok=True)
    try:
        from importlib import import_module

        s3prl_download = import_module("s3prl.util.download")
        s3prl_download.set_dir(s3prl_download_dir)
    except Exception:
        pass

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def resolve_modelscope_model_path(model_id: str) -> str:
    provider, _, model_name = str(model_id).partition("/")
    if not provider or not model_name:
        return str(model_id)
    candidates = [
        CACHE_DIR / "modelscope" / "models" / provider / model_name,
        CACHE_DIR / "modelscope" / provider / model_name,
    ]
    for candidate in candidates:
        if (candidate / "configuration.json").is_file() or (candidate / "config.yaml").is_file():
            return str(candidate)
    return str(model_id)


def disable_windows_error_dialogs() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        sem_failcriticalerrors = 0x0001
        sem_nogpfaultbox = 0x0002
        sem_noopenfileerrorbox = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            sem_failcriticalerrors | sem_nogpfaultbox | sem_noopenfileerrorbox
        )
    except Exception:
        pass


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_audio(audio_path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    target = SAMPLE_RATE * 5
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sr


def ensure_normalized_audio(audio_path: Path) -> None:
    import soundfile as sf

    audio, sr = load_audio(audio_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(DEFAULT_AUDIO), audio, sr, subtype="PCM_16")


class VramMonitor:
    def __init__(self, interval_seconds: float = 0.001) -> None:
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, int | None]] = []
        self._pynvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None
            self._handle = None

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _bytes_to_mib(value: int | None) -> int | None:
        if value is None:
            return None
        return int(round(value / (1024 * 1024)))

    def snapshot(self) -> dict[str, int | None]:
        if self._pynvml is None or self._handle is None:
            return {"global_mib": None, "process_mib": None}

        global_used = None
        process_used = None
        try:
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            global_used = self._bytes_to_mib(int(mem.used))
            total = int(mem.total)
        except Exception:
            total = 0

        process_bytes: int | None = None
        getters = [
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ]
        for getter_name in getters:
            getter = getattr(self._pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                for proc in getter(self._handle):
                    if int(proc.pid) == self.pid:
                        used = getattr(proc, "usedGpuMemory", None)
                        if used is None:
                            continue
                        used = int(used)
                        if total and used > total:
                            continue
                        process_bytes = max(process_bytes or 0, used)
            except Exception:
                continue

        process_used = self._bytes_to_mib(process_bytes)
        return {"global_mib": global_used, "process_mib": process_used}

    def start(self) -> None:
        self.samples = []
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.peak()

    def _loop(self) -> None:
        while self._running.is_set():
            self.samples.append(self.snapshot())
            time.sleep(self.interval_seconds)

    def peak(self) -> dict[str, int | None]:
        peak_global = max(
            (s["global_mib"] for s in self.samples if s["global_mib"] is not None),
            default=None,
        )
        peak_process = max(
            (s["process_mib"] for s in self.samples if s["process_mib"] is not None),
            default=None,
        )
        return {"global_mib": peak_global, "process_mib": peak_process}


def torch_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def tensor_shape(value: Any) -> list[int] | str | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(v) for v in value.shape]
    if isinstance(value, (list, tuple)):
        return [len(value)]
    return type(value).__name__


class SpeechBrainAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = CACHE_DIR / "speechbrain" / sanitize(model_id)
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=str(savedir),
            run_opts={"device": device},
        )
        self.torch = torch

    def infer(self, audio: Any, sample_rate: int) -> Any:
        wav = self.torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            emb = self.model.encode_batch(wav, normalize=False)
        return emb.detach().cpu()


class PyannoteModelAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from pyannote.audio import Inference, Model

        token = (
            os.getenv("HF_ACCESS_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        cache_dir = CACHE_DIR / "pyannote"
        original_torch_load = torch.load

        def trusted_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
            load_kwargs["weights_only"] = False
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = trusted_torch_load
        load_kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
        if token:
            load_kwargs["use_auth_token"] = token
        if model_id == "pyannote/embedding":
            load_kwargs["strict"] = False

        try:
            self.model = Model.from_pretrained(model_id, **load_kwargs)
        finally:
            torch.load = original_torch_load
        if self.model is None:
            raise RuntimeError(
                f"{model_id} could not be loaded. HF_TOKEN is "
                f"{'set' if token else 'not set'}, but pyannote.audio returned None."
            )
        inference_kwargs: dict[str, Any] = {"window": "whole", "device": torch.device(device)}
        if token:
            inference_kwargs["use_auth_token"] = token
        self.inference = Inference(self.model, **inference_kwargs)

    def infer(self, audio: Any, sample_rate: int) -> Any:
        import torch

        waveform = torch.from_numpy(audio).unsqueeze(0)
        return self.inference({"waveform": waveform, "sample_rate": sample_rate})


class PyannotePretrainedEmbeddingAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from pyannote.audio.pipelines.speaker_verification import (
            PretrainedSpeakerEmbedding,
        )

        token = (
            os.getenv("HF_ACCESS_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.torch = torch
        self.device = torch.device(device)
        model_kwargs: dict[str, Any] = {"device": self.device}
        if token:
            model_kwargs["use_auth_token"] = token
        self.model = PretrainedSpeakerEmbedding(model_id, **model_kwargs)

    def infer(self, audio: Any, sample_rate: int) -> Any:
        waveform = self.torch.from_numpy(audio).reshape(1, 1, -1)
        return self.model(waveform)


class WeSpeakerNativeAdapter:
    def __init__(self, model_name: str, device: str) -> None:
        import wespeaker

        self.model = wespeaker.load_model(model_name)
        if model_name in {"campplus", "eres2net"}:
            self.model.set_wavform_norm(True)
            self.model.set_window_type("povey")
        self.model.set_resample_rate(SAMPLE_RATE)
        self.model.set_vad(False)
        self.model.set_device(device)

    def infer(self, audio: Any, sample_rate: int) -> Any:
        import torch

        pcm = torch.from_numpy(audio).unsqueeze(0)
        pcm = pcm.to(torch.float)
        if sample_rate != self.model.resample_rate:
            pcm = __import__("torchaudio").transforms.Resample(
                orig_freq=sample_rate, new_freq=self.model.resample_rate
            )(pcm)
        feats = self.model.compute_features(
            pcm,
            sample_rate=self.model.resample_rate,
            cmn=True,
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        return outputs[0].detach().cpu()


class ModelScopeSpeakerVerificationAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        ms_device = "gpu" if device.startswith("cuda") else "cpu"
        model = resolve_modelscope_model_path(model_id)
        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=model,
            device=ms_device,
        )

    def infer(self, audio: Any, sample_rate: int) -> Any:
        result = self.pipeline([audio], output_emb=True)
        return result.get("embs")


class NemoTitanetAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.torch = torch
        self.device = device
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_id)
        self.model = self.model.to(device)
        self.model.eval()

    def infer(self, audio: Any, sample_rate: int) -> Any:
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            sf.write(str(path), audio, sample_rate, subtype="PCM_16")
            with self.torch.inference_mode():
                emb = self.model.get_embedding(str(path))
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu()
            return emb
        finally:
            path.unlink(missing_ok=True)


class EspnetAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        from espnet2.bin.spk_inference import Speech2Embedding
        from espnet_model_zoo.downloader import ModelDownloader

        cache_dir = Path(os.environ.get("ESPNET_MODEL_ZOO_CACHE", CACHE_DIR / "espnet_model_zoo"))
        downloader = ModelDownloader(cachedir=cache_dir)
        kwargs = downloader.download_and_unpack(model_id)
        self.model = Speech2Embedding(device=device, dtype="float32", **kwargs)

    def infer(self, audio: Any, sample_rate: int) -> Any:
        with __import__("torch").inference_mode():
            emb = self.model(audio)
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu()
        return emb


class RawNet3LocalAdapter:
    def __init__(self, model_dir: str, device: str) -> None:
        import torch

        model_root = Path(model_dir)
        if not model_root.exists():
            raise FileNotFoundError(
                f"RawNet3 source not found at {model_root}. Clone Jungjee/RawNet first."
            )
        sys.path.insert(0, str(model_root))
        from models.RawNet3 import RawNet3
        from models.RawNetBasicBlock import Bottle2neck

        self.torch = torch
        self.device = device
        self.model = RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1,
        )
        weights = model_root / "models" / "weights" / "model.pt"
        state = torch.load(str(weights), map_location="cpu")["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)

    def infer(self, audio: Any, sample_rate: int) -> Any:
        wav = self.torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            emb = self.model(wav)
        return emb.detach().cpu()


class WavLMXVectorAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def infer(self, audio: Any, sample_rate: int) -> Any:
        inputs = self.extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs)
        return output.embeddings.detach().cpu()


class ResemblyzerAdapter:
    def __init__(self, model_id: str, device: str) -> None:
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder(device=device)

    def infer(self, audio: Any, sample_rate: int) -> Any:
        return self.encoder.embed_utterance(audio)


ADAPTERS: dict[str, Callable[[str, str], Any]] = {
    "speechbrain": SpeechBrainAdapter,
    "pyannote_model": PyannoteModelAdapter,
    "pyannote_pretrained_embedding": PyannotePretrainedEmbeddingAdapter,
    "wespeaker_native": WeSpeakerNativeAdapter,
    "modelscope_sv": ModelScopeSpeakerVerificationAdapter,
    "nemo_titanet": NemoTitanetAdapter,
    "espnet": EspnetAdapter,
    "rawnet3_local": RawNet3LocalAdapter,
    "wavlm_xvector": WavLMXVectorAdapter,
    "resemblyzer": ResemblyzerAdapter,
}


def benchmark_child(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    configure_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = ENGINES[args.engine]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, sr = load_audio(Path(args.audio))
    monitor = VramMonitor()
    before_load = monitor.snapshot()
    result: dict[str, Any] = {
        "engine_id": args.engine,
        "name": engine["name"],
        "kind": engine["kind"],
        "model": engine["model"],
        "device": device,
        "audio_path": str(Path(args.audio).resolve()),
        "sample_rate": sr,
        "audio_seconds": round(len(audio) / sr, 4),
        "status": "ok",
        "before_load_global_vram_mib": before_load["global_mib"],
        "before_load_process_vram_mib": before_load["process_mib"],
    }

    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        load_start = time.perf_counter()
        adapter = ADAPTERS[engine["kind"]](engine["model"], device)
        torch_sync()
        result["load_time_ms"] = round((time.perf_counter() - load_start) * 1000, 3)
        after_load = monitor.snapshot()
        result["after_load_global_vram_mib"] = after_load["global_mib"]
        result["after_load_process_vram_mib"] = after_load["process_mib"]
        result["after_load_global_delta_mib"] = (
            after_load["global_mib"] - before_load["global_mib"]
            if after_load["global_mib"] is not None
            and before_load["global_mib"] is not None
            else None
        )
        result["after_load_process_delta_mib"] = (
            after_load["process_mib"] - (before_load["process_mib"] or 0)
            if after_load["process_mib"] is not None
            else None
        )

        warm_start = time.perf_counter()
        warm_embedding = adapter.infer(audio, sr)
        torch_sync()
        result["warmup_time_ms"] = round((time.perf_counter() - warm_start) * 1000, 3)
        result["embedding_shape"] = tensor_shape(warm_embedding)

        times_ms: list[float] = []
        monitor.start()
        for _ in range(args.repeats):
            start = time.perf_counter()
            embedding = adapter.infer(audio, sr)
            torch_sync()
            times_ms.append((time.perf_counter() - start) * 1000)
        peak = monitor.stop()
        result["timed_runs_ms"] = [round(v, 3) for v in times_ms]
        result["median_eval_time_ms"] = round(float(np.median(times_ms)), 3)
        result["mean_eval_time_ms"] = round(float(np.mean(times_ms)), 3)
        result["min_eval_time_ms"] = round(float(np.min(times_ms)), 3)
        result["max_eval_time_ms"] = round(float(np.max(times_ms)), 3)
        result["peak_infer_global_vram_mib"] = peak["global_mib"]
        result["peak_infer_process_vram_mib"] = peak["process_mib"]
        result["peak_infer_global_delta_mib"] = (
            peak["global_mib"] - before_load["global_mib"]
            if peak["global_mib"] is not None
            and before_load["global_mib"] is not None
            else None
        )
        result["peak_infer_process_delta_mib"] = (
            peak["process_mib"] - (before_load["process_mib"] or 0)
            if peak["process_mib"] is not None
            else None
        )
        if device.startswith("cuda"):
            result["torch_peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 3
            )
            result["torch_peak_reserved_mib"] = round(
                torch.cuda.max_memory_reserved() / (1024 * 1024), 3
            )
        result["final_embedding_shape"] = tensor_shape(embedding)
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=40)
    finally:
        monitor.close()
        if "adapter" in locals():
            del adapter
        gc.collect()
        if "torch" in sys.modules:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result["status"] == "ok" else 2


def flatten_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "engine_id",
        "name",
        "status",
        "device",
        "median_eval_time_ms",
        "mean_eval_time_ms",
        "min_eval_time_ms",
        "max_eval_time_ms",
        "warmup_time_ms",
        "load_time_ms",
        "before_load_global_vram_mib",
        "after_load_global_vram_mib",
        "peak_infer_global_vram_mib",
        "after_load_global_delta_mib",
        "peak_infer_global_delta_mib",
        "before_load_process_vram_mib",
        "after_load_process_vram_mib",
        "peak_infer_process_vram_mib",
        "after_load_process_delta_mib",
        "peak_infer_process_delta_mib",
        "torch_peak_allocated_mib",
        "torch_peak_reserved_mib",
        "embedding_shape",
        "error_type",
        "error",
    ]
    return {field: result.get(field) for field in fields}


def benchmark_parent(args: argparse.Namespace) -> int:
    configure_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_normalized_audio(Path(args.audio))

    engines = args.engines or list(ENGINES.keys())
    results: list[dict[str, Any]] = []
    RESULTS_JSONL.write_text("", encoding="utf-8")

    for engine_id in engines:
        if engine_id not in ENGINES:
            raise SystemExit(f"Unknown engine: {engine_id}")
        output = OUT_DIR / f"{engine_id}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--engine",
            engine_id,
            "--audio",
            str(DEFAULT_AUDIO),
            "--output",
            str(output),
            "--device",
            args.device,
            "--repeats",
            str(args.repeats),
        ]
        print(f"\n=== {ENGINES[engine_id]['name']} ===", flush=True)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=args.timeout_seconds,
            )
            elapsed = round((time.perf_counter() - start) * 1000, 3)
            if output.exists():
                result = json.loads(output.read_text(encoding="utf-8"))
            else:
                result = {
                    "engine_id": engine_id,
                    "name": ENGINES[engine_id]["name"],
                    "status": "crashed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                }
            result["child_wall_time_ms"] = elapsed
            result["returncode"] = proc.returncode
            if proc.returncode != 0 and result.get("status") == "ok":
                result["status"] = "error"
            if proc.stderr:
                result["stderr_tail"] = proc.stderr[-4000:]
            if proc.stdout:
                result["stdout_tail"] = proc.stdout[-4000:]
        except subprocess.TimeoutExpired as exc:
            result = {
                "engine_id": engine_id,
                "name": ENGINES[engine_id]["name"],
                "status": "timeout",
                "timeout_seconds": args.timeout_seconds,
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:],
            }
        results.append(result)
        with RESULTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")
        print(
            f"{result.get('status')} "
            f"median={result.get('median_eval_time_ms')}ms "
            f"peak_delta={result.get('peak_infer_global_delta_mib')}MiB",
            flush=True,
        )

    SUMMARY_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [flatten_for_csv(result) for result in results]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for result in results if result.get("status") == "ok")
    print(f"\nFinished: {ok}/{len(results)} engines succeeded", flush=True)
    print(f"JSON: {SUMMARY_JSON}", flush=True)
    print(f"CSV:  {SUMMARY_CSV}", flush=True)
    return 0 if ok == len(results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--engine")
    parser.add_argument("--engines", nargs="*")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--output", default=str(OUT_DIR / "child_result.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disable_windows_error_dialogs()
    configure_env()
    if args.child:
        if not args.engine:
            raise SystemExit("--engine is required with --child")
        return benchmark_child(args)
    return benchmark_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

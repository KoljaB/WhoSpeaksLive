"""Small HTTP sidecar for optional local transcript translation models.

The server intentionally keeps model inference outside the browser process.
Creating the provider is cheap; PyTorch, Transformers, and model weights are
loaded lazily by :class:`TransformersTranslationProvider` on the first POST.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .translation_service import (
    MADLAD_MODEL_ID,
    NLLB_MODEL_ID,
    TRANSLATEGEMMA_MODEL_ID,
    TranslationProvider,
    TranslationProviderConfig,
    create_translation_provider,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8799
MAX_REQUEST_BYTES = 1_048_576


@dataclass(frozen=True)
class LocalModelProfile:
    name: str
    family: str
    model: str
    description: str


LOCAL_MODEL_PROFILES: dict[str, LocalModelProfile] = {
    "translate-gemma-4b": LocalModelProfile(
        name="translate-gemma-4b",
        family="translategemma",
        model=TRANSLATEGEMMA_MODEL_ID,
        description="Recommended quality-first local model; gated by the Gemma terms.",
    ),
    "nllb-200-600m": LocalModelProfile(
        name="nllb-200-600m",
        family="nllb",
        model=NLLB_MODEL_ID,
        description="Broad, lighter research model; CC-BY-NC-4.0 weights.",
    ),
    "madlad-400-3b": LocalModelProfile(
        name="madlad-400-3b",
        family="madlad",
        model=MADLAD_MODEL_ID,
        description="Very broad Apache-2.0 translation model.",
    ),
}


@dataclass(frozen=True)
class TranslationServerConfig:
    model_profile: str = "translate-gemma-4b"
    model: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    max_input_tokens: int = 512
    trust_remote_code: bool = False
    local_files_only: bool = False

    @property
    def profile(self) -> LocalModelProfile:
        try:
            return LOCAL_MODEL_PROFILES[self.model_profile]
        except KeyError as exc:
            allowed = ", ".join(LOCAL_MODEL_PROFILES)
            raise ValueError(f"unknown model profile {self.model_profile!r}; choose one of: {allowed}") from exc

    @property
    def resolved_model(self) -> str:
        return str(self.model or self.profile.model).strip()

    def provider_config(self) -> TranslationProviderConfig:
        return TranslationProviderConfig(
            kind=self.profile.family,
            model=self.resolved_model,
            device=self.device,
            dtype=self.dtype,
            max_new_tokens=max(1, int(self.max_new_tokens)),
            max_input_tokens=max(8, int(self.max_input_tokens)),
            trust_remote_code=bool(self.trust_remote_code),
            local_files_only=bool(self.local_files_only),
        )


class TranslationHTTPError(Exception):
    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = str(code)
        self.message = str(message)


ProviderFactory = Callable[[TranslationProviderConfig | Mapping[str, Any]], TranslationProvider]


class TranslationSidecar:
    """Synchronous sidecar state; a lock serializes access to one local model."""

    def __init__(
        self,
        config: TranslationServerConfig,
        *,
        provider: TranslationProvider | None = None,
        provider_factory: ProviderFactory = create_translation_provider,
    ) -> None:
        self.config = config
        self.provider = provider or provider_factory(config.provider_config())
        self._translation_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        status = self.provider.status().to_dict()
        metadata = status.get("model_metadata")
        license_info = metadata.get("license") if isinstance(metadata, Mapping) else None
        available = bool(status.get("available"))
        ready = bool(status.get("ready"))
        if ready:
            readiness = "ready"
        elif available:
            readiness = "not_loaded"
        else:
            readiness = "unavailable"
        return {
            "ok": available,
            "readiness": readiness,
            "provider": status.get("provider") or self.provider.provider_id,
            "model": status.get("model") or self.provider.model_id,
            "model_profile": self.config.model_profile,
            "detail": status.get("detail") or "",
            "capabilities": status.get("capabilities") or self.provider.capabilities.to_dict(),
            "license": license_info,
            "model_metadata": metadata,
        }

    def translate_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        text_value = payload.get("source_text")
        if text_value is None:
            text_value = payload.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise TranslationHTTPError(
                HTTPStatus.BAD_REQUEST,
                "invalid_source_text",
                "source_text (or text) must be a non-empty string",
            )
        source_language = self._required_string(payload, "source_language")
        target_language = self._required_string(payload, "target_language")
        context = self._context(payload.get("context", ()))
        if not self.provider.supports_language_pair(source_language, target_language):
            raise TranslationHTTPError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "unsupported_language_pair",
                f"the configured model does not support {source_language} -> {target_language}",
            )

        started = time.perf_counter()
        try:
            with self._translation_lock:
                translated_text = self.provider.translate(
                    text_value,
                    source_language,
                    target_language,
                    context=context,
                )
        except TranslationHTTPError:
            raise
        except Exception as exc:
            raise TranslationHTTPError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "translation_failed",
                str(exc) or exc.__class__.__name__,
            ) from exc
        latency_seconds = max(0.0, time.perf_counter() - started)
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise TranslationHTTPError(
                HTTPStatus.BAD_GATEWAY,
                "empty_translation",
                "the translation provider returned no text",
            )
        status = self.provider.status().to_dict()
        return {
            "translated_text": translated_text.strip(),
            "source_language": source_language,
            "target_language": target_language,
            "provider": status.get("provider") or self.provider.provider_id,
            "model": status.get("model") or self.provider.model_id,
            "latency_seconds": latency_seconds,
        }

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TranslationHTTPError(
                HTTPStatus.BAD_REQUEST,
                f"invalid_{key}",
                f"{key} must be a non-empty string",
            )
        return value.strip()

    @staticmethod
    def _context(value: Any) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            raise TranslationHTTPError(
                HTTPStatus.BAD_REQUEST,
                "invalid_context",
                "context must be a string or an array of strings",
            )
        if any(not isinstance(item, str) for item in value):
            raise TranslationHTTPError(
                HTTPStatus.BAD_REQUEST,
                "invalid_context",
                "every context item must be a string",
            )
        return tuple(item for item in value if item)

    def close(self) -> None:
        with self._translation_lock:
            self.provider.close()


def make_handler(sidecar: TranslationSidecar, *, quiet: bool = True) -> type[BaseHTTPRequestHandler]:
    class TranslationHandler(BaseHTTPRequestHandler):
        server_version = "WhoSpeaksTranslation/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(sidecar.health())
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/v1/translate":
                self._send_error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
                return
            try:
                payload = self._read_json_body()
                result = sidecar.translate_payload(payload)
            except TranslationHTTPError as exc:
                self._send_error(exc.status, exc.code, exc.message)
                return
            except Exception as exc:
                self._send_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    str(exc) or exc.__class__.__name__,
                )
                return
            self._send_json(result)

        def _read_json_body(self) -> Mapping[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError as exc:
                raise TranslationHTTPError(
                    HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length header"
                ) from exc
            if length <= 0:
                raise TranslationHTTPError(HTTPStatus.BAD_REQUEST, "empty_body", "request body is empty")
            if length > MAX_REQUEST_BYTES:
                raise TranslationHTTPError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"request body exceeds {MAX_REQUEST_BYTES} bytes",
                )
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TranslationHTTPError(
                    HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be valid UTF-8 JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise TranslationHTTPError(
                    HTTPStatus.BAD_REQUEST, "invalid_payload", "request JSON must be an object"
                )
            return payload

        def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
            self._send_json({"error": {"code": code, "message": message}}, status=status)

        def _send_json(self, payload: Mapping[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            if not quiet:
                super().log_message(format, *args)

    return TranslationHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve one optional local translation model. Install whospeaks[translation] first; "
            "TranslateGemma requires Transformers >=4.57.3 and acceptance of the Gemma terms."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model-profile",
        choices=tuple(LOCAL_MODEL_PROFILES),
        default="translate-gemma-4b",
        help="Local model preset (default: translate-gemma-4b).",
    )
    parser.add_argument("--model", default="", help="Override the Hugging Face model id for the preset family.")
    parser.add_argument("--device", default="auto", help="Torch device such as auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype such as auto, float32, float16, or bfloat16.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose-http", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TranslationServerConfig:
    return TranslationServerConfig(
        model_profile=args.model_profile,
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )


def run_server(args: argparse.Namespace) -> None:
    sidecar = TranslationSidecar(config_from_args(args))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sidecar, quiet=not args.verbose_http))
    health = sidecar.health()
    print(
        f"WhoSpeaks translation server: http://{args.host}:{args.port} "
        f"({health['model_profile']}, {health['model']}; model loads on first request)",
        flush=True,
    )
    license_info = health.get("license")
    if isinstance(license_info, Mapping):
        print(
            f"Model license: {license_info.get('display_name') or license_info.get('identifier')} - "
            f"{license_info.get('url') or ''}",
            flush=True,
        )
        notice = str(license_info.get("notice") or "").strip()
        if notice:
            print(notice, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sidecar.close()


def main(argv: Sequence[str] | None = None) -> None:
    run_server(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LOCAL_MODEL_PROFILES",
    "LocalModelProfile",
    "TranslationHTTPError",
    "TranslationServerConfig",
    "TranslationSidecar",
    "build_arg_parser",
    "config_from_args",
    "main",
    "make_handler",
    "run_server",
]

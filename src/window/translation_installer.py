"""Download and verify model files for an isolated translation sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from window.translation_server import LOCAL_MODEL_PROFILES


def prepare_model(model_profile: str, model_dir: Path, *, download: bool = True) -> dict[str, object]:
    profile = LOCAL_MODEL_PROFILES[model_profile]
    model_dir = model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    if download:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=profile.model, local_dir=str(model_dir))
    required = ("config.json",)
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Model {model_profile} is incomplete in {model_dir}; missing {', '.join(missing)}."
        )
    payload: dict[str, object] = {
        "ok": True,
        "model_profile": model_profile,
        "family": profile.family,
        "repository": profile.model,
        "model_dir": str(model_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one WhoSpeaks local translation model.")
    parser.add_argument("--model-profile", choices=tuple(LOCAL_MODEL_PROFILES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_model(args.model_profile, args.model_dir, download=not args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

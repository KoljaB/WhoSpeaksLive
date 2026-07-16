"""Run packaged service scripts without depending on a source checkout path."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from importlib import resources


SERVICE_SCRIPTS = {
    "mlx-asr": ("faster-whisper-asr", "mlx_asr_server.py"),
    "embeddings": ("voice-embeddings-server", "embeddings_server.py"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=SERVICE_SCRIPTS)
    args = parser.parse_args(argv)
    directory, filename = SERVICE_SCRIPTS[args.service]
    script = resources.files(__package__).joinpath(directory, filename)
    script_dir = os.fspath(script.parent)
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)
    runpy.run_path(os.fspath(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

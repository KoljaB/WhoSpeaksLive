from __future__ import annotations

import argparse
import json
from pathlib import Path

from window.live_speaker_parity_replay import validate_and_replay_world_tape


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a production live-speaker World Tape and replay its shared server core."
    )
    parser.add_argument("tape_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = validate_and_replay_world_tape(args.tape_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    validation_ok = bool(report.get("validation", {}).get("valid"))
    core_ok = bool(report.get("server_core_replay", {}).get("exact_match"))
    return 0 if validation_ok and core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

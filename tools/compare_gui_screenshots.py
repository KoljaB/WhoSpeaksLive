"""Create deterministic visual-diff artifacts for WhoSpeaks launcher screenshots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageOps, ImageStat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--allow-resize",
        action="store_true",
        help="Resize a mismatched reference; omitted by default so exact-size audits fail closed.",
    )
    return parser


def compare(
    reference_path: Path,
    actual_path: Path,
    output_dir: Path,
    *,
    allow_resize: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(reference_path) as source_reference, Image.open(actual_path) as source_actual:
        reference_size = source_reference.size
        actual = source_actual.convert("RGB")
        if reference_size != actual.size and not allow_resize:
            raise ValueError(
                "Exact-size audit refused: "
                f"reference is {reference_size[0]}x{reference_size[1]}, "
                f"actual is {actual.width}x{actual.height}."
            )
        reference = source_reference.convert("RGB")
        if reference.size != actual.size:
            reference = reference.resize(actual.size, Image.Resampling.LANCZOS)

    difference = ImageChops.difference(reference, actual)
    stat = ImageStat.Stat(difference)
    channel_means = [round(float(value), 4) for value in stat.mean]
    channel_rms = [round(float(value), 4) for value in stat.rms]
    grayscale = ImageOps.grayscale(difference)
    histogram = grayscale.histogram()
    pixel_count = max(1, actual.width * actual.height)

    def above(threshold: int) -> float:
        return round(sum(histogram[threshold + 1 :]) / pixel_count, 6)

    reference.save(output_dir / "reference-aligned.png")
    actual.save(output_dir / "actual.png")
    Image.blend(reference, actual, 0.5).save(output_dir / "overlay-50.png")
    difference.save(output_dir / "difference-rgb.png")
    heat_source = ImageEnhance.Contrast(grayscale).enhance(2.5)
    ImageOps.colorize(heat_source, black="#071116", white="#ff5d73", mid="#f2c14e").save(
        output_dir / "difference-heatmap.png"
    )

    metrics: dict[str, object] = {
        "reference": str(reference_path),
        "actual": str(actual_path),
        "reference_source_size": list(reference_size),
        "comparison_size": list(actual.size),
        "reference_resized": reference_size != actual.size,
        "mean_absolute_error_rgb": channel_means,
        "mean_absolute_error_average": round(sum(channel_means) / 3, 4),
        "rms_error_rgb": channel_rms,
        "rms_error_average": round(math.sqrt(sum(value * value for value in channel_rms) / 3), 4),
        "changed_pixel_ratio_over_16": above(16),
        "changed_pixel_ratio_over_32": above(32),
        "note": "Pixel metrics are diagnostic, not a pass/fail threshold; generated references contain non-literal typography and geometry.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    metrics = compare(
        args.reference.resolve(),
        args.actual.resolve(),
        args.output_dir.resolve(),
        allow_resize=args.allow_resize,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

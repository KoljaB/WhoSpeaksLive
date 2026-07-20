"""Plan causal continuous-audio live-window embedding experiments.

This module deliberately knows nothing about transcript sentence boundaries.  It
models right-aligned probe windows on one shared media-time grid and provides the
counts and runtime forecasts needed before a provider process is started.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
from typing import Iterator, Sequence


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_HOP_SECONDS = Decimal("0.2")
FULL_WINDOW_UNIVERSE_SECONDS = tuple(
    Decimal(tenths) / Decimal(10) for tenths in range(7, 31)
)


@dataclass(frozen=True)
class StagePreset:
    name: str
    description: str
    audio_seconds: tuple[Decimal, ...]
    window_seconds: tuple[Decimal, ...]
    provider_count: int


STAGE_PRESETS: dict[str, StagePreset] = {
    "calibration_30s": StagePreset(
        name="calibration_30s",
        description="30-second throughput calibration across short, middle, and long windows",
        audio_seconds=(Decimal("30"),),
        window_seconds=(Decimal("0.7"), Decimal("1.5"), Decimal("3.0")),
        provider_count=15,
    ),
    "provider_screen_2m": StagePreset(
        name="provider_screen_2m",
        description="Two-minute all-provider screen including the production 1.0-second baseline",
        audio_seconds=(Decimal("120"),),
        window_seconds=(
            Decimal("0.7"),
            Decimal("1.0"),
            Decimal("1.5"),
            Decimal("3.0"),
        ),
        provider_count=15,
    ),
    "coarse_5m": StagePreset(
        name="coarse_5m",
        description="Five-minute coarse length screen for five surviving providers",
        audio_seconds=(Decimal("300"),),
        window_seconds=(
            Decimal("0.7"),
            Decimal("1.0"),
            Decimal("1.3"),
            Decimal("1.5"),
            Decimal("1.9"),
            Decimal("2.3"),
            Decimal("2.7"),
            Decimal("3.0"),
        ),
        provider_count=5,
    ),
    "full_5m": StagePreset(
        name="full_5m",
        description="Five-minute full candidate universe; diagnostic only, not the default",
        audio_seconds=(Decimal("300"),),
        window_seconds=FULL_WINDOW_UNIVERSE_SECONDS,
        provider_count=15,
    ),
}


@dataclass(frozen=True)
class ProviderRate:
    provider: str
    embeddings_per_second: float
    load_seconds: float = 0.0
    source: str = "measured"
    measurement_contract: str = "unspecified"
    gating_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not math.isfinite(self.embeddings_per_second) or self.embeddings_per_second <= 0:
            raise ValueError("embeddings_per_second must be finite and greater than zero")
        if not math.isfinite(self.load_seconds) or self.load_seconds < 0:
            raise ValueError("load_seconds must be finite and non-negative")


@dataclass(frozen=True)
class RuntimeForecast:
    provider_count: int
    raw_seconds: float
    conservative_seconds: float
    safety_factor: float
    planning_limit_seconds: float
    hard_limit_seconds: float
    status: str
    forecast_authority: str


@dataclass(frozen=True)
class ExperimentPlan:
    stage: str
    description: str
    sample_rate: int
    hop_samples: int
    audio_samples: tuple[int, ...]
    audio_start_samples: tuple[int, ...]
    window_samples: tuple[int, ...]
    provider_count: int
    edge_policy: str = "full_windows_only"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.hop_samples <= 0:
            raise ValueError("hop_samples must be greater than zero")
        if not self.audio_samples or any(value <= 0 for value in self.audio_samples):
            raise ValueError("audio_samples must contain positive excerpt lengths")
        if len(self.audio_start_samples) != len(self.audio_samples):
            raise ValueError("audio_start_samples must match audio_samples one-to-one")
        if any(value < 0 for value in self.audio_start_samples):
            raise ValueError("audio_start_samples must be non-negative")
        if not self.window_samples or any(value <= 0 for value in self.window_samples):
            raise ValueError("window_samples must contain positive window lengths")
        if tuple(sorted(set(self.window_samples))) != self.window_samples:
            raise ValueError("window_samples must be unique and sorted")
        if self.provider_count <= 0:
            raise ValueError("provider_count must be greater than zero")
        if self.edge_policy != "full_windows_only":
            raise ValueError("only the explicit full_windows_only policy is supported")

    @property
    def audio_seconds(self) -> tuple[float, ...]:
        return tuple(value / self.sample_rate for value in self.audio_samples)

    @property
    def audio_start_seconds(self) -> tuple[float, ...]:
        return tuple(value / self.sample_rate for value in self.audio_start_samples)

    @property
    def hop_seconds(self) -> float:
        return self.hop_samples / self.sample_rate

    @property
    def window_seconds(self) -> tuple[float, ...]:
        return tuple(value / self.sample_rate for value in self.window_samples)

    @property
    def counts_by_window(self) -> dict[int, int]:
        return {
            window_samples: sum(
                full_window_count(
                    total_samples=audio_samples,
                    hop_samples=self.hop_samples,
                    window_samples=window_samples,
                    source_start_samples=source_start_samples,
                )
                for audio_samples, source_start_samples in zip(
                    self.audio_samples,
                    self.audio_start_samples,
                    strict=True,
                )
            )
            for window_samples in self.window_samples
        }

    @property
    def embeddings_per_provider(self) -> int:
        return sum(self.counts_by_window.values())

    @property
    def total_embeddings(self) -> int:
        return self.embeddings_per_provider * self.provider_count

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "corpus_kind": "continuous_causal_live_windows",
            "stage": self.stage,
            "description": self.description,
            "sample_rate": self.sample_rate,
            "hop_samples": self.hop_samples,
            "hop_seconds": self.hop_seconds,
            "edge_policy": self.edge_policy,
            "audio_samples": list(self.audio_samples),
            "audio_seconds": list(self.audio_seconds),
            "audio_start_samples": list(self.audio_start_samples),
            "audio_start_seconds": list(self.audio_start_seconds),
            "window_samples": list(self.window_samples),
            "window_seconds": list(self.window_seconds),
            "provider_count": self.provider_count,
            "counts_by_window_samples": {
                str(key): value for key, value in self.counts_by_window.items()
            },
            "embeddings_per_provider": self.embeddings_per_provider,
            "total_embeddings": self.total_embeddings,
        }


def seconds_to_samples(
    value: Decimal | float | int | str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    label: str = "seconds",
    allow_zero: bool = False,
) -> int:
    """Convert a duration to an exact integer sample count."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    try:
        seconds = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not a valid decimal duration: {value!r}") from exc
    samples = seconds * sample_rate
    if not samples.is_finite() or samples < 0 or (samples == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{label} must be finite and {qualifier}")
    integral = samples.to_integral_value()
    if samples != integral:
        raise ValueError(
            f"{label}={seconds} does not resolve to an integer sample count at {sample_rate} Hz"
        )
    return int(integral)


def shared_right_edges(
    total_samples: int,
    hop_samples: int,
    *,
    source_start_samples: int = 0,
) -> range:
    """Return local edges aligned to the original media timeline's hop grid."""

    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be greater than zero")
    if source_start_samples < 0:
        raise ValueError("source_start_samples must be non-negative")
    first_absolute_edge = max(
        hop_samples,
        ((source_start_samples + hop_samples - 1) // hop_samples) * hop_samples,
    )
    last_absolute_edge = (
        (source_start_samples + total_samples) // hop_samples
    ) * hop_samples
    if last_absolute_edge < first_absolute_edge:
        return range(0)
    first_local_edge = first_absolute_edge - source_start_samples
    last_local_edge = last_absolute_edge - source_start_samples
    return range(first_local_edge, last_local_edge + 1, hop_samples)


def full_window_count(
    *,
    total_samples: int,
    hop_samples: int,
    window_samples: int,
    source_start_samples: int = 0,
) -> int:
    """Count complete [right-window, right] slices on the shared edge grid."""

    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if hop_samples <= 0 or window_samples <= 0:
        raise ValueError("hop_samples and window_samples must be greater than zero")
    if source_start_samples < 0:
        raise ValueError("source_start_samples must be non-negative")
    first_valid_absolute_sample = source_start_samples + window_samples
    first_valid_tick = max(
        1,
        (first_valid_absolute_sample + hop_samples - 1) // hop_samples,
    )
    last_valid_tick = (source_start_samples + total_samples) // hop_samples
    return max(0, last_valid_tick - first_valid_tick + 1)


def iter_full_window_bounds(
    *,
    total_samples: int,
    hop_samples: int,
    window_samples: int,
    source_start_samples: int = 0,
) -> Iterator[tuple[int, int]]:
    """Yield sample-exact causal bounds on a grid shared by every window length."""

    for right_edge in shared_right_edges(
        total_samples,
        hop_samples,
        source_start_samples=source_start_samples,
    ):
        if right_edge >= window_samples:
            yield right_edge - window_samples, right_edge


def build_plan(
    *,
    stage: str,
    description: str,
    audio_seconds: Sequence[Decimal | float | int | str],
    audio_start_seconds: Sequence[Decimal | float | int | str] | None = None,
    window_seconds: Sequence[Decimal | float | int | str],
    provider_count: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_seconds: Decimal | float | int | str = DEFAULT_HOP_SECONDS,
) -> ExperimentPlan:
    audio_samples = tuple(
        seconds_to_samples(value, sample_rate=sample_rate, label="audio_seconds")
        for value in audio_seconds
    )
    starts = (
        audio_start_seconds
        if audio_start_seconds is not None
        else tuple(Decimal("0") for _ in audio_samples)
    )
    audio_start_samples = tuple(
        seconds_to_samples(
            value,
            sample_rate=sample_rate,
            label="audio_start_seconds",
            allow_zero=True,
        )
        for value in starts
    )
    windows = tuple(
        sorted(
            {
                seconds_to_samples(value, sample_rate=sample_rate, label="window_seconds")
                for value in window_seconds
            }
        )
    )
    return ExperimentPlan(
        stage=stage,
        description=description,
        sample_rate=sample_rate,
        hop_samples=seconds_to_samples(
            hop_seconds,
            sample_rate=sample_rate,
            label="hop_seconds",
        ),
        audio_samples=audio_samples,
        audio_start_samples=audio_start_samples,
        window_samples=windows,
        provider_count=provider_count,
    )


def forecast_runtime(
    *,
    embeddings_per_provider: int,
    rates: Sequence[ProviderRate],
    safety_factor: float = 1.5,
    planning_limit_hours: float = 18.0,
    hard_limit_hours: float = 24.0,
) -> RuntimeForecast:
    """Forecast sequential provider time and classify it against both budget limits."""

    if embeddings_per_provider < 0:
        raise ValueError("embeddings_per_provider must be non-negative")
    if not rates:
        raise ValueError("at least one provider rate is required")
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be finite and at least 1.0")
    if planning_limit_hours <= 0 or hard_limit_hours <= planning_limit_hours:
        raise ValueError("hard_limit_hours must be greater than planning_limit_hours > 0")

    raw_seconds = sum(
        rate.load_seconds + embeddings_per_provider / rate.embeddings_per_second
        for rate in rates
    )
    conservative_seconds = raw_seconds * safety_factor
    planning_limit_seconds = planning_limit_hours * 3600
    hard_limit_seconds = hard_limit_hours * 3600
    if conservative_seconds > hard_limit_seconds:
        status = "hard_limit_exceeded"
    elif conservative_seconds > planning_limit_seconds:
        status = "planning_limit_exceeded"
    else:
        status = "within_planning_limit"
    return RuntimeForecast(
        provider_count=len(rates),
        raw_seconds=raw_seconds,
        conservative_seconds=conservative_seconds,
        safety_factor=safety_factor,
        planning_limit_seconds=planning_limit_seconds,
        hard_limit_seconds=hard_limit_seconds,
        status=status,
        forecast_authority=(
            "gating_measurement"
            if all(rate.gating_eligible for rate in rates)
            else "non_gating_estimate"
        ),
    )


def load_benchmark_rates(results_root: Path) -> list[ProviderRate]:
    """Read provider rates from benchmark-style ``*/result.json`` artifacts."""

    rates: list[ProviderRate] = []
    for result_path in sorted(results_root.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
        if str(payload.get("status", "")).lower() != "ok":
            continue
        count = int(payload.get("window_count") or 0)
        wall_seconds = float(payload.get("wall_time_ms") or 0.0) / 1000.0
        load_seconds = float(payload.get("load_time_ms") or 0.0) / 1000.0
        inference_seconds = wall_seconds - load_seconds
        provider = str(payload.get("engine_id") or result_path.parent.name)
        if count <= 0 or inference_seconds <= 0:
            continue
        rates.append(
            ProviderRate(
                provider=provider,
                embeddings_per_second=count / inference_seconds,
                load_seconds=max(0.0, load_seconds),
                source=str(result_path),
                measurement_contract="historical_dialogue1_dense_windows_v1",
                gating_eligible=False,
            )
        )
    if not rates:
        raise RuntimeError(f"No usable */result.json benchmark rates found under {results_root}")
    return rates


def _parse_decimal_list(raw: str, *, label: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            values.append(Decimal(text))
        except InvalidOperation as exc:
            raise argparse.ArgumentTypeError(f"{label} contains an invalid decimal: {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must contain at least one value")
    return tuple(values)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _select_conservative_rates(
    rates: Sequence[ProviderRate],
    provider_count: int,
    embeddings_per_provider: int,
) -> list[ProviderRate]:
    if provider_count > len(rates):
        raise ValueError(
            f"Plan requests {provider_count} providers but only {len(rates)} measured rates are available"
        )
    # Without an explicit shortlist, use the providers with the largest projected
    # wall time.  This remains conservative even for tiny jobs where model load
    # dominates post-load throughput.
    return sorted(
        rates,
        key=lambda item: item.load_seconds
        + embeddings_per_provider / item.embeddings_per_second,
        reverse=True,
    )[:provider_count]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run counts and runtime budgets for causal continuous-audio live windows."
    )
    parser.add_argument("--preset", choices=sorted(STAGE_PRESETS), default="calibration_30s")
    parser.add_argument(
        "--audio-seconds",
        help="Comma-separated continuous excerpt durations; overrides the preset.",
    )
    parser.add_argument(
        "--audio-start-seconds",
        help=(
            "Comma-separated absolute source start times for the excerpts. "
            "Defaults to zero; required for real non-zero source excerpts."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        help="Comma-separated right-aligned window lengths; overrides the preset.",
    )
    parser.add_argument(
        "--provider-count",
        type=_positive_int,
        help="Override the preset provider count.",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--hop-seconds", default=str(DEFAULT_HOP_SECONDS))
    parser.add_argument(
        "--benchmark-results-root",
        type=Path,
        help="Optional directory containing provider subdirectories with result.json files.",
    )
    parser.add_argument("--safety-factor", type=float, default=1.5)
    parser.add_argument("--planning-limit-hours", type=float, default=18.0)
    parser.add_argument("--hard-limit-hours", type=float, default=24.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _format_text(plan: ExperimentPlan, forecast: RuntimeForecast | None) -> str:
    lines = [
        f"stage: {plan.stage}",
        f"audio: {sum(plan.audio_seconds):.3f}s across {len(plan.audio_samples)} excerpt(s)",
        f"hop: {plan.hop_seconds:.3f}s, causal right-aligned, {plan.edge_policy}",
        "windows: " + ", ".join(f"{value:.1f}s" for value in plan.window_seconds),
        f"providers: {plan.provider_count}",
        f"embeddings/provider: {plan.embeddings_per_provider:,}",
        f"total embeddings: {plan.total_embeddings:,}",
    ]
    if forecast is not None:
        lines.extend(
            [
                f"measured rates used: {forecast.provider_count}",
                f"raw forecast: {forecast.raw_seconds / 3600:.2f}h",
                f"conservative forecast: {forecast.conservative_seconds / 3600:.2f}h "
                f"(x{forecast.safety_factor:.2f})",
                f"budget status: {forecast.status}",
                f"forecast authority: {forecast.forecast_authority}",
            ]
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    preset = STAGE_PRESETS[args.preset]
    audio_seconds = (
        _parse_decimal_list(args.audio_seconds, label="--audio-seconds")
        if args.audio_seconds
        else preset.audio_seconds
    )
    window_seconds = (
        _parse_decimal_list(args.window_seconds, label="--window-seconds")
        if args.window_seconds
        else preset.window_seconds
    )
    audio_start_seconds = (
        _parse_decimal_list(args.audio_start_seconds, label="--audio-start-seconds")
        if args.audio_start_seconds
        else tuple(Decimal("0") for _ in audio_seconds)
    )
    if len(audio_start_seconds) != len(audio_seconds):
        parser.error("--audio-start-seconds must contain one value per --audio-seconds excerpt")
    provider_count = (
        args.provider_count if args.provider_count is not None else preset.provider_count
    )
    customized = any(
        value is not None
        for value in (
            args.audio_seconds,
            args.audio_start_seconds,
            args.window_seconds,
            args.provider_count,
        )
    )
    plan = build_plan(
        stage=f"custom_from_{preset.name}" if customized else preset.name,
        description=(
            f"Custom overrides of preset {preset.name}"
            if customized
            else preset.description
        ),
        audio_seconds=audio_seconds,
        audio_start_seconds=audio_start_seconds,
        window_seconds=window_seconds,
        provider_count=provider_count,
        sample_rate=args.sample_rate,
        hop_seconds=args.hop_seconds,
    )

    forecast: RuntimeForecast | None = None
    selected_rates: list[ProviderRate] = []
    if args.benchmark_results_root:
        selected_rates = _select_conservative_rates(
            load_benchmark_rates(args.benchmark_results_root),
            provider_count,
            plan.embeddings_per_provider,
        )
        forecast = forecast_runtime(
            embeddings_per_provider=plan.embeddings_per_provider,
            rates=selected_rates,
            safety_factor=args.safety_factor,
            planning_limit_hours=args.planning_limit_hours,
            hard_limit_hours=args.hard_limit_hours,
        )

    if args.json:
        payload = plan.as_dict()
        if forecast is not None:
            payload["runtime_forecast"] = asdict(forecast)
            payload["forecast_rates"] = [asdict(rate) for rate in selected_rates]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_text(plan, forecast))
    return 2 if forecast is not None and forecast.status != "within_planning_limit" else 0


if __name__ == "__main__":
    raise SystemExit(main())

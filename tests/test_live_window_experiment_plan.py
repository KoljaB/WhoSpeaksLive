from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embeddings.live_window_experiment_plan import (
    FULL_WINDOW_UNIVERSE_SECONDS,
    ProviderRate,
    STAGE_PRESETS,
    build_plan,
    forecast_runtime,
    full_window_count,
    iter_full_window_bounds,
    load_benchmark_rates,
    seconds_to_samples,
    shared_right_edges,
)


class LiveWindowExperimentPlanTests(unittest.TestCase):
    def test_five_minute_full_grid_has_exact_shared_timeline_count(self) -> None:
        plan = build_plan(
            stage="test",
            description="test",
            audio_seconds=(Decimal("300"),),
            window_seconds=FULL_WINDOW_UNIVERSE_SECONDS,
            provider_count=15,
        )

        self.assertEqual(plan.embeddings_per_provider, 35_796)
        self.assertEqual(plan.total_embeddings, 536_940)

    def test_stage_presets_are_intentionally_small(self) -> None:
        calibration = STAGE_PRESETS["calibration_30s"]
        calibration_plan = build_plan(
            stage=calibration.name,
            description=calibration.description,
            audio_seconds=calibration.audio_seconds,
            window_seconds=calibration.window_seconds,
            provider_count=calibration.provider_count,
        )
        provider_screen = STAGE_PRESETS["provider_screen_2m"]
        provider_plan = build_plan(
            stage=provider_screen.name,
            description=provider_screen.description,
            audio_seconds=provider_screen.audio_seconds,
            window_seconds=provider_screen.window_seconds,
            provider_count=provider_screen.provider_count,
        )
        coarse = STAGE_PRESETS["coarse_5m"]
        coarse_plan = build_plan(
            stage=coarse.name,
            description=coarse.description,
            audio_seconds=coarse.audio_seconds,
            window_seconds=coarse.window_seconds,
            provider_count=coarse.provider_count,
        )

        self.assertEqual(calibration_plan.embeddings_per_provider, 426)
        self.assertEqual(calibration_plan.total_embeddings, 6_390)
        self.assertEqual(provider_plan.embeddings_per_provider, 2_372)
        self.assertEqual(provider_plan.total_embeddings, 35_580)
        self.assertEqual(coarse_plan.embeddings_per_provider, 11_933)
        self.assertEqual(coarse_plan.total_embeddings, 59_665)

    def test_every_window_is_right_aligned_and_contains_no_future_sample(self) -> None:
        sample_rate = 16_000
        hop_samples = seconds_to_samples("0.2", sample_rate=sample_rate)
        window_samples = seconds_to_samples("0.7", sample_rate=sample_rate)
        total_samples = seconds_to_samples("2.0", sample_rate=sample_rate)
        bounds = list(
            iter_full_window_bounds(
                total_samples=total_samples,
                hop_samples=hop_samples,
                window_samples=window_samples,
            )
        )

        self.assertEqual(bounds[0], (1_600, 12_800))
        self.assertEqual(bounds[-1], (20_800, 32_000))
        self.assertTrue(all(start >= 0 for start, _ in bounds))
        self.assertTrue(all(end <= total_samples for _, end in bounds))
        self.assertTrue(all(end - start == window_samples for start, end in bounds))
        self.assertTrue(all(end % hop_samples == 0 for _, end in bounds))

    def test_lengths_share_the_same_right_edge_grid(self) -> None:
        total_samples = seconds_to_samples("3.0")
        hop_samples = seconds_to_samples("0.2")
        short_edges = {
            end
            for _, end in iter_full_window_bounds(
                total_samples=total_samples,
                hop_samples=hop_samples,
                window_samples=seconds_to_samples("0.7"),
            )
        }
        long_edges = {
            end
            for _, end in iter_full_window_bounds(
                total_samples=total_samples,
                hop_samples=hop_samples,
                window_samples=seconds_to_samples("1.5"),
            )
        }

        self.assertTrue(long_edges.issubset(short_edges))
        self.assertEqual(max(short_edges), max(long_edges))
        self.assertEqual(tuple(shared_right_edges(total_samples, hop_samples))[-1], total_samples)

    def test_non_aligned_excerpt_preserves_original_media_grid_phase(self) -> None:
        total_samples = seconds_to_samples("2.0")
        source_start_samples = seconds_to_samples("0.15")
        hop_samples = seconds_to_samples("0.2")
        window_samples = seconds_to_samples("0.7")
        bounds = list(
            iter_full_window_bounds(
                total_samples=total_samples,
                hop_samples=hop_samples,
                window_samples=window_samples,
                source_start_samples=source_start_samples,
            )
        )

        self.assertEqual(bounds[0], (2_400, 13_600))
        self.assertEqual(bounds[-1], (18_400, 29_600))
        self.assertEqual(len(bounds), 6)
        self.assertTrue(
            all((source_start_samples + end) % hop_samples == 0 for _, end in bounds)
        )
        self.assertEqual(
            full_window_count(
                total_samples=total_samples,
                hop_samples=hop_samples,
                window_samples=window_samples,
                source_start_samples=source_start_samples,
            ),
            len(bounds),
        )

        plan = build_plan(
            stage="offset",
            description="offset",
            audio_seconds=("2.0",),
            audio_start_seconds=("0.15",),
            window_seconds=("0.7",),
            provider_count=1,
        )
        self.assertEqual(plan.embeddings_per_provider, 6)
        self.assertEqual(plan.audio_start_samples, (2_400,))

    def test_separate_excerpts_do_not_hide_their_startup_cost(self) -> None:
        one_excerpt = build_plan(
            stage="one",
            description="one",
            audio_seconds=("120",),
            window_seconds=("3.0",),
            provider_count=1,
        )
        four_excerpts = build_plan(
            stage="four",
            description="four",
            audio_seconds=("30", "30", "30", "30"),
            window_seconds=("3.0",),
            provider_count=1,
        )

        self.assertLess(four_excerpts.embeddings_per_provider, one_excerpt.embeddings_per_provider)

    def test_runtime_forecast_uses_provider_rates_load_and_safety_margin(self) -> None:
        rates = [
            ProviderRate("fast", embeddings_per_second=10.0, load_seconds=5.0),
            ProviderRate("slow", embeddings_per_second=5.0, load_seconds=10.0),
        ]
        forecast = forecast_runtime(
            embeddings_per_provider=3_600,
            rates=rates,
            safety_factor=1.5,
            planning_limit_hours=1.0,
            hard_limit_hours=2.0,
        )

        self.assertAlmostEqual(forecast.raw_seconds, 1_095.0)
        self.assertAlmostEqual(forecast.conservative_seconds, 1_642.5)
        self.assertEqual(forecast.status, "within_planning_limit")
        self.assertEqual(forecast.forecast_authority, "non_gating_estimate")

    def test_runtime_forecast_distinguishes_planning_and_hard_limits(self) -> None:
        rate = ProviderRate("provider", embeddings_per_second=1.0)

        planning_exceeded = forecast_runtime(
            embeddings_per_provider=20_000,
            rates=[rate],
            safety_factor=1.0,
            planning_limit_hours=5.0,
            hard_limit_hours=6.0,
        )
        hard_exceeded = forecast_runtime(
            embeddings_per_provider=25_000,
            rates=[rate],
            safety_factor=1.0,
            planning_limit_hours=5.0,
            hard_limit_hours=6.0,
        )

        self.assertEqual(planning_exceeded.status, "planning_limit_exceeded")
        self.assertEqual(hard_exceeded.status, "hard_limit_exceeded")

    def test_benchmark_rate_loader_uses_post_load_window_time(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider_dir = root / "engine"
            provider_dir.mkdir()
            (provider_dir / "result.json").write_text(
                json.dumps(
                    {
                        "engine_id": "engine",
                        "status": "ok",
                        "window_count": 200,
                        "load_time_ms": 2_000,
                        "wall_time_ms": 12_000,
                    }
                ),
                encoding="utf-8",
            )

            rates = load_benchmark_rates(root)

        self.assertEqual(len(rates), 1)
        self.assertEqual(rates[0].provider, "engine")
        self.assertAlmostEqual(rates[0].embeddings_per_second, 20.0)
        self.assertAlmostEqual(rates[0].load_seconds, 2.0)

    def test_invalid_non_integral_sample_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seconds_to_samples("0.00001", sample_rate=16_000)

    def test_full_window_count_matches_iterator(self) -> None:
        total_samples = seconds_to_samples("5")
        hop_samples = seconds_to_samples("0.2")
        window_samples = seconds_to_samples("1.3")
        expected = len(
            list(
                iter_full_window_bounds(
                    total_samples=total_samples,
                    hop_samples=hop_samples,
                    window_samples=window_samples,
                )
            )
        )

        self.assertEqual(
            full_window_count(
                total_samples=total_samples,
                hop_samples=hop_samples,
                window_samples=window_samples,
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()

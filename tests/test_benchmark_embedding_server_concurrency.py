from __future__ import annotations

import unittest

import numpy as np

from tools import benchmark_embedding_server_concurrency as benchmark


class EmbeddingServerConcurrencyBenchmarkTests(unittest.TestCase):
    def test_reuse_diagnostics_are_optional_and_recursive(self) -> None:
        self.assertEqual(benchmark._reuse_diagnostics({"components": [{"provider": "x"}]}), [])
        self.assertEqual(
            benchmark._reuse_diagnostics({
                "work": "joined",
                "components": [
                    {"provider": "a", "result_source": "calculated"},
                    {"provider": "b", "result_source": "cached"},
                ],
                "embedding": ["cache"],
            }),
            [
                {"path": "response.work", "state": "joined"},
                {"path": "response.components[0].result_source", "state": "calculated"},
                {"path": "response.components[1].result_source", "state": "cache"},
            ],
        )

    def test_component_vector_is_reconstructed_from_weighted_stack(self) -> None:
        first = np.asarray([1.0, 0.0], dtype=np.float32)
        speechbrain = np.asarray([0.0, 0.6, 0.8], dtype=np.float32)
        stack = benchmark.normalize_vector(np.concatenate([first, speechbrain * 0.28]))
        result = {
            "canonical_stack": [
                {"provider": "espnet_ecapa_wavlm_joint", "weight": 1.0},
                {"provider": "speechbrain_resnet", "weight": 0.28},
            ],
            "components": [
                {"provider": "espnet_ecapa_wavlm_joint", "dim": 2},
                {"provider": "speechbrain_resnet", "dim": 3},
            ],
        }
        reconstructed = benchmark._component_embeddings(result, stack)
        np.testing.assert_allclose(reconstructed["speechbrain_resnet"], speechbrain, atol=1e-6)

    def test_coalesce_analysis_reports_parity_and_wall_saving(self) -> None:
        speechbrain = np.asarray([0.0, 1.0], dtype=np.float32)
        records = [
            {
                "benchmark": "coalesce:control",
                "mode": "standalone_control",
                "role": "standalone_speechbrain",
                "provider": "speechbrain_resnet",
                "slice_id": "control",
                "repetition": 0,
                "ok": True,
                "wall_seconds": 0.10,
                "reuse_diagnostics": [{"path": "response.work", "state": "calculated"}],
                "_embedding": speechbrain,
            },
            {
                "benchmark": "coalesce:promoted_public",
                "mode": "concurrent_release",
                "role": "final_stack",
                "provider": benchmark.PROMOTED_PUBLIC_PROVIDER,
                "slice_id": "pair",
                "repetition": 0,
                "ok": True,
                "wall_seconds": 0.12,
                "reuse_diagnostics": [],
                "_component_embeddings": {"speechbrain_resnet": speechbrain},
            },
            {
                "benchmark": "coalesce:promoted_public",
                "mode": "concurrent_release",
                "role": "standalone_speechbrain",
                "provider": "speechbrain_resnet",
                "slice_id": "pair",
                "repetition": 0,
                "ok": True,
                "wall_seconds": 0.02,
                "reuse_diagnostics": [{"path": "response.work", "state": "joined"}],
                "_embedding": speechbrain.copy(),
            },
        ]
        result = benchmark._coalesce_analysis(records)
        self.assertAlmostEqual(result["parity_cosine_distribution"]["min"], 1.0)
        mode = result["modes"]["coalesce:promoted_public:concurrent_release"]
        self.assertAlmostEqual(mode["standalone_p50_seconds_saved_vs_calculated"], 0.08)
        self.assertAlmostEqual(mode["standalone_p50_speedup_vs_calculated"], 5.0)
        self.assertEqual(mode["reuse_state_response_counts"]["joined"], 1)

    def test_parser_accepts_coalesce_and_httpx(self) -> None:
        args = benchmark.build_parser().parse_args([
            "--sections", "coalesce",
            "--http-client", "httpx",
            "--coalesce-followup-delay-seconds", "0.025",
        ])
        self.assertEqual(args.sections, {"coalesce"})
        self.assertEqual(args.http_client, "httpx")
        self.assertEqual(args.coalesce_followup_delay_seconds, 0.025)


if __name__ == "__main__":
    unittest.main()

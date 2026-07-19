from __future__ import annotations

import unittest

from embeddings.provider_identity import (
    PROMOTED_PUBLIC_PROVIDER,
    summarize_embedding_stack,
)


class EmbeddingProviderIdentityTests(unittest.TestCase):
    def test_known_stack_uses_friendly_label_but_retains_exact_identity(self) -> None:
        summary = summarize_embedding_stack(PROMOTED_PUBLIC_PROVIDER, 640)

        self.assertEqual(summary["label"], "Promoted public stack")
        self.assertEqual(summary["dimensions"], 640)
        self.assertEqual(summary["provider_count"], 3)
        self.assertEqual(summary["identifier"], PROMOTED_PUBLIC_PROVIDER)

    def test_arbitrary_ensemble_gets_bounded_generated_summary(self) -> None:
        provider = "alpha_provider=1.0+beta_provider=0.5+gamma_provider=0.25+delta_provider=0.1"
        summary = summarize_embedding_stack(provider, 512)

        self.assertEqual(summary["label"], "Custom ensemble · 4 providers")
        self.assertEqual(summary["kind"], "custom_ensemble")
        self.assertEqual(summary["provider_count"], 4)
        self.assertEqual(summary["identifier"], provider)


if __name__ == "__main__":
    unittest.main()

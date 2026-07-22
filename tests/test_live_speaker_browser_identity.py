from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_MODULE = ROOT / "src" / "window" / "assets" / "web" / "live" / "live_context.js"
TRANSLATION_MODULE = ROOT / "src" / "window" / "assets" / "web" / "live" / "transcript_translation.js"


class LiveSpeakerBrowserIdentityTest(unittest.TestCase):
    def test_registry_is_bidirectional_monotonic_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            context_copy = temporary_path / "live_context.mjs"
            store_copy = temporary_path / "app_store.mjs"
            context_copy.write_text(
                CONTEXT_MODULE.read_text(encoding="utf-8").replace(
                    '"./app_store.js"', '"./app_store.mjs"'
                ),
                encoding="utf-8",
            )
            store_copy.write_text(
                (CONTEXT_MODULE.parent / "app_store.js").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            module_url = context_copy.as_uri()
            program = f"""
              import {{LiveSpeakerPresentationRegistry, mergePublicSpeakerSnapshot}} from {json.dumps(module_url)};
          const registry = new LiveSpeakerPresentationRegistry();
          const first = registry.apply({{
            alias_generation: 1,
            final_internal_speaker_id: "S3",
            surviving_public_speaker_id: "LIVE_TRACKLET_1",
          }});
          const stale = registry.apply({{
            alias_generation: 1,
            final_internal_speaker_id: "S4",
            surviving_public_speaker_id: "LIVE_TRACKLET_2",
          }});
          const manyToOne = registry.apply({{
            alias_generation: 2,
            final_internal_speaker_id: "S4",
            surviving_public_speaker_id: "LIVE_TRACKLET_1",
          }});
          if (!first || stale || manyToOne) process.exit(2);
          if (registry.toPublic("S3") !== "LIVE_TRACKLET_1") process.exit(3);
          if (registry.toInternal("LIVE_TRACKLET_1") !== "S3") process.exit(4);
          registry.reset("next-run");
          if (registry.toPublic("S3") !== "S3" || registry.aliasGeneration !== 0) process.exit(5);
          const hydrated = registry.hydrate(
            {{S5: "LIVE_TRACKLET_2"}},
            {{LIVE_TRACKLET_2: "S5"}},
            7,
          );
          if (!hydrated || registry.aliasGeneration !== 7) process.exit(6);
          if (registry.toInternal("LIVE_TRACKLET_2") !== "S5") process.exit(7);
          const current = [
            {{id: "LIVE_TRACKLET_2"}},
            {{id: "LIVE_TRACKLET_3", source: "live_provisional"}},
          ];
          const merged = mergePublicSpeakerSnapshot(current, [{{id: "LIVE_TRACKLET_2"}}], registry);
          if (merged.length !== 2 || !merged.some(item => item.id === "LIVE_TRACKLET_3")) process.exit(8);
          registry.reset("fresh-run");
          const resetSpeakers = registry.stripTemporarySpeakers(current);
          const afterResetSnapshot = registry.mergeSnapshot(resetSpeakers, [{{id: "S1"}}]);
          if (afterResetSnapshot.length !== 1 || afterResetSnapshot[0].id !== "S1") process.exit(9);
            """
            result = subprocess.run(
                ["node", "--input-type=module", "-e", program],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stderr or result.stdout)

    def test_final_backed_tracklet_is_not_provisional(self):
        with tempfile.TemporaryDirectory() as temporary:
            module_copy = Path(temporary) / "transcript_translation.mjs"
            module_copy.write_text(TRANSLATION_MODULE.read_text(encoding="utf-8"), encoding="utf-8")
            program = f"""
              import {{installTranscriptTranslation}} from {json.dumps(module_copy.as_uri())};
              const api = {{}};
              installTranscriptTranslation({{api}});
              if (!api.isLiveProvisionalSpeaker({{id: "LIVE_TRACKLET_1", source: "live_provisional"}})) process.exit(2);
              if (api.isLiveProvisionalSpeaker({{
                id: "LIVE_TRACKLET_1",
                source: "live_provisional",
                presentation_aliased: true,
                internal_speaker_id: "S2",
              }})) process.exit(3);
              if (!api.isLiveProvisionalSpeaker({{id: "LIVE_NEW_1"}})) process.exit(4);
            """
            result = subprocess.run(
                ["node", "--input-type=module", "-e", program],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()

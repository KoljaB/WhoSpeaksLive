from __future__ import annotations

import unittest

import numpy as np

from window.live_speaker_open_set_tracklets import (
    OPEN_SET_TRACKLET_PRESET,
    PROFILE_CONTRADICTION_TRACKLET_PRESET,
    OpenSetTrackletConfig,
    OpenSetTrackletOverlay,
    OpenSetTrackletStep,
    open_set_tracklet_config_for_preset,
)


def _step(
    short,
    *,
    long=None,
    profiles=(),
    media_time=1.0,
    base=None,
):
    return OpenSetTrackletStep(
        media_time=media_time,
        speech=True,
        probe_scheduled=True,
        release_signal=False,
        short_embedding=np.asarray(short, dtype=np.float32),
        long_embedding=np.asarray(long if long is not None else short, dtype=np.float32),
        profiles=tuple(profiles),
        base_visible_speaker=base,
        base_action="hold" if base else "none",
        base_reason="base",
    )


class OpenSetTrackletOverlayTest(unittest.TestCase):
    def test_reuse_idle_ttl_defaults_to_eight_seconds(self):
        self.assertEqual(8.0, OpenSetTrackletConfig().reuse_idle_ttl_seconds)

    def test_versioned_profile_contradiction_preset_is_fail_closed(self):
        incumbent = open_set_tracklet_config_for_preset(OPEN_SET_TRACKLET_PRESET)
        candidate = open_set_tracklet_config_for_preset(
            PROFILE_CONTRADICTION_TRACKLET_PRESET
        )

        self.assertFalse(incumbent.profile_contradiction_enabled)
        self.assertTrue(candidate.profile_contradiction_enabled)
        self.assertEqual(4, candidate.profile_contradiction_min_profiles)
        self.assertEqual(0.36, candidate.profile_contradiction_short_ceiling)
        self.assertEqual(0.36, candidate.profile_contradiction_long_ceiling)
        with self.assertRaises(ValueError):
            OpenSetTrackletConfig(preset=PROFILE_CONTRADICTION_TRACKLET_PRESET)
        with self.assertRaises(ValueError):
            open_set_tracklet_config_for_preset("unversioned")

    def test_profile_contradiction_requires_four_valid_profiles(self):
        overlay = OpenSetTrackletOverlay(open_set_tracklet_config_for_preset(
            PROFILE_CONTRADICTION_TRACKLET_PRESET
        ))
        incumbent = OpenSetTrackletOverlay(open_set_tracklet_config_for_preset(
            OPEN_SET_TRACKLET_PRESET
        ))
        profiles = tuple(
            {"label": f"S{index + 1}", "centroid": np.eye(5)[index]}
            for index in range(3)
        )

        item = _step(
            np.eye(5)[4], profiles=profiles, media_time=1.0, base="S1"
        )
        result = overlay.step(item)
        incumbent_result = incumbent.step(item)

        self.assertEqual(incumbent_result.visible_speaker, result.visible_speaker)
        self.assertEqual(incumbent_result.reason, result.reason)
        self.assertEqual(incumbent_result.action, result.action)
        self.assertFalse(result.diagnostics["profile_contradiction_pending"])

    def test_two_consistent_profile_contradictions_create_tracklet(self):
        overlay = OpenSetTrackletOverlay(open_set_tracklet_config_for_preset(
            PROFILE_CONTRADICTION_TRACKLET_PRESET
        ))
        profiles = tuple(
            {"label": f"S{index + 1}", "centroid": np.eye(5)[index]}
            for index in range(4)
        )

        first = overlay.step(_step(
            np.eye(5)[4], profiles=profiles, media_time=1.0, base="S1"
        ))
        second = overlay.step(_step(
            np.eye(5)[4], profiles=profiles, media_time=1.75, base="S2"
        ))

        self.assertIsNone(first.visible_speaker)
        self.assertEqual("open_set_pending_unknown", first.reason)
        self.assertTrue(first.diagnostics["profile_contradiction_active"])
        self.assertTrue(first.diagnostics["profile_contradiction_pending"])
        self.assertEqual("LIVE_TRACKLET_1", second.visible_speaker)
        self.assertEqual("open_set_tracklet_confirmed", second.reason)
        self.assertTrue(second.created_speaker)

    def test_strong_known_evidence_resets_profile_contradiction(self):
        overlay = OpenSetTrackletOverlay(open_set_tracklet_config_for_preset(
            PROFILE_CONTRADICTION_TRACKLET_PRESET
        ))
        profiles = tuple(
            {"label": f"S{index + 1}", "centroid": np.eye(5)[index]}
            for index in range(4)
        )
        overlay.step(_step(
            np.eye(5)[4], profiles=profiles, media_time=1.0, base="S1"
        ))

        strong = overlay.step(_step(
            np.eye(5)[0], profiles=profiles, media_time=1.75, base="S1"
        ))
        restarted = overlay.step(_step(
            np.eye(5)[4], profiles=profiles, media_time=2.5, base="S2"
        ))

        self.assertEqual("S1", strong.visible_speaker)
        self.assertFalse(strong.diagnostics["profile_contradiction_pending"])
        self.assertIsNone(restarted.visible_speaker)
        self.assertEqual("open_set_pending_unknown", restarted.reason)

    def test_silence_resets_profile_contradiction_history(self):
        overlay = OpenSetTrackletOverlay(open_set_tracklet_config_for_preset(
            PROFILE_CONTRADICTION_TRACKLET_PRESET
        ))
        profiles = tuple(
            {"label": f"S{index + 1}", "centroid": np.eye(5)[index]}
            for index in range(4)
        )
        overlay.step(_step(
            np.eye(5)[4], profiles=profiles, media_time=1.0, base="S1"
        ))
        silence = _step(
            np.eye(5)[4], profiles=profiles, media_time=1.4, base="S1"
        )
        silence = OpenSetTrackletStep(**{**silence.__dict__, "speech": False})
        overlay.step(silence)

        restarted = overlay.step(_step(
            np.eye(5)[4], profiles=profiles, media_time=1.75, base="S2"
        ))

        self.assertIsNone(restarted.visible_speaker)
        self.assertEqual("open_set_pending_unknown", restarted.reason)

    def test_non_dedicated_step_cannot_create_or_update_tracklet_history(self):
        overlay = OpenSetTrackletOverlay()
        nondedicated = _step([1.0, 0.0], media_time=1.0)
        nondedicated = OpenSetTrackletStep(
            **{**nondedicated.__dict__, "probe_scheduled": False}
        )
        result = overlay.step(nondedicated)

        self.assertIsNone(result.visible_speaker)
        self.assertEqual(0, result.diagnostics["pending_count"])
        self.assertEqual(0, result.diagnostics["tracklet_count"])

    def test_two_causal_novel_probes_create_stable_public_identity(self):
        overlay = OpenSetTrackletOverlay()
        first = overlay.step(_step([1.0, 0.0], media_time=1.0))
        second = overlay.step(_step([0.99, 0.05], media_time=1.75))

        self.assertIsNone(first.visible_speaker)
        self.assertEqual("open_set_pending_unknown", first.reason)
        self.assertEqual("LIVE_TRACKLET_1", second.visible_speaker)
        self.assertTrue(second.created_speaker)
        self.assertTrue(second.provisional_speaker)

    def test_final_profile_aliases_atomically_to_existing_tracklet(self):
        overlay = OpenSetTrackletOverlay()
        overlay.step(_step([1.0, 0.0], media_time=1.0))
        overlay.step(_step([1.0, 0.0], media_time=1.75))
        merged = overlay.step(_step(
            [1.0, 0.0],
            profiles=({"label": "S3", "centroid": [1.0, 0.0], "name": "Speaker 3"},),
            media_time=2.5,
            base="S3",
        ))

        self.assertEqual({"S3": "LIVE_TRACKLET_1"}, overlay.final_to_public)
        self.assertEqual({"LIVE_TRACKLET_1": "S3"}, overlay.public_to_final)
        self.assertEqual("LIVE_TRACKLET_1", merged.visible_speaker)
        self.assertEqual(1, len(merged.aliases))
        self.assertEqual("S3", merged.aliases[0].final_internal_speaker_id)

    def test_profile_merge_is_bijective(self):
        overlay = OpenSetTrackletOverlay()
        overlay.step(_step([1.0, 0.0], media_time=1.0))
        overlay.step(_step([1.0, 0.0], media_time=1.75))
        overlay.step(_step(
            [1.0, 0.0],
            profiles=({"label": "S1", "centroid": [1.0, 0.0]},),
            media_time=2.5,
            base="S1",
        ))
        conflict = overlay.step(_step(
            [1.0, 0.0],
            profiles=(
                {"label": "S1", "centroid": [1.0, 0.0]},
                {"label": "S2", "centroid": [1.0, 0.0]},
            ),
            media_time=3.25,
            base="S2",
        ))

        self.assertNotIn("S2", overlay.final_to_public)
        self.assertEqual(1, conflict.diagnostics["alias_conflicts"])

    def test_old_unaliased_profile_cannot_claim_a_later_tracklet(self):
        overlay = OpenSetTrackletOverlay()
        profile = ({"label": "S1", "centroid": [0.0, 1.0]},)
        first_publication = overlay.step(_step(
            [1.0, 0.0], profiles=profile, media_time=1.0, base="S1"
        ))
        tracklet_created_later = overlay.step(_step(
            [1.0, 0.0], profiles=profile, media_time=1.75, base="S1"
        ))

        self.assertFalse(first_publication.aliases)
        self.assertEqual("LIVE_TRACKLET_1", tracklet_created_later.visible_speaker)
        self.assertNotIn("S1", overlay.final_to_public)

    def test_newly_published_profile_can_claim_an_existing_tracklet(self):
        overlay = OpenSetTrackletOverlay()
        old_profile = ({"label": "S1", "centroid": [0.0, 1.0]},)
        overlay.step(_step([1.0, 0.0], profiles=old_profile, media_time=1.0))
        overlay.step(_step([1.0, 0.0], profiles=old_profile, media_time=1.75))
        newly_published = overlay.step(_step(
            [1.0, 0.0],
            profiles=(
                {"label": "S1", "centroid": [0.0, 1.0]},
                {"label": "S2", "centroid": [1.0, 0.0]},
            ),
            media_time=2.5,
            base="S2",
        ))

        self.assertEqual(1, len(newly_published.aliases))
        self.assertEqual("LIVE_TRACKLET_1", overlay.final_to_public["S2"])

    def test_fresh_overlay_reset_allows_first_publication_again(self):
        stale_run = OpenSetTrackletOverlay()
        stale_run.step(_step(
            [1.0, 0.0],
            profiles=({"label": "S1", "centroid": [0.0, 1.0]},),
            media_time=1.0,
        ))

        fresh_run = OpenSetTrackletOverlay()
        fresh_run.step(_step([1.0, 0.0], media_time=1.0))
        fresh_run.step(_step([1.0, 0.0], media_time=1.75))
        published = fresh_run.step(_step(
            [1.0, 0.0],
            profiles=({"label": "S1", "centroid": [1.0, 0.0]},),
            media_time=2.5,
            base="S1",
        ))

        self.assertEqual(1, len(published.aliases))
        self.assertEqual("LIVE_TRACKLET_1", fresh_run.final_to_public["S1"])

    def test_retired_profile_reappearance_cannot_reclaim_tracklet(self):
        overlay = OpenSetTrackletOverlay()
        overlay.step(_step([1.0, 0.0], media_time=1.0))
        overlay.step(_step([1.0, 0.0], media_time=1.75))
        profile = ({"label": "S2", "centroid": [1.0, 0.0]},)
        overlay.step(_step([1.0, 0.0], profiles=profile, media_time=2.5, base="S2"))
        overlay.step(_step([1.0, 0.0], profiles=(), media_time=3.25))
        reappeared = overlay.step(_step(
            [1.0, 0.0], profiles=profile, media_time=4.0, base="S2"
        ))

        self.assertFalse(reappeared.aliases)
        self.assertNotIn("S2", overlay.final_to_public)

    def test_alias_retires_when_final_profile_disappears(self):
        overlay = OpenSetTrackletOverlay()
        overlay.step(_step([1.0, 0.0], media_time=1.0))
        overlay.step(_step([1.0, 0.0], media_time=1.75))
        overlay.step(_step(
            [1.0, 0.0],
            profiles=({"label": "S1", "centroid": [1.0, 0.0]},),
            media_time=2.5,
            base="S1",
        ))
        retired = overlay.step(_step([1.0, 0.0], profiles=(), media_time=3.25))

        self.assertEqual({}, overlay.final_to_public)
        self.assertEqual(1, len(retired.aliases))
        self.assertTrue(retired.aliases[0].retired)

    def test_relaxed_dual_scale_reuse_is_not_used_for_pending_confirmation(self):
        config = OpenSetTrackletConfig(
            reuse_short_min=0.90,
            weak_reactivation_short_min=0.20,
            weak_reactivation_long_min=0.40,
        )
        overlay = OpenSetTrackletOverlay(config)

        self.assertTrue(overlay._reuse_pass(0.5, 0.9, allow_relaxed=True))
        self.assertFalse(overlay._reuse_pass(0.5, 0.9, allow_relaxed=False))

    def test_stale_unbound_tracklet_is_excluded_from_reuse_and_pending_dedup(self):
        boundary = OpenSetTrackletOverlay()
        boundary.step(_step([1.0, 0.0], media_time=1.0))
        boundary.step(_step([1.0, 0.0], media_time=1.75))
        still_eligible = boundary.step(_step([1.0, 0.0], media_time=9.75))
        self.assertEqual("LIVE_TRACKLET_1", still_eligible.visible_speaker)
        self.assertEqual("open_set_tracklet_reuse", still_eligible.reason)

        expired = OpenSetTrackletOverlay()
        expired.step(_step([1.0, 0.0], media_time=1.0))
        expired.step(_step([1.0, 0.0], media_time=1.75))
        pending = expired.step(_step([1.0, 0.0], media_time=9.751))
        confirmed = expired.step(_step([1.0, 0.0], media_time=10.5))

        self.assertIsNone(pending.visible_speaker)
        self.assertEqual("open_set_pending_unknown", pending.reason)
        self.assertEqual("LIVE_TRACKLET_2", confirmed.visible_speaker)
        self.assertTrue(confirmed.created_speaker)
        self.assertEqual(2, confirmed.diagnostics["tracklet_count"])

    def test_stale_tracklet_keeps_alias_ability_but_not_reuse_authority(self):
        overlay = OpenSetTrackletOverlay()
        overlay.step(_step([1.0, 0.0], media_time=1.0))
        overlay.step(_step([1.0, 0.0], media_time=1.75))
        published = overlay.step(_step(
            [1.0, 0.0],
            profiles=({"label": "S2", "centroid": [1.0, 0.0]},),
            media_time=10.0,
            base="S2",
        ))

        self.assertEqual({"S2": "LIVE_TRACKLET_1"}, overlay.final_to_public)
        self.assertEqual(1, len(published.aliases))
        self.assertEqual("LIVE_TRACKLET_1", published.visible_speaker)
        self.assertEqual("base", published.reason)
        self.assertEqual(1, published.diagnostics["tracklet_count"])

    def test_idle_bound_prototype_expires_but_base_keeps_public_alias(self):
        overlay = OpenSetTrackletOverlay()
        overlay.step(_step([1.0, 0.0], media_time=1.0))
        overlay.step(_step([1.0, 0.0], media_time=1.75))
        profile = ({"label": "S2", "centroid": [1.0, 0.0]},)
        overlay.step(_step(
            [1.0, 0.0], profiles=profile, media_time=2.5, base="S2"
        ))
        after_ttl = overlay.step(_step(
            [1.0, 0.0], profiles=profile, media_time=10.501, base="S2"
        ))

        self.assertEqual({"S2": "LIVE_TRACKLET_1"}, overlay.final_to_public)
        self.assertFalse(after_ttl.aliases)
        self.assertEqual("LIVE_TRACKLET_1", after_ttl.visible_speaker)
        self.assertEqual("base", after_ttl.reason)
        self.assertEqual(1, after_ttl.diagnostics["tracklet_count"])


if __name__ == "__main__":
    unittest.main()

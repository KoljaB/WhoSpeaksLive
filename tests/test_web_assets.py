from __future__ import annotations

from pathlib import Path
import unittest

from window.web_assets import read_web_asset, read_web_text, render_live_index, web_asset_content_type


class PackagedWebAssetTests(unittest.TestCase):
    def test_live_index_uses_external_css_and_es_modules(self) -> None:
        rendered = render_live_index({"source": "demo", "language": {"name": "English"}})

        self.assertIn('/assets/web/live/styles.css', rendered)
        self.assertIn('type="module" src="/assets/web/live/app.js"', rendered)
        self.assertNotIn("__BOOTSTRAP_JSON__", rendered)
        self.assertNotIn("<style>", rendered)

    def test_bootstrap_json_cannot_close_its_script_node(self) -> None:
        rendered = render_live_index({"source": "</script><script>alert(1)</script>"})

        bootstrap = rendered.split('id="bootstrap-data"', 1)[1].split("</script>", 1)[0]
        self.assertNotIn("</script>", bootstrap)
        self.assertIn("\\u003c/script\\u003e", bootstrap)

    def test_asset_whitelist_and_content_types(self) -> None:
        self.assertTrue(read_web_asset("live/app.js").startswith(b"import "))
        self.assertIn(b"installMeetingChat", read_web_asset("live/meeting_chat.js"))
        self.assertIn(b"installHelpSystem", read_web_asset("live/help_system.js"))
        self.assertEqual(web_asset_content_type("live/app.js"), "text/javascript")
        self.assertEqual(web_asset_content_type("live/styles.css"), "text/css")
        with self.assertRaises(FileNotFoundError):
            read_web_asset("../../pyproject.toml")

    def test_report_and_fact_lens_pages_are_packaged_resources(self) -> None:
        report = read_web_text("reports/index.html")
        fact_lens = read_web_text("fact_lens/index.html")

        self.assertIn('/assets/web/reports/styles-base.css', report)
        self.assertIn('type="module" src="/assets/web/reports/app.js"', report)
        self.assertTrue(read_web_asset("reports/app.js").startswith(b"import "))
        self.assertIn("WhoSpeaksLive Fact Lens", fact_lens)

    def test_live_page_exposes_grounded_meeting_chat_controls(self) -> None:
        page = read_web_text("live/index.html")
        script = read_web_text("live/meeting_chat.js")

        self.assertIn('data-speaker-tab="ask"', page)
        self.assertIn('id="askSelectedMeetings"', page)
        self.assertIn("Ask selected sessions", page)
        self.assertIn("Ask this session", page)
        self.assertNotIn("Ask this meeting", page)
        self.assertIn('id="meetingChatForm"', page)
        self.assertIn('id="meetingChatMessages"', page)
        self.assertIn('id="meetingChatProgressBar"', page)
        self.assertIn('id="meetingChatProgressElapsed"', page)
        self.assertIn("meeting.started_at", script)
        self.assertIn("Not established from the selected transcript", script)
        self.assertIn("meetingChatProgressBar.value", script)
        self.assertIn("row_index", script)

        saved_reports = read_web_text("live/saved_reports.js")
        self.assertIn('const legacyChat = /^ROW-(\\d+)$/.exec(value);', saved_reports)

    def test_meeting_intelligence_tabs_reflect_service_availability(self) -> None:
        page = read_web_text("live/index.html")
        chat = read_web_text("live/meeting_chat.js")
        review = read_web_text("live/transcript_review.js")
        styles = read_web_text("live/styles.css")

        self.assertIn('data-speaker-tab="ask"', page)
        self.assertIn('fetch("/api/meeting-intelligence/status"', chat)
        self.assertIn('setMeetingCapability("ask"', chat)
        self.assertIn('setMeetingCapability("intelligence"', chat)
        self.assertIn("function setMeetingCapability", review)
        self.assertIn("requestedButton && requestedButton.disabled", review)
        self.assertIn(".speaker-tab.unavailable, .speaker-tab:disabled", styles)

    def test_live_page_exposes_context_help(self) -> None:
        page = read_web_text("live/index.html")
        app = read_web_text("live/app.js")
        script = read_web_text("live/help_system.js")
        handler = (Path(__file__).resolve().parents[1] / "src" / "window" / "live_http_handler.py").read_text(encoding="utf-8")

        self.assertNotIn('id="helpButton"', page)
        self.assertNotIn('class="inline-help-button"', page)
        self.assertIn('id="helpDrawer"', page)
        self.assertIn('event.key === "F1"', script)
        self.assertIn('import("./help_system.js")', app)
        self.assertIn("the core live app remains available", app)
        self.assertIn("disabledReason", script)
        self.assertIn("Speakers are meeting-local voice clusters", script)
        self.assertIn(r'/assets/web/live/[a-z][a-z0-9_]*\.(?:css|js)', handler)

    def test_live_video_can_float_and_uses_audio_master_sync(self) -> None:
        page = read_web_text("live/index.html")
        media = read_web_text("live/media_capture.js")
        policy = read_web_text("live/media_sync.js")

        self.assertIn('id="popoutMedia"', page)
        self.assertIn("requestPictureInPicture", media)
        self.assertIn("startAudioMasterVideoSync", media)
        self.assertIn("video.currentTime = action.targetTime", media)
        self.assertNotIn("audio.currentTime = action.targetTime", media)
        self.assertIn("enterDriftSeconds: 0.1", policy)
        self.assertIn("exitDriftSeconds: 0.04", policy)
        self.assertIn("hardSeekSeconds: 0.5", policy)

    def test_empty_speaker_removal_is_server_guarded(self) -> None:
        page = read_web_text("live/index.html")
        panel = read_web_text("live/speaker_panel.js")

        self.assertIn('id="speakerUndoCorrection"', page)
        self.assertIn('post("/api/speakers/remove-empty", {speaker_ids: [internalSpeakerId]})', panel)
        self.assertIn("expected_sentence_count: sentenceTotal", panel)
        self.assertIn("expected_source_sentence_count: speakerCurrentSessionSentenceCount(sourceSpeakerId)", panel)
        self.assertIn("expected_target_sentence_count: speakerCurrentSessionSentenceCount(targetSpeakerId)", panel)
        self.assertIn('{emptyOnly: true}', panel)


if __name__ == "__main__":
    unittest.main()

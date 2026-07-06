# Public Diarization Events

The window diarization server exposes a stable automation event stream at:

```text
GET /api/events
```

This endpoint uses Server-Sent Events (SSE). A client opens one HTTP request and
the server pushes JSON event envelopes as transcript and speaker state changes
happen. The browser-facing raw stream at `/events` is unchanged; `/api/events`
is the normalized API intended for Python tools, summarizers, and integrations.

Each event envelope has this shape:

```json
{
  "id": 12,
  "time": 1783370000.123,
  "schema": "whospeaks.events.v1",
  "type": "transcript.final_unknown",
  "source_event": "sentence",
  "session_id": "3d9f...",
  "payload": {}
}
```

Common event types:

```text
speaker.snapshot
speaker.created
speaker.renamed
speaker.updated
speaker.removed
speaker.state_changed
transcript.pending
transcript.final
transcript.final_unknown
transcript.speaker_assigned
transcript.speaker_revised
transcript.speaker_cleared
transcript.updated
live_speaker.changed
live_speaker.updated
system.status
system.error
session.stopped
```

Transcript consumers should store rows by `payload.id` or `payload.index`.
Speaker labels can be revised after a final row arrives, so treat transcript
events as upserts rather than append-only messages.

Minimal Python client:

```python
from window.event_client import DiarizationEventClient

client = DiarizationEventClient("http://localhost:8796")

@client.on("transcript.final_unknown")
def on_unknown(event):
    print("Unknown:", event["payload"]["text"])

@client.on("transcript.speaker_revised")
def on_revision(event):
    payload = event["payload"]
    print(payload["previous_speaker"], "->", payload["new_speaker"], payload["text"])

@client.on("speaker.state_changed")
def on_speakers(event):
    print("Speakers:", [speaker["display_name"] for speaker in event["payload"]["speakers"]])

client.run_forever()
```

Use `GET /api/events?snapshot=0` to skip the initial `speaker.snapshot` event.

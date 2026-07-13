# Live Window Workflow

The live window workflow turns one media session into a speaker-labeled transcript while showing fast speaker feedback during playback.

## The Main Screen

The browser UI has three important areas:

- Video or media area: the source being replayed.
- Live transcript: the currently active and recently finalized transcript rows.
- Speaker panel: known and discovered speakers, speaking time, sentence count, and the active Live tag.

The speaker panel is the fastest visual signal. The transcript is optimized to avoid distracting color flicker inside a running sentence.

## Start A Run

1. Launch the app.
2. Open the browser URL.
3. Pick or load media.
4. Press Start.
5. Keep the media playing until the content you care about has finished.

The app grows an ASR window over media time. When a sentence boundary is accepted, the sentence is emitted to the transcript and speaker memory is updated.

## Live Speaker Feedback

The active Live tag is driven by short, frequent embedding probes over recent audio. It is meant to answer: "Who is probably speaking right now?"

The live transcript uses the dominant live speaker over the stable part of the sentence. Short pauses do not immediately turn a row gray, and late speaker changes near the end of a long sentence are discounted so the row usually keeps the speaker that dominated most of the sentence.

On a single local GPU, live speaker probes share capacity with final ASR. The default probe cadence includes backpressure so live feedback should not starve final transcription. For transcript-quality tests where timing matters more than live speaker highlighting, launch with `--no-live-speaker-assignment`; realtime text preview stays enabled and final speaker assignment still runs on committed rows.

## Final Sentence Assignment

Final sentences are assigned after ASR has enough text and timing information. The speaker ID is based on embeddings from the sentence audio and the current speaker memory.

The final loop uses a cooldown after a successful split. That means it can emit all sentences found in a pass, then wait briefly before trying another split instead of immediately retrying 200 ms later.

## Music And Non-Speech Gaps

The final ASR path uses VAD plus ASR no-speech metadata to avoid adding text for music-only or silence-heavy windows. VAD, or voice activity detection, is the fast audio check that decides whether a window contains speech-like audio. Whisper-like ASR models also return `no_speech_prob` for each segment, which estimates whether the segment was actually non-speech.

When a segment has high `no_speech_prob`, the app drops that ASR segment instead of adding it to the transcript. This is not a transcript text filter and it does not search for known hallucinated phrases. Very short low-confidence segments can still be kept so short real responses such as "yes", "ok", or "ja" are not lost.

See [Configuration](configuration.md#asr-no-speech-filtering) for the tuning flags and defaults.

## Speaker Management

During a run you can:

- Rename a speaker.
- Clear speaker memory.
- Add reference audio for a named speaker.
- Save a speaker group locally.
- Export a portable speaker group JSON.
- Import a speaker group JSON from disk.

Saved speakers are most useful after a complete pass over a representative clip.

## Saved Sessions

The Sessions tab stores completed or autosaved runs so they can be reopened later for transcript review, speaker cleanup, filtering, and export without rerunning ASR or embeddings.

The session filters are visibility filters:

- `Active`: saved sessions that are not archived.
- `Archived`: saved sessions that have been archived.
- `All`: both active and archived sessions.

Active does not mean currently recording, currently open, or resumable. A newly saved YouTube run appears in Active by default because archive is only a visibility flag. Archiving a session hides it from Active, keeps it in Archived and All, and does not delete transcript, speaker, embedding, or audio-reference data.

Session names can be edited directly in the Sessions tab: click the session title, type the new name, and press Enter or click away to save it. The row menu still exposes Rename and Delete for explicit actions.

New sessions use the clearest title available. Preset YouTube clips use their preset title; custom YouTube runs fall back to the YouTube video id plus the run start time instead of only showing `youtube.com`. The row metadata shows the saved start-to-end time range in a compact form, followed by duration and speaker count.

Use `+ New session` in the Sessions tab to create a new active session immediately and return to a clean Ready state. Pressing Start also creates a new active session immediately if one is not already prepared; transcript rows, speakers, embeddings, and audio references are filled into that same session as the run progresses.

## Ask Sessions

The **Ask** tab answers questions from transcript evidence for the current live session, an opened saved session, or up to 20 checked saved sessions. Checked sessions take priority over an opened or running session; when nothing is checked, the opened saved session is used, followed by the current running session. Click **Ask selected sessions** beside the bulk session actions to open the selected scope directly.

Each answer links to its session, speaker, and timestamp. Clicking a citation opens the corresponding saved session, scrolls to the transcript row, and highlights it. Live answers use only finalized rows and show the transcript cutoff used for the answer.

Short single-session transcripts are sent to the configured large language model directly. Long or multi-session scopes use hybrid search, which combines semantic similarity from text embeddings with exact keyword matching, so they require a configured text-embedding endpoint. See [Ask sessions](ask-sessions.md) for setup, scope rules, indexing behavior, and troubleshooting.

## Practical Tips

- Let the full clip finish before saving a speaker group.
- Use Archive to remove old saved sessions from the default Active list without deleting them.
- Use a remote GPU server for smoother local UI work when using large ASR and embedding models.
- Disable live speaker assignment during local bottleneck tests to separate final ASR speed from live embedding contention.
- If a newly detected speaker appears at the end of a sentence, wait for the sentence split before judging final assignment quality.
- Use validation runs when comparing parameter changes.

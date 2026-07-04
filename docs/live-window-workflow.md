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

## Final Sentence Assignment

Final sentences are assigned after ASR has enough text and timing information. The speaker ID is based on embeddings from the sentence audio and the current speaker memory.

The final loop uses a cooldown after a successful split. That means it can emit all sentences found in a pass, then wait briefly before trying another split instead of immediately retrying 200 ms later.

## Speaker Management

During a run you can:

- Rename a speaker.
- Clear speaker memory.
- Add reference audio for a named speaker.
- Save a speaker group locally.
- Export a portable speaker group JSON.
- Import a speaker group JSON from disk.

Saved speakers are most useful after a complete pass over a representative clip.

## Practical Tips

- Let the full clip finish before saving a speaker group.
- Use a remote GPU server for smoother local UI work when using large ASR and embedding models.
- If a newly detected speaker appears at the end of a sentence, wait for the sentence split before judging final assignment quality.
- Use validation runs when comparing parameter changes.

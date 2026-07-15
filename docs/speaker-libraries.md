# Legacy Speaker-Group Files

Speaker-group JSON files are the older portable workflow for seeding meeting-local Speaker profiles; use [People And Voice Recognition](people-and-recognition.md) for persistent real-person identities.

## Choose The Right Model

| People | Speaker-group file |
| --- | --- |
| Persistent Person identity | Meeting-local Speaker seed |
| Several independently managed Voice samples | One exported set of profiles |
| Suggestion, confirmation, and roster controls | Explicit load/import before a run |
| Stored in `people.json` and `voice-samples/` | Stored as a portable JSON file |

Speaker-group export does **not** contain People, Person settings, Person-owned Voice samples, or retained manual audio. It is not a People backup.

## Concepts

A legacy speaker profile contains a meeting label, display metadata, speech duration, sentence count, and a centroid. A centroid is the average embedding vector used to represent that meeting Speaker.

When the final and live-speaker embedding providers differ, a group can contain separate live profiles. Embeddings from different provider contracts are not directly comparable.

## Save And Load

Save a group only after the app has processed representative speech for each Speaker. Load it before another run when the older workflow is specifically required.

Local groups share the speaker-library directory with the People library by default:

```text
runtime/speakers/
```

Move the directory with `WHOSPEAKS_SPEAKER_LIBRARY_DIR` or `--speaker-library-dir`. Back up the complete directory—not only group files—when People are also in use.

## Import And Export

Export creates a portable Speaker-group JSON. Import loads that JSON into current meeting memory.

Use it to:

- reproduce an older workflow;
- seed a controlled validation run; or
- move a legacy meeting-local profile set.

Do not use it as the primary workflow for recurring participants or as a privacy backup.

## Migrate A Group To People

1. Load the legacy group once.
2. Let the meeting Speakers appear.
3. Use **Link to Person…** for each recurring participant.
4. Create or select the corresponding Person.
5. Verify in **Settings → People** that a compatible Voice sample was safely enrolled.
6. Turn on **Include in automatic recognition** only for plausible upcoming attendees.
7. Use People in later meetings; the group no longer needs to be loaded routinely.

If an immediate live assignment does not activate after loading a legacy group, check that `--live-speaker-embedding-provider` matches the provider used when the group was saved.

# People and Voice profiles

The goal is to recognize returning people conservatively while keeping meeting-local Speakers separate from durable People and their Voice samples.

## The model

A **Speaker** such as `S2` is one cluster inside one meeting. It stays meeting-local, including when the meeting is saved. A **Person** is a durable identity with a stable ID; two People may have the same display name. A **Voice profile** is all recognition evidence owned by one Person, and a **Voice sample** is one independently manageable manual or meeting-derived source.

```text
Meeting
└── Speaker S2 ── linked to ── Person Alice

Person Alice
└── Voice profile
    ├── Manual Voice sample · headset
    ├── Manual Voice sample · telephone
    ├── Meeting Voice sample · laptop
    └── Meeting Voice sample · room microphone
```

Naming a Speaker changes a meeting label only. It does not create a Person or store biometric evidence. Use **Recognize in future meetings…** to create or select a Person and explicitly link the Speaker.

## Suggestions and confirmation

Recognition is suggestion-first. A compatible, sufficiently strong and distinct match appears as **Likely Alice**, with **Confirm** and **Not Alice** actions. A confirmed link appears as **Linked to Alice · Recognition active**. `Unknown` remains valid even when only one Person is expected.

The matcher filters disabled People, the meeting's expected-People roster, disabled or quarantined samples, source policy, provider contract, vector dimension, and invalid data before scoring. It scores each Voice sample once. The strongest sample is primary evidence, with at most a small capped contribution from one independent, non-duplicate corroborating source. An absolute threshold and a margin over the nearest competing Person are both required.

The initial conservative constants are covered by the deterministic cross-condition fixture in `tests/fixtures/person_voice_matching_cases.json`: near-duplicate representations at cosine similarity 0.995 or above do not corroborate; corroboration begins at 0.62 and can add at most 0.025. Provider-specific acceptance thresholds remain controlled by the live configuration.

## Manual Voice samples

Create or select a Person, then choose **Add Voice sample**. Upload and microphone recording both require that Person's ID. The app decodes the audio, rejects silence, very short or severely clipped input, embeds speech windows, removes statistical outliers, and stores one source-level sample.

Multiple manual samples may coexist. Each can be disabled, re-enabled, relabeled through the API, or permanently deleted. Automatic meeting learning never replaces or edits a manual sample.

The original manual audio is retained locally under the Person and sample-owned `voice-samples/` directory so it can be audited or re-embedded. Public API state reports that retention but never exposes the absolute path. **Delete** removes the retained audio and representations. **Forget voice data** removes every sample while preserving the Person and historical transcript labels. A cleanup failure is reported and the profile is not claimed as deleted.

## Recognition policy and expected People

Each Person has independent controls:

- **Manually added Voice samples**: include manual sources in matching.
- **Confirmed meeting samples**: include learned meeting sources in matching.
- **Learn from confirmed meetings**: allow future confirmed meetings to update their own meeting sample.

Both source categories and learning are on by default. Turning learning off does not hide stored meeting samples. Excluding meeting samples from matching does not turn learning off.

**Expected this meeting** is a session-scoped candidate roster. It does not change the Person's global recognition-enabled state, and a new session clears the roster.

## Confirmed meeting learning

An explicit link freezes a confirmation-time seed. Reliable finalized sentence evidence must pass duration, quality, overlap, competing-Speaker, competing-Person, cohesion, outlier, and fixed-seed gates. Checkpoints replace at most one effective sample for the Person, session, provider, and capture condition; other meetings and all manual samples stay unchanged.

Derived samples record their trusted anchors. Disabling or deleting their only trusted anchor quarantines the dependent evidence rather than letting it validate itself recursively. Corrections and unlinking recompute or remove only the affected session sample.

## Saved meetings

Saved review uses a dedicated operation addressed by `session_id` and `speaker_id`; it never routes saved `S1` to an unrelated live `S1`. The operation reads the saved transcript's current corrected assignments, stored embedding records, Speaker profiles, and provider metadata. It robustly reconstructs one sample, writes the saved Speaker–Person link and Person sample idempotently, and recomputes that sample after later row corrections.

When compatible evidence is missing, **Recognize in future meetings…** remains visible but disabled with a stable factual explanation. Saved enrollment does not require ownership of an unrelated live session.

Saved-session and People writes use a fixed lock order—session store, then People library—and a durable transaction-intent file. Retrying the same operation repairs an interrupted second write without duplicating a sample.

## Storage migration and privacy

People library v2 stores `Person → VoiceSample → EmbeddingRepresentation`. On the first mutation of a readable v1 library, WhoSpeaks writes a timestamped v1 backup before atomically replacing the original with v2. Existing v1 templates become meeting samples; they are never guessed to be manual samples by matching names. Newer unsupported versions fail clearly and are never downgraded.

For local recovery, stop WhoSpeaks, preserve the failed `people.json` for diagnosis, and copy the newest adjacent `people.v1.<timestamp>.bak.json` back to `people.json`. The backup remains v1 and will be migrated again on the next guarded write.

Provider and model identity is part of every representation's compatibility contract. A provider upgrade does not compare incompatible vectors or penalize them with an artificial score; the sample stays visible as incompatible. Retained manual audio makes an explicit future re-embedding operation possible, but the first v2 implementation never rewrites it silently.

Public Person and saved-session payloads omit embedding vectors, encoded centroids, and absolute local paths. Historical confirmed transcript assignments are not silently rewritten when samples or People are disabled or deleted.

## Legacy Speaker groups

Speaker-group import and export remain available for compatibility. A legacy group may seed meeting-local diarization memory, but the normal cross-meeting workflow is Person-owned. Legacy references are not automatically attached to a Person by name.

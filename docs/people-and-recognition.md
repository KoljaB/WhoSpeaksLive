# People And Voice Recognition

People let WhoSpeaks recognize the same real person across otherwise independent meetings without forcing an uncertain identity onto a detected voice.

> **Before enrolling a voice:** Voice recordings and voice embeddings are sensitive, voice-derived identity data. Obtain appropriate authorization, tell participants when cross-meeting recognition is used, and do not treat a WhoSpeaks suggestion as identity authentication. See [Security And Data Privacy](security-and-data-privacy.md).

## The Three Things To Know

- A **Speaker** is a voice cluster inside one meeting, such as `Speaker 2`. It is local to that meeting.
- A **Person** is a durable identity, such as `Alice`, that can be linked to Speakers in many meetings.
- A **Voice sample** is reusable evidence owned by one Person. A Person may have several samples from different microphones, rooms, or calls.

These are separate layers:

```text
UNKNOWN transcript row
  -> Speaker 2             WhoSpeaks has clustered the voice in this meeting
  -> Speaker 2 linked to Alice
  -> Alice owns one or more reusable Voice samples
```

`UNKNOWN` means a transcript row has not yet been assigned to a meeting Speaker. `Speaker 2` without a Person link means the voice cluster is known locally, but its persistent identity is not.

Renaming `Speaker 2` to `Alice` changes the meeting label only. Use **Link to Person…** when you want a durable identity and future recognition.

## Quick Start: Learn Alice From A Live Meeting

1. Start transcription and let Alice speak naturally. Three clean sentences or at least six seconds of coherent speech gives the enrollment safety checks useful evidence.
2. Open **Speakers** and expand Alice's detected Speaker.
3. Choose **Link to Person…**.
4. Select an existing Person or enter `Alice` to create one.
5. Read the status message:
   - **Linked … and saved a Voice sample** means identity and reusable evidence were both stored.
   - **Linked …; more clean speech is needed** means the meeting identity is saved, but no unsafe or undersized Voice sample was enrolled. Keep the meeting running; later clean evidence can be learned if that Person allows learning from confirmed meetings.
6. Open **Settings → People**. Turn on **Include in automatic recognition** if Alice is a plausible attendee in upcoming meetings. New People start with this setting off.

Linking never changes that roster setting for you. A Person can therefore be safely stored without participating in future automatic suggestions.

## Alternative: Add Alice Before A Meeting

1. Open **Speakers** and choose **+ Add person**.
2. Enter Alice's name.
3. Open **Settings → People → Alice → Add Voice sample**.
4. Upload an audio file or record from the microphone.
5. Turn on **Include in automatic recognition** when you want Alice considered.

An empty Person has a name and settings but cannot be recognized until at least one compatible Voice sample is active.

For a good manual sample:

- use one speaker only;
- provide at least 1.5 seconds of usable speech and preferably eight seconds or more;
- avoid overlap, music, imitation, clipping, and heavy background noise;
- capture representative conditions rather than many near-duplicates;
- add another sample when the real use case differs substantially, such as headset versus telephone audio.

Manual uploads and microphone recordings retain the original audio on the WhoSpeaks backend so the sample can be audited or re-enrolled later.

> **Provider warning:** The Quickstart smoke provider is for checking installation, not building a durable People library. Choose the intended production embedding provider before enrolling important Voice samples. Samples are not automatically re-embedded after a provider change.

## What Happens In The Next Meeting

Recognition is suggestion-first:

1. WhoSpeaks first creates a meeting-local Speaker.
2. A short first utterance may remain unidentified.
3. Each finalized sentence adds evidence.
4. Automatic Person matching starts after at least two finalized sentences or about four seconds of speech.
5. If one eligible Person clears both the similarity and separation checks, the UI shows **Likely Alice**.
6. Choose **Confirm**, **Not Alice**, or use **Link to Person…** manually.

Matching is recalculated as finalized evidence accumulates, so a suggestion may appear later in the same meeting. A match is never forced merely because only one Person is eligible.

**Confirm** links this Speaker to Alice and stores a safe meeting-derived Voice sample when sufficient coherent evidence exists. **Not Alice** rejects Alice only for this detected Speaker in the current meeting; it does not disable Alice, delete data, or affect later meetings.

One Person is reserved to one current Speaker suggestion or confirmed link. This prevents the same identity from being proposed for several simultaneous Speaker clusters.

## Automatic Recognition Eligibility

All three conditions are required:

| Included in automatic recognition | Recognition active | Compatible active sample | Result |
| --- | --- | --- | --- |
| On | On | Yes | Eligible for a suggestion, never guaranteed |
| Off | On | Yes | No automatic suggestion |
| On | Off | Yes | No automatic suggestion |
| On | On | No | No automatic suggestion |

An empty included roster means no People participate in automatic recognition. Manual linking remains available.

**Include in automatic recognition** is a persistent roster, not a one-meeting reset. WhoSpeaks remembers it across sessions and restarts until you change it. Keep it limited to plausible attendees. A large roster creates more comparisons and more close alternatives, which increases ambiguity and often causes conservative non-matches.

**Recognition active** is also persistent. Adding or learning a Voice sample does not silently turn it back on. When off, the Person cannot be returned as an automatic match. Existing samples may still act as collision guards during safe enrollment so another Person is not accidentally taught the same voice.

## Recognition Sources And Continued Learning

Each Person has independent source controls:

- **Use manual Voice samples:** allow uploaded or microphone-recorded samples in matching.
- **Use confirmed meeting samples:** allow safe meeting-derived samples in matching.
- **Learn from confirmed meetings:** allow later safe checkpoints to refresh the current meeting's sample as corrections and additional evidence arrive.

Explicitly linking or confirming a Speaker may create the initial safe meeting sample even when continued learning is off. The continued-learning setting governs later automatic updates.

Meeting enrollment requires coherent evidence and checks it against competing Speakers and other People. A direct identity link is still allowed when that evidence is insufficient, but no fallback profile centroid is stored as a Voice sample.

Automatically derived meeting samples are capped at eight per embedding provider for each Person; lower-quality automatic samples may be pruned as stronger conditions arrive. Manual samples and explicitly user-confirmed samples are not silently downgraded to automatic evidence.

## What Every Control Does

| Control | What changes | What stays |
| --- | --- | --- |
| Rename Speaker | Meeting display name | Person links and Voice samples |
| Link to Person | This Speaker's durable identity; a safe source sample when available | Roster and Recognition active settings |
| Confirm | Accepts the current suggestion and safe evidence | Other meetings and roster settings |
| Not Alice | Rejects Alice for this Speaker in this meeting | Alice, her samples, and later meetings |
| Unlink | Removes this Speaker–Person link and this meeting's Person-owned sample | Person, independent samples, transcript text, and displayed meeting name |
| Disable sample | Excludes one sample from matching | The sample and retained audio |
| Delete sample | Removes one sample and retained manual audio; a deleted meeting source is suppressed during background recomputation | Person, historical transcript labels, and saved meeting evidence |
| Recognition active | Allows or pauses matching for the whole Person | Links and stored data |
| Include in automatic recognition | Adds or removes the Person from the persistent candidate roster | Links and stored data |
| Forget voice data | Removes all Person-owned samples and retained manual audio; unlinks that Person from live and saved sessions | Person name, meeting transcripts, and unlabelled historical meeting evidence |
| Delete person | Removes the Person, Person-owned samples, retained manual audio, and saved identity links | Historical transcript labels and unlabelled meeting evidence |
| Reset live speaker detection… | Clears current detected Speakers and the live transcript presentation | People and Person-owned Voice samples |
| Delete saved session | Removes the saved meeting, Person-owned samples derived from it, and suppression markers that name it | Other People and other meetings |

Deletion removes data from active WhoSpeaks-managed storage. It is not secure media erasure and cannot remove external audio files, downloaded exports, migration backups, filesystem snapshots, or third-party backups. Follow the complete-erasure guidance in [Security And Data Privacy](security-and-data-privacy.md#complete-erasure).

## Saved-Session Review

Opening a saved session changes the app from live operation to historical review:

- **+ Add person** and **Reset live speaker detection…** are disabled because they operate on live state.
- Use a saved Speaker's **Link to Person…** action to select an existing Person or create a new one.
- Linking is disabled when the session lacks a recorded provider, compatible embeddings, enough coherent speech, or sufficient separation from another Speaker or Person. The UI shows the specific reason.
- **Unlink** removes that saved Speaker's Person link and its source meeting sample, while keeping transcript labels.

Saved link and unlink operations use a durable transaction-intent file. If the process stops between the People-library write and the saved-session write, reopening the session completes the pending operation idempotently instead of creating another Person or sample.

Saved corrections recalculate every linked Person sample in that session because reassigning transcript rows changes the evidence boundaries. A deleted meeting sample remains suppressed during this background recalculation. To intentionally recreate it, unlink and link the saved Speaker again.

## Provider Compatibility

A Voice representation records its embedding-provider contract and vector length. Matching ignores samples whose provider or dimension differs from the active session.

WhoSpeaks cannot detect a model replacement hidden behind the same provider name with the same vector length. Treat a provider name as a stable contract: use a new provider identifier when changing an embedding model incompatibly.

Raw manual audio is not automatically re-embedded after a provider change. Re-enroll a compatible sample deliberately.

## Storage, Backup, And Moving Machines

People are not part of legacy Speaker-group exports. A complete People backup must preserve the entire speaker-library directory, including `people.json`, `voice-samples/`, and any migration backups. Preserve the session directory as the same point-in-time set if historical links and reconstructible meeting evidence matter.

Stop every WhoSpeaks process using those directories before copying them. Restore both directories together, restart WhoSpeaks, and verify provider compatibility.

See [Security And Data Privacy](security-and-data-privacy.md) for the full storage map, remote-processing flows, access controls, backup procedure, and deletion limits.

## Troubleshooting

### Alice is saved but never suggested

Check, in order:

1. **Include in automatic recognition** is on.
2. **Recognition active** is on.
3. At least one Voice sample is active and compatible.
4. The relevant manual/meeting source category is enabled.
5. The meeting has at least two finalized sentences or about four seconds for that Speaker.
6. The embedding provider has not changed.
7. Alice was not rejected with **Not Alice** for this Speaker.
8. The roster is small enough to avoid unnecessary ambiguity.

### The suggestion appears late

This is expected for short or fragmented speech. Matching starts only after the evidence gate and is rerun as more finalized speech arrives. Noise, overlap, and changing microphones can require more evidence.

### The wrong Person is suggested

Choose **Not Person**, then use **Link to Person…** if the correct identity is known. Review duplicate People, weak samples, and an overly broad included roster. Do not confirm merely to make the UI look complete.

### A Person exists but says Recognition unavailable

The Person has no active sample compatible with the current provider and vector dimension. Enable an appropriate sample, restore the intended provider, or enroll a new compatible sample.

### Controls are gray after opening a session

You are reviewing historical state. Return to the live session for **+ Add person** and live reset controls. Use the saved Speaker's own Link or Unlink action for historical identity work.

### Saved Link to Person is disabled

Read the reason beside the control. Common causes are missing embeddings, a missing provider record, insufficient coherent speech, or evidence too close to another Speaker or Person.

## Legacy Speaker-Group Files

People are the normal cross-meeting identity workflow. Speaker-group JSON files are retained for compatibility and portable meeting-local seeding.

| People | Legacy Speaker-group file |
| --- | --- |
| Durable real-person identity | Portable meeting-local profile seed |
| Multiple Voice samples per Person | One manually loaded/exported group |
| Automatic suggestions and a persistent roster | Explicit load/import each time |
| Stored in the People library | Stored as a separate JSON file |
| Not included in group export | Does not back up People |

To migrate an old group, load it once, let its Speakers appear, link each Speaker to a new or existing Person, verify that a compatible Person Voice sample was saved, and use People for later sessions. Most new workflows do not need both systems.

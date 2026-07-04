# Speaker Libraries

Speaker libraries let the app reuse known voices across sessions instead of rediscovering every speaker from scratch.

## Concepts

A speaker profile contains a speaker label, display metadata, speech duration, sentence count, and a centroid. A centroid is the average embedding vector used to represent that speaker.

When the final embedding provider and live-speaker embedding provider differ, the app can keep separate live-speaker profiles. This matters because embeddings from different providers are not directly comparable.

## Saving Speakers

Save a speaker group after the app has processed enough speech for each speaker. A complete run usually gives better profiles than a short partial run.

Saved speaker groups go under the speaker library directory:

```text
runtime/speakers/
```

You can move this directory with `WHOSPEAKS_SPEAKER_LIBRARY_DIR`.

## Loading Speakers

Load a speaker group before starting a new run when you want immediate assignment for known voices. Loaded groups should provide:

- Final profiles for completed sentence assignment.
- Live profiles when the group was saved with a separate live-speaker embedding provider that matches the current live provider.

If the current live provider differs from the saved live provider, the app avoids using incompatible live profiles.

## Import And Export

Export creates a portable JSON speaker group. Import reads that JSON back into the current speaker library.

Use export when you want to:

- Move speakers to another checkout.
- Share a known-speaker set with another machine.
- Preserve a snapshot before experiments.

## Best Practices

- Keep provider settings consistent when comparing sessions.
- Use descriptive speaker names after a full run, then save the group.
- Prefer export for backups; prefer save/load for day-to-day local use.
- If live speaker assignment does not activate immediately after loading a group, check that `--live-speaker-embedding-provider` matches the provider used when the group was saved.

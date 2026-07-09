# Technical Description

WhoSpeaksLive solves speaker diarization by repeatedly converting recent audio into text, turning voice audio into embeddings, and comparing those embeddings against speaker memory.

## From Audio To Sentences

Audio arrives from a replayed media source. The app tracks playback time and extracts windows of audio.

ASR, or automatic speech recognition, turns each audio window into words with timestamps. The final diarization loop does not emit every partial word immediately. Instead, it waits for a complete sentence or a stable boundary, then emits a final sentence with start and end times.

After a sentence split succeeds, the loop waits for the configured interval before trying another final split. This keeps the app from repeatedly reprocessing almost the same boundary a few hundred milliseconds later.

## From Sentences To Speakers

For each accepted sentence, the app extracts the matching audio and sends it to an embedding provider. A speaker embedding is a vector: a list of numbers that captures voice characteristics in the coordinate system of a specific model.

The speaker memory compares the new vector with existing speaker profiles:

- If it is close to a known speaker, the sentence is assigned to that speaker.
- If it is not close enough, the app may create a new speaker.
- If the evidence is weak, the sentence may stay unknown until later refinement.

The final speaker profile is updated over time as more sentences are assigned.

## Live Speaker Versus Final Speaker

The app has a fast live-speaker path and a final sentence path.

The live path uses short recent audio windows to answer "who is speaking now?" It drives the Live tag and active transcript styling. The final path waits for sentence context and usually has more audio.

These paths can use different embedding providers. When they do, the app must not compare live embeddings directly against final embeddings. Instead, it keeps live-compatible speaker profiles built with the live provider.

## Provider Stacks

An embedding provider stack combines multiple providers in one vector.

Example:

```text
espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34
```

Each provider produces its own normalized vector. The app weights each component, concatenates them, and normalizes the combined vector. All profiles made with that stack should be compared only with embeddings made by the same stack.

## Why Validation Matters

Small timing changes can improve live feel while hurting final accuracy, or improve final accuracy while increasing live latency. Validation keeps those tradeoffs visible by scoring a reproducible run against a canonical transcript.

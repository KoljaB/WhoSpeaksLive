# Speaker Model Presets

Speaker model presets select the embedding models used to identify speakers; each preset configures both the final sentence path and the optional live-speaker path.

An embedding is a numeric voice representation. WhoSpeaks compares embeddings made by the same model or exact model stack to decide whether two speech segments likely belong to the same speaker.

## Where the setting appears

In the full-screen `whospeaks` launcher, open **Settings** and select **Speaker model preset**. The main-screen **Live speaker labels** checkbox is independent:

- On uses the preset's live provider for provisional speaker labels while speech is in progress.
- Off disables provisional live-speaker scoring, but the preset's final provider still assigns speakers to completed transcript sentences.

Changing this preset does not select Nemotron or Kroko. Those are live-text transcription engines, not speaker-embedding models. It also does not change **Deployment**; the selected providers run through the configured local or remote embeddings backend.

## Exact preset definitions

The launcher labels, stable preset IDs, and provider expressions below match the current application definitions.

| Launcher label | Preset ID | Final sentence provider | Live-speaker provider | Intended use |
| --- | --- | --- | --- | --- |
| **Low VRAM - SpeechBrain ECAPA** | `smoke` | `speechbrain_ecapa` | `speechbrain_ecapa` | Lowest-complexity first run. Use it to prove installation, media flow, embeddings, and speaker assignment before comparing quality. |
| **Single model - ESPnet ECAPA** | `single_espnet` | `espnet_ecapa_wavlm_joint` | `espnet_ecapa_wavlm_joint` | Isolate one ESPnet model on both paths. Useful for controlled provider evaluation without an ensemble. |
| **Low VRAM final + fast live** | `smoke_fast_live` | `speechbrain_ecapa` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Keep the simple final provider while using the responsive two-model live stack. This loads more models than `smoke`. |
| **High quality - public ensemble** | `public_quality` | `espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Reproducible public four-model final candidate plus the fast live stack. Expect more model downloads, startup time, RAM, and VRAM than single-model presets. |
| **Recommended - public ensemble** | `promoted_public` | `espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37` | `speechbrain_resnet` | Current promoted `whospeaks-window` final stack plus the real-GUI-validated live provider. “Recommended” does not mean the starter always selects it or that it wins on every recording or dataset. |

### `smoke`: Low VRAM - SpeechBrain ECAPA

This is the safest bring-up preset because both paths use one public model. Final and live embeddings share the same vector space, so the application does not need a separate live-compatible speaker profile set.

“Low VRAM” is relative to the multi-provider presets. Actual memory use still depends on the embeddings backend, device, installed framework versions, concurrent ASR models, and audio workload. This preset is a functional baseline, not the claimed accuracy leader.

### `single_espnet`: Single model - ESPnet ECAPA

This uses one ESPnet ECAPA/WavLM model for both final and live assignment. It is intended for measuring that provider in isolation. It should not be described as equivalent to either weighted public ensemble.

### `smoke_fast_live`: Low VRAM final + fast live

Final sentence assignment remains on SpeechBrain ECAPA. Provisional live labels use the pyannote/WeSpeaker ResNet34 PyTorch plus ONNX stack. Because final and live providers differ, WhoSpeaks maintains live-compatible speaker profiles rather than comparing live vectors directly with final SpeechBrain vectors.

This is useful when responsive live labels matter but the final provider is not the quality experiment. Its “Low VRAM final” wording describes only the final side; enabling the separate two-model live stack has an additional memory and startup cost.

### `public_quality`: High quality - public ensemble

The final embedding concatenates four independently normalized public-provider vectors after applying their configured component weights. The result is normalized again. The weights are vector scaling factors, not probabilities, confidence values, percentages, or a requirement to sum to one.

This preset remains a public quality candidate for reproducible comparison. “High quality” does not mean it is proven best for every microphone, language, speaker population, or acoustic condition.

### `promoted_public`: Recommended - public ensemble

The final path uses the current three-provider stack promoted by `whospeaks-window`; the live path uses SpeechBrain ResNet. Finalized sentences are encoded in both vector spaces so the same speaker profiles can be compared with the shifting live windows.

Provider presets select embedding models only. The 0.7/1.5-second windows, Bayesian tracker, 0.4-second cadence, 2.5-second hold, and open-set tracklet policy are global application defaults and do not change when a provider preset changes.

Keep this preset distinct from `public_quality`: current validation has not established one public ensemble as the universal winner for all target data. The launcher label marks the currently promoted option, while `public_quality` preserves the alternative reproducible public candidate. A fresh starter profile can still begin with the `smoke` preset for safer bring-up.

## What a weighted provider expression means

Expressions use `+` to join providers and `=` to give each component a non-negative scaling weight:

```text
provider_a=1.0+provider_b=0.5
```

For each audio segment, WhoSpeaks:

1. obtains and normalizes one vector from each provider;
2. multiplies each vector by its configured weight;
3. concatenates the weighted vectors;
4. normalizes the combined vector.

Only compare speaker profiles with embeddings produced by the same exact provider expression. After changing presets, rebuild or reload speaker references compatible with the new final and live stacks rather than assuming old vectors are interchangeable.

## Selecting a preset from the command line

Use the stable preset ID, not the launcher label:

```powershell
whospeaks config --provider-preset promoted_public
whospeaks launch --print
```

Available IDs are:

```text
smoke
single_espnet
smoke_fast_live
public_quality
promoted_public
```

The launcher exposes these five named presets. Advanced CLI users can set `--embedding-provider` and `--live-speaker-embedding-provider` independently; a combination that does not exactly match a named preset is recorded as `custom`.

## Operational cautions

- Local embeddings require enough memory and all dependencies for every provider in the selected expression.
- A remote embeddings server must advertise and successfully load every selected provider.
- First use may download model files and take substantially longer than later starts.
- Some upstream models may require accepting model terms or providing a Hugging Face token.
- Provider names and weights affect speaker-vector compatibility; they are not cosmetic tuning labels.
- Validate presets on recordings representative of the actual microphones, rooms, languages, overlap, noise, and speaker population before making comparative accuracy claims.

For the underlying live-versus-final design and vector construction, see [Technical description](technical-description.md#live-speaker-versus-final-speaker). For remote provider loading, see [External servers](external-servers.md#provider-readiness-checks).

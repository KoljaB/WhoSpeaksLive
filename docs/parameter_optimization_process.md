# Speaker Diarization Parameter Optimization Process

This process finds better live-compatible speaker diarization parameters by replaying cached live sentence windows against ElevenLabs canonical baselines and keeping only changes that can be reproduced by the real live path.

The only score that counts is the score produced by the same live-compatible pipeline used by:

tools/youtube_window_diarize_gui.py

A score is invalid unless all of the following are true:

1. The exact code path is available to the live GUI or shared production/live diarization path.
2. The system uses no speaker prior knowledge, no answer key, no canonical labels, no video-specific identities.
3. The method can run online/streaming, without future transcript/audio access except what the live system would have at that moment.
4. Cached replay may only be used as a deterministic simulator of the live path.
5. Any postprocessing used for scoring must also be wired into the live path.
6. Do not report any cached-only, experimental-only, oracle, non-live, or not-yet-wired score as “best score” or “achieved score.”
7. If a score does not satisfy all conditions, label it INVALID_FOR_TARGET.

Your task now:

A. Audit the repository and identify every score in docs, logs, and code.
B. Separate them into:
   - VALID_LIVE_COMPATIBLE
   - INVALID_EXPERIMENTAL_REPLAY
   - INVALID_ORACLE_OR_PRIOR
   - UNKNOWN
C. Update docs/best_parameters.md so it contains only the best VALID_LIVE_COMPATIBLE parameter set.
D. Add a section called “Invalid research-only results” and move the 0.835299 result there with a clear warning.
E. Add or update a test/evaluation script that prints whether a score is VALID_FOR_TARGET=true or false.
F. Continue optimization, but only count scores where VALID_FOR_TARGET=true.
G. Do not claim the target is reached until VALID_FOR_TARGET=true and score >= 0.85.

## Goal

The optimizer should answer one question:

> Given the exact sentences and sentence audio clips produced by our live system, which embedding provider stack and clustering parameters assign speakers closest to the ElevenLabs baselines?

We optimize speaker assignment only. Sentence text, sentence boundaries, word timestamps, and sentence audio clips are fixed by the live extraction cache for a run.

## Non-Negotiable Invariants

- Sentence windows must come from the real live system, not from ElevenLabs canonical boundaries.
- The fast optimizer must replay the same online clustering logic as the live system.
- Cached sentence WAVs must match the live embedding input:
  `pad_audio(trim_silence(raw_sentence_audio), min_embed_seconds)`.
- The fast cache must pass the parity gate before parameter optimization is trusted.
- Provider stacks must fit the live VRAM budget. A provider weight does not reduce VRAM because the provider still loads.
- Any final parameter set must be live-compatible and exposed in both the fast path and `youtube_window_diarize_gui.py`.

## Main Inputs

### ElevenLabs Canonical Baselines

Canonical files are used as the target truth for scoring:

```text
output_elevenlabs_<video_id>\<video_id>.canonical_diarization.json
```

They provide canonical speaker segments, text, and timings. They are not used to create live sentence boundaries.

### Live Sentence Feature Cache

The optimizer uses cached outputs created from the live validation replay:

```text
tools\.window_diarize_feature_cache\live_window_corpus_60_90_cuda
```

Each video cache should contain:

```text
manifest.json
sentences.jsonl
audio\sentence_XXXX.wav
embeddings\<provider>.npz
```

Each sentence row should preserve live metadata such as:

```text
index
text
start
end
audio_length_seconds
spoken_word_seconds
speech_audio_ratio
window_left
window_right
next_left
words
first_word_start
last_word_end
next_word_start
gap_to_next_word_seconds
boundary_strategy
sentence_boundary_pre_padding_seconds
sentence_boundary_post_padding_seconds
sentence_boundary_gap_ratio
audio_file
raw_clip_samples
embedding_clip_samples
embedding_audio_seconds
```

### Validation Outputs

The matching validation JSONs live under:

```text
tools\.window_diarize_validation\live_window_corpus_60_90_cuda
```

These store the original live replay `final_payloads`, clustering args, and metrics. They are used for parity checks and for understanding what the real live path produced.

## Tools

### Build Live Sentence Corpus

```text
tools\build_live_window_sentence_corpus.py
```

This runs `youtube_window_diarize_gui.py --validate-window-replay` for each video and then calls `precompute_window_diarize_features.py --no-embeddings`.

Use this when sentence windows need to be generated or regenerated.

### Build Sentence Feature Cache

```text
tools\precompute_window_diarize_features.py
```

This converts live validation `final_payloads` into:

- `sentences.jsonl`
- per-sentence embedding-input WAV files
- optional provider embeddings

For sentence extraction only, run with `--no-embeddings`.

### Precompute Embedding Providers

```text
tools\precompute_window_embedding_providers.py
```

This fills `embeddings\<provider>.npz` for each cached video. Run providers one by one when debugging speed, VRAM, or failures.

### Parity Check

```text
tools\compare_window_cache_to_validation.py
```

This confirms that the fast replay path gives the same speaker assignments as the validation run that created the cache.

The baseline parity command is:

```powershell
.\.venv\Scripts\python.exe tools\compare_window_cache_to_validation.py --dataset-dir tools\.window_diarize_feature_cache\live_window_corpus_60_90_cuda --top 0
```

Expected result for a valid unchanged cache is:

```text
speaker mismatches: 0
text/time mismatches: 0
```

If parity fails, stop optimization and fix the cache or fast replay first.

### Optimizer

```text
tools\optimize_window_diarize_cache.py
```

This loads all cached sentence rows and embedding matrices once, replays many candidate configs through the online clustering logic, and scores the results against ElevenLabs.

## What We Optimize

The optimizer can change only parameters that affect cached replay speaker assignment.

### Provider Stack

- provider names
- provider weights
- stack size
- stack-level VRAM budget

Example:

```text
jungjee_rawnet3=1 + speaker3d_campplus=0.758
```

### Speaker Clustering Parameters

The current search space includes:

```text
same_speaker_similarity
similarity_temperature
speaker_softmax_temperature
new_speaker_threshold
duplicate_profile_similarity
unknown_short_threshold
min_first_speaker_seconds
min_new_speaker_seconds
late_new_speaker_min_seconds
max_speakers
min_margin
margin_temperature
update_unknown_max
new_speaker_confirmation_count
new_speaker_confirmation_similarity
max_pending_new_speakers
min_new_speaker_words
min_speech_audio_ratio
retro_reassign_min_similarity
retro_reassign_min_margin
```

### What We Do Not Optimize In This Stage

These are fixed once the live sentence cache exists:

- ASR model output
- sentence text
- sentence boundaries
- word timestamps
- raw sentence audio windows
- trim and padding behavior used to create embedding audio

Changing those requires regenerating the live sentence cache and all embeddings.

## Scoring

The optimizer uses a robust balanced score. It rewards duration and segment accuracy while penalizing unknown duration and bad speaker counts.

### Per-Video Score

```text
0.70 * duration_accuracy
+ 0.20 * segment_accuracy
- 0.20 * unknown_duration_rate
- 0.08 * abs(predicted_speaker_count - canonical_speaker_count) / canonical_speaker_count
- 0.04 * max(0, predicted_speaker_count - canonical_speaker_count) / canonical_speaker_count
```

Definitions:

- `duration_accuracy`: time-weighted speaker assignment accuracy after mapping predicted speakers to canonical speakers.
- `segment_accuracy`: segment-count accuracy after speaker mapping.
- `unknown_duration_rate`: fraction of spoken duration assigned to unknown.
- `predicted_speaker_count`: number of speakers created by our clustering logic.
- `canonical_speaker_count`: number of speakers in the ElevenLabs baseline.

### Global Score

```text
0.55 * mean_video_score
+ 0.30 * worst_video_score
+ 0.15 * bottom3_mean_video_score
```

This prevents a candidate from looking good only because it improves easy videos while failing hard ones.

## Optimization Loop

### 1. Confirm Data Readiness

For every video:

- Canonical ElevenLabs baseline exists.
- Live validation JSON exists.
- Feature cache exists.
- `sentences.jsonl` has rows.
- Sentence WAVs exist.
- Required provider embedding `.npz` files exist.

Do not optimize on partial data unless the run is explicitly a smoke test.

### 2. Run Parity Gate

Run `compare_window_cache_to_validation.py` before every serious optimization round.

If there are mismatches, the fast replay is not equivalent to the live path. Fix that before testing parameters.

### 3. Establish Baseline Score

Always evaluate the current live default config as row zero.

This gives the reference score and per-video metrics. A candidate is only meaningful relative to this baseline.

### 4. Filter Providers By Live Constraints

Use provider benchmark summaries to exclude providers that are too heavy.

For current live use, keep stack VRAM under the chosen cap, for example:

```text
2048 MiB total provider VRAM
```

Important:

- Sum VRAM for every provider in the stack.
- Weight does not reduce VRAM.
- Providers with unknown VRAM should be treated cautiously or excluded from strict live-compatible runs.

### 5. Broad Search

Run a broad deterministic search first:

- single-provider candidates
- provider stacks of size 1 to 3
- deterministic random configs with a fixed seed
- provider weights from the configured weight grid
- clustering parameters sampled from the search ranges

The point is provider and threshold diversity, not perfect local tuning.

Example command shape:

```powershell
.\.venv\Scripts\python.exe tools\optimize_window_diarize_cache.py --dataset-dir tools\.window_diarize_feature_cache\live_window_corpus_60_90_cuda --budget 2000 --seed 1337 --output tools\.window_diarize_validation\live_window_corpus_60_90_cuda\optimizer_run.json
```

### 6. Successive Halving

Use a smaller first-pass video subset to reject obviously bad configs quickly.

The historical first-pass set is:

```text
JWS-qfR6K3w
ZY0DG8rUnCA
KdOXM3I_5hk
acbnyagl8jo
```

Then rerun the best candidates across the full corpus.

### 7. Local Refinement

Take the top configs and mutate:

- small threshold changes
- small temperature changes
- small provider weight changes
- discrete parameter neighbors

Rerun the best mutations across the full corpus.

Do not refine one provider stack too early. Local mutation can get stuck in a local minimum.

### 8. Provider-Diverse Reranking

After local refinement, inspect whether top results are all from one provider stack.

If so, run a provider-diverse pass:

- keep several strong stacks
- optimize each enough to be comparable
- rerank across all videos

This avoids missing a stack that needs different thresholds.

### 9. Inspect Per-Video Metrics

Do not look only at global score.

Inspect:

- duration accuracy
- segment accuracy
- unknown duration rate
- predicted speaker count
- canonical speaker count
- over-speaker penalties
- per-video regressions

Known warning signs:

- high global score but one video collapses
- predicted speakers far above canonical count
- unknown duration drops only because too many new speakers are created
- Cunk-like multi-speaker dialogue gets worse while talk-show clips improve

### 10. Choose Candidate

A good candidate should:

- beat baseline global robust score
- improve or preserve hard videos where possible
- avoid large unknown duration increases
- avoid runaway speaker creation
- fit provider VRAM budget
- use only parameters implemented in the live path

Strict old guardrails like "never regress Cunk by more than 0.01" are optional. They are useful for conservative checks but should not automatically reject a clearly better global candidate unless that is the current goal.

### 11. Live Verification

After selecting top candidates, verify with the real live replay path:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --validate-window-replay --url <url> --validation-canonical <canonical.json> --validation-output <out.json> ...
```

Run at least:

- Cunk
- Coin Toss
- the worst or most unstable current video

Compare live replay assignments with fast replay assignments. If they diverge, the optimizer is no longer trustworthy until parity is restored.

### 12. Promote Parameters

Only after live verification:

- update defaults in `youtube_window_diarize_gui.py`
- update optimizer `DEFAULT_CONFIG` and `DEFAULT_PROVIDER_WEIGHTS`
- record output JSON path and score in `docs\parameter_optimization_insights.md`
- keep the optimizer result JSON for reproducibility

## Output To Preserve

Every serious optimizer run should produce JSON with:

- baseline config and score
- ranked configs
- provider weights
- clustering parameters
- provider VRAM metadata
- excluded providers and reason
- per-video metrics
- unknown duration rate
- predicted vs canonical speaker count
- score deltas vs baseline
- reproducible command line

## Common Failure Modes

### Fast Replay Does Not Match Live Replay

Cause:

- wrong audio preprocessing
- stale embeddings
- changed live clustering defaults
- regenerated validation but not features

Fix:

- rerun parity
- regenerate sentence cache if sentence windows changed
- regenerate embeddings if sentence audio changed

### Score Improves By Creating Too Many Speakers

Cause:

- new-speaker thresholds too permissive
- confirmation too weak
- duplicate-profile threshold too low

Fix:

- inspect predicted speaker count penalties
- tighten new speaker creation
- increase duplicate-profile protection

### Unknown Rate Is Too High

Cause:

- `min_speech_audio_ratio` too strict
- unknown short thresholds too aggressive
- existing-speaker assignment threshold too strict

Fix:

- tune unknown-related params carefully
- verify short sentence assignments manually on hard clips

### Provider Stack Scores Well But Cannot Run Live

Cause:

- summed provider VRAM exceeds live budget
- provider startup too slow
- provider conflicts with ASR GPU use

Fix:

- exclude by stack VRAM
- rerun optimization with live-compatible provider set

## Current Practical Baseline

As of the current notes, the score-first cached default for the 15-video corpus is:

```text
espnet_ecapa_wavlm_joint=0.74 + jungjee_rawnet3=0.99 + wespeaker_campplus=0.34 + speechbrain_resnet=0.38 + resemblyzer=0.12
```

with the key clustering defaults:

```text
same_speaker_similarity=0.37
new_speaker_threshold=0.38
duplicate_profile_similarity=0.4
unknown_short_threshold=0.3225
min_new_speaker_seconds=1.6
margin_temperature=0.03
similarity_temperature=0.0648
speaker_softmax_temperature=0.0443
update_unknown_max=0.61
new_speaker_confirmation_similarity=0.5149
min_new_speaker_words=3
min_speech_audio_ratio=0.0
```

This is a score-first default, not the best <=2 GB provider-stack candidate. The best live-compatible provider stack found under about 2 GB provider VRAM remains approximately:

```text
jungjee_rawnet3=1 + speaker3d_campplus=0.75
```

Do not assume either is final. Treat the active default as the score baseline to beat with reproducible cache and live verification, and use a VRAM-capped optimizer run when the deployment target requires the <=2 GB provider budget.

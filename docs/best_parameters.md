# Best Parameters

Last updated: 2026-07-03

## Current Best Cached-Replay Score

Best no-canonical experimental replay found during the 2026-07-03 continuation:

```text
global_robust_score:      0.835299
mean_video_score:         0.855105
worst_video_score:        0.809006
bottom3_mean_video_score: 0.815263
```

This result uses the provider stack below plus no-canonical prototype reassignment, turn-taking HMM smoothing, low-cohesion profile absorption, and safe profile-merge postprocessing:

```text
espnet_ecapa_wavlm_joint=0.74
jungjee_rawnet3=0.99
wespeaker_campplus=0.34
speechbrain_resnet=0.35
resemblyzer=0.12

postprocess=prototype_all_rows_local_refine
merge_mode=safe_merge
prototype max_per_profile=32
prototype_min_duration=0.15
prototype_max_unknown=1.0
prototype top_k=12
prototype centroid_blend=0.555
prototype only_low_confidence=false
prototype reassign_unknown=true
prototype max_reassign_duration=8.0
prototype min_unknown=0.79
prototype max_margin=0.324
prototype min_similarity=-0.039
prototype min_delta=0.108

turn_hmm max_per_profile=32
turn_hmm prototype_min_duration=0.15
turn_hmm prototype_max_unknown=1.0
turn_hmm top_k=20
turn_hmm centroid_blend=0.475
turn_hmm emission_weight=1.0
turn_hmm base_switch_penalty=0.025
turn_hmm question_switch_bonus=0.021
turn_hmm backchannel_switch_bonus=0.296
turn_hmm after_backchannel_switch_bonus=0.02
turn_hmm short_turn_switch_bonus=0.0
turn_hmm current_label_bias=0.211
turn_hmm max_score_loss=0.133
turn_hmm max_dialogue_gap=1.4
turn_hmm short_duration=0.45

low_cohesion_absorb max_per_profile=32
low_cohesion_absorb prototype_min_duration=0.15
low_cohesion_absorb top_k=12
low_cohesion_absorb centroid_blend=0.372
low_cohesion_absorb min_profile_seconds=4.0
low_cohesion_absorb max_profile_seconds=45.0
low_cohesion_absorb max_profile_share=0.16
low_cohesion_absorb max_own_mean=0.56
low_cohesion_absorb min_nearest_centroid=0.28
low_cohesion_absorb max_nearest_centroid=0.42
low_cohesion_absorb max_score_loss=0.18
low_cohesion_absorb min_row_similarity=0.1
low_cohesion_absorb small_profile_share_for_loss=0.12
low_cohesion_absorb small_profile_max_score_loss=0.1

recycle_absorbed_profile_islands max_donor_share=0.10
recycle_absorbed_profile_islands max_donor_own_mean=0.56
recycle_absorbed_profile_islands min_donor_nearest=0.28
recycle_absorbed_profile_islands max_neighbor_gap=2.0
recycle_absorbed_profile_islands max_island_duration=0.8
recycle_absorbed_profile_islands max_current_score=0.35
recycle_absorbed_profile_islands max_score_margin=0.12
recycle_absorbed_profile_islands max_recycle_seconds_per_donor=2.0
recycle_absorbed_profile_islands duration_weight=0.05

safe_merge thresholds: 0.53 global, 0.40 for profiles <=12 seconds, only when profile_count <=5
```

Reproducibility output:

```text
runtime\optimization\recycle_absorbed_profile_islands_full.json
```

Additional clustering override versus the parameter-only defaults:

```text
unknown_short_threshold=0.27
adaptive retro_reassign_min_similarity=0.20 only when base_predicted_speaker_count == 6, otherwise 0.05
```

Important caveat: this is still an experimental cached replay result. It does not use canonical speaker identity or video-specific rules, but the prototype, turn-HMM, and low-cohesion absorption stages were tested as postprocesses and still need a live-compatible online implementation before promotion as the product default.

## Current Best Live-Compatible Parameter-Only Cached-Replay Score

```text
global_robust_score:      0.813546
mean_video_score:         0.835910
worst_video_score:        0.784538
bottom3_mean_video_score: 0.789563
total_unknown_rate:       0.006729
```

## Live Prototype Refinement Scores

These scores use the current live default provider stack (`speechbrain_resnet=0.38`) and the new live-compatible prototype refinement path. The checkbox-off mode keeps already assigned speaker labels stable and only fills rows that were UNKNOWN first. The checkbox-on mode also allows later prototype evidence to revise already assigned transcript rows.

```text
no prototype refinement:        0.813546
checkbox off, stable labels:    0.817220
checkbox on, allow reassignment: 0.817988
```

Reproducibility output:

```text
runtime\optimization\live_refinement_modes_speechbrain038.json
```

## Provider Stack

```text
espnet_ecapa_wavlm_joint=0.74
jungjee_rawnet3=0.99
wespeaker_campplus=0.34
speechbrain_resnet=0.38
resemblyzer=0.12
```

As a single CLI value:

```text
espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12
```

Measured provider VRAM sum is about `4660 MiB`, based on the benchmark summary values in `D:\Projekte\WhoSpeaks\benchmarks\voice_embeddings\summary_all.json`.

## Clustering Parameters

```text
same_speaker_similarity=0.37
similarity_temperature=0.0648
speaker_softmax_temperature=0.0443
new_speaker_threshold=0.38
duplicate_profile_similarity=0.4
unknown_short_threshold=0.3225
min_first_speaker_seconds=1.3098
min_new_speaker_seconds=1.6
late_new_speaker_min_seconds=3.4127
max_speakers=12
min_margin=0.0386
margin_temperature=0.03
update_unknown_max=0.61
new_speaker_confirmation_count=1
new_speaker_confirmation_similarity=0.5149
max_pending_new_speakers=6
min_new_speaker_words=3
min_speech_audio_ratio=0.0
retro_reassign_min_similarity=0.05
retro_reassign_min_margin=0.0
```

## Reproducibility

Best-result source:

```text
runtime\optimization\provider_weight_axis_5gb_pass3_seed_resemblyzer012.json
```

Plateau check:

```text
runtime\optimization\provider_weight_axis_5gb_pass4_seed_speechbrain038.json
```

The current best uses no canonical speaker identity during assignment. The score is from cached live sentence-window replay against canonical baselines. Fresh live `--validate-window-replay` verification is still required before treating it as fully live-verified.

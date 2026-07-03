# Best Parameters

Last updated: 2026-07-03

## Current Best Cached-Replay Score

```text
global_robust_score:      0.813546
mean_video_score:         0.835910
worst_video_score:        0.784538
bottom3_mean_video_score: 0.789563
total_unknown_rate:       0.006729
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

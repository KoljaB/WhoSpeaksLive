# ElevenLabs Scribe Regression Dataset

This is the canonical local regression corpus for ElevenLabs references and live-compatible cached replay.

## Quick Facts

| Item | Value |
| --- | ---: |
| Videos | 28 |
| Baseline videos with canonical diarization | 28 |
| Baseline videos with all provider embeddings | 27 |
| Baseline segments | 2610 |
| Live videos with sentence/audio caches | 28 |
| Live videos with all provider embeddings | 27 |
| Live videos still missing provider embeddings | 1 |
| Live sentences | 3947 |
| Embedding providers | 15 |
| Default optimizer evaluation videos | 22 |
| Evaluation blacklist entries | 6 |

## Layout

```text
elevenlabs_scribe_27/
  README.md
  manifest.json
  aliases.json
  blacklist.json
  coverage.csv
  source_lists/
  videos/
    <canonical_youtube_id>/
      metadata.json
      baseline/
        canonical_diarization.json
        best_diarization.{md,json,csv}
        source_files/
      baseline_embeddings/
        <provider>.embeddings.npz
        <provider>.segments.json
      live_window/
        manifest.json
        sentences.jsonl
        audio/
        embeddings/
```

Every folder under `videos/` is named by canonical YouTube video ID. Source aliases are recorded in `aliases.json` and each video's `metadata.json`.

## Alias Policy

`cunk` is not a canonical video folder in this dataset. The Cunk baseline source key maps to canonical YouTube ID `JWS-qfR6K3w`, so the normalized folder is `videos/JWS-qfR6K3w`.

## Current Live Embedding Gaps

`6BuK09sWn9s` intentionally carries the three providers from the current live stack (`espnet_ecapa_wavlm_joint`, `speechbrain_resnet`, and `wespeaker_campplus`); the other historical provider caches are not required for its regression score.

## Evaluation Blacklist

`blacklist.json` lists videos that stay in the dataset but are excluded from optimizer scoring/search by default because their canonical baseline is known bad. Use `--include-blacklisted` with `runtime\optimization\optimize_canonical_27.py` when intentionally inspecting those videos.

## Complete Live Embedding Videos

27 videos currently have complete 15-provider live sentence embeddings.

## Source Roots

- `baseline_root`: `D:\Projekte\SpeakerDiarization\data\baselines\elevenlabs_scribe`
- `baseline_embeddings_root`: `D:\Projekte\SpeakerDiarization\data\baselines\elevenlabs_scribe\_sentence_embeddings`
- `live_complete_root`: `D:\Projekte\SpeakerDiarization\data\live_sentence_boundaries\live_window_corpus_60_90_cuda_complete_23`
- `live_missing_1x_root`: `D:\Projekte\SpeakerDiarization\data\live_sentence_boundaries\live_window_corpus_60_90_cuda_missing_1x`
- `live_more_4_root`: `D:\Projekte\SpeakerDiarization\data\live_sentence_boundaries\live_window_corpus_60_90_cuda_more_4`
- `whospeaks_more_root`: `D:\Projekte\WhoSpeaks\tools\.window_diarize_feature_cache\live_window_corpus_60_90_cuda_more`

## Rebuild

Run the first command from the repo root to regenerate the original normalized corpus. The second command restores the German saved-session regression when its local source session is available:

```powershell
python data\datasets\_build_elevenlabs_scribe_27.py
python data\datasets\_add_session_regression_video.py --video-id 6BuK09sWn9s --source-url https://www.youtube.com/watch?v=6BuK09sWn9s --session-dir runtime\sessions\f22cf4eb4580463b99fdffb11bf58262 --reference-md runtime\outputs\elevenlabs_scribe\6BuK09sWn9s\6BuK09sWn9s.elevenlabs_best_diarization.md
```

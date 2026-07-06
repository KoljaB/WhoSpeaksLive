# ElevenLabs Scribe 27-Video Dataset

This is the canonical local dataset root for the 27-video ElevenLabs Scribe baseline corpus.

## Quick Facts

| Item | Value |
| --- | ---: |
| Videos | 27 |
| Baseline videos with canonical diarization | 27 |
| Baseline videos with all provider embeddings | 27 |
| Baseline segments | 2586 |
| Live videos with sentence/audio caches | 27 |
| Live videos with all provider embeddings | 27 |
| Live videos still missing provider embeddings | 0 |
| Live sentences | 3889 |
| Embedding providers | 15 |
| Default optimizer evaluation videos | 21 |
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

No live embedding gaps remain.

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

Run this from the repo root to regenerate the normalized corpus from the current source folders:

```powershell
python data\datasets\_build_elevenlabs_scribe_27.py
```

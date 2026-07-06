# ElevenLabs Scribe Baseline Dataset

This dataset is the local reference set for scoring live speaker diarization against ElevenLabs Scribe baselines.

The canonical consolidated root is `data/datasets/elevenlabs_scribe_27`. Use that root first for optimization, scoring, coverage checks, and future dataset maintenance.

## Quick Facts

| Item | Value |
| --- | ---: |
| Baseline videos | 27 |
| ElevenLabs canonical diarization segments | 2,586 |
| Embedding providers | 15 |
| Baseline video/provider embedding pairs | 405 / 405 |
| Baseline segment embedding rows | 38,790 / 38,790 |
| Baseline embedding validation issues | 0 |
| Live-window sentence/audio cache videos | 27 |
| Baseline videos covered by live-window sentence/audio cache | 27 / 27 |
| Live-window sentences | 3,889 |
| Live-window videos with all provider embeddings | 27 / 27 |
| Live-window video/provider embedding pairs | 405 / 405 |
| Live-window sentence embedding rows | 58,335 / 58,335 |

## Canonical Roots

Use this path as the stable entry point:

- Canonical 27-video dataset:
  `data/datasets/elevenlabs_scribe_27`

The older roots below are source/provenance inputs for rebuilding the canonical dataset:

- Baseline transcripts and canonical ElevenLabs diarization:
  `data/baselines/elevenlabs_scribe`
- Baseline segment embeddings:
  `data/baselines/elevenlabs_scribe/_sentence_embeddings`
- Live-system sentence boundaries and live sentence embeddings:
  `data/live_sentence_boundaries/live_window_corpus_60_90_cuda_complete_23`

The baseline root, baseline embedding root, historical live-window root, and `D:\Projekte\WhoSpeaks` cache roots are intentionally treated as inputs. The normalized dataset resolves naming aliases, groups all files by canonical YouTube ID, and records remaining gaps in one manifest.

## Filesystem Layout

```text
data/
  datasets/
    _build_elevenlabs_scribe_27.py
    elevenlabs_scribe_27/
      README.md
      manifest.json
      aliases.json
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
              sentence_0000.wav
              sentence_0001.wav
              ...
            embeddings/
              <provider>.npz
```

## Baseline Transcript Files

Each normalized video folder under `data/datasets/elevenlabs_scribe_27/videos/<canonical_youtube_id>/baseline` contains the best available ElevenLabs Scribe baseline exports.

The most important file is `baseline/canonical_diarization.json`. It is the canonical reference used for scoring. It contains:

- media metadata, including `source_url`, `audio_file`, and `duration_sec`
- ElevenLabs provenance, including provider, model, transcription ID, and diarization threshold when available
- speaker definitions with stable `SPEAKER_N` labels
- word-level timing
- canonical diarization `segments`

Each canonical segment includes:

- `segment_id`
- `speaker_id`
- `start_sec`
- `end_sec`
- `duration_sec`
- `text`
- `word_start_index`
- `word_end_index`

The normalized `best_diarization.*` files are human-readable and tabular exports of the same baseline reference. The original source files, including raw ElevenLabs Scribe responses, are preserved under `baseline/source_files`.

## Baseline Segment Embeddings

Baseline embeddings live under each normalized video folder at `baseline_embeddings`.

Each video has one pair of files per provider:

- `<provider>.embeddings.npz`: compressed NumPy arrays.
- `<provider>.segments.json`: row metadata for the embedding file.

The row order in `<provider>.embeddings.npz` matches the `segments` list in `<provider>.segments.json`. The NPZ files include:

- `embeddings`: one vector per canonical ElevenLabs segment
- `start_sec`
- `end_sec`
- `audio_start_sec`
- `audio_end_sec`

`audio_start_sec` and `audio_end_sec` are the actual audio slice boundaries used for embedding. They usually match the canonical segment boundaries, but can be expanded for very short segments or chunked internally for long segments.

## Embedding Providers

All 15 providers are complete for all 27 videos.

| Provider | Dim | Complete Videos | Rows |
| --- | ---: | ---: | ---: |
| `espnet_ecapa_wavlm_joint` | 192 | 27 | 2,586 |
| `espnet_rawnet3` | 192 | 27 | 2,586 |
| `jungjee_rawnet3` | 256 | 27 | 2,586 |
| `nemo_titanet_large` | 192 | 27 | 2,586 |
| `pyannote_embedding` | 512 | 27 | 2,586 |
| `pyannote_wespeaker_resnet34_lm` | 256 | 27 | 2,586 |
| `resemblyzer` | 256 | 27 | 2,586 |
| `speaker3d_campplus` | 192 | 27 | 2,586 |
| `speaker3d_eres2netv2` | 192 | 27 | 2,586 |
| `speechbrain_ecapa` | 192 | 27 | 2,586 |
| `speechbrain_resnet` | 256 | 27 | 2,586 |
| `speechbrain_xvector` | 512 | 27 | 2,586 |
| `wavlm_base_sv` | 512 | 27 | 2,586 |
| `wespeaker_campplus` | 192 | 27 | 2,586 |
| `wespeaker_resnet34_lm_onnx` | 256 | 27 | 2,586 |

Long ElevenLabs segments can be too large for some embedding models as one forward pass. The corpus builder supports `--max-embed-chunk-seconds` so long audio is embedded in shorter windows and averaged back into one normalized vector per original baseline segment. This preserves the one-row-per-baseline-segment contract.

## Live-System Sentence Boundaries

The normalized live-window corpus lives under each video folder at `live_window`.

This corpus contains sentence splits produced by the live system, not by ElevenLabs. Each video cache contains:

- `sentences.jsonl`: one JSON object per detected live sentence.
- `audio/sentence_XXXX.wav`: the exact audio clip for that live sentence.
- `embeddings/<provider>.npz`: one embedding row per live sentence for each provider.
- `manifest.json`: per-video validation and provenance.

Each `sentences.jsonl` row contains fields such as:

- `index`
- `text`
- `start`
- `end`
- `words`
- `window_left`
- `window_right`
- `boundary_strategy`
- `audio_file`
- `embedding_audio_seconds`

The normalized dataset now contains live sentence/audio caches for all 27 baseline videos. One video uses an alias: the Cunk baseline source folder is named `cunk`, while its canonical YouTube ID and normalized folder are `JWS-qfR6K3w`.

The durable programmatic map is `data/datasets/elevenlabs_scribe_27/manifest.json`, with the alias repeated in `aliases.json`. Use those files when joining baseline folders to live-window caches so `cunk` is not incorrectly flagged missing.

No live-window provider embedding gaps remain. The four videos that originally lived only in `D:\Projekte\WhoSpeaks\tools\.window_diarize_feature_cache\live_window_corpus_60_90_cuda_more` are now staged locally in `data/live_sentence_boundaries/live_window_corpus_60_90_cuda_more_4`, with all 15 provider embeddings generated through the remote Linux embeddings server.

## Video Inventory

| Canonical Video ID | Baseline Key | Origin | Baseline Segments | Live Sentences | Live Embedding Providers | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `1NBVQB-Srpw` | `1NBVQB-Srpw` | imported from WhoSpeaks | 85 | 105 | 15 |  |
| `20v1OxUXcQY` | `20v1OxUXcQY` | imported from WhoSpeaks | 98 | 138 | 15 |  |
| `acbnyagl8jo` | `acbnyagl8jo` | imported from WhoSpeaks | 94 | 115 | 15 |  |
| `aHGd6LqAVzw` | `aHGd6LqAVzw` | imported from WhoSpeaks | 57 | 92 | 15 |  |
| `blcKeLDDzSM` | `blcKeLDDzSM` | imported from WhoSpeaks | 61 | 116 | 15 |  |
| `bPpcfH_HHH8` | `bPpcfH_HHH8` | imported from WhoSpeaks | 101 | 102 | 15 | live sentence/audio copied from WhoSpeaks; live embeddings generated locally in `live_window_corpus_60_90_cuda_more_4` |
| `cLdy4P-XJPE` | `cLdy4P-XJPE` | generated for requested video | 52 | 134 | 15 | Brooks and Capehart on whether the Supreme Court has stood up to Trump |
| `Dd7FixvoKBw` | `Dd7FixvoKBw` | imported from WhoSpeaks | 57 | 66 | 15 |  |
| `DsyfYJ5Ou3g` | `DsyfYJ5Ou3g` | imported from WhoSpeaks | 70 | 90 | 15 |  |
| `e3h6es6zh1c` | `e3h6es6zh1c` | imported from WhoSpeaks | 50 | 85 | 15 | live sentence/audio copied from WhoSpeaks; live embeddings generated locally in `live_window_corpus_60_90_cuda_more_4` |
| `F2-2RBi1qzY` | `F2-2RBi1qzY` | imported from WhoSpeaks | 100 | 71 | 15 |  |
| `gj7BRMuB-n4` | `gj7BRMuB-n4` | imported from WhoSpeaks | 155 | 209 | 15 |  |
| `JWS-qfR6K3w` | `cunk` | imported from WhoSpeaks | 45 | 65 | 15 | Cunk baseline alias; source file prefix is `cunk_on_earth_clip` |
| `k1tsGGz-Qw0` | `k1tsGGz-Qw0` | imported from WhoSpeaks | 83 | 127 | 15 | live sentence/audio copied from WhoSpeaks; live embeddings generated locally in `live_window_corpus_60_90_cuda_more_4` |
| `KdOXM3I_5hk` | `KdOXM3I_5hk` | imported from WhoSpeaks | 245 | 191 | 15 |  |
| `L-CfFo5aQGU` | `L-CfFo5aQGU` | generated for requested video | 132 | 267 | 15 | Face The Nation: Kaine, Mayor Panel |
| `mBeT_AoCXvc` | `mBeT_AoCXvc` | generated for requested video | 96 | 149 | 15 | America's mayors panel |
| `mWABb5Dy9BQ` | `mWABb5Dy9BQ` | imported from WhoSpeaks | 219 | 220 | 15 |  |
| `oFBuCp19L7M` | `oFBuCp19L7M` | imported from WhoSpeaks | 95 | 114 | 15 |  |
| `onHUfyRP1BE` | `onHUfyRP1BE` | generated for requested video | 9 | 64 | 15 | Academic Panel COI / Kyle Diorio pt. 2/3 |
| `pD4IdQTmneI` | `pD4IdQTmneI` | generated for requested video | 15 | 64 | 15 | political analyst clip |
| `PhofRoLXqhE` | `PhofRoLXqhE` | generated for requested video | 100 | 286 | 15 | Washington Week with The Atlantic, 2026-07-03 |
| `S_o3y7CzDUY` | `S_o3y7CzDUY` | generated for requested video | 76 | 274 | 15 | Why is Finland's economy stuck in a rut? |
| `vIfGgDnmBXg` | `vIfGgDnmBXg` | imported from WhoSpeaks | 145 | 155 | 15 | live sentence/audio copied from WhoSpeaks; live embeddings generated locally in `live_window_corpus_60_90_cuda_more_4` |
| `WNZn37Uc700` | `WNZn37Uc700` | imported from WhoSpeaks | 174 | 178 | 15 |  |
| `y_5WfLjvOK4` | `y_5WfLjvOK4` | generated for requested video | 91 | 327 | 15 | Face the Nation foreign policy panel |
| `ZY0DG8rUnCA` | `ZY0DG8rUnCA` | imported from WhoSpeaks | 81 | 85 | 15 |  |

## Manifests To Trust First

Use these manifests for programmatic discovery:

- `data/datasets/elevenlabs_scribe_27/manifest.json`
- `data/datasets/elevenlabs_scribe_27/coverage.csv`
- `data/datasets/elevenlabs_scribe_27/aliases.json`
- `data/datasets/elevenlabs_scribe_27/videos/<canonical_youtube_id>/metadata.json`
- `data/baselines/elevenlabs_scribe/manifest.json`
- `data/baselines/elevenlabs_scribe/_sentence_embeddings/manifest.json`
- `data/live_sentence_boundaries/live_window_corpus_60_90_cuda_complete_23/manifest.json`
- `data/live_sentence_boundaries/live_window_corpus_60_90_cuda_complete_23/baseline_coverage.json`

The canonical dataset manifest validates:

- 27 canonical YouTube video folders
- 27 baseline videos
- 27 baseline videos with all 15 provider embeddings
- 27 live sentence/audio caches
- 27 live caches with all 15 provider embeddings
- 0 live caches with missing provider embeddings
- 3,889 total live sentences

The aggregate embedding manifest validates:

- 27 videos
- 15 providers
- 405 complete video/provider pairs
- 38,790 total embedding rows
- 0 validation issues

The live-window manifest validates:

- 23 live-window videos
- 15 providers
- 345 complete video/provider pairs
- 51,300 total embedding rows
- 0 validation issues

The `live_window_corpus_60_90_cuda_more_4` embedding manifest validates:

- 4 live-window videos
- 15 providers
- 60 complete video/provider pairs
- 7,035 total embedding rows
- 0 failed providers

## Operational Notes

- Use `data/datasets/elevenlabs_scribe_27` as the default dataset root for optimization and scoring work.
- Use `runtime/optimization/optimize_canonical_27.py` for cached optimization against this canonical layout. It wraps the historical WhoSpeaks optimizer and loads `videos/<canonical_youtube_id>/live_window` directly.
- Use `runtime/optimization/canonical_27_optimization_ready.md` for ready commands and verified 27-video baseline scores.
- Treat `data/baselines/elevenlabs_scribe`, `data/baselines/elevenlabs_scribe/_sentence_embeddings`, `data/live_sentence_boundaries/*`, and `D:\Projekte\WhoSpeaks\tools\.window_diarize_feature_cache\*` as source roots for rebuilds and provenance.
- Do not create a `videos/cunk` folder. The canonical dataset folder is `videos/JWS-qfR6K3w`, with `cunk` recorded as the source baseline key.
- Do not mix the baseline segment embeddings with the live-window sentence embeddings. They have different segmentation and different row counts.
- For scoring against ElevenLabs, use each video's `baseline/canonical_diarization.json` and `baseline_embeddings`.
- For optimizing live-system sentence boundaries and live speaker assignment, use each video's `live_window/sentences.jsonl`, `live_window/audio`, and `live_window/embeddings`.

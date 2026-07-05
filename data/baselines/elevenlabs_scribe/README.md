# ElevenLabs Scribe Baselines

This folder contains ElevenLabs Scribe diarization baselines imported from `D:\Projekte\WhoSpeaks` plus locally generated requested-video baselines.

Each video has its own subfolder named after the YouTube video ID or source key. The source folders were named `output_elevenlabs_*` in `WhoSpeaks`; the prefix was removed here so consumers can address baselines by key.

## Contents

- `*.audio.elevenlabs_scribe_v2.word*.md`: human-readable Scribe transcript.
- `*.audio.elevenlabs_scribe_v2.word*.raw.json`: raw ElevenLabs Scribe response.
- `*.audio.elevenlabs_scribe_v2.word*.turns.json` and `*.turns.csv`: turn-level transcript exports.
- `*.canonical_diarization.json`: canonical diarization reference for validation.
- `*.elevenlabs_best_diarization.*`: best available baseline diarization export in Markdown, JSON, and CSV.
- `*.youtube_title.txt`: fetched YouTube title for locally generated requested-video baselines.
- `_source_lists/`: copied YouTube test-list files from `WhoSpeaks/docs` plus local requested-video URL lists.
- `manifest.json`: generated manifest with origins, source directories or URLs, file counts, SHA-256 hashes, and missing IDs from source lists.

## Import Notes

Imported baseline folders from `WhoSpeaks`: 19.

Generated requested-video baseline folders: 8.

Total baseline folders: 27.

Total baseline files: 241.

The requested-video baselines generated on 2026-07-05 are:

- `onHUfyRP1BE`
- `pD4IdQTmneI`
- `cLdy4P-XJPE`
- `y_5WfLjvOK4`
- `mBeT_AoCXvc`
- `L-CfFo5aQGU`
- `PhofRoLXqhE`
- `S_o3y7CzDUY`

`youtube_videos_to_test.txt` references `JWS-qfR6K3w`, but `WhoSpeaks` did not contain an `output_elevenlabs_JWS-qfR6K3w` folder. The available Cunk baseline was imported as `cunk`.

Several entries from `20_more_youtube_videos_to_test.txt` also had no matching `output_elevenlabs_*` folder in `WhoSpeaks`; see `manifest.json` for the exact missing IDs.

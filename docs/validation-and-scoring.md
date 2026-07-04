# Validation And Scoring

Validation turns diarization changes into comparable numbers instead of relying only on visual inspection.

## What Validation Measures

Validation can compare generated output against a canonical transcript. A canonical transcript is a reference file with expected speakers, sentence text, and timing.

The project includes a small deterministic Cunk fixture under `tests/fixtures/`.

Common signals:

- Final sentence speaker accuracy.
- Browser-observed live speaker coverage.
- Wrong live speaker time.
- Missing live speaker time.
- Live speaker latency after turn changes.
- Flicker inside a speaker turn.

## Window Replay Validation

Use window replay validation when you want to test a configuration end to end:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --validate-window-replay
```

Add backend and provider flags when validating the same remote setup used in production:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --validate-window-replay --asr-backend remote --remote-asr-url http://192.168.178.22:8650 --embeddings-backend remote --remote-embeddings-url http://192.168.178.22:8660 --embedding-provider "espnet_ecapa_wavlm_joint=0.74+jungjee_rawnet3=0.99+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12" --live-speaker-embedding-provider "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50"
```

## Browser Live Observation

Browser live observation samples the rendered DOM state instead of only reading backend events. This checks what the user actually sees.

Use:

```powershell
--browser-live-observation-output runtime/outputs/window-diarize-validation/browser-live.json
```

Important knobs:

- `--browser-live-observation-interval-seconds`
- `--browser-live-observation-max-sample-gap-seconds`
- `--browser-live-observation-flicker-gap-seconds`

## Interpreting Scores

Treat a score as meaningful only when:

- The same canonical file is used.
- The same media and replay timing are used.
- New-speaker creation sentences are handled consistently.
- Provider strings are recorded.
- The run is reproducible without cached per-sentence decisions.

Do not compare a live-speaker score against a final-sentence score as if they measure the same thing. Live scoring rewards fast visible assignment. Final scoring rewards stable completed sentence labels.

## Regression Tests

Run the core regression suite before committing behavior changes:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core_regressions
```

The suite includes checks for public contracts such as defaults, speaker memory behavior, browser UI safety, imported speaker groups, live profile updates, and scoring semantics.

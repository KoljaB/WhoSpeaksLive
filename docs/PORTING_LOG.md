# youtube_window_diarize_gui.py Porting Log

## Scope

Port `D:\Projekte\WhoSpeaks\tools\youtube_window_diarize_gui.py` into this workspace and make it runnable with the same functional surface where practical:

- browser-synced growing-window diarization GUI
- local media download/cache handling through `yt-dlp`
- local and remote ASR modes
- realtime preview worker support
- speaker library/reference audio support
- replay validation default reference
- default stacked speaker-embedding provider

## Source Inspection

- Source project: `D:\Projekte\WhoSpeaks`
- Target workspace: `D:\Projekte\SpeakerDiarization`
- Main GUI file: `tools\youtube_window_diarize_gui.py`
- Direct local imports:
  - `tools\realtime_speakerdiarize.py`
  - `tools\speaker_embedding_cluster.py`
  - `tools\textcolors\speaker_color_allocation.py`
  - `tools\youtube_local_filefeed_replay.py`
- Subprocess/helper dependency found during inspection:
  - `tools\kroko_realtime_preview_worker.py`
- Embedding-provider dependency:
  - `tools\benchmark_voice_embeddings.py`

## Problems And Fixes

### `rg` unusable in this PowerShell session

Problem: `rg --files D:\Projekte\WhoSpeaks` failed with a Windows application-association error.

Fix: Switched to Git-tracked file listing and targeted PowerShell searches so generated folders such as `.venv`, `.cache`, and output directories did not dominate discovery.

### Source project contains large generated/runtime folders

Problem: A raw recursive listing entered `.venv` and model-cache directories and timed out after producing thousands of irrelevant files.

Fix: Treated generated folders as non-source by default and traced runtime dependencies from imports, subprocess calls, constants, and default paths.

### Two-environment runtime design

Problem: The GUI starts from the main environment, but `default_embedding_python()` intentionally prefers `ROOT\.venv-voice-embeddings\Scripts\python.exe` for the default stacked embedding provider.

Fix: Recreate the same two-venv layout in the target workspace: `.venv` for the GUI/ASR path and `.venv-voice-embeddings` for embedding providers.

## Actions

- Inspected source layout and target workspace.
- Identified the direct local runtime dependency set.
- Copied the GUI and direct helper modules:
  - `tools\youtube_window_diarize_gui.py`
  - `tools\realtime_speakerdiarize.py`
  - `tools\speaker_embedding_cluster.py`
  - `tools\youtube_local_filefeed_replay.py`
  - `tools\kroko_realtime_preview_worker.py`
  - `tools\benchmark_voice_embeddings.py`
  - `tools\textcolors\speaker_color_allocation.py`
- Copied supporting project files: `.gitignore`, `.env`, `README.md`, and `requirements.txt`.
- Vendored local editable package sources that the source venv referenced by absolute path:
  - `RealtimeSTT`
  - `RealtimeSTT_server`
  - `stream2sentence`
- Copied runtime data needed by the default flow:
  - default cached YouTube media for `JWS-qfR6K3w`
  - uploaded speaker reference WAVs
  - `output_elevenlabs_cunk\cunk_on_earth_clip.canonical_diarization.json`
  - ESPnet model-zoo cache for `espnet_ecapa_wavlm_joint`
  - WeSpeaker CAM++ cache
  - RawNet3 source and weights
  - faster-whisper large-v2 cache
  - default Kroko Community 64-L streaming model
  - WavLM large checkpoint used by ESPnet/S3PRL
  - NLTK tokenizer/tagger/cmudict data
- Created local venvs:
  - `.venv` with the copied GUI/ASR package set
  - `.venv-voice-embeddings` with the copied embedding package set
  - `.venvs\kroko-install-test` with the copied Python 3.12 Kroko preview package set
- Patched local defaults so the copied GUI prefers this workspace:
  - `REALTIME_PREVIEW` defaults now point at `.venvs\kroko-install-test`.
  - Kroko model default now points at `test-model-cache\kroko-onnx`.
  - faster-whisper download root now defaults to `.cache\faster-whisper`.
  - `NLTK_DATA` now defaults to `.cache\nltk`.
  - Silero VAD model lookup now checks local copied venvs.
  - embedding subprocesses now start with `cwd=ROOT` so ESPnet's `./hub` model config resolves correctly.
  - S3PRL download directory is initialized to `.cache\s3prl\download`.
- Added `test-model-cache/` to `.gitignore`.

### `Copy-Item -LiteralPath ...\*` did not copy vendored package contents

Problem: The first vendoring attempt left `RealtimeSTT`, `RealtimeSTT_server`, and `stream2sentence` effectively empty because `-LiteralPath` does not expand the trailing wildcard.

Fix: Re-copied those package directories with `robocopy`, excluding only bytecode caches.

### `stream2sentence` tried to download NLTK data

Problem: `youtube_window_diarize_gui.py --help` imported successfully but printed `Could not download nltk punkt_tab data` because the sandboxed environment has no network path.

Fix: Copied the existing local NLTK data from `C:\Users\Start\AppData\Roaming\nltk_data` into `.cache\nltk` with an approved elevated `robocopy`, then set `NLTK_DATA` before importing `stream2sentence`.

### Kroko preview package was Python-version specific

Problem: The source GUI venv is Python 3.11, but the available `kroko_onnx` native extension is compiled for Python 3.12.

Fix: Created `.venvs\kroko-install-test` with Python 3.12 and mirrored the source Kroko preview package set there. The GUI now defaults to that local Python for preview.

### ESPnet default embedding tried to download WavLM

Problem: The default embedding stack loaded ESPnet, but ESPnet's S3PRL frontend tried to fetch `wavlm_large.pt` from Hugging Face.

Fix: Copied the source cached WavLM checkpoint into `hub\` because the ESPnet model config uses `download_dir: ./hub`. Also set the embedding helper working directory to the project root and initialized S3PRL's cache directory for local fallback.

## Verification

- `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help`
  - Passed.
- Local package resolution check for `stream2sentence`, `RealtimeSTT`, `faster_whisper`, and `numpy`
  - Passed; imports resolve inside `D:\Projekte\SpeakerDiarization`.
- Kroko preview worker startup:
  - Passed; emitted `{"ready":true}` using the copied model and local Python 3.12 preview venv.
- Silero VAD startup:
  - Passed with `raw_onnx_ifless` and the copied Silero ONNX model.
- Server startup smoke:
  - Passed; reached `http://127.0.0.1:8796/` with cached media and no browser. The command was intentionally stopped by timeout because the server runs forever.
- Default stacked embedding helper:
  - Passed on a copied speaker reference WAV with `espnet_ecapa_wavlm_joint=0.725+jungjee_rawnet3=1+wespeaker_campplus=0.35`.
- faster-whisper large-v2 cache load:
  - Passed on CPU/int8 from `.cache\faster-whisper`.
- `py_compile` over copied Python files:
  - Passed.

## Run Command

From `D:\Projekte\SpeakerDiarization`:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py
```

For a no-browser startup check:

```powershell
.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --no-browser --skip-download --no-startup-warmup-before-url
```

## Stepwise Refactor Log - 2026-07-01

The goal of this pass was to extract clear single-responsibility units from the two large scripts while keeping their existing import surface working.

### Extracted units

- `tools\audio_utils.py`
  - Moved shared sample-rate constants, JSON dumping, audio coercion, silence trimming, padding, WAV IO, file loading, vector normalization, cosine similarity, sigmoid, clamp, and softmax.
  - Kept compatibility by importing these names back into `tools\realtime_speakerdiarize.py`.
- `tools\embedding_providers.py`
  - Moved embedding cache setup, provider name normalization, provider factories, concrete provider adapters, stacked providers, `EmbeddingSubprocessClient`, and `run_embedding_helper`.
  - Fixed the helper-client default script path so subprocesses still execute `tools\realtime_speakerdiarize.py --embedding-helper` instead of the new module.
- `tools\realtime_speaker_memory.py`
  - Moved `SpeakerProfile`, `SpeakerDecision`, and `SpeakerMemory`.
  - Kept realtime reassignment/context helpers in the original script because they depend on realtime records and CLI args.
- `tools\window_domain.py`
  - Moved GUI data classes and sentence-boundary defaults: `MediaFiles`, `TimedWord`, `MappedWord`, `SentencePart`, `WindowTranscript`, `VadWindowState`, `PendingUnknownSentence`, and `EmbeddingSentenceJob`.
- `tools\window_media.py`
  - Moved media cache probing, media resolution/download fallback, per-URL media resolution, and browser-stream ID resolution.
- `tools\window_remote_asr.py`
  - Moved the remote ASR HTTP client and its response-to-`TimedWord` parser.

### Tests after each extraction

- After `audio_utils.py`:
  - `.\.venv\Scripts\python.exe -m py_compile tools\audio_utils.py tools\realtime_speakerdiarize.py tools\youtube_window_diarize_gui.py`
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help`
  - Import smoke for `SAMPLE_RATE`, `load_audio_file`, `trim_silence`, and `audio_to_float_mono`.
- After `embedding_providers.py`:
  - `py_compile` over `audio_utils.py`, `embedding_providers.py`, `realtime_speakerdiarize.py`, and `youtube_window_diarize_gui.py`.
  - GUI `--help`.
  - Import smoke for `EmbeddingSubprocessClient`, `default_embedding_python`, `run_embedding_helper`, and `create_embedding_provider`.
  - Mocked JSON-line helper protocol smoke for `run_embedding_helper`.
- After `realtime_speaker_memory.py`:
  - `py_compile` over all extracted modules plus both large scripts.
  - GUI `--help`.
  - Direct speaker-memory behavior smoke: S1 creation, S1 reassignment for a similar vector, and S2 creation for a distinct vector.
- After `window_domain.py`:
  - `py_compile` over all extracted modules plus both large scripts.
  - GUI `--help`.
  - Direct dataclass/default import smoke.
- After `window_media.py`:
  - `py_compile` over all extracted modules plus both large scripts.
  - GUI `--help`.
  - Direct media smoke with explicit local WAV files, no network or download.
- After `window_remote_asr.py`:
  - `py_compile` over all extracted modules plus both large scripts.
  - GUI `--help`.
  - Direct remote-ASR response parsing smoke with synthetic JSON.

### Problems found and fixes

- Problem: A compatibility import smoke was first run from `tools\` with the root-relative `.venv` path.
  - Fix: Reran from `tools\` with `..\.venv\Scripts\python.exe`.
- Problem: The first embedding helper live smoke used `tools\.window_diarize\refactor_embedding_smoke.wav`, which was not writable.
  - Fix: Reran with a root-level temporary WAV in the writable workspace.
- Problem: The live `speechbrain_ecapa` helper smoke reached the helper process but failed because the offline Hugging Face cache does not contain the requested SpeechBrain files.
  - Fix at this stage: Treated this as an environment/cache limitation for this refactor pass and used a mocked provider protocol smoke to verify the extracted helper logic. Later fixed by loading the workspace `.env` before forcing offline mode; see the large runtime extraction log below.
- Problem: A live `wespeaker_campplus` helper smoke used available local WeSpeaker assets but exceeded the five-minute timeout.
  - Fix at this stage: Did not use heavy model startup as a per-step regression test; kept compile/help/import and mocked protocol tests for the extraction boundary. Later removed WeSpeaker from the window GUI default provider; explicit `wespeaker_campplus` support remains available.
- Problem: Moving the remote ASR client could have removed `word_attr`, but local faster-whisper word parsing still uses it later in the GUI script.
  - Fix: Left `word_attr` in `youtube_window_diarize_gui.py` and gave `window_remote_asr.py` its own private `_word_attr`.

## Large Runtime Extraction Log - 2026-07-01

The goal of this pass was to reduce the two remaining large scripts by moving complete runtime responsibilities, not just small helpers.

### Extracted units

- `tools\realtime_gui_html.py`
  - Moved the realtime browser GUI HTML/JavaScript template out of `tools\realtime_speakerdiarize.py`.
  - The realtime script now imports `HTML` from this module.
- `tools\window_gui_html.py`
  - Moved the growing-window browser GUI HTML/JavaScript template out of `tools\youtube_window_diarize_gui.py`.
  - The window GUI script now imports `HTML` from this module.
- `tools\realtime_runtime.py`
  - Moved the realtime live runtime: transcript splitting records, reassignment/context helpers, `RealtimeSpeakerEngine`, YouTube URL parsing, trace logging, video clocking, audio input selection, event bus, WASAPI controller, HTTP request handler, and GUI server.
  - `tools\realtime_speakerdiarize.py` keeps CLI parsing, validation/replay tooling, and compatibility exports.
- `tools\window_runtime.py`
  - Moved the growing-window runtime: GUI defaults, speaker sensitivity presets, speaker library helpers, preset video list, local preview transcribers, event buses, Kroko chunk inference, sentence/window helpers, embedding candidate helpers, and `WindowDiarizer`.
  - `tools\youtube_window_diarize_gui.py` keeps HTTP routing, request body/file handling, CLI parsing, validation entry points, and process startup.

### Size result

- `tools\realtime_speakerdiarize.py`: 1,340 lines.
- `tools\youtube_window_diarize_gui.py`: 953 lines.
- `tools\realtime_runtime.py`: 1,529 lines.
- `tools\window_runtime.py`: 2,319 lines.
- `tools\realtime_gui_html.py`: 709 lines.
- `tools\window_gui_html.py`: 1,312 lines.

### Tests after each larger extraction

- After `realtime_gui_html.py`:
  - `.\.venv\Scripts\python.exe -m py_compile tools\realtime_gui_html.py tools\realtime_speakerdiarize.py tools\youtube_window_diarize_gui.py`
  - `.\.venv\Scripts\python.exe tools\realtime_speakerdiarize.py --help`
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help`
- After `window_gui_html.py`:
  - `.\.venv\Scripts\python.exe -m py_compile tools\window_gui_html.py tools\youtube_window_diarize_gui.py tools\realtime_speakerdiarize.py`
  - `.\.venv\Scripts\python.exe tools\realtime_speakerdiarize.py --help`
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help`
- After `realtime_runtime.py`:
  - `.\.venv\Scripts\python.exe -m py_compile tools\realtime_runtime.py tools\realtime_speakerdiarize.py tools\youtube_window_diarize_gui.py tools\youtube_local_filefeed_replay.py`
  - `.\.venv\Scripts\python.exe tools\realtime_speakerdiarize.py --help`
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help`
  - Compatibility import smoke for `extract_youtube_video_id` from `realtime_speakerdiarize`.
- After `window_runtime.py`:
  - `.\.venv\Scripts\python.exe -m py_compile tools\window_runtime.py tools\youtube_window_diarize_gui.py tools\realtime_runtime.py tools\realtime_speakerdiarize.py tools\youtube_local_filefeed_replay.py`
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --help`
  - `.\.venv\Scripts\python.exe tools\realtime_speakerdiarize.py --help`

### Problems found and fixes

- Problem: The extracted embedding environment setup did not read the workspace `.env`, so a present Hugging Face token was ignored and the helper forced offline mode.
  - Fix: Added project `.env`, `.env.local`, and `tools\.env` loading in `tools\embedding_providers.py`, normalized `HF_TOKEN`, `HF_ACCESS_TOKEN`, and `HUGGINGFACE_TOKEN`, and only forced `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` when no token is available.
- Problem: A live `speechbrain_ecapa` helper smoke could not fetch missing Hugging Face model files while offline mode was forced.
  - Fix: Reran the helper after the `.env` fix with network allowed. The helper returned a normalized 192-dimensional embedding; the token was not printed.
- Problem: A live `wespeaker_campplus` smoke was too slow and is not needed for the current default path.
  - Fix: Stopped using WeSpeaker as a regression smoke and changed `DEFAULT_WINDOW_EMBEDDING_PROVIDER` to `speechbrain_ecapa`, with `WHOSPEAKS_WINDOW_EMBEDDING_PROVIDER` still available to opt into the older stack or any other supported provider.
- Problem: Extracting `realtime_runtime.py` moved `extract_youtube_video_id`, but `tools\youtube_local_filefeed_replay.py` still imports that symbol from `realtime_speakerdiarize.py`.
  - Fix: Re-exported `extract_youtube_video_id` from `tools\realtime_speakerdiarize.py` via the import from `realtime_runtime.py`.
- Problem: The first `window_runtime.py` extraction missed names that `parse_args()` still used: `infer_kroko_preview_chunk_seconds` and `NEW_SPEAKER_SENSITIVITY_PRESETS`.
  - Fix: Imported both names from `tools\window_runtime.py` back into `tools\youtube_window_diarize_gui.py`, then reran compile/help checks.

### Startup checks

- Full no-browser cached-media startup with explicit `--embedding-provider speechbrain_ecapa`:
  - Command timed out only because the GUI server keeps running.
  - It reached cached media resolution, faster-whisper CUDA warmup, stream2sentence tokenizer warmup, SpeechBrain speaker embedding warmup, Silero ONNX VAD warmup, ASR warmup transcription, and printed the GUI URL.
- Quick default no-browser startup without pre-warmup:
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --no-browser --skip-download --no-startup-warmup-before-url --port 8797`
  - Command timed out only because the GUI server keeps running.
  - Startup config printed `embedding_provider=speechbrain_ecapa` and the server reached `http://127.0.0.1:8797/`.

## Runtime Facade Split Log - 2026-07-01

The goal of this pass was to make `realtime_runtime.py` and `window_runtime.py` small compatibility facades by extracting the remaining large runtime responsibilities into dedicated modules.

### Realtime extracted units

- `tools\realtime_transcript.py`
  - Moved timestamp normalization, transcript part records, sentence boundary timing, and `split_transcript_by_timestamps`.
- `tools\realtime_speaker_engine.py`
  - Moved queued sentence jobs, processed sentence records, speaker assignment/reassignment helpers, context assignment helpers, and `RealtimeSpeakerEngine`.
- `tools\realtime_capture.py`
  - Moved YouTube URL parsing, trace logging, video clock state, audio input/device selection, `EventBus`, and `YouTubeWasapiController`.
- `tools\realtime_server.py`
  - Moved the realtime HTTP request handler and `GuiServer`.
- `tools\realtime_runtime.py`
  - Replaced with a 63-line explicit re-export facade for compatibility.

### Window extracted units

- `tools\window_config.py`
  - Moved workspace paths, GUI defaults, VAD/default model path discovery, speaker sensitivity presets, speaker-library path helpers, and preset YouTube video metadata.
- `tools\window_preview.py`
  - Moved the preview transcriber interface, mock preview, in-process Kroko preview, subprocess Kroko preview, and Kroko chunk-size inference.
- `tools\window_events.py`
  - Moved normal and recording event buses.
- `tools\window_text.py`
  - Moved word attribute extraction, text reconstruction, mapped word spans, stream2sentence splitting, sentence boundary timing, content-word filtering, and embedding-candidate text checks.
- `tools\window_diarizer.py`
  - Moved `WindowDiarizer`, the main growing-window orchestration controller.
- `tools\window_runtime.py`
  - Replaced with a 127-line explicit re-export facade for compatibility.

### Size result

- `tools\realtime_runtime.py`: 63 lines.
- `tools\realtime_transcript.py`: 268 lines.
- `tools\realtime_speaker_engine.py`: 547 lines.
- `tools\realtime_capture.py`: 623 lines.
- `tools\realtime_server.py`: 112 lines.
- `tools\window_runtime.py`: 127 lines.
- `tools\window_config.py`: 255 lines.
- `tools\window_preview.py`: 253 lines.
- `tools\window_events.py`: 44 lines.
- `tools\window_text.py`: 257 lines.
- `tools\window_diarizer.py`: 1,581 lines.

### Tests after each extraction

- After `realtime_transcript.py`:
  - `py_compile` over the new transcript module, `realtime_runtime.py`, `realtime_speakerdiarize.py`, and `youtube_window_diarize_gui.py`.
  - Realtime `--help`.
  - Window GUI `--help`.
- After `realtime_speaker_engine.py`:
  - `py_compile` over the new engine module, `realtime_runtime.py`, `realtime_speakerdiarize.py`, and `youtube_window_diarize_gui.py`.
  - Realtime `--help`.
  - Window GUI `--help`.
- After `realtime_capture.py`:
  - `py_compile` over the new capture module, `realtime_runtime.py`, `realtime_speakerdiarize.py`, and `youtube_window_diarize_gui.py`.
  - Realtime `--help`.
  - Window GUI `--help`.
- After `realtime_server.py`:
  - `py_compile` over all new realtime runtime modules plus both public scripts.
  - Realtime `--help`.
  - Window GUI `--help`.
- After `window_config.py`, `window_preview.py`, `window_events.py`, `window_text.py`, and `window_diarizer.py`:
  - Repeated `py_compile` for the newly extracted module plus `window_runtime.py`, `youtube_window_diarize_gui.py`, and `realtime_speakerdiarize.py`.
  - Window GUI `--help`.
  - Realtime `--help`.
- After replacing both runtime files with explicit re-export facades:
  - `py_compile` over all extracted realtime/window runtime modules, both public scripts, and `youtube_local_filefeed_replay.py`.
  - Window GUI `--help`.
  - Realtime `--help`.
  - Compatibility import smoke for `extract_youtube_video_id` from `realtime_speakerdiarize`.

### Problems found and fixes

- Problem: The first attempted `window_config.py` extraction used the wrong class marker and stopped before writing files.
  - Fix: Reran the extraction with the correct `class RealtimePreviewTranscriber` marker.
- Problem: The no-browser startup path failed after moving Kroko preview helpers because `window_preview.py` used `re.search` in `infer_kroko_preview_chunk_seconds` but did not import `re`.
  - Fix: Added the missing `re` import and reran compile/help.
- Problem: The next no-browser startup reached server creation but failed because `window_events.py` emitted status messages with `_console_print` and `datetime` that were no longer in the same module.
  - Fix: Imported `_console_print` from `window_config.py` and `datetime` from the standard library.

### Startup check

- Quick default no-browser startup without pre-warmup:
  - `.\.venv\Scripts\python.exe tools\youtube_window_diarize_gui.py --no-browser --skip-download --no-startup-warmup-before-url --port 8798`
  - Command timed out only because the GUI server keeps running.
  - Startup config printed `embedding_provider=speechbrain_ecapa` and the server reached `http://127.0.0.1:8798/`.

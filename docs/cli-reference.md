# CLI Reference

This page documents every command-line parameter found in the repository parser definitions, plus the environment variables that change runtime defaults.

A parameter is an input that changes how a command behaves. Boolean parameters shown as `--name` / `--no-name` accept either form; use the positive form to enable the behavior and the `--no-` form to disable it.

For normal runs, start with [Configuration](configuration.md). Use this page when you need to understand an exact flag from `--help`, a launch profile, or a validation command.

## Terms

- ASR means automatic speech recognition: turning audio into transcript text.
- VAD means voice activity detection: marking which parts of audio sound like speech.
- Embedding means a numeric voice fingerprint used to compare speakers.
- Similarity means how close two embeddings are; higher usually means "more likely the same speaker."
- Margin means the lead between the best speaker match and the runner-up.
- UNKNOWN means the system has speech but not enough evidence to assign a known speaker.
- EMA means exponential moving average: smoothing where newer evidence counts more than older evidence.
- Canonical means the trusted reference transcript or speaker labels used for validation.

## Installed Commands

These commands are installed by the package entry points in `pyproject.toml`.

### `whospeaks`

Starter command for setup, health checks, saved profiles, and launching `whospeaks-window`.

| Scope | Parameter | Default | What it does |
| --- | --- | --- | --- |
| global | `--no-interactive` | off unless passed | Print the dashboard once and exit instead of opening the interactive starter CLI. |
| doctor | `--mode` | `auto` | Select which topology to check: auto-detect, all-local, remote controller plus servers, or server-only. Choices: `auto`, `local`, `remote`, `server`. |
| doctor | `--remote-asr-url` | empty | Remote ASR server URL to check instead of the saved profile value. |
| doctor | `--remote-embeddings-url` | empty | Remote embeddings server URL to check instead of the saved profile value. |
| doctor | `--port` |  | TCP port for the local server or evaluation target. |
| doctor | `--json` | off unless passed | Print the health-check result as machine-readable JSON. |
| doctor | `--deep` | off unless passed | Run expensive provider/cache checks such as remote /load. |
| doctor | `--strict` | off unless passed | Return non-zero when required checks fail. |
| doctor | `--fix` | off unless passed | Offer the recommended pip install action after checks. |
| doctor | `--yes` | off unless passed | Do not prompt before running the pip install action. |
| doctor | `--dry-run` | off unless passed | Print installer commands without running them. |
| setup | `--mode` | `local` | Choose the setup profile to save: local all-in-one, remote controller, or GPU server. Choices: `local`, `remote`, `server`. |
| setup | `--provider-preset` | empty | Apply a named final/live embedding provider stack while saving the setup profile. |
| setup | `--install` | off unless passed | Run the recommended pip extra installer. |
| setup | `--deep` | off unless passed | Run expensive provider/cache checks during setup. |
| setup | `--yes` | off unless passed | Do not prompt before running the pip install action. |
| setup | `--dry-run` | off unless passed | Print installer commands without running them. |
| launch | `--print` | off unless passed | Print the launch command and exit. |
| launch | `--dry-run` | off unless passed | Alias for --print. |
| launch | `--provider-preset` | empty | Temporarily apply a named provider stack to the printed or executed launch command. |
| launch | `--extra-args` | empty | Additional whospeaks-window arguments appended to the profile. |
| config | `--set` | `[]` | Save one profile field as `NAME=VALUE`; repeat the flag to set multiple fields. |
| config | `--reset` | off unless passed | Reset the saved starter profile to defaults before applying any `--set` values. |

### `whospeaks-meeting-intelligence`

Standalone browser server for generating and reviewing LLM-based meeting intelligence reports from saved sessions or a demo transcript. See [Meeting intelligence server](meeting-intelligence-server.md) for the workflow and examples.

| Area | Parameter | Default | What it does |
| --- | --- | --- | --- |
| Server | `--host` | `127.0.0.1` | Interface address the report server binds to. |
| Server | `--port` | 8798 | TCP port for the report browser UI. |
| Data | `--session-dir` | `DEFAULT_SESSION_DIR` | Directory containing durable WhoSpeaksLive saved sessions. |
| Data | `--cache-dir` | `DEFAULT_CACHE_DIR` | Directory where generated meeting intelligence reports are cached. |
| Data | `--demo-transcript` | empty | Add one transcript-only demo session from a WhoSpeaksLive transcript text file. |
| LLM | `--env-file` | repo `.env` | Local environment file loaded before LLM defaults are resolved. Existing process environment variables are not overwritten. |
| LLM | `--llm-provider` | `llama_cpp` | OpenAI-compatible provider preset. Choices: `llama_cpp`, `ollama`, `lm_studio`, `openai`, `openrouter`. |
| LLM | `--llm-base-url` | provider default | OpenAI-compatible base URL, without `/chat/completions`. |
| LLM | `--llm-model` | provider default | Model name sent to the LLM server. |
| LLM | `--llm-api-key` | empty | API key for providers that require authentication. |
| LLM | `--timeout-seconds` | 900.0 | HTTP timeout for one report-generation LLM request. |
| LLM | `--max-tokens` | 4096 | Maximum tokens for evidence-extraction calls. |
| LLM | `--section-max-tokens` | 4096 | Maximum tokens for per-section report calls. |
| Pipeline | `--max-segment-rows` | 80 | Maximum transcript rows per evidence-extraction segment. Values below 12 are raised to 12. |
| Development | `--mock-llm` | off unless passed | Use deterministic mock responses instead of contacting an LLM server. |
| Development | `--auto-generate` | off unless passed | Generate a report automatically when a selected session has no current report. |

The provider flags define the initial browser state. The browser can switch provider, model, and base URL at runtime through `/api/llm-config`, and can load account-visible provider models through `/api/llm-models`. API keys remain server-side.

### `whospeaks-window`

Main browser app for media download/playback, final ASR, speaker assignment, live feedback, and validation.

| Area | Parameter | Default | What it does |
| --- | --- | --- | --- |
| Media, server, and session lease | `--url` | `DEFAULT_URL` | Media URL to download or play; defaults to the demo clip. |
| Media, server, and session lease | `--work-dir` | `DEFAULT_WORK_DIR` | Scratch directory for downloads and intermediate files. |
| Media, server, and session lease | `--output-dir` | `DEFAULT_OUTPUT_DIR` | Directory for run outputs such as media, traces, validation JSON, or reports. |
| Media, server, and session lease | `--session-dir` | `DEFAULT_SESSION_DIR` | Directory used for durable saved WhoSpeaks Live sessions. |
| Media, server, and session lease | `--audio-file` |  | Use this existing audio file instead of downloading audio. |
| Media, server, and session lease | `--video-file` |  | Use this existing video file for browser playback instead of downloading video. |
| Media, server, and session lease | `--skip-download` | off unless passed | Reuse existing local files and do not call `yt-dlp`. |
| Media, server, and session lease | `--yt-dlp` |  | Path to the `yt-dlp` executable. |
| Media, server, and session lease | `--host` | `127.0.0.1` | Interface address the local HTTP server binds to. |
| Media, server, and session lease | `--port` | 8795 | TCP port for the local server or evaluation target. |
| Media, server, and session lease | `--no-browser` | off unless passed | Start the server without opening a browser tab. |
| Media, server, and session lease | `--demo-seat-lease`<br>`--no-demo-seat-lease` | false | Require one browser tab to take the public demo seat before controlling a shared run. |
| Media, server, and session lease | `--session-lease-idle-timeout-seconds` | 120.0 | Release an acquired demo seat after this many seconds if no run has started. |
| Media, server, and session lease | `--session-lease-heartbeat-timeout-seconds` | 45.0 | Release and stop an active demo seat if the owner tab stops sending heartbeats. |
| Media, server, and session lease | `--session-lease-completed-release-delay-seconds` | 10.0 | Seconds to keep a completed one-seat demo session before releasing it for the next user. |
| Media, server, and session lease | `--session-lease-max-run-seconds` | 900.0 | Hard maximum owner runtime for one public demo seat. |
| Media, server, and session lease | `--asr-backend` | `local` | ASR backend for final growing-window transcription. Choices: `local`, `remote`. |
| Media, server, and session lease | `--remote-asr-url` | `DEFAULT_REMOTE_ASR_URL` | Base URL of the remote faster-whisper large-v2 ASR server. |
| ASR and language | `--remote-asr-timeout-seconds` | 120.0 | HTTP timeout for each remote ASR request. |
| ASR and language | `--model` | `large-v2` | Final ASR model used for committed transcription. |
| ASR and language | `--language` | `default_language_code()` | Realtime language for final ASR, Kroko preview model selection, and sentence splitting. |
| ASR and language | `--sentence-tokenizer` |  | Sentence tokenizer for stream2sentence. Defaults to the language-specific realtime choice. |
| ASR and language | `--device` | `cuda` | Device for local ASR or embedding execution, such as `cuda`, `cpu`, or `auto`. |
| ASR and language | `--compute-type` | `float16` | faster-whisper numeric precision, such as `float16` or `int8`. |
| ASR and language | `--download-root` | `default_faster_whisper_download_root()` | Model cache root used by faster-whisper or preview model downloads. |
| ASR and language | `--beam-size` | 5 | Beam size for final ASR; higher can improve text at the cost of speed. |
| ASR and language | `--interval-seconds` | 0.7 | Fixed delay between transcription passes, also used as cooldown after a successful sentence split. 0 runs continuously with no overlap. |
| ASR and language | `--min-playback-advance-seconds` | 0.75 | Minimum browser playback-time advance required before starting the next pass. |
| ASR and language | `--min-window-seconds` | 2.0 | Minimum amount of media time collected before a final ASR pass. |
| ASR and language | `--unstable-tail-seconds` | 1.35 | Minimum seconds after a candidate sentence's last word before committing a punctuation-ending sentence. |
| ASR and language | `--vad-sentence-splitting`<br>`--no-vad-sentence-splitting` | true | Use local VAD to force-finalize a window after trailing silence. |
| VAD, ASR gating, and sentence commits | `--vad-backend` | `silero` | VAD backend for sentence-window finalization. Choices: `silero`, `rms`. |
| VAD, ASR gating, and sentence commits | `--vad-silero-backend` | `default_silero_vad_backend(default_vad_model_path)` | Silero implementation used when --vad-backend silero is active. Choices: `auto`, `raw_onnx_ifless`, `raw_onnx`, `official_onnx`, `pytorch_cpu`. |
| VAD, ASR gating, and sentence commits | `--vad-silero-onnx-model-path` | `default_vad_model_path` | Path to a Silero ONNX model file. Defaults to the local RealtimeSTT model cache when available. |
| VAD, ASR gating, and sentence commits | `--vad-silero-onnx-threads` | 2 | CPU threads used by the raw ONNX Silero VAD session. |
| VAD, ASR gating, and sentence commits | `--vad-silero-speech-threshold` | 0.5 | Silero speech probability required to mark a 512-sample chunk as speech. |
| VAD, ASR gating, and sentence commits | `--vad-silence-seconds` | 1.1 | Trailing silence required before VAD forces the current window to finalize. |
| VAD, ASR gating, and sentence commits | `--vad-final-window-post-silence-seconds` | 0.75 | On a VAD split, transcribe the previous final window only this far after VAD speech end. |
| VAD, ASR gating, and sentence commits | `--vad-next-window-start-silence-seconds` | 0.7 | On a VAD split, advance the next window start to at least this far after VAD speech end. |
| VAD, ASR gating, and sentence commits | `--vad-speech-rms-threshold` | 0.003 | RMS threshold used by --vad-backend rms or by the RMS fallback. |
| VAD, ASR gating, and sentence commits | `--vad-frame-seconds` | 0.03 | Frame size used by the local energy VAD. |
| VAD, ASR gating, and sentence commits | `--vad-merge-gap-seconds` | 0.18 | Short silence gaps below this length are merged into surrounding speech. |
| VAD, ASR gating, and sentence commits | `--vad-min-speech-seconds` | 0.25 | Minimum detected speech in a window before VAD can trigger a split. |
| VAD, ASR gating, and sentence commits | `--vad-gate-secondary-backend` | `webrtc` | Realtime-safe secondary VAD required to confirm ASR/preview speech gates. Choices: `off`, `webrtc`. |
| VAD, ASR gating, and sentence commits | `--vad-gate-webrtc-mode` | 3 | WebRTC VAD aggressiveness for ASR/preview gate confirmation (0-3). |
| VAD, ASR gating, and sentence commits | `--vad-gate-min-consensus-seconds` | 0.12 | Minimum secondary-VAD overlap required to accept a primary VAD speech span. |
| VAD, ASR gating, and sentence commits | `--vad-gate-min-consensus-ratio` | 0.05 | Minimum secondary-VAD overlap ratio required to accept a primary VAD speech span. |
| VAD, ASR gating, and sentence commits | `--asr-vad-gate`<br>`--no-asr-vad-gate` | true | Before final ASR, trim leading/trailing non-speech and transcribe one padded speech-bounded clip instead of the full music/silence-containing window. |
| VAD, ASR gating, and sentence commits | `--asr-vad-gate-pre-padding-seconds` | 0.2 | Audio kept before each VAD speech island sent to final ASR. |
| VAD, ASR gating, and sentence commits | `--asr-vad-gate-post-padding-seconds` | 0.35 | Audio kept after each VAD speech island sent to final ASR. |
| VAD, ASR gating, and sentence commits | `--asr-vad-gate-merge-gap-seconds` | 0.85 | When internal gap cutting is enabled, merge padded ASR speech islands separated by at most this many seconds. |
| VAD, ASR gating, and sentence commits | `--asr-vad-gate-min-clip-seconds` | 0.2 | Drop padded ASR speech clips shorter than this duration. |
| VAD, ASR gating, and sentence commits | `--asr-vad-gate-cut-internal-gaps`<br>`--no-asr-vad-gate-cut-internal-gaps` | false | Experimental: cut long non-speech gaps inside a final ASR window. Disabled by default to avoid splitting sentences. |
| VAD, ASR gating, and sentence commits | `--asr-no-speech-filter`<br>`--no-asr-no-speech-filter` | true | Drop ASR segments whose Whisper no_speech_prob is above the configured threshold. |
| VAD, ASR gating, and sentence commits | `--asr-no-speech-prob-threshold` | 0.65 | Whisper no_speech_prob threshold above which ASR segment words are discarded. |
| VAD, ASR gating, and sentence commits | `--asr-no-speech-hard-threshold` | 0.85 | Whisper no_speech_prob threshold above which even very short ASR segments are discarded. |
| VAD, ASR gating, and sentence commits | `--asr-no-speech-keep-short-max-words` | 2 | Keep ASR segments at or above the no_speech_prob threshold when they have at most this many words and stay below the hard threshold. |
| VAD, ASR gating, and sentence commits | `--asr-no-speech-keep-short-max-seconds` | 0.45 | Keep ASR segments at or above the no_speech_prob threshold when they are at most this long and stay below the hard threshold. |
| VAD, ASR gating, and sentence commits | `--sentence-boundary-pre-padding-seconds` | `DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS` | Audio kept before the next word when cutting between two consecutive completed sentences. |
| VAD, ASR gating, and sentence commits | `--sentence-boundary-post-padding-seconds` | `DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS` | Audio kept after the last word when cutting between two consecutive completed sentences. |
| VAD, ASR gating, and sentence commits | `--sentence-boundary-gap-ratio` | `DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO` | For tight word gaps, fraction of the gap assigned to the previous sentence. |
| VAD, ASR gating, and sentence commits | `--final-flush-epsilon-seconds` | 0.5 | Treat playback as ended when browser time is within this many seconds of audio duration. |
| VAD, ASR gating, and sentence commits | `--start-warmup-stale-seconds` | 10.0 | Refresh ASR and embedding warmups on Start when the previous runtime warmup is older than this. Use 0 to always refresh. |
| VAD, ASR gating, and sentence commits | `--startup-warmup-before-url`<br>`--no-startup-warmup-before-url` | true | Warm ASR, embeddings, and VAD before printing/serving the browser URL. |
| Embedding backend and speaker library | `--embedding-provider` | `DEFAULT_WINDOW_EMBEDDING_PROVIDER` | Provider or weighted provider stack used for final speaker assignment. |
| Embedding backend and speaker library | `--embedding-python` | `default_embedding_python()` | Python executable used by the local embedding helper process. |
| Embedding backend and speaker library | `--embedding-device` | `cuda` | Device used by local speaker embedding models. |
| Embedding backend and speaker library | `--live-speaker-embedding-provider` | `pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50` | Provider or weighted provider stack used only for fast live speaker assignment. Empty uses --embedding-provider. |
| Embedding backend and speaker library | `--live-speaker-assignment`<br>`--no-live-speaker-assignment` | true | Enable live speaker highlighting/scoring during realtime preview. Use --no-live-speaker-assignment to keep live text preview without live speaker scoring. |
| Embedding backend and speaker library | `--embeddings-backend`<br>`--embedding-backend`<br>`-embeddings-backend` | `local` | Speaker embedding backend. Use remote to send embedding requests to the Linux GPU server. Choices: `local`, `remote`. |
| Embedding backend and speaker library | `--remote-embeddings-url`<br>`--remote-embedding-url` | `DEFAULT_REMOTE_EMBEDDINGS_URL` | Base URL of the remote voice embeddings server. |
| Embedding backend and speaker library | `--remote-embeddings-timeout-seconds`<br>`--remote-embedding-timeout-seconds` | `DEFAULT_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS` | HTTP timeout for remote embedding health, load, and embed requests. |
| Embedding backend and speaker library | `--remote-embeddings-device`<br>`--remote-embedding-device` | `auto` | Device query parameter sent to the remote embeddings server. |
| Embedding backend and speaker library | `--embedding-helper-response-timeout-seconds` | `DEFAULT_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS` | Maximum time to wait for an embedding helper response. First startup of the default high-quality stacked provider may need several minutes while models download and initialize. |
| Embedding backend and speaker library | `--speaker-library-dir` | `DEFAULT_SPEAKER_LIBRARY_DIR` | Directory for saved speaker groups and uploaded reference audio. |
| Embedding backend and speaker library | `--new-speaker-sensitivity` |  | Optional five-step new-speaker spawning sensitivity preset. Position 3 matches the tuned defaults. Choices: `{1,2,3,4,5}`. |
| Speaker assignment thresholds | `--same-speaker-similarity` | 0.43 | Similarity needed before a sentence can update or reuse an existing speaker. |
| Speaker assignment thresholds | `--similarity-temperature` | 0.061 | Softmax temperature for similarity scores; lower values make the best match dominate. |
| Speaker assignment thresholds | `--speaker-softmax-temperature` | 0.0557 | Temperature used when converting speaker similarities into probabilities. |
| Speaker assignment thresholds | `--new-speaker-threshold` | 0.4309 | Unknown-speaker probability needed before creating a new speaker. |
| Speaker assignment thresholds | `--duplicate-profile-similarity` | 0.4247 | Similarity above which a candidate new profile is treated as an existing speaker duplicate. |
| Speaker assignment thresholds | `--unknown-short-threshold` | 0.287 | Unknown probability above which short uncertain fragments stay UNKNOWN. |
| Speaker assignment thresholds | `--min-first-speaker-seconds` | 1.8373 | Minimum speech duration required before creating the first speaker profile. |
| Speaker assignment thresholds | `--min-new-speaker-seconds` | 2.0358 | Minimum sentence duration required before creating an additional speaker. |
| Speaker assignment thresholds | `--late-new-speaker-min-seconds` | 3.1604 | Longer duration required for late new-speaker creation after profiles already exist. |
| Speaker assignment thresholds | `--max-speakers` | 12 | Maximum number of speaker profiles the run may create automatically. |
| Speaker assignment thresholds | `--min-margin` | 0.0372 | Minimum lead the best speaker match needs over the runner-up. |
| Speaker assignment thresholds | `--margin-temperature` | 0.0361 | Softmax temperature for margin-based speaker confidence. |
| Speaker assignment thresholds | `--update-unknown-max` | 0.4289 | Maximum UNKNOWN probability allowed when updating an existing speaker profile. |
| Speaker assignment thresholds | `--new-speaker-confirmation-count` | 1 | Number of mutually similar far-away sentence embeddings required before creating a new speaker. |
| Speaker assignment thresholds | `--new-speaker-confirmation-similarity` | 0.5801 | Minimum cosine similarity between pending new-speaker candidates before creating a speaker. |
| Speaker assignment thresholds | `--max-pending-new-speakers` | 6 | Maximum queued new-speaker candidates kept before confirmation. |
| Speaker assignment thresholds | `--known-speaker-min-similarity` | 0.5563 | When non-negative, existing speakers below this top similarity are treated as gray-zone UNKNOWN instead of confidently assigned. |
| Speaker assignment thresholds | `--known-speaker-gray-zone-min-unknown-probability` | 0.064 | Minimum unknown probability required before --known-speaker-min-similarity defers an assignment to UNKNOWN. |
| Speaker assignment thresholds | `--profile-update-min-similarity` | 0.5011 | When non-negative, update existing speaker centroids only if top similarity is at least this value. |
| Speaker assignment thresholds | `--profile-update-min-margin` | 0.0037 | When non-negative, update existing speaker centroids only if top-vs-runner-up margin is at least this value. |
| Speaker assignment thresholds | `--low-similarity-unknown-floor-similarity` | 0.56 | When non-negative, raise unknown probability for known-speaker comparisons below this top similarity. |
| Speaker assignment thresholds | `--low-similarity-unknown-floor-probability` | 0.1885 | Unknown probability floor used with --low-similarity-unknown-floor-similarity. |
| Speaker assignment thresholds | `--gray-zone-promote-max-similarity` | 0.55 | Maximum candidate-vs-known centroid similarity allowed before a gray-zone pending voice can become a new speaker. |
| Speaker assignment thresholds | `--min-new-speaker-words` | 3 | Minimum content words required for a sentence to create or confirm a new speaker profile. |
| Speaker assignment thresholds | `--retro-reassign-min-similarity` | 0.02 | Minimum cosine similarity for assigning an earlier UNKNOWN sentence to an existing speaker. |
| Speaker assignment thresholds | `--retro-reassign-min-margin` | 0.0 | Minimum top-vs-runner-up similarity gap for retro UNKNOWN reassignment when multiple speakers exist. |
| Speaker assignment thresholds | `--speaker-refinement`<br>`--no-speaker-refinement` | true | Enable prototype-based live refinement. Stable mode only fills UNKNOWN rows later. |
| Speaker assignment thresholds | `--speaker-refinement-unknown-tentative`<br>`--no-speaker-refinement-unknown-tentative` | true | Allow prototype refinement to show tentative speaker hints on UNKNOWN transcript rows. |
| Speaker assignment thresholds | `--speaker-refinement-unknown-commit`<br>`--no-speaker-refinement-unknown-commit` | true | Allow later evidence to commit UNKNOWN transcript rows to a known or newly confirmed speaker. |
| Speaker assignment thresholds | `--allow-speaker-reassignment`<br>`--no-allow-speaker-reassignment` | true | Allow prototype refinement to change already committed non-UNKNOWN speaker labels. |
| Speaker refinement cleanup | `--speaker-refinement-max-per-profile` | 32 | Maximum refinement examples kept for each speaker profile. |
| Speaker refinement cleanup | `--speaker-refinement-min-duration` | 0.15 | Shortest final sentence that can participate in refinement. |
| Speaker refinement cleanup | `--speaker-refinement-max-unknown` | 1.0 | Highest UNKNOWN probability still eligible for refinement. |
| Speaker refinement cleanup | `--speaker-refinement-top-k` | 12 | Number of nearest refinement candidates considered for each decision. |
| Speaker refinement cleanup | `--speaker-refinement-centroid-blend` | 0.555 | Blend weight between a speaker's existing centroid and refinement examples. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-min-similarity` | 0.2 | Minimum similarity needed before refinement can fill an UNKNOWN row. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-min-margin` | 0.0 | Minimum lead over the runner-up before refinement can fill an UNKNOWN row. |
| Speaker refinement cleanup | `--speaker-refinement-known-max-duration` | 8.0 | Longest already-known sentence that can be changed by refinement. |
| Speaker refinement cleanup | `--speaker-refinement-known-min-similarity` | -0.039 | Minimum similarity needed before refinement can change a known speaker row. |
| Speaker refinement cleanup | `--speaker-refinement-known-min-delta` | 0.04 | Minimum improvement required before refinement changes a known speaker row. |
| Speaker refinement cleanup | `--speaker-refinement-final-passes` | 1 | Bounded extra speaker refinement passes after the final sentence is committed. |
| Speaker refinement cleanup | `--speaker-refinement-small-island-merge`<br>`--no-speaker-refinement-small-island-merge` | true | Merge a tiny one-off speaker island when the same speaker appears immediately before and after it. |
| Speaker refinement cleanup | `--speaker-refinement-small-island-max-duration` | 5.0 | Maximum total duration of a small isolated speaker island that may be merged. |
| Speaker refinement cleanup | `--speaker-refinement-small-island-max-segments` | 3 | Maximum number of consecutive segments in a small isolated island that may be merged. |
| Speaker refinement cleanup | `--speaker-refinement-tiny-fragmented-merge`<br>`--no-speaker-refinement-tiny-fragmented-merge` | true | Merge a very small fragmented speaker profile into its dominant neighboring speaker at finalization. |
| Speaker refinement cleanup | `--speaker-refinement-tiny-fragmented-max-duration` | 6.0 | Maximum total duration of a fragmented speaker profile that may be merged away. |
| Speaker refinement cleanup | `--speaker-refinement-tiny-fragmented-max-segments` | 8 | Maximum total segment count of a fragmented speaker profile that may be merged away. |
| Speaker refinement cleanup | `--speaker-refinement-tiny-fragmented-min-islands` | 2 | Minimum number of separated islands before fragmented-profile cleanup applies. |
| Speaker refinement cleanup | `--speaker-refinement-tiny-fragmented-max-islands` | 3 | Maximum number of islands still considered a tiny fragmented profile. |
| Speaker refinement cleanup | `--speaker-refinement-tiny-fragmented-min-neighbor-share` | 0.5 | Minimum share of neighboring evidence that must point to the same replacement speaker. |
| Speaker refinement cleanup | `--speaker-refinement-terminal-outro-merge`<br>`--no-speaker-refinement-terminal-outro-merge` | true | Merge a singleton terminal promotional outro back to the stable opening speaker. |
| Speaker refinement cleanup | `--speaker-refinement-terminal-outro-max-duration` | 12.0 | Maximum duration of the final isolated outro fragment that may be merged. |
| Speaker refinement cleanup | `--speaker-refinement-terminal-outro-lookback-segments` | 2 | Number of prior segments checked when deciding whether the ending fragment is an outro continuation. |
| Speaker refinement cleanup | `--speaker-refinement-terminal-outro-min-target-duration` | 5.0 | Minimum duration of the target speaker section before terminal-outro merge is allowed. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-same-speaker-fill`<br>`--no-speaker-refinement-unknown-same-speaker-fill` | true | Fill a short UNKNOWN island only when it is flanked by the same speaker on both sides. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-same-speaker-max-duration` | 3.0 | Maximum duration of an UNKNOWN island that can be filled from matching speakers on both sides. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-same-speaker-max-segments` | 1 | Maximum segment count of an UNKNOWN island that can be filled from matching speakers on both sides. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-previous-speaker-fill`<br>`--no-speaker-refinement-unknown-previous-speaker-fill` | true | Fill a short non-embedding UNKNOWN tail only when it is contiguous with the previous speaker and separated from the next speaker by a pause. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-previous-speaker-max-duration` | 0.75 | Maximum duration of an UNKNOWN tail that can be filled from the previous speaker. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-previous-speaker-max-segments` | 1 | Maximum segment count of an UNKNOWN tail that can be filled from the previous speaker. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-previous-speaker-max-previous-gap` | 0.35 | Maximum gap from the previous speaker for previous-speaker fill. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-previous-speaker-min-next-gap` | 0.3 | Minimum gap to the next speaker required before previous-speaker fill is allowed. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-next-speaker-fill`<br>`--no-speaker-refinement-unknown-next-speaker-fill` | true | Fill a short non-embedding UNKNOWN head only when it is separated from the previous speaker by a pause and contiguous with the next speaker. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-next-speaker-max-duration` | 1.75 | Maximum duration of an UNKNOWN head that can be filled from the next speaker. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-next-speaker-max-segments` | 1 | Maximum segment count of an UNKNOWN head that can be filled from the next speaker. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-next-speaker-max-next-gap` | 0.05 | Maximum gap to the next speaker for next-speaker fill. |
| Speaker refinement cleanup | `--speaker-refinement-unknown-next-speaker-min-previous-gap` | 0.15 | Minimum gap from the previous speaker required before next-speaker fill is allowed. |
| Speaker refinement cleanup | `--speaker-refinement-long-low-confidence-retro-split`<br>`--no-speaker-refinement-long-low-confidence-retro-split` | true | Split a long, very low-confidence retro assignment into a new final speaker. |
| Speaker refinement cleanup | `--speaker-refinement-long-low-confidence-retro-min-duration` | 4.0 | Minimum duration before low-confidence retroactive split cleanup can apply. |
| Speaker refinement cleanup | `--speaker-refinement-long-low-confidence-retro-max-similarity` | 0.06 | Maximum similarity to existing speakers for low-confidence retroactive split cleanup. |
| Speaker refinement cleanup | `--speaker-refinement-long-low-confidence-retro-max-margin` | 0.04 | Maximum top-speaker margin still considered low-confidence for retroactive split cleanup. |
| Speaker refinement cleanup | `--speaker-refinement-long-low-confidence-retro-max-splits` | 1 | Maximum number of low-confidence retroactive splits performed per run. |
| Speaker refinement cleanup | `--min-embed-seconds` | 0.5 | Minimum audio duration before creating a speaker embedding. |
| Realtime preview setup and VAD gate | `--min-speech-audio-ratio` | 0.0 | Minimum sum(word durations) / sentence audio duration required before embedding a sentence. |
| Realtime preview setup and VAD gate | `--realtime-preview-engine` | `kroko_onnx` | Realtime preview engine: kroko_onnx, mock, or off. |
| Realtime preview setup and VAD gate | `--realtime-preview-model` |  | Kroko/Banafo model name for replace-only realtime preview text. Overrides --realtime-preview-model-preset. |
| Realtime preview setup and VAD gate | `--realtime-preview-model-preset` | `DEFAULT_KROKO_PREVIEW_MODEL_PRESET` | Named Kroko preview model preset. Use pro-16l for Kroko-EN-Pro-16-L-Streaming-001.data. |
| Realtime preview setup and VAD gate | `--realtime-preview-model-path` |  | Exact Kroko `.data` model file for realtime preview; bypasses preset lookup. |
| Realtime preview setup and VAD gate | `--realtime-preview-auto-download`<br>`--no-realtime-preview-auto-download` | `DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD` | Download missing public Kroko Community preview models from Hugging Face before starting preview. |
| Realtime preview setup and VAD gate | `--realtime-preview-download-root` |  | Directory where missing public Kroko preview models are downloaded. |
| Realtime preview setup and VAD gate | `--realtime-preview-python` | `DEFAULT_KROKO_PREVIEW_PYTHON` | Python executable that runs the Kroko preview worker. |
| Realtime preview setup and VAD gate | `--realtime-preview-realtimestt-root` | `DEFAULT_REALTIMESTT_ROOT` | RealtimeSTT checkout used to find Kroko and VAD assets. |
| Realtime preview setup and VAD gate | `--realtime-preview-provider` | `cpu` | Execution provider passed to the Kroko preview worker, usually `cpu`. |
| Realtime preview setup and VAD gate | `--realtime-preview-num-threads` | 2 | CPU thread count for Kroko realtime preview decoding. |
| Realtime preview setup and VAD gate | `--realtime-preview-startup-timeout-seconds` |  | Maximum time to wait for the realtime preview engine before disabling preview. Defaults to 45s for pro-16l and 12s otherwise. |
| Realtime preview setup and VAD gate | `--realtime-preview-request-timeout-seconds` | 5.0 | Maximum time to wait for one realtime preview decode request. |
| Realtime preview setup and VAD gate | `--realtime-preview-interval-seconds` |  | Seconds between realtime preview decode attempts; empty uses the model preset default. |
| Realtime preview setup and VAD gate | `--realtime-preview-min-audio-seconds` |  | Minimum buffered audio required before preview decoding starts; empty uses the preset default. |
| Realtime preview setup and VAD gate | `--realtime-preview-min-advance-seconds` |  | Minimum playback advance before preview text is recomputed; empty uses the preset default. |
| Realtime preview setup and VAD gate | `--realtime-preview-feed-chunk-seconds` |  | Audio seconds fed to Kroko per streaming accept call. By default this is inferred from the Kroko model name. |
| Realtime preview setup and VAD gate | `--realtime-preview-vad-gate`<br>`--no-realtime-preview-vad-gate` | true | Start Kroko preview only after VAD speech onset and reset it after sustained non-speech. |
| Realtime preview setup and VAD gate | `--realtime-preview-vad-gate-pre-padding-seconds` | 0.35 | Buffered audio kept before VAD speech onset when starting Kroko preview. |
| Realtime preview setup and VAD gate | `--realtime-preview-vad-gate-post-padding-seconds` | 0.35 | Audio kept after VAD speech end before resetting Kroko preview. |
| Realtime preview setup and VAD gate | `--realtime-preview-vad-gate-close-silence-seconds` | 1.1 | Sustained VAD non-speech required before closing and resetting a Kroko preview session. |
| Realtime preview setup and VAD gate | `--realtime-preview-reset-overlap-seconds` | 0.15 | Audio pre-roll kept before the committed sentence boundary when resetting preview after final sentence commits. |
| Realtime preview setup and VAD gate | `--realtime-preview-diarize-min-audio-seconds` | 1.5 | Minimum live unresolved audio duration before scoring it against known speakers. |
| Live speaker feedback | `--realtime-preview-diarize-min-advance-seconds` | 0.75 | Minimum live playback advance before recomputing the live speaker embedding. |
| Live speaker feedback | `--realtime-preview-diarize-min-similarity` | 0.45 | Minimum cosine similarity for assigning a live preview row to an existing speaker. |
| Live speaker feedback | `--realtime-preview-diarize-min-margin` | 0.08 | Minimum top-vs-runner-up margin for assigning a live preview row when multiple speakers exist. |
| Live speaker feedback | `--realtime-preview-diarize-min-known-probability` | 0.5 | Minimum known-speaker probability before the live row label switches from Unknown to a speaker. |
| Live speaker feedback | `--live-speaker-embedding-min-interval-seconds` | 0.75 | Minimum wall-clock spacing between live speaker embedding requests from preview/probe paths. |
| Live speaker feedback | `--live-speaker-embedding-target-utilization` | 0.25 | Target fraction of wall time live speaker embeddings may occupy; use 1.0 to disable latency backoff. |
| Live speaker feedback | `--live-speaker-verify-on-change`<br>`--no-live-speaker-verify-on-change` | false | Use the full embedding stack to confirm visible live speaker changes proposed by the fast provider. |
| Live speaker feedback | `--live-speaker-verify-min-interval-seconds` | 2.0 | Minimum wall-clock spacing between full-stack live speaker change verification requests. |
| Live speaker feedback | `--live-speaker-ema-window-seconds` | 1.0 | Wall-clock window used for smoothing live speaker probabilities. |
| Live speaker feedback | `--live-speaker-ema-count` | 1 | Maximum number of recent live speaker probability snapshots blended by EMA. |
| Live speaker feedback | `--live-speaker-ema-alpha` | 0.55 | EMA weight for the newest live speaker probability snapshot. |
| Live speaker feedback | `--live-speaker-probe`<br>`--no-live-speaker-probe` | true | When enabled, score the last live audio window against known speakers for fallback speaker highlighting. |
| Live speaker feedback | `--live-speaker-probe-interval-seconds` | 0.75 | Seconds between fallback live-speaker probes. |
| Live speaker feedback | `--live-speaker-probe-attack-interval-seconds` | 0.0 | Optional faster probe interval while acquiring a speaker or resolving UNKNOWN; 0 disables. |
| Live speaker feedback | `--live-speaker-probe-window-seconds` | 1.0 | Recent audio window scored by the fallback live-speaker probe. |
| Live speaker feedback | `--live-speaker-probe-hold-seconds` | 1.0 | Seconds the browser keeps a fallback live-speaker highlight after a matching probe. |
| Live speaker feedback | `--live-speaker-probe-min-advance-seconds` | 0.75 | Minimum playback advance before rescoring the fallback live-speaker probe window. |
| Live speaker feedback | `--live-speaker-probe-attack-min-advance-seconds` | 0.0 | Optional faster minimum playback advance during attack cadence; 0 uses the attack interval. |
| Live speaker feedback | `--live-speaker-probe-min-speech-seconds` | 0.15 | Minimum RMS-gated speech inside the probe window before embedding it. |
| Live speaker feedback | `--live-speaker-probe-speech-backend` | `rms` | Speech gate used by live-speaker probe windows. 'vad' reuses the configured VAD backend. Choices: `rms`, `vad`. |
| Live speaker feedback | `--live-speaker-probe-clear-on-silence`<br>`--no-live-speaker-probe-clear-on-silence` | true | Clear the fallback live speaker when the recent audio window has no RMS-gated speech. |
| Live speaker feedback | `--live-speaker-clear-on-vad-split`<br>`--no-live-speaker-clear-on-vad-split` | false | Clear the fallback live speaker when the main VAD finalizes a sentence window after trailing silence. |
| Live speaker feedback | `--live-speaker-probe-clear-window-seconds` | 1.0 | Recent audio duration checked for silence before clearing the fallback live speaker. |
| Live speaker feedback | `--live-speaker-probe-clear-silence-count` | 1 | Clear the fallback live speaker after this many consecutive silent clear windows. |
| Live speaker feedback | `--live-speaker-probe-clear-unknown-count` | 2 | Clear the fallback live speaker after this many consecutive speech probes score as UNKNOWN; use 0 to disable. |
| Live speaker feedback | `--live-speaker-probe-unknown-clear-debounce-seconds` | 0.0 | Delay UNKNOWN fallback-live-speaker clear events in the browser by this many seconds; 0 clears immediately. |
| Live speaker feedback | `--live-speaker-probe-unknown-keepalive`<br>`--no-live-speaker-probe-unknown-keepalive` | false | Keep the current fallback live speaker highlighted during pre-clear UNKNOWN probes. |
| Live speaker feedback | `--live-speaker-probe-unknown-release-smoothing` | `none` | Smooth current-speaker versus UNKNOWN evidence before releasing the fallback live speaker. Choices: `none`, `sma`, `ema`. |
| Live speaker feedback | `--live-speaker-probe-unknown-release-count` | 3 | Number of recent UNKNOWN release samples used by SMA/EMA release smoothing. |
| Live speaker feedback | `--live-speaker-probe-unknown-release-ema-alpha` | 0.5 | EMA weight for the newest UNKNOWN release sample. |
| Live speaker feedback | `--live-speaker-probe-unknown-release-margin` | 0.0 | Tolerance added to the current speaker probability before UNKNOWN release wins. |
| Live speaker feedback | `--live-speaker-provisional-new-speaker`<br>`--no-live-speaker-provisional-new-speaker` | false | Emit a temporary live speaker id for speech that does not yet match any known speaker. |
| Live speaker feedback | `--live-speaker-provisional-min-audio-seconds` | 1.0 | Minimum live probe audio duration before creating a provisional live speaker. |
| Live speaker feedback | `--live-speaker-provisional-min-unknown-probability` | 0.5 | Minimum unknown probability required before creating a provisional live speaker. |
| Live speaker feedback | `--live-speaker-weak-profile-assist`<br>`--no-live-speaker-weak-profile-assist` | false | Allow a stricter, similarity-based live assignment for very young known-speaker profiles. |
| Live speaker feedback | `--live-speaker-weak-profile-max-speech-seconds` | 2.5 | Maximum accumulated profile speech seconds considered weak for live-speaker assist. |
| Live speaker feedback | `--live-speaker-weak-profile-min-similarity` | 0.4 | Minimum top similarity for weak-profile live-speaker assist. |
| Live speaker feedback | `--live-speaker-weak-profile-min-margin` | 0.12 | Minimum top-vs-runner-up margin for weak-profile live-speaker assist. |
| Live speaker feedback | `--live-speaker-weak-profile-max-unknown-probability` | 0.55 | Maximum UNKNOWN probability allowed for weak-profile live-speaker assist. |
| Live speaker feedback | `--section-gap-new-speaker`<br>`--no-section-gap-new-speaker` | false | Allow a long media gap plus moderate similarity to create a new section speaker. |
| Live speaker feedback | `--section-gap-new-speaker-min-gap-seconds` | 60.0 | Minimum media-time gap since the matched speaker last ended before section-gap splitting. |
| Live speaker feedback | `--section-gap-new-speaker-min-prior-speech-seconds` | 8.0 | Minimum existing profile speech seconds required before section-gap splitting can clone it. |
| Live speaker feedback | `--section-gap-new-speaker-min-duration-seconds` | 5.0 | Minimum current sentence duration for section-gap new-speaker splitting. |
| Live speaker feedback | `--section-gap-new-speaker-min-similarity` | 0.35 | Minimum similarity to an old speaker for section-gap new-speaker splitting. |
| Live speaker feedback | `--section-gap-new-speaker-max-similarity` | 0.58 | Maximum similarity to an old speaker before section-gap splitting treats it as the same speaker. |
| Live speaker feedback | `--section-gap-new-speaker-min-margin` | 0.08 | Minimum top-vs-runner-up margin for section-gap new-speaker splitting. |
| Live speaker feedback | `--unknown-pair-new-speaker`<br>`--no-unknown-pair-new-speaker` | false | Create a new speaker when a recent UNKNOWN sentence pairs with a longer weak existing-speaker match. |
| Live speaker feedback | `--unknown-pair-new-speaker-max-gap-seconds` | 4.0 | Maximum gap between a pending UNKNOWN sentence and the current sentence for pair-based new-speaker creation. |
| Live speaker feedback | `--unknown-pair-new-speaker-min-unknown-duration-seconds` | 0.2 | Minimum duration of the pending UNKNOWN sentence used for pair-based new-speaker creation. |
| Live speaker feedback | `--unknown-pair-new-speaker-min-current-duration-seconds` | 2.5 | Minimum current sentence duration for pair-based new-speaker creation. |
| Live speaker feedback | `--unknown-pair-new-speaker-min-pair-similarity` | 0.45 | Minimum embedding similarity between UNKNOWN and current sentence for pair-based new-speaker creation. |
| Live speaker feedback | `--unknown-pair-new-speaker-max-existing-similarity` | 0.55 | Maximum similarity to an existing speaker before pair-based new-speaker creation is suppressed. |
| Live speaker feedback | `--unknown-pair-new-speaker-max-existing-margin` | 0.2 | Maximum existing-speaker margin allowed for pair-based new-speaker creation. |
| Live speaker feedback | `--unknown-pair-new-speaker-min-unknown-probability` | 0.1 | Minimum UNKNOWN probability on the current sentence for pair-based new-speaker creation. |
| Live speaker feedback | `--live-speaker-raw-change-snap`<br>`--no-live-speaker-raw-change-snap` | true | Allow strong unsmoothed live probabilities to switch away from the active speaker before EMA catches up. |
| Live speaker feedback | `--live-speaker-raw-change-min-probability` | 0.7 | Minimum raw known-speaker probability required for a live speaker-change snap. |
| Live speaker feedback | `--live-speaker-raw-change-min-margin` | 0.25 | Minimum raw probability lead over the active speaker required for a live speaker-change snap. |
| Live speaker feedback | `--live-speaker-sentence-hint`<br>`--no-live-speaker-sentence-hint` | true | Let fresh final sentence assignments seed the visible live speaker when no stronger live tag is active. |
| Live speaker feedback | `--live-speaker-highlight-transcript`<br>`--no-live-speaker-highlight-transcript` | true | Allow realtime transcript rows to drive the speaker-list live highlight when no fallback live-speaker probe is active. |
| Live speaker feedback | `--live-speaker-highlight-transcript-max-lag-seconds` | -1.0 | Maximum playback lag after a realtime transcript row end for that row to drive the speaker-list live highlight; negative disables the limit. |
| Live speaker feedback | `--live-speaker-highlight-transcript-override-min-probability` | 1.1 | Minimum raw realtime transcript speaker probability needed to override an active fallback live-speaker highlight; values above 1 disable override. |
| Live speaker feedback | `--live-speaker-highlight-transcript-override-min-margin` | 0.0 | Minimum raw probability lead over UNKNOWN needed for transcript speaker highlight override. |
| Live speaker feedback | `--live-speaker-sentence-hint-override`<br>`--no-live-speaker-sentence-hint-override` | true | Allow fresh final sentence assignments to replace the current fallback live speaker. |
| Live speaker feedback | `--live-speaker-sentence-hint-max-lag-seconds` | 1.25 | Maximum playback lag after a final sentence end for emitting a live-speaker sentence hint. |
| Live speaker feedback | `--live-speaker-sentence-hint-new-speaker-max-lag-seconds` | 1.25 | Maximum playback lag for a newly created speaker's first live-speaker sentence hint. |
| Live speaker feedback | `--live-speaker-sentence-hint-new-speaker-hold-seconds` | -1.0 | Optional hold duration for newly created speaker sentence hints; negative uses the normal hint hold. |
| Live speaker feedback | `--live-speaker-sentence-hint-new-speaker-max-top-similarity` | 1.0 | Only emit delayed new-speaker sentence hints when the new profile's top existing-speaker similarity is at or below this value. |
| Live speaker feedback | `--live-speaker-sentence-hint-hold-seconds` | 0.3 | Browser hold duration for live-speaker sentence hints. |
| Live speaker feedback | `--live-speaker-sentence-hint-hold-through-sentence`<br>`--no-live-speaker-sentence-hint-hold-through-sentence` | false | When a final sentence is assigned before playback has passed its end, keep its live hint through that end plus the hint hold. |
| Live speaker feedback | `--live-speaker-sentence-hint-min-duration-seconds` | 0.0 | Minimum final sentence duration required before it may emit a live-speaker sentence hint. |
| Live speaker feedback | `--realtime-preview-engine-options-json` | empty | Extra JSON object merged into the RealtimeSTT Kroko engine options. |
| Validation and browser observation | `--keep-segment-audio` | off unless passed | Keep per-segment audio clips for inspection instead of deleting them after processing. |
| Validation and browser observation | `--validate-window-replay` | off unless passed | Run the window replay validation path instead of starting a normal browser session. |
| Validation and browser observation | `--validation-canonical` | `DEFAULT_CUNK_CANONICAL` | Canonical transcript/speaker JSON used as the validation target. |
| Validation and browser observation | `--validation-output` | `DEFAULT_VALIDATION_OUTPUT` | Path for the validation summary JSON. |
| Validation and browser observation | `--validation-trace-output` |  | Optional JSONL trace output written during validation. |
| Validation and browser observation | `--validation-replay-speed` | 1.0 | Playback speed multiplier for validation replay. |
| Validation and browser observation | `--validation-update-interval-seconds` | 0.1 | How often validation replay advances and publishes state. |
| Validation and browser observation | `--validation-final-wait-seconds` | 90.0 | Maximum time to wait for final sentences after replay ends. |
| Validation and browser observation | `--validation-match-mode` | `auto` | How validation maps output rows to canonical speakers: automatic, timestamp, or text. Choices: `auto`, `timestamp`, `text`. |
| Validation and browser observation | `--browser-live-observation-output` |  | When set, the browser samples the rendered live-speaker DOM state and writes a strict browser-observed score JSON here. |
| Validation and browser observation | `--browser-live-observation-interval-seconds` | `DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS` | Seconds between browser DOM live-speaker samples. |
| Validation and browser observation | `--browser-live-observation-max-sample-gap-seconds` | `DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS` | Maximum playback span represented by one browser DOM sample interval. |
| Validation and browser observation | `--browser-live-observation-flicker-gap-seconds` | `DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS` | Minimum in-turn live-speaker gap counted as visible flicker. |
| Validation and browser observation | `--validation-keep-preview` | off unless passed | Keep realtime preview enabled during validation. Final sentence metrics usually do not need this. |

### `whospeaks-realtime`

Legacy realtime capture and speaker diarization command.

| Area | Parameter | Default | What it does |
| --- | --- | --- | --- |
| Server and capture | `--host` | `127.0.0.1` | Interface address the local HTTP server binds to. |
| Server and capture | `--port` | 8767 | TCP port for the local server or evaluation target. |
| Server and capture | `--no-browser` | off unless passed | Start the server without opening a browser tab. |
| Server and capture | `--input-device-index` |  | Audio input device index to capture from. |
| Server and capture | `--allow-default-input` | off unless passed | Allow the system default microphone/input when no matching loopback device is found. |
| ASR and realtime recorder | `--model` | `large-v2` | Final ASR model used for committed transcription. |
| ASR and realtime recorder | `--rt-model` | `tiny.en` | Realtime preview ASR model used by the legacy realtime path. |
| ASR and realtime recorder | `--language` | `default_language_code()` | Language code shared by ASR, preview model selection, and sentence splitting. |
| ASR and realtime recorder | `--device` | `cuda` | Device for local ASR or embedding execution, such as `cuda`, `cpu`, or `auto`. |
| ASR and realtime recorder | `--compute-type` | `float16` | faster-whisper numeric precision, such as `float16` or `int8`. |
| ASR and realtime recorder | `--download-root` |  | Model cache root used by faster-whisper or preview model downloads. |
| ASR and realtime recorder | `--split-marks` | `off` | Controls visible markers inserted around realtime split boundaries. |
| ASR and realtime recorder | `--realtime-processing-pause` | 0.1 | Pause between realtime processing iterations. |
| ASR and realtime recorder | `--post-speech-silence-duration` | 1.25 | Silence duration after speech before the realtime recorder closes a phrase. |
| ASR and realtime recorder | `--stop-trailing-silence-seconds` | 2.0 | Trailing silence appended when stopping capture so the last phrase can finalize. |
| ASR and realtime recorder | `--stop-drain-seconds` | 25.0 | Maximum time to wait for final realtime work during shutdown. |
| ASR and realtime recorder | `--stop-embedding-drain-seconds` | 10.0 | Maximum time to wait for embedding work during shutdown. |
| ASR and realtime recorder | `--final-video-latency-seconds` | 0.8 | Offset used to align final transcript timing with video playback latency. |
| ASR and realtime recorder | `--min-length-of-recording` | 0.0 | Minimum phrase recording length accepted by the realtime recorder. |
| ASR and realtime recorder | `--silero-sensitivity` | 0.05 | Silero VAD sensitivity for the legacy realtime recorder; lower values are stricter. |
| ASR and realtime recorder | `--webrtc-sensitivity` | 3 | WebRTC VAD aggressiveness for the legacy realtime recorder, usually 0-3. |
| ASR and realtime recorder | `--beam-size` | 5 | Beam size for final ASR; higher can improve text at the cost of speed. |
| ASR and realtime recorder | `--beam-size-realtime` | 1 | Beam size for realtime preview decoding. |
| ASR and realtime recorder | `--batch-size` | 0 | RealtimeSTT final decoder batch size. 0 uses the regular decoder; RealtimeSTT's default 16 is faster but dropped words on the Cunk clip. |
| ASR and realtime recorder | `--realtime-batch-size` | 0 | RealtimeSTT preview decoder batch size. 0 uses the regular decoder. |
| Transcript splitting | `--no-final-word-timestamps` | on unless passed | Disable word timestamps in final realtime transcripts. |
| Transcript splitting | `--no-split-final-transcripts` | on unless passed | Keep each final realtime transcript as one segment instead of splitting it into sentences. |
| Transcript splitting | `--word-split-gap-seconds` | 0.0 | Word gap that can force a split between sentence fragments. |
| Transcript splitting | `--max-timestamp-split-seconds` | 0.0 | Maximum segment duration before timestamp-based splitting is allowed; 0 disables it. |
| Transcript splitting | `--max-word-timestamp-seconds` | 1.2 | Maximum single-word timestamp duration accepted before split heuristics intervene. |
| Transcript splitting | `--min-timestamp-split-words` | 1 | Minimum word count required for timestamp-based splitting. |
| Transcript splitting | `--split-audio-padding-seconds` | 0.0 | Audio padding kept around split realtime segments. |
| Transcript splitting | `--sentence-boundary-pre-padding-seconds` | `DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS` | Audio kept before the next word when cutting between two sentence groups. |
| Transcript splitting | `--sentence-boundary-post-padding-seconds` | `DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS` | Audio kept after the last word when cutting between two sentence groups. |
| Transcript splitting | `--sentence-boundary-gap-ratio` | `DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO` | For tight word gaps, fraction of the gap assigned to the previous sentence. |
| Transcript splitting | `--split-on-soft-punctuation` | off unless passed | Allow commas and other soft punctuation to create realtime splits. |
| Transcript splitting | `--allow-non-sentence-live-splits` | on unless passed | Allow legacy realtime preview to split on fragments that are not complete sentences. Internal switch hidden from `--help`. |
| Speaker assignment | `--embedding-provider` | `DEFAULT_EMBEDDING_PROVIDER` | Provider or weighted provider stack used for final speaker assignment. |
| Speaker assignment | `--embedding-python` | `default_embedding_python()` | Python executable used by the local embedding helper process. |
| Speaker assignment | `--embedding-device` | `cuda` | Device used by local speaker embedding models. |
| Speaker assignment | `--same-speaker-similarity` | 0.45 | Similarity needed before a sentence can update or reuse an existing speaker. |
| Speaker assignment | `--similarity-temperature` | 0.07 | Softmax temperature for similarity scores; lower values make the best match dominate. |
| Speaker assignment | `--speaker-softmax-temperature` | 0.075 | Temperature used when converting speaker similarities into probabilities. |
| Speaker assignment | `--new-speaker-threshold` | 0.58 | Unknown-speaker probability needed before creating a new speaker. |
| Speaker assignment | `--duplicate-profile-similarity` | 0.4 | Similarity above which a candidate new profile is treated as an existing speaker duplicate. |
| Speaker assignment | `--unknown-short-threshold` | 0.86 | Unknown probability above which short uncertain fragments stay UNKNOWN. |
| Speaker assignment | `--min-first-speaker-seconds` | 1.2 | Minimum speech duration required before creating the first speaker profile. |
| Speaker assignment | `--min-new-speaker-seconds` | 2.0 | Minimum sentence duration required before creating an additional speaker. |
| Speaker assignment | `--late-new-speaker-min-seconds` | 3.5 | Longer duration required for late new-speaker creation after profiles already exist. |
| Speaker assignment | `--min-embed-seconds` | 0.5 | Minimum audio duration before creating a speaker embedding. |
| Speaker assignment | `--max-speakers` | 10 | Maximum number of speaker profiles the run may create automatically. |
| Speaker assignment | `--min-margin` | 0.05 | Minimum lead the best speaker match needs over the runner-up. |
| Speaker assignment | `--margin-temperature` | 0.035 | Softmax temperature for margin-based speaker confidence. |
| Speaker assignment | `--update-unknown-max` | 0.55 | Maximum UNKNOWN probability allowed when updating an existing speaker profile. |
| Speaker assignment | `--no-reassign-uncertain-sentences` | on unless passed | Disable the pass that reassigns short uncertain sentences after more context is available. |
| Speaker assignment | `--reassign-max-seconds` | 2.2 | Maximum duration of an uncertain sentence eligible for reassignment. |
| Speaker assignment | `--reassign-unknown-min` | 0.7 | Minimum UNKNOWN probability for a sentence to be considered for reassignment. |
| Speaker assignment | `--reassign-unknown-max` | 0.82 | Maximum UNKNOWN probability still eligible for reassignment. |
| Speaker assignment | `--reassign-min-similarity` | 0.42 | Minimum speaker similarity needed during uncertain-sentence reassignment. |
| Speaker assignment | `--reassign-short-max-seconds` | 1.2 | Maximum duration of the stricter short-fragment reassignment path. |
| Speaker assignment | `--reassign-short-min-similarity` | 0.3 | Minimum similarity for short-fragment reassignment. |
| Speaker assignment | `--reassign-short-min-margin` | 0.1 | Minimum winner margin for short-fragment reassignment. |
| Speaker assignment | `--no-context-assign-short-fragments` | on unless passed | Disable nearby-speaker context assignment for very short uncertain fragments. |
| Speaker assignment | `--context-assign-max-seconds` | 1.0 | Maximum duration of a short fragment eligible for nearby-speaker context assignment. |
| Speaker assignment | `--context-assign-candidate-unknown-min` | 0.9 | UNKNOWN probability required before a fragment becomes a context-assignment candidate. |
| Speaker assignment | `--context-assign-window` | 4 | Number of nearby stable sentences considered on each side. |
| Speaker assignment | `--context-assign-stable-unknown-max` | 0.7 | Maximum UNKNOWN probability for neighboring sentences to count as stable. |
| Speaker assignment | `--context-assign-stable-min-seconds` | 0.5 | Minimum duration for neighboring sentences to count as stable. |
| Speaker assignment | `--context-assign-same-speaker-confidence` | 0.92 | Confidence needed when both sides suggest the same speaker. |
| Speaker assignment | `--context-assign-disagree-confidence` | 0.78 | Confidence needed when neighboring speakers disagree. |
| Speaker assignment | `--context-assign-disagree-min-similarity` | 0.28 | Minimum similarity used when resolving disagreeing neighbors. |
| Speaker assignment | `--context-assign-disagree-margin` | 0.08 | Minimum margin used when resolving disagreeing neighbors. |
| Speaker assignment | `--context-assign-one-sided-confidence` | 0.82 | Confidence needed when only one side has stable speaker context. |
| Speaker assignment | `--context-assign-one-sided-block-margin` | 0.12 | Margin that blocks one-sided context assignment when evidence is ambiguous. |
| Speaker assignment | `--no-context-assign-one-sided` | on unless passed | Only context-assign short fragments when stable speakers on both sides agree. |
| Speaker assignment | `--segment-audio-dir` | `CACHE_DIR / 'realtime_speakerdiarize_segments'` | Directory where kept realtime segment audio clips are written. |
| Speaker assignment | `--keep-segment-audio` | off unless passed | Keep per-segment audio clips for inspection instead of deleting them after processing. |
| Traces | `--trace-log` |  | Optional JSONL trace path for backend/frontend UI events. |
| Traces | `--analyze-trace` |  | Analyze a previously captured JSONL trace and exit. |
| Traces | `--trace-analysis-output` |  | Write structured trace-vs-canonical analysis JSON when --analyze-trace is used. |
| Traces | `--trace-summary-only` | off unless passed | When analyzing a trace, print only the summary instead of verbose row details. |
| Traces | `--trace-match-mode` | `auto` | How --analyze-trace maps trace rows to canonical speakers. Choices: `auto`, `timestamp`, `text`. |
| Traces | `--trace-session` | `latest` | Which session to analyze from a multi-session trace: latest, all, or an explicit session id. Defaults to latest. |
| Validation and replay | `--validate-cunk` | off unless passed | Run the Cunk fixture validation path. |
| Validation and replay | `--validate-cunk-word-splits` | off unless passed | Validate word-split behavior on the Cunk fixture. |
| Validation and replay | `--validate-cunk-realtime-replay` | off unless passed | Replay the Cunk fixture through the realtime pipeline. |
| Validation and replay | `--validation-audio` | `OUTPUTS_DIR / 'pyannote-cunk' / 'cunk_on_earth_clip.mp3'` | Audio file used by realtime validation. |
| Validation and replay | `--validation-canonical` | `CUNK_CANONICAL` | Canonical transcript/speaker JSON used as the validation target. |
| Validation and replay | `--validation-output` | `REALTIME_VALIDATION_OUTPUT_DIR / 'latest.json'` | Path for the validation summary JSON. |
| Validation and replay | `--mixed-overlap-min-seconds` | 0.05 | Minimum overlap counted as a mixed-speaker overlap during validation. |
| Validation and replay | `--replay-speed` | 8.0 | Replay speed multiplier; higher values feed audio faster than realtime. |
| Validation and replay | `--replay-chunk-seconds` | 0.1 | Audio chunk duration fed to the replay loop each step. |
| Validation and replay | `--replay-trailing-silence-seconds` | 2.0 | Silence appended after replay audio so pending speech can finalize. |
| Validation and replay | `--replay-drain-seconds` | 25.0 | Maximum time to wait for queued realtime work after replay input ends. |
| Validation and replay | `--replay-embedding-drain-seconds` | 15.0 | Maximum time to wait for queued embedding work after replay input ends. |
| Validation and replay | `--no-replay-sleep` | on unless passed | Feed replay audio as fast as possible instead of wall-clock pacing. |
| Validation and replay | `--embedding-helper` | off unless passed | Run the realtime embedding helper subprocess instead of the main app. Internal switch hidden from `--help`. |

### `whospeaks-filefeed-replay`

Downloads or reuses a media file and feeds it through realtime replay validation.

| Parameter | Default | What it does |
| --- | --- | --- |
| `--url` | `DEFAULT_URL` | Media URL to download or play; defaults to the demo clip. |
| `--work-dir` | `DEFAULT_WORK_DIR` | Scratch directory for downloads and intermediate files. |
| `--output-dir` | `DEFAULT_OUTPUT_DIR` | Directory for run outputs such as media, traces, validation JSON, or reports. |
| `--audio-file` |  | Use this existing audio file instead of downloading audio. |
| `--video-file` |  | Use this existing video file for browser playback instead of downloading video. |
| `--download-video` | off unless passed | Download video as well as audio for replay workflows. |
| `--skip-download` | off unless passed | Reuse existing local files and do not call `yt-dlp`. |
| `--yt-dlp` |  | Path to the `yt-dlp` executable. |
| `--python` | `Path(sys.executable)` | Python executable used to launch the downstream replay command. |
| `--model` | `default_realtimestt_model()` | Final ASR model used for committed transcription. |
| `--rt-model` | `default_realtimestt_rt_model()` | Realtime preview ASR model used by the legacy realtime path. |
| `--download-root` | `default_download_root()` | Model cache root used by faster-whisper or preview model downloads. |
| `--embedding-python` | `DEFAULT_EMBEDDING_PYTHON` | Python executable used by the local embedding helper process. |
| `--validation-canonical` | `DEFAULT_CANONICAL` | Canonical transcript/speaker JSON used as the validation target. |
| `--trace-log` |  | JSONL trace path to read or write, depending on the command. |
| `--validation-output` |  | Path for the validation summary JSON. |
| `--replay-speed` | 1.0 | Replay speed multiplier; higher values feed audio faster than realtime. |
| `--replay-chunk-seconds` | 0.1 | Audio chunk duration fed to the replay loop each step. |
| `--replay-trailing-silence-seconds` | 2.0 | Silence appended after replay audio so pending speech can finalize. |
| `--replay-drain-seconds` | 25.0 | Maximum time to wait for queued realtime work after replay input ends. |
| `--replay-embedding-drain-seconds` | 15.0 | Maximum time to wait for queued embedding work after replay input ends. |
| `--no-replay-sleep` | off unless passed | Disables replay sleep. |
| `--no-run` | off unless passed | Prepare/download files and print the downstream command without running it. |

### `whospeaks-embedding-benchmark`

Benchmarks local speaker embedding engines with one audio sample.

| Parameter | Default | What it does |
| --- | --- | --- |
| `--child` | off unless passed | Run one benchmark engine as a child process; normally used by the benchmark driver. |
| `--engine` |  | One embedding engine/provider to benchmark in child mode. |
| `--engines` |  | Specific list of benchmark engines/providers to run; omitted means the default set. |
| `--audio` | `str(DEFAULT_AUDIO)` | Audio file used as benchmark input. |
| `--output` | `str(OUT_DIR / 'child_result.json')` | Output JSON path for benchmark, scoring, or evaluation results. |
| `--device` | `auto` | Device for local ASR or embedding execution, such as `cuda`, `cpu`, or `auto`. |
| `--repeats` | 5 | Number of repeated benchmark runs per engine. |
| `--timeout-seconds` | 900 | Maximum seconds to wait before treating the operation as failed. |

### `whospeaks-browser-live-eval`

Runs browser-observed live-speaker evaluation against a local window server.

| Parameter | Default | What it does |
| --- | --- | --- |
| `--port` | 8796 | TCP port for the local server or evaluation target. |
| `--output` | `ROOT / 'runtime' / 'validation' / 'browser-live-speaker-observed.json'` | Output JSON path for benchmark, scoring, or evaluation results. |
| `--server-log` | `ROOT / 'runtime' / 'validation' / 'browser-live-speaker-server.log'` | Log file captured from the server during browser live-speaker evaluation. |
| `--timeout-seconds` | 480.0 | Maximum seconds to wait before treating the operation as failed. |
| `--headless` | off unless passed | Run browser evaluation without showing a browser window. |

## Repository Helper Modules

These modules are not installed as top-level commands, but they define parameters and are used by development, scoring, corpus-building, or worker workflows.

### `src/embeddings/build_sentence_embedding_corpus.py`

| Parameter | Default | What it does |
| --- | --- | --- |
| `--input-root` | `DEFAULT_INPUT_ROOT` | Root directory containing canonical input transcripts or sentence audio. |
| `--output-root` | `DEFAULT_OUTPUT_ROOT` | Root directory where generated corpus embeddings are written. |
| `--backend` | `remote` | Choose whether corpus embeddings are computed locally or by the remote embeddings server. Choices: `local`, `remote`. |
| `--remote-embeddings-url` | `DEFAULT_REMOTE_EMBEDDINGS_URL` | Remote embeddings server URL used for corpus embedding requests. |
| `--device` | `cuda` | Device for local ASR or embedding execution, such as `cuda`, `cpu`, or `auto`. |
| `--timeout-seconds` | 600.0 | Maximum seconds to wait before treating the operation as failed. |
| `--min-slice-seconds` | 0.1 | Minimum slice seconds used by the pipeline. |
| `--max-embed-chunk-seconds` | 0.0 | Split longer segment audio into fixed windows and average embeddings. Zero disables chunking. |
| `--providers` | empty | Comma-separated provider ids. Empty means all remote providers. |
| `--max-videos` | 0 | Optional smoke-test limit. Zero means all videos. |
| `--stop-on-provider-error` | off unless passed | Stop the batch when a provider fails instead of recording the error and continuing. |

### `src/embeddings/build_live_sentence_embedding_corpus.py`

| Parameter | Default | What it does |
| --- | --- | --- |
| `--input-root` | `DEFAULT_INPUT_ROOT` | Root directory containing canonical input transcripts or sentence audio. |
| `--output-root` | `DEFAULT_OUTPUT_ROOT` | Root directory where generated corpus embeddings are written. |
| `--remote-embeddings-url` | `DEFAULT_REMOTE_EMBEDDINGS_URL` | Remote embeddings server URL used for live-sentence corpus embedding requests. |
| `--device` | `cuda` | Device for local ASR or embedding execution, such as `cuda`, `cpu`, or `auto`. |
| `--timeout-seconds` | 300.0 | Per remote request timeout. Failed providers are recorded and skipped. |
| `--providers` | empty | Comma-separated provider ids. Empty means all providers reported by the remote server. |
| `--max-videos` | 0 | Optional smoke-test limit. |
| `--max-providers` | 0 | Optional smoke-test limit. |
| `--stop-on-provider-error` | off unless passed | Stop the batch when a provider fails instead of recording the error and continuing. |

### `src/window/browser_live_speaker_scoring.py`

| Parameter | Default | What it does |
| --- | --- | --- |
| `--observations` |  | Browser observation JSON used by scoring helpers. |
| `--canonical` |  | Canonical diarization JSON used by scoring helpers. |
| `--output` |  | Output JSON path for benchmark, scoring, or evaluation results. |
| `--max-sample-gap-seconds` | `DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS` | Largest observation sample gap counted as continuous evidence. |
| `--flicker-gap-seconds` | `DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS` | Smallest visible speaker gap counted as flicker. |

### `src/window/live_speaker_probe_scoring.py`

| Parameter | Default | What it does |
| --- | --- | --- |
| `--trace` |  | Trace JSONL file used by live-speaker probe scoring. |
| `--canonical` |  | Canonical diarization JSON used by scoring helpers. |
| `--output` |  | Output JSON path for benchmark, scoring, or evaluation results. |
| `--latency-grace-seconds` | `DEFAULT_LATENCY_GRACE_SECONDS` | Correct live assignments within this many seconds of a turn start receive full latency credit. |
| `--latency-cap-seconds` | `DEFAULT_LATENCY_CAP_SECONDS` | Latency at or beyond this many seconds after the grace window receives zero latency credit. |
| `--turn-merge-gap-seconds` | `DEFAULT_TURN_MERGE_GAP_SECONDS` | Merge adjacent same-speaker canonical segments separated by at most this gap before latency scoring. |
| `--unknown-clear-debounce-seconds` | 0.0 | Delay UNKNOWN live_speaker_clear handling by this many seconds when reconstructing visible active speaker state. |

### `src/window/fact_lens_sidecar.py`

| Parameter | Default | What it does |
| --- | --- | --- |
| `--source-url` | `DEFAULT_SOURCE_URL` | Event source URL for the fact-lens sidecar. |
| `--host` | `DEFAULT_DASHBOARD_HOST` | Interface address the local HTTP server binds to. |
| `--port` | `DEFAULT_DASHBOARD_PORT` | TCP port for the local server or evaluation target. |
| `--enable-llm` | off unless passed | Enable LLM claim extraction. Disabled by default; without this, the sidecar only displays final transcript sentences. |
| `--llm-base-url` | `DEFAULT_LLM_BASE_URL` | OpenAI-compatible base URL used by the fact-lens sidecar. |
| `--llm-model` | `DEFAULT_LLM_MODEL` | Model name requested by the fact-lens sidecar. |
| `--llm-client` | `whospeaks-fact-lens` | LLM client implementation selected by the fact-lens sidecar. |
| `--llm-lane` | `shared` | Processing lane name used to separate fact-lens requests. |
| `--llm-timeout` | 12.0 | Maximum seconds to wait for one fact-lens LLM request. |
| `--llm-max-tokens` | 768 | Maximum response tokens requested from the fact-lens LLM. |
| `--schema-mode` | `both` | How strictly the fact-lens sidecar asks the LLM for structured output. Choices: `json_schema`, `response_format`, `both`. |
| `--debounce-seconds` | 0.35 | Delay used to merge rapid transcript events before fact-lens processing. |
| `--context-size` | 8 | Number of recent transcript items sent as context to fact-lens. |
| `--queue-size` | 32 | Maximum queued fact-lens items before older work is dropped. |
| `--max-sentences` | 80 | Maximum sentences retained by the fact-lens sidecar. |
| `--max-cards` | 80 | Maximum claim/fact cards retained by the fact-lens sidecar. |
| `--sse-timeout` | 30.0 | Read timeout for server-sent events. |
| `--reconnect-seconds` | 2.0 | Delay before reconnecting a dropped event stream. |
| `--offline-demo` | off unless passed | Run fact-lens with local demo events instead of a live source URL. |
| `--offline-interval-seconds` | 2.0 | Seconds between offline demo events. |
| `--mock-llm` | off unless passed | Use deterministic mock LLM responses for fact-lens development. |
| `--quiet` | off unless passed | Reduce fact-lens console output. |

### `src/workers/kroko_realtime_preview_worker.py`

| Parameter | Default | What it does |
| --- | --- | --- |
| `--engine` | `kroko_onnx` | Realtime preview engine implementation loaded by the worker. |
| `--model` | `Kroko-EN-Community-64-L-Streaming-001.data` | Kroko/Banafo streaming model name used by the preview worker. |
| `--model-path` | empty | Exact model file path for the Kroko preview worker. |
| `--download-root` | empty | Directory searched or used for Kroko preview model downloads. |
| `--provider` | `cpu` | Execution provider selected for the Kroko preview worker. |
| `--num-threads` | 2 | CPU thread count for the Kroko preview worker. |
| `--language` | `en` | Language code shared by ASR, preview model selection, and sentence splitting. |
| `--engine-options-json` | empty | JSON object passed through as extra realtime preview engine options. |
| `--realtimestt-root` | empty | RealtimeSTT root used by the Kroko preview worker. |

## Environment Variables

Environment variables set defaults before command-line arguments are parsed. A CLI flag usually wins for that run.

| Variable | What it does |
| --- | --- |
| `APPDATA` | Windows base directory used for the default starter profile when `WHOSPEAKS_CONFIG` is unset. |
| `ESPNET_MODEL_ZOO_CACHE` | Cache directory for ESPnet model-zoo downloads used by embedding benchmarks. |
| `HF_ACCESS_TOKEN` | Hugging Face token used for gated speaker embedding models; interchangeable with `HF_TOKEN`. |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | Hugging Face Hub warning toggle; the repo defaults it to `1` to reduce noisy Windows cache warnings. |
| `HF_HUB_OFFLINE` | Forces Hugging Face Hub offline mode when set; embedding providers default it to offline when no token is available. |
| `HF_TOKEN` | Hugging Face token used for gated speaker embedding models; checked before `HF_ACCESS_TOKEN` in provider code. |
| `HUGGINGFACE_TOKEN` | Fallback Hugging Face token name for gated model access. |
| `MKL_NUM_THREADS` | MKL CPU thread count; embedding helpers default it to `1` for predictable local model behavior. |
| `NLTK_DATA` | Tokenizer data path; the app sets it to the runtime NLTK cache if you do not set it. |
| `OMP_NUM_THREADS` | OpenMP CPU thread count; embedding helpers default it to `1` for predictable local model behavior. |
| `PYTHONIOENCODING` | Python process output encoding; helpers default it to `utf-8`. |
| `REALTIMESTT_ROOT` | Root of the vendored or external RealtimeSTT checkout used for Kroko/Silero discovery. |
| `TOKENIZERS_PARALLELISM` | Hugging Face tokenizers parallelism toggle; helpers default it to `false` to avoid noisy warnings and thread spikes. |
| `TRANSFORMERS_OFFLINE` | Forces Transformers offline mode when set; embedding providers default it to offline when no token is available. |
| `WHOSPEAKS_ASR_LANGUAGE` | Fallback language for ASR when `--language` and `WHOSPEAKS_LANGUAGE` are not set. |
| `WHOSPEAKS_CACHE_DIR` | Mutable cache root for downloaded models and tokenizer data. |
| `WHOSPEAKS_CONFIG` | Exact path to the saved `whospeaks` starter profile JSON. |
| `WHOSPEAKS_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS` | Default timeout for local embedding helper replies. |
| `WHOSPEAKS_FAST_WHISPER_DOWNLOAD_ROOT` | Default faster-whisper model cache root for `--download-root`. |
| `WHOSPEAKS_KROKO_PREVIEW_AUTO_DOWNLOAD` | Default for preview model auto-download; use false/off/0 to require local files. |
| `WHOSPEAKS_KROKO_PREVIEW_MODEL_DIR` | Extra path list searched for Kroko preview model files. |
| `WHOSPEAKS_KROKO_PREVIEW_MODEL_PATH` | Exact Kroko preview `.data` model file to use by default. |
| `WHOSPEAKS_KROKO_PREVIEW_MODEL_PRESET` | Default Kroko preview preset such as `community-64l` or `pro-16l`. |
| `WHOSPEAKS_KROKO_PREVIEW_MODEL_REPO` | Hugging Face repo used when public Kroko preview models are auto-downloaded. |
| `WHOSPEAKS_KROKO_PREVIEW_PYTHON` | Python executable used for the Kroko realtime preview worker. |
| `WHOSPEAKS_LANGUAGE` | Default language for final ASR, realtime preview model selection, and sentence splitting. |
| `WHOSPEAKS_MODAL_ASR_MODEL` | Modal ASR server model name. |
| `WHOSPEAKS_MODAL_EMBEDDING_PROVIDER` | Embedding provider stack used by the Modal window deployment. |
| `WHOSPEAKS_MODAL_EXTRA_ARGS` | Additional command-line arguments appended inside the Modal window deployment. |
| `WHOSPEAKS_MODAL_GPU` | GPU type requested by the Modal deployment. |
| `WHOSPEAKS_MODAL_LANGUAGE` | Language passed to the Modal window deployment. |
| `WHOSPEAKS_MODAL_SCALEDOWN_WINDOW_SECONDS` | Idle scale-down window for the Modal deployment. |
| `WHOSPEAKS_MI_LLM_API_KEY` | Generic API key fallback for the meeting intelligence server. |
| `WHOSPEAKS_MI_LLM_BASE_URL` | Default OpenAI-compatible base URL for the meeting intelligence server. |
| `WHOSPEAKS_MI_LLM_MODEL` | Default model name for the meeting intelligence server. |
| `OPENAI_API_KEY` | API key used by the meeting intelligence `openai` provider; can come from the process environment or the configured `.env` file. |
| `OPENROUTER_API_KEY` | API key used by the meeting intelligence `openrouter` provider; can come from the process environment or the configured `.env` file. |
| `WHOSPEAKS_MODEL_DIR` | Mutable model root under the runtime directory. |
| `WHOSPEAKS_PROJECT_ROOT` | Override for the repository root used to resolve vendor, tests, and runtime paths. |
| `WHOSPEAKS_REMOTE_ASR_URL` | Default remote ASR server URL. |
| `WHOSPEAKS_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS` | Default timeout for remote embedding health/load/embed requests. |
| `WHOSPEAKS_REMOTE_EMBEDDINGS_URL` | Default remote embeddings server URL. |
| `WHOSPEAKS_RUNTIME_DIR` | Top-level mutable runtime directory for media, caches, outputs, and speakers. |
| `WHOSPEAKS_SENTENCE_TOKENIZER` | Default sentence tokenizer override, for example `nltk+rule-based`. |
| `WHOSPEAKS_SILERO_VAD_ONNX_MODEL_PATH` | Default Silero ONNX VAD model path. |
| `WHOSPEAKS_SPEAKER_LIBRARY_DIR` | Directory for saved speaker groups and uploaded reference audio. |
| `XDG_CONFIG_HOME` | Non-Windows base directory for the default starter profile when `WHOSPEAKS_CONFIG` is unset. |

## Coverage Check

This reference was built from a repo-wide search for `argparse.add_argument(...)` and environment reads such as `os.environ.get(...)`. When adding a new user-facing parameter, update this page in the same change so `--help` and the docs stay aligned.

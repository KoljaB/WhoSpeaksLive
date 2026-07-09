# Meeting Intelligence Server

The meeting intelligence server turns saved speaker-labeled transcripts into an evidence-grounded meeting report in a separate browser UI.

Use this server when you want to review a finished transcript, generate a structured summary, inspect decisions and action items, and click evidence back into the transcript. It is separate from the main live window. The live window can store lightweight meeting intelligence data for saved sessions, while this server is the standalone report-generation and review surface for the multi-pass LLM report.

## What It Does

The server loads saved WhoSpeaksLive sessions, or an explicit demo transcript, and runs a multi-step report pipeline:

1. Prepare transcript rows and speaker labels.
2. Split the transcript into manageable segments.
3. Extract evidence anchors from each segment.
4. Generate report sections from those evidence anchors.
5. Save the report in a local cache.
6. Show progress, report sections, evidence, and transcript rows in the browser.

An evidence anchor is a small auditable support span. Each anchor stores the transcript row IDs that support a claim. In the browser, clicking an evidence chip opens the Transcript tab, scrolls to the matching rows, and highlights them.

## Start With A Mock LLM

Mock mode is deterministic and does not contact a model server. Use it to verify that the browser UI, session loading, cache writing, report deletion, and evidence links work.

```powershell
whospeaks-meeting-intelligence `
  --host 127.0.0.1 `
  --port 8798 `
  --demo-transcript D:\path\to\whospeakslive_transcript.txt `
  --mock-llm
```

Open:

```text
http://127.0.0.1:8798/
```

Select a session and click `Generate report`.

## Start With llama.cpp Or Another OpenAI-Compatible Server

The real report pipeline uses an OpenAI-compatible `/chat/completions` endpoint. LLM means large language model. The meeting intelligence server does not discover a model automatically; pass the provider, base URL, and model name at startup.

Example with a local or tunneled llama.cpp server:

```powershell
whospeaks-meeting-intelligence `
  --host 127.0.0.1 `
  --port 8798 `
  --demo-transcript D:\path\to\whospeakslive_transcript.txt `
  --llm-provider llama_cpp `
  --llm-base-url http://127.0.0.1:18081/v1 `
  --llm-model gemma-4-12b-it-Q6_K.gguf `
  --max-tokens 2048 `
  --section-max-tokens 4096 `
  --max-segment-rows 80
```

The base URL must not include `/chat/completions`; the client appends that path.

You can also run it as a Python module from the repository:

```powershell
python -m window.meeting_intelligence_server `
  --host 127.0.0.1 `
  --port 8798 `
  --llm-provider llama_cpp `
  --llm-base-url http://127.0.0.1:18081/v1 `
  --llm-model gemma-4-12b-it-Q6_K.gguf
```

## LLM Providers

Supported provider values:

| Provider | Default base URL | Default model | Notes |
| --- | --- | --- | --- |
| `llama_cpp` | `http://127.0.0.1:8081/v1` | `local` | Good for local or SSH-tunneled llama.cpp servers. |
| `ollama` | `http://127.0.0.1:11434/v1` | `gemma3` | Uses Ollama's OpenAI-compatible endpoint. |
| `lm_studio` | `http://127.0.0.1:1234/v1` | `local-model` | Uses LM Studio's local OpenAI-compatible server. |
| `openai` | `https://api.openai.com/v1` | `gpt-4.1-mini` | Requires an API key. |
| `openrouter` | `https://openrouter.ai/api/v1` | `google/gemma-3-12b-it` | Requires an API key. |

Command-line flags override provider defaults:

```powershell
--llm-base-url http://127.0.0.1:18081/v1
--llm-model gemma-4-12b-it-Q6_K.gguf
--llm-api-key <token>
```

Environment variables can also set defaults:

| Variable | What it does |
| --- | --- |
| `WHOSPEAKS_MI_LLM_BASE_URL` | Default meeting-intelligence LLM base URL. |
| `WHOSPEAKS_MI_LLM_MODEL` | Default meeting-intelligence model name. |
| `WHOSPEAKS_MI_LLM_API_KEY` | Generic API key fallback for providers that need authentication. |
| `OPENAI_API_KEY` | API key used by the `openai` provider. |
| `OPENROUTER_API_KEY` | API key used by the `openrouter` provider. |

## Inputs And Caches

By default, the server reads saved sessions from the normal WhoSpeaksLive session directory. You can override this:

```powershell
--session-dir D:\path\to\sessions
```

To add a transcript-only demo session, pass:

```powershell
--demo-transcript D:\path\to\whospeakslive_transcript.txt
```

For the private repository demo data used during local development, this may point to `docs-private\demo-meeting\whospeakslive_transcript.txt`.

Generated reports are cached locally. Override the cache directory when you want an isolated test cache:

```powershell
--cache-dir D:\Projekte\SpeakerDiarization\runtime\meeting_intelligence_reports_browser
```

The cache is provider-aware. A cached report generated with `mock_meeting_llm` is not treated as valid when the server is currently configured for `llama_cpp:gemma-4-12b-it-Q6_K.gguf`.

## Browser Workflow

1. Open `http://127.0.0.1:8798/`.
2. Select a session in the left sidebar.
3. Click `Generate report`.
4. Watch the progress panel while evidence and sections are generated.
5. Review Summary, Decisions, Action items, Questions, Risks, Evidence, and Transcript tabs.
6. Click evidence chips to jump to highlighted transcript rows.
7. Click `Delete`, then `Confirm`, to delete only the cached report for that session.
8. Click `Generate report` again to recreate it.

Deleting a report does not delete the transcript, speakers, embeddings, saved session, or media files. It removes only the cached meeting intelligence report.

## API Endpoints

The browser uses these local endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/config` | `GET` | Return provider, model, base URL, and runtime config. |
| `/api/sessions` | `GET` | List demo and saved sessions. |
| `/api/report?session_id=...` | `GET` | Return the current cached report and transcript rows. |
| `/api/generate-async` | `POST` | Start background report generation. |
| `/api/generate-status?job_id=...` | `GET` | Poll report-generation progress. |
| `/api/generate` | `POST` | Synchronous generation endpoint kept for tests and simple clients. |
| `/api/delete-report` | `POST` | Delete the cached report for a session. |

`/api/generate-async` returns a job object. The browser polls `/api/generate-status` and updates the progress bar with stages such as `prepare`, `segment`, `evidence`, `section`, `finalize`, and `completed`.

## Troubleshooting

If the page says there is no current report, click `Generate report`. If the report was generated with a different provider or model, the server treats it as stale and will not display it as current.

If generation appears stuck, check the progress panel first. Real local LLM generation can take several minutes for larger transcripts, especially when many sections are enabled.

If the server cannot connect to the model, verify the base URL:

```powershell
curl.exe -sS http://127.0.0.1:18081/v1/models
```

If evidence opens the Transcript tab but does not highlight a row, regenerate the report. Older reports or sessions without stable row IDs can reference normalized row IDs; current browser builds resolve both stored and normalized IDs.

If you only want to test the UI, use `--mock-llm`. If you want a realistic report, use a real OpenAI-compatible server and enough context/output tokens for the selected model.

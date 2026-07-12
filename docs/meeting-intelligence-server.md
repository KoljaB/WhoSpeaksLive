# Meeting Intelligence Server

The meeting intelligence server turns saved speaker-labeled transcripts into an evidence-grounded meeting report in a separate browser UI.

Use this server when you want to review a finished transcript, generate a structured summary, inspect decisions and action items, and click evidence back into the transcript. It is separate from the main live window. The live window can store lightweight meeting intelligence data for saved sessions, while this server is the standalone report-generation and review surface for the multi-pass LLM report.

## What It Does

The server loads saved WhoSpeaksLive sessions, or an explicit demo transcript, and runs a multi-step report pipeline for the selected report template:

1. Prepare transcript rows and speaker labels.
2. Split the transcript into manageable segments.
3. Send every configured section objective and output-field definition to each evidence pass.
4. Extract evidence anchors from each segment and tag them with relevant section keys.
5. Generate each flat report section from its full definition, relevant evidence, and global context.
6. Save the report and its complete template snapshot in a local template-specific cache.
7. Show progress, report sections, evidence, and transcript rows in the browser.

An evidence anchor is a small auditable support span. Each anchor stores the transcript row IDs that support a claim. In the browser, clicking an evidence chip opens the Transcript tab, scrolls to the matching rows, and highlights them.

## Predefined And Custom Reports

The **Report template** selector works with ordinary JSON report templates. A template is a reusable report design containing an ordered, flat list of sections. Predefined and user-created Custom templates use the same validation, evidence extraction, section generation, and rendering path.

Use **Predefined** to select the supplied Standard Meeting Intelligence report and the ten domain examples for works councils, podcasts, medical case conferences, film production, incident response, mediation, investigative journalism, qualitative research, shift handover, and committee minutes. Click **Inspect** to open a Predefined report read-only, then **Clone** to create an editable Custom copy.

Use **New** to open **Report builder** and build a Custom template from scratch. Custom reports also offer **Edit** and **Delete**. Each section can configure its objective, item limit, evidence requirement, card/table/timeline/quote layout, relevance/chronological/severity sorting, and typed output fields. Reports have no nested subsection level. The builder's primary action is **Save template**.

See [Custom Reports](custom-reports.md) for the builder workflow, complete template schema, predefined report coverage, language and privacy policies, and MVP limitations.

## Start With A Mock LLM

Mock mode is deterministic and does not contact a model server. Use it to verify that the browser UI, session loading, cache writing, report deletion, and evidence links work.

```powershell
whospeaks-meeting-intelligence `
  --host 127.0.0.1 `
  --port 8798 `
  --demo-transcript D:\path\to\whospeakslive_transcript.txt `
  --mock-llm
```

## Spanish Executive-Meeting Reports

For a Spanish meeting, start the live window with `--language es` and start this server with `--report-language es`. The language setting is an explicit contract for all generated report content: evidence labels, executive summary, decisions, action items, questions, risks, and deadlines. It does not translate quoted transcript text.

```powershell
whospeaks-meeting-intelligence `
  --host 127.0.0.1 `
  --port 8798 `
  --report-language es `
  --auto-generate `
  --llm-provider llama_cpp `
  --llm-base-url http://127.0.0.1:18081/v1 `
  --llm-model YOUR_SPANISH_CAPABLE_MODEL
```

Reports generated in another language are treated as stale, so a previously cached English report cannot be shown as the current Spanish report. `--report-language` accepts every language code supported by WhoSpeaks final transcription, including `de`, `en`, `es`, `fr`, `it`, `pt`, and the broader Whisper-language set listed in [Configuration](configuration.md#language). The server resolves aliases and regional codes through the shared language configuration, so `German` and `de-AT` both select `de`.

A template whose language is set to **Inherit** uses `--report-language`. A fixed-language template overrides that server default; for example, the predefined French medical-case report always requests French output. The effective language is included in cache validation.

With `--auto-generate`, the server checks saved sessions every 10 seconds and queues a report as soon as the live window finalizes a new one with status `Saved`. Existing saved sessions are deliberately ignored when the server starts, so historical meetings are never unexpectedly sent to the LLM. No report-page selection or Generate click is needed. Change the cadence with `--auto-generate-poll-seconds`.

Automatic generation currently uses the Standard Meeting Intelligence template. Generate other Predefined or Custom reports explicitly in the browser or API.

The `whospeaks` launcher can start both services from one command. It inherits the report language from the live profile unless `--report-language` is supplied:

```powershell
whospeaks launch --with-reports --report-llm-provider openai --report-llm-model gpt-4.1-nano
```

Use `whospeaks reports --print` to inspect or run only the report-server command.

Open:

```text
http://127.0.0.1:8798/
```

Select a session and a Report template, then click `Generate report`.

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
| `openai` | `https://api.openai.com/v1` | `gpt-5.6-luna` | Requires an API key. The browser can load the models available to the configured account and select cheaper models such as `gpt-4.1-nano` when available. |
| `openrouter` | `https://openrouter.ai/api/v1` | `google/gemma-3-12b-it` | Requires an API key. |

Command-line flags override provider defaults:

```powershell
--llm-base-url http://127.0.0.1:18081/v1
--llm-model gemma-4-12b-it-Q6_K.gguf
--llm-api-key <token>
```

The startup provider is only the initial browser state. The browser sidebar also has provider, model, and base URL controls. Click `Models` to ask the selected provider's `/models` endpoint for account-visible model IDs, then click `Apply` to switch the runtime provider without restarting the server.

Switching provider or model does not rewrite old cached reports. The cache is provider-aware, so a report generated with `llama_cpp:gemma-4-12b-it-Q6_K.gguf` becomes stale when the runtime provider is `openai:gpt-4.1-nano`, and becomes current again if you switch back.

Environment variables can also set defaults:

| Variable | What it does |
| --- | --- |
| `WHOSPEAKS_MI_LLM_BASE_URL` | Default meeting-intelligence LLM base URL. |
| `WHOSPEAKS_MI_LLM_MODEL` | Default meeting-intelligence model name. |
| `WHOSPEAKS_MI_LLM_API_KEY` | Generic API key fallback for providers that need authentication. |
| `OPENAI_API_KEY` | API key used by the `openai` provider. |
| `OPENROUTER_API_KEY` | API key used by the `openrouter` provider. |

The server loads a local `.env` file by default before building the LLM config. This is a convenience for local development; process-level environment variables still win because `.env` values only fill missing keys. Keep real keys in the untracked `.env`, not in docs or committed files:

```env
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=
```

Use `--env-file D:\path\to\.env` to load a different file for one run.

## OpenAI Structured Output Compatibility

OpenAI structured output uses strict JSON schemas. A strict schema is a JSON Schema where every object has a closed property set and every declared property is listed as required. The server normalizes schemas sent through OpenAI `response_format` so future report sections do not fail because of nested open-ended objects.

This normalization is provider-path specific:

- OpenAI and OpenRouter receive the normalized strict `response_format` schema.
- llama.cpp, Ollama, and LM Studio keep the local OpenAI-compatible request shape and can still receive the extra `json_schema` field when configured with schema mode `both`.
- HTTP 400 errors include the provider's response body when available, so schema or model-capability failures are visible in the browser progress panel instead of only saying `Bad Request`.

## Inputs And Caches

By default, the server reads saved sessions from the normal WhoSpeaksLive session directory. You can override this:

```powershell
--session-dir D:\path\to\sessions
```

To add a transcript-only demo session, pass:

```powershell
--demo-transcript D:\path\to\whospeakslive_transcript.txt
```

Generated reports are cached locally. Override the cache directory when you want an isolated test cache:

```powershell
--cache-dir D:\Projekte\SpeakerDiarization\runtime\meeting_intelligence_reports_browser
```

Custom report templates are also saved locally. Override their directory separately:

```powershell
--template-dir D:\path\to\report_templates
```

The cache supports multiple reports for one session by using the session ID and template ID together. A Podcast Production report therefore does not overwrite a Standard Meeting Intelligence report made from the same transcript.

A cache is current only while transcript and speaker revision, provider and model, effective report language, template ID, and template revision all match. For example, a report generated with `mock_meeting_llm` is not current under `llama_cpp:gemma-4-12b-it-Q6_K.gguf`, and editing a Custom template makes its previous report stale. Switching back to matching settings can make an unchanged cache current again.

## Browser Workflow

1. Open `http://127.0.0.1:8798/`.
2. Select a session in the left sidebar.
3. Select a report design under **Predefined** or **Custom** in the **Report template** selector; its current cache for the selected session loads automatically.
4. Optionally use **Inspect** or **Clone** for a Predefined report, **Edit** or **Delete** for a Custom report, or **New** to open **Report builder**.
5. Click `Generate report`.
6. Watch the modal progress overlay while template-aware evidence and sections are generated; select **View report** after completion to dismiss it.
7. Review the dynamically generated section navigation, Evidence, and Transcript views.
8. Click evidence chips to jump to highlighted transcript rows.
9. Switch templates and generate another report for the same session when needed.
10. Click `Delete`, then `Confirm`, to delete only the selected template's cached report.
11. Click `Generate report` again to recreate it.

Deleting a report does not delete the transcript, speakers, embeddings, saved session, template, reports made with other templates, or media files. It removes only the selected cached meeting-intelligence report.

## API Endpoints

The browser uses these local endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/config` | `GET` | Return provider, model, base URL, and runtime config. |
| `/api/llm-config` | `POST` | Switch runtime provider, model, and base URL. API keys stay server-side. |
| `/api/llm-models?provider=...&base_url=...` | `GET` | Return text-generation model IDs from the provider's `/models` endpoint, sorted with cheaper `nano` and `mini` names first when present. |
| `/api/sessions` | `GET` | List demo and saved sessions. |
| `/api/templates` | `GET` | List inspectable Predefined and saved Custom report templates. |
| `/api/template?template_id=...` | `GET` | Return one complete template definition. |
| `/api/templates/save` | `POST` | Validate and save a Custom template; accepts the template object directly or under `template`. |
| `/api/templates/clone` | `POST` | Clone `template_id` under a supplied `name` as an editable Custom template. |
| `/api/templates/delete` | `POST` | Delete a Custom template by `template_id`; Predefined templates are immutable. |
| `/api/report?session_id=...&template_id=...` | `GET` | Return the selected template's current cached report and transcript rows. The standard template is the default. |
| `/api/generate-async` | `POST` | Start background generation for `session_id` and `template_id`. |
| `/api/generate-status?job_id=...` | `GET` | Poll report-generation progress. |
| `/api/generate` | `POST` | Synchronously generate `session_id` with `template_id`; kept for tests and simple clients. |
| `/api/delete-report` | `POST` | Delete the cache for one `session_id` and `template_id`. |

`/api/generate-async` returns a job object. The browser polls `/api/generate-status` and updates a modal progress overlay with stages such as `prepare`, `segment`, `evidence`, `section`, `finalize`, and `completed`. The overlay is not part of the scrollable report layout.

Template-changing endpoints accept JSON request bodies. Examples:

```json
{"template_id":"builtin.english-podcast-production","name":"My podcast report"}
```

```json
{"session_id":"SESSION_ID","template_id":"custom.my-podcast-report"}
```

## Troubleshooting

If the page says there is no current report, confirm that the intended template is selected and click `Generate report`. A cache is stale when its transcript revision, provider/model, effective language, or template revision differs from the current selection.

If generation appears stuck, check the progress overlay first. Real local LLM generation can take several minutes for larger transcripts, especially when many sections are enabled.

If the server cannot connect to the model, verify the base URL:

```powershell
curl.exe -sS http://127.0.0.1:18081/v1/models
```

For OpenAI, verify that the server process can see `OPENAI_API_KEY`. If you add a global Windows environment variable after the server has already started, restart the server so the new process inherits it. In the browser, the provider status shows only whether the key is configured; the key value is never sent to the browser.

If OpenAI generation fails with HTTP 400, read the progress-panel detail. Current builds include the OpenAI error message. Common causes are unsupported model capabilities, invalid model IDs, or schema restrictions. The report schemas are normalized for OpenAI strict structured output, so nested open-object schema failures should be treated as bugs and covered by tests.

If evidence opens the Transcript tab but does not highlight a row, regenerate the report. Older reports or sessions without stable row IDs can reference normalized row IDs; current browser builds resolve both stored and normalized IDs.

If a local-only report refuses to generate with OpenAI or OpenRouter, that is the template's routing policy working as intended. Select llama.cpp, Ollama, or LM Studio, or clone the template and deliberately change the policy after reviewing the privacy implications.

If you only want to test the UI, use `--mock-llm`. If you want a realistic report, use a real OpenAI-compatible server and enough context/output tokens for the selected model.

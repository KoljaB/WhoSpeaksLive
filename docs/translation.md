# Live translation

WhoSpeaks can translate each stable transcript sentence in the background while keeping the recognized source text authoritative.

Here, **live translation** means sentence-live: a job starts as soon as final ASR commits a stable sentence, not on every changing draft token. Translating draft fragments would create flicker, repeatedly spend model time on the same words, and make multi-target backlogs much worse.

## Browser experience

Translation is intentionally compact. The `Live transcript` heading has one translation control instead of doubling every row by default. Each browser can independently choose:

- `Original`: show only the recognized source text.
- One target language: show only that translation, falling back to the original while it is queued or if it fails.
- `All selected`: show several target languages together, with an optional original line.

Target languages are session-wide because they decide which work the backend performs. Display mode, primary language, and whether `All selected` includes the original are browser-local preferences. Search includes available translations; copy and TXT export follow the visible mode; JSON export keeps the canonical `text` field and adds translations separately.

A source-text SHA-256 hash and revision travel with each request. If a newer source revision exists when an older result returns, that result is discarded. Translation errors never stop transcription.

## Providers

| Profile | When to use it | Model terms and limits |
| --- | --- | --- |
| `translate-gemma-4b` | Recommended quality-first local default. It is a dedicated translation model and fits desktop/server deployment better than a general report LLM. | [TranslateGemma 4B IT](https://huggingface.co/google/translategemma-4b-it) covers 55 languages, uses the Gemma terms, and requires accepting those terms before Hugging Face downloads the gated weights. The checkpoint is named 4B but currently contains about 5B parameters. |
| `nllb-200-600m` | Lighter local option and the broadest practical choice among the smaller profiles, especially for lower-resource languages. | [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) is CC-BY-NC-4.0. Its model card describes research and single-sentence use and says it was not released for production deployment. WhoSpeaks does not redistribute the weights; users download them separately and remain responsible for attribution and non-commercial use. |
| `madlad-400-3b` | Very broad permissively licensed alternative when TranslateGemma does not cover a language. | [MADLAD-400 3B MT](https://huggingface.co/google/madlad400-3b-mt) is Apache-2.0 and publishes 419 language tags. Its model card reports evaluation on 204 languages, so a language tag is not itself a quality guarantee. |
| Chrome Translator API | Prefer this on supported Chrome desktop installations to translate on-device without a per-character API charge. | Enable **Prefer Chrome** in the launcher. WhoSpeaks checks each language pair at runtime, lets Chrome download a required language pack, and asks the selected backend provider only when Chrome is unavailable or fails. |
| `azure_translator` | Managed-cloud default with broad language coverage and efficient text translation. | Uses `AZURE_TRANSLATOR_KEY` by default and optionally `translation_region`. The key itself is never saved in the WhoSpeaks profile. |
| `google_cloud` | Broad managed coverage and a useful fallback for language pairs missing elsewhere. | The current adapter uses the API-key authenticated Cloud Translation Basic v2 JSON API. The browser displays the required “Powered by Google Translate” attribution next to the translation controls. |
| `deepl` | Premium translation-specific option with useful sentence context. | Uses the API Free endpoint by default; set a Pro endpoint override when appropriate. Previous transcript sentences are sent through DeepL's context field, not concatenated into the text being translated. |
| `libretranslate` | Open REST contract for a private or managed LibreTranslate deployment. | Defaults to `http://127.0.0.1:5000`; an API key is optional for self-hosted instances and normally required by managed hosting. |
| `reports_llm` | Reuse the OpenAI-compatible LLM already configured for meeting reports. Useful for experiments or when that server is already loaded. | Quality, privacy, latency, and cost depend on that configured provider. The prompt treats transcript text as data and requests translation-only output. |
| `openai_compatible` | Use any other llama.cpp, Ollama, LM Studio, hosted, or cloud endpoint that exposes `/v1/chat/completions`. | Configure the base URL, model ID, and API-key environment variable. |

Machine translation is assistive text, not a certified translation. Names, numbers, specialized terminology, ambiguity, and sarcasm can still be wrong.

## Why a sidecar is recommended

A **sidecar** is a separate server process dedicated to one supporting function. Local translation models have large dependencies and can consume substantial GPU memory. Running them in the translation sidecar means:

- the browser/transcription process does not import or allocate the model;
- translation can use a separate Python environment or a different machine;
- a translation crash or model-load failure cannot stop transcription;
- one lock serializes access to the model, while the live app keeps a bounded queue and cached per-language results.

The first request loads the selected model lazily and can therefore be much slower than later sentences.

## Installation and startup

The guided installer can create a model-specific sidecar environment, install the appropriate CUDA or CPU PyTorch build, download the selected weights, and update the saved launcher profile:

```powershell
whospeaks install-translation --model-profile nllb-200-600m --torch auto --yes
```

The same command works on Windows and Linux. Default locations are:

- Windows: `%LOCALAPPDATA%\WhoSpeaks\translation\<profile>\venv` and `%LOCALAPPDATA%\WhoSpeaks\models\translation\<profile>`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/whospeaks/translation/<profile>/venv` and `${XDG_DATA_HOME:-~/.local/share}/whospeaks/models/translation/<profile>`

Available profiles are `nllb-200-600m`, `translate-gemma-4b`, and `madlad-400-3b`. Each gets its own environment, so installing one does not replace the dependencies of another. Select the active server in the launcher Translation tab, or repeat `install-translation` for another profile. Use `--venv` and `--model-dir` for server-specific storage layouts, `--torch cuda|cpu|auto` for the runtime, and `--skip-model-download` to verify an already populated model directory without network access.

The general installer exposes the same choice:

```powershell
whospeaks install --target core --translation-model-profile nllb-200-600m --yes
```

`core`, `local`, and `server` targets may all include a local translation sidecar. A server-only profile can start it with `whospeaks translation`; a controller profile starts it automatically with `whospeaks launch`.

For a manual source-checkout installation, create the isolated environment directly:

```powershell
py -3.11 -m venv .venv-translation
.\.venv-translation\Scripts\python.exe -m pip install -e ".[translation]"
```

For TranslateGemma, sign in to Hugging Face, accept the terms on the [checkpoint page](https://huggingface.co/google/translategemma-4b-it), and authenticate the isolated environment before the first download:

```powershell
.\.venv-translation\Scripts\hf.exe auth login
```

On Linux the executable is `.venv-translation/bin/hf`. A failed gated download leaves the environment intact; authenticate, accept the model terms, and rerun the same installer command.

Then configure the normal WhoSpeaks profile:

```powershell
whospeaks config `
  --translation-enabled `
  --translation-provider sidecar `
  --translation-python .\.venv-translation\Scripts\python.exe `
  --translation-model-profile translate-gemma-4b `
  --translation-target-languages "en,de" `
  --translation-max-targets 4
whospeaks launch
```

The launcher uses `translation_python` to start the isolated sidecar in another console when the saved provider is `sidecar`. To start or inspect it separately:

```powershell
whospeaks translation --print
whospeaks-translation-server --model-profile translate-gemma-4b --device auto --port 8799
```

Direct live-window configuration is also available:

```powershell
whospeaks-window `
  --translation-provider sidecar `
  --translation-base-url http://127.0.0.1:8799 `
  --translation-target-language en `
  --translation-target-language de `
  --translation-max-targets 4
```

Use `--translation-provider transformers` only when intentionally loading the model inside the live-window process.

## Managed and REST API providers

WhoSpeaks stores only the **name** of an environment variable, never the secret value. Configure a provider in the launcher Translation tab, or use the CLI. For example, DeepL API Free on Windows:

```powershell
$env:DEEPL_API_KEY = "your-key"
whospeaks config --translation-enabled --translation-provider deepl `
  --translation-target-languages "en,de"
whospeaks launch
```

Azure additionally accepts a region:

```powershell
$env:AZURE_TRANSLATOR_KEY = "your-key"
whospeaks config --translation-enabled --translation-provider azure_translator `
  --translation-region westeurope --translation-target-languages "en,de"
```

Google Cloud Translation Basic uses `GOOGLE_TRANSLATE_API_KEY`. LibreTranslate uses `LIBRETRANSLATE_API_KEY` when its endpoint requires authentication and accepts an endpoint override:

```powershell
whospeaks config --translation-enabled --translation-provider libretranslate `
  --translation-base-url http://translate.internal:5000
```

The default secret-variable names are:

| Provider | Environment variable | Default endpoint |
| --- | --- | --- |
| DeepL | `DEEPL_API_KEY` | `https://api-free.deepl.com/v2` |
| Google Cloud Translation | `GOOGLE_TRANSLATE_API_KEY` | Cloud Translation Basic v2 |
| Azure Translator | `AZURE_TRANSLATOR_KEY` | Global Translator endpoint |
| LibreTranslate | `LIBRETRANSLATE_API_KEY` | `http://127.0.0.1:5000` |
| OpenAI-compatible | `OPENAI_API_KEY` | Must be configured |

Use `--translation-api-key-env MY_SECRET_NAME` to select another environment variable. Use `--translation-base-url` for DeepL Pro, sovereign/private cloud endpoints, proxies, or self-hosted LibreTranslate.

### Chrome first with automatic fallback

Select a normal backend provider and enable **Prefer Chrome; use selected provider as fallback**. For each stable sentence and target language, the browser checks Chrome's Translator API. A supported or downloadable pair is translated on-device and the completed result is returned to WhoSpeaks for normal session persistence. An unsupported pair or browser-side failure is queued through the selected backend provider instead. Jobs are serialized in the browser so simultaneous language-pack/model work does not overwhelm Chrome.

## Capacity and multiple languages

Every stable sentence creates one job per selected target language. With a single serialized local model, two targets need roughly twice the inference work and four targets roughly four times the work. Cached duplicates return immediately, but new speech does not. The default maximum is four simultaneous targets and is configurable up to sixteen.

The bounded worker queue protects transcription responsiveness. A larger, still-bounded deferred backlog admits long-session backfills as worker slots open instead of dropping them immediately; only work beyond both capacities becomes a per-language error, while the original transcript continues. Practical limits depend on model, hardware, sentence rate, and target count; begin with one target and increase while observing queue depth, deferred jobs, and latency at `GET /api/translation/status`.

## HTTP interfaces

The live app exposes:

- `GET /api/translation/status`
- `POST /api/translation/configure` with `{"target_languages":["en","de"]}`
- `POST /api/translation/browser-result` for a revision-checked Chrome result
- `POST /api/translation/browser-fallback` when Chrome cannot translate a selected pair
- raw server-sent events named `translation`
- public semantic events such as `translation.queued`, `translation.completed`, and `translation.failed`

The local sidecar exposes:

- `GET /health`
- `POST /v1/translate` with `source_text`, `source_language`, `target_language`, and optional `context`

Completed results are stored in the session's separate `translations.json`. A saved-session review can switch among translations that were generated for that session; selecting a language that was never generated falls back to the original instead of pretending work is still running. Older saved sessions without this file continue to open with an empty translation list.

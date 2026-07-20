"""Expanded, decision-oriented help for desktop-launcher settings.

Tooltips and the persistent help strip deliberately stay short.  These texts are
shown only on F1 and explain operational consequences, compatibility, and the
cases in which a user should change a value.
"""

from __future__ import annotations


def _paragraphs(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip())


SETTINGS_MORE_HELP: dict[str, str] = {
    "settings_sections": _paragraphs(
        "Changing category only changes the visible form. Edits in other categories remain pending until you save or discard them.",
        "The selected deployment can hide settings that do not apply to that topology; hidden values are preserved in the draft, not silently reset. Save validates the complete profile, including fields outside the current category.",
    ),
    "save_changes": _paragraphs(
        "Save validates the complete launch profile before writing it. A failed validation keeps every edit in the form, opens the relevant category, focuses the exact control, and leaves the saved profile unchanged.",
        "Saving changes configuration only. It does not restart running services; stop and launch WhoSpeaks again when a changed runtime, model, endpoint, or port must take effect.",
    ),
    "discard_changes": _paragraphs(
        "Discard reloads the last saved profile across every category, including edits in categories that are currently hidden. It does not stop services or undo configuration that was already saved.",
        "Use this when you want a known persisted baseline. There is no per-field undo after the draft is discarded.",
    ),
    "mode": _paragraphs(
        "Local on NVIDIA GPU starts the browser app and runs the large final-ASR and speaker-embedding stack on this computer. Local on CPU only also runs everything here, but uses Kroko or Nemotron for live and fixed text, Whisper Base only for final word alignment, and SpeechBrain ECAPA for speaker embeddings. It allocates no CUDA memory or VRAM.",
        "Remote client starts the browser app here but sends final-ASR and speaker-embedding work to configured backend URLs, normally on a GPU server. Model server installs those backends for another WhoSpeaks computer and does not start the browser app.",
        "Deployment controls which settings and dependencies are required; it is not merely a label. Remote services reduce local model load but move audio over the network, so plain HTTP endpoints should be limited to a trusted LAN or protected by a VPN or authenticated TLS proxy.",
    ),
    "language": _paragraphs(
        "This is the source language contract shared by final ASR, sentence splitting, live preview, speaker-labeled transcripts, and the default report language. It is not automatic language detection.",
        "Nemotron and Kroko live preview support fewer languages than final Whisper ASR. Saving validates the selected engine against this language and marks an incompatible choice; for another supported Whisper language, set Live text to Off and keep final transcription enabled.",
    ),
    "realtime_preview_engine": _paragraphs(
        "Live text is provisional text shown while a sentence is still being spoken. Final ASR remains authoritative and replaces the preview when the sentence is committed.",
        "Nemotron uses the sherpa-onnx runtime and offers 560 ms and 160 ms presets. Kroko/Banafo uses a separate native worker and has a smaller validated language set. Off removes preview startup and model cost without disabling final transcription or speaker recognition.",
    ),
    "realtime_preview_model_preset": _paragraphs(
        "For Nemotron, 160 ms emits updates sooner while 560 ms uses a longer chunk intended for steadier preview text. This setting changes provisional latency, not the final faster-whisper model.",
        "For Kroko, Community 64L is the public auto-download path for supported languages; Pro 16L is configured only for English and must already be available. Switching live-text engine replaces the available presets so incompatible combinations cannot be saved.",
    ),
    "live_speaker_assignment": _paragraphs(
        "When enabled, WhoSpeaks scores provisional audio with the Live provider and shows a tentative speaker beside live text. The final provider still re-evaluates committed sentences and can correct that label.",
        "Disabling this saves live embedding work and hides provisional names. It does not disable final speaker assignment, speaker libraries, or the final provider.",
    ),
    "host": _paragraphs(
        "127.0.0.1 binds the browser controller only to this machine and is the safest default. A LAN address or 0.0.0.0 allows other devices to reach the service, subject to firewall rules.",
        "The browser API can expose transcripts, sessions, media controls, and local service actions. Do not expose it directly to the public internet; use a trusted network or an authenticated reverse proxy when remote access is required.",
    ),
    "port": _paragraphs(
        "This TCP port serves the live browser UI and its API. The launcher checks availability before startup, but another process can still claim the port between the check and launch.",
        "Changing the port changes the URL you open and any bookmarks or integrations that call the browser API. It does not change the Meeting Intelligence or translation ports.",
    ),
    "model": _paragraphs(
        "This selects the local faster-whisper model used for committed, high-accuracy transcription. Larger models generally need more download space, RAM or VRAM, and warm-up time; smaller and distilled models trade some accuracy for lower resource use.",
        "A custom Hugging Face model ID must be compatible with faster-whisper/CTranslate2. In a remote-controller profile the remote ASR server chooses its own loaded model, so this local field is not used.",
    ),
    "device": _paragraphs(
        "Automatic lets the local final-ASR runtime select a usable accelerator. CUDA forces an NVIDIA CUDA path; CPU avoids GPU use but is normally slower for large models.",
        "This setting applies to local final ASR only. It does not select the device used by remote servers, speaker embeddings, live preview, or the translation sidecar.",
    ),
    "compute_type": _paragraphs(
        "Compute type is the numeric representation used by faster-whisper/CTranslate2. float16 is a common CUDA choice; int8 variants reduce memory and can help CPU or constrained-GPU setups; float32 is broadly compatible but uses more memory.",
        "Not every device supports every type efficiently. Automatic is the safest portability choice; if startup reports an unsupported compute type, choose a type supported by that device rather than treating it as an accuracy control.",
    ),
    "vad_backend": _paragraphs(
        "Voice activity detection (VAD) decides when audio contains speech and when enough silence has occurred to finalize a sentence window. It does not transcribe words or identify speakers.",
        "RMS uses signal energy, has minimal dependencies, and is predictable in clean audio. Silero uses a neural speech detector and may reject steady background noise better, but adds model/runtime cost and can behave differently on very quiet or unusual voices.",
    ),
    "realtime_preview_model_dir": _paragraphs(
        "Leave this blank for WhoSpeaks to discover or download the selected unpacked Nemotron/sherpa-onnx model in its managed model locations. A manual directory is useful for offline installations, shared caches, or pre-provisioned machines.",
        "The directory must contain the files expected by the selected preset, including model, token, and configuration data. Point to the unpacked model directory, not the downloaded archive or its parent cache folder.",
    ),
    "realtime_preview_python": _paragraphs(
        "Kroko/Banafo can run in a separate Python environment because its native packages may require a different Python version from the main launcher. Select the environment's actual python executable, not its folder.",
        "Leave this blank for the managed/current runtime. Nemotron uses the main WhoSpeaks environment and ignores this Kroko-specific path.",
    ),
    "embedding_python": _paragraphs(
        "This optional executable runs local speaker-embedding helper processes in a separate environment. It is useful when PyTorch, ONNX, or provider dependencies are isolated from the launcher environment.",
        "The environment must contain WhoSpeaks-compatible helper code and every dependency required by the chosen Final and Live provider expressions. Remote embeddings profiles do not launch this local helper.",
    ),
    "embedding_device": _paragraphs(
        "This selects the processor used by the local speaker-embedding helper. CPU-only mode fixes it to CPU so no CUDA allocation or VRAM is required.",
        "Local on NVIDIA GPU keeps CUDA as its default. SpeechBrain ECAPA remains fast enough for final and provisional live speaker embeddings in the CPU-only profile.",
    ),
    "cpu_alignment_model": _paragraphs(
        "CPU mode keeps the Kroko or Nemotron transcript fixed and uses Whisper only as a forced aligner: it locates those known words in the buffered audio instead of decoding new text.",
        "Base is the measured quality default. Tiny roughly halves alignment CPU time, but its sentence-end and word-start boundaries are less precise.",
    ),
    "cpu_alignment_threads": _paragraphs(
        "This is a hard worker-pool limit for the CTranslate2 alignment model. The production default is two threads, while Kroko and speaker embeddings use their own separately bounded pools.",
        "Raising the value can reduce the short delay after an endpoint, but it also raises instantaneous CPU use and is not recommended for background desktop operation.",
    ),
    "provider_preset": _paragraphs(
        "A speaker embedding is a numeric representation of a voice. The preset selects the exact model or weighted model stack used for committed sentences and for provisional live labels; it does not select an ASR model.",
        "Different provider expressions create incompatible vector spaces. After changing a preset, rebuild or reload speaker references made for the new exact stack. Multi-model presets also increase first-start downloads, warm-up time, RAM/VRAM use, and remote-server provider requirements.",
    ),
    "embedding_provider": _paragraphs(
        "The Final provider creates embeddings from committed sentence audio and is the authoritative path for saved speaker assignment. An expression can join normalized provider vectors with + and scale a component with =, for example provider_a=1.0+provider_b=0.5; weights scale vectors and are not probabilities.",
        "The exact expression defines compatibility with enrolled speaker references. Changing a provider name, component, order, or weight can make existing vectors unsuitable. For remote embeddings, the server must advertise and successfully load every component in this expression.",
    ),
    "live_speaker_embedding_provider": _paragraphs(
        "The Live provider scores shorter, still-changing audio windows to show provisional speaker names quickly. It may use a faster stack than the Final provider; WhoSpeaks keeps live-compatible speaker profiles instead of comparing vectors from different model spaces directly.",
        "This value affects responsiveness, model load, and provisional labels only. Final committed sentences are reassigned with the Final provider. If Live speaker labels are off, this provider is retained in the profile but is not used for live scoring.",
    ),
    "remote_asr_url": _paragraphs(
        "Enter the base HTTP or HTTPS URL of the remote faster-whisper ASR service, normally including its port, for example http://192.168.1.20:8650. Do not append a transcription route; WhoSpeaks calls the service contract below this base URL.",
        "Readiness checks verify the health response, while launch-time requests send sentence audio and the selected language. Plain HTTP does not encrypt that audio, so keep it on a trusted LAN or protect it with a VPN or authenticated TLS proxy.",
    ),
    "remote_embeddings_url": _paragraphs(
        "Enter the base HTTP or HTTPS URL of the voice-embeddings service, normally including its port, for example http://192.168.1.20:8660. The service must support the configured Final and Live provider expressions, not merely respond to health checks.",
        "Speaker audio windows are sent to this endpoint. Provider load failures, missing model terms, or incompatible model stacks can therefore fail after basic reachability succeeds; Diagnostics reports provider readiness separately.",
    ),
    "reports_enabled": _paragraphs(
        "Meeting Intelligence starts a separate local web service for saved-session review, evidence-grounded reports, and Ask sessions. The live window proxies its chat APIs so the user does not need to open a second browser page.",
        "Turning it off leaves capture, transcription, speakers, and session saving intact. Existing sessions and cached reports are not deleted.",
    ),
    "reports_port": _paragraphs(
        "This port serves the Meeting Intelligence browser and API, including report generation and Ask. It must differ from the live browser port and from other local services.",
        "Changing it updates how the live app reaches Meeting Intelligence. It does not move the configured LLM or text-embedding endpoint.",
    ),
    "report_language": _paragraphs(
        "Follow live language asks generated report content to use the transcription language. Selecting another language requests summaries, decisions, action items, and answers in that language; quoted transcript evidence remains source text.",
        "Language is part of report-cache validity. A report generated under a different effective language is treated as stale rather than silently reused.",
    ),
    "report_llm_provider": _paragraphs(
        "This provider is shared by evidence extraction, report sections, and grounded Ask answers. llama.cpp, Ollama, and LM Studio target local OpenAI-compatible servers; OpenAI, OpenRouter, and other compatible endpoints may send transcript content to an external service.",
        "The selection fills a conventional base URL and model choices but does not install or start the LLM. Cloud providers require their documented environment-variable key; local providers must already be listening and have a suitable instruction model loaded.",
    ),
    "report_llm_base_url": _paragraphs(
        "Use the provider's API root, commonly ending in /v1, not the full /chat/completions route; the Meeting Intelligence client appends the operation path. Provider selection supplies normal defaults.",
        "Override this for a remote machine, reverse proxy, non-default local port, or compatible hosted service. Changing the URL can change where transcript text is processed, so review the endpoint's privacy and authentication boundary.",
    ),
    "report_llm_model": _paragraphs(
        "The model must support instruction following and reliable structured output for evidence and report schemas. A model that can chat but cannot consistently produce the requested JSON may still fail report generation.",
        "For OpenAI, the launcher requests the account-visible model catalog when OPENAI_API_KEY is available and filters it to text-generation candidates. Other compatible providers accept their exposed model ID; changing model makes mismatched cached reports stale rather than overwriting them.",
    ),
    "text_embedding_preset": _paragraphs(
        "Text embeddings are numeric representations of transcript meaning; they are unrelated to the voice embeddings used for speaker recognition. They enable semantic retrieval for long sessions and questions spanning multiple sessions.",
        "Not configured still permits short, single-session chat that fits directly into the LLM context. OpenAI presets send transcript chunks to OpenAI's embeddings API. Custom accepts an OpenAI-compatible embedding endpoint, model, and key-variable name.",
    ),
    "text_embedding_base_url": _paragraphs(
        "This is the OpenAI-compatible API root used to create and query the transcript search index. It is contacted by Meeting Intelligence, not by the browser directly.",
        "Changing endpoints does not make an old index compatible. Re-index sessions when the endpoint or model produces a different vector space, and consider whether transcript chunks may leave the local machine.",
    ),
    "text_embedding_model": _paragraphs(
        "Choose an embedding model exposed by the configured endpoint, not a chat model. The model determines vector dimensions and meaning, so all documents and queries in one index must use the same model.",
        "Changing this value requires rebuilding affected semantic indexes. Larger embedding models can improve retrieval in some domains but increase API cost, storage, and indexing time.",
    ),
    "text_embedding_api_key_env": _paragraphs(
        "Enter an environment-variable name such as OPENAI_API_KEY, never the secret itself. At runtime Meeting Intelligence reads the variable from its process environment and sends the resulting credential only to the configured embedding endpoint.",
        "The launcher saves only this name. If WhoSpeaks is started from a desktop shortcut or service, ensure that process actually inherits the variable; a key visible in another terminal is not automatically available everywhere.",
    ),
    "report_auto_generate": _paragraphs(
        "When enabled, Meeting Intelligence watches for newly finalized saved meetings and queues the standard report automatically. Existing historical sessions are deliberately not submitted merely because the service starts.",
        "Disable this when reports should be generated only after review or when LLM calls have privacy or cost implications. Sessions remain available for manual report generation later.",
    ),
    "translation_enabled": _paragraphs(
        "Translation starts only for committed stable sentences; draft live words are not repeatedly translated. The original transcript remains authoritative, and a translation error never stops transcription.",
        "Each selected target creates separate work. Enable translation only when targets and a viable provider are configured, especially for local models whose first request may include a long lazy-load delay.",
    ),
    "translation_browser_preferred": _paragraphs(
        "For each supported source-target pair, Chrome's Translator API is tried first and may download an on-device language pack. Successful browser translation stays on that device and avoids a backend request.",
        "The selected provider remains the automatic fallback for unsupported pairs or browser failures. This option therefore does not remove the need to configure a fallback when translation must be reliable across browsers and languages.",
    ),
    "translation_provider": _paragraphs(
        "Local sidecar isolates a translation model in another process; Local in live process loads it beside capture and can consume the same memory. DeepL, Google Cloud, and Azure are managed APIs; LibreTranslate targets a private or hosted REST service; Meeting Intelligence LLM reuses the report model; OpenAI-compatible uses a chat-completions endpoint.",
        "Provider choice changes which model, URL, key, region, Python, port, and device fields are relevant. It also changes privacy, licensing, latency, first-load behavior, and per-request cost; the launcher shows only the fields consumed by the selected path.",
    ),
    "translation_target_languages": _paragraphs(
        "Select one or more output languages. The source language cannot also be a target, duplicates are removed, and Maximum targets limits how many selections may be saved.",
        "Support still depends on the chosen provider or local model; a WhoSpeaks code means the app can represent the language, not that every provider guarantees the pair. Supported codes are listed below.",
        "{language_codes}",
    ),
    "translation_max_targets": _paragraphs(
        "This is both validation and a capacity guard. Each committed sentence creates one job per target, so two targets require roughly twice the translation work and four roughly four times the work for a serialized local model.",
        "Begin with one target and increase while observing translation latency and queue status. Raising the limit does not make the model parallel or increase provider quotas.",
    ),
    "translation_model_profile": _paragraphs(
        "TranslateGemma 4B is the quality-first local default and requires accepting Gemma model terms. NLLB-200 distilled 600M is lighter and broad but carries CC-BY-NC-4.0 terms and was released for research/single-sentence use rather than production. MADLAD-400 3B is a broader Apache-2.0 alternative with higher resource needs.",
        "The profile supplies tested model and runtime defaults for Local sidecar or Local in live process. Language coverage in a model card is not a guarantee of quality for your terminology, names, or audio domain.",
    ),
    "translation_model": _paragraphs(
        "Normally leave this blank so the selected local profile or managed provider uses its default. Set it only when an OpenAI-compatible endpoint requires an explicit model ID or when intentionally overriding a local Hugging Face checkpoint.",
        "The override must match the selected provider's API and runtime. It does not install weights, accept model terms, or validate that the model supports every selected language.",
    ),
    "translation_base_url": _paragraphs(
        "This is the selected provider's API root. Defaults are filled for managed services and LibreTranslate; override it for DeepL Pro, private or sovereign endpoints, reverse proxies, or a self-hosted compatible service.",
        "For OpenAI-compatible translation, configure the API root rather than a browser page. Treat the URL as a data boundary because stable transcript sentences are sent there when Chrome does not handle the pair.",
    ),
    "translation_api_key_env": _paragraphs(
        "Enter the environment-variable name holding the selected provider's credential, for example DEEPL_API_KEY, GOOGLE_TRANSLATE_API_KEY, AZURE_TRANSLATOR_KEY, or LIBRETRANSLATE_API_KEY. Never paste the key itself into this field.",
        "Only the name is persisted. The launched process must inherit that variable, and changing provider normally changes the expected default variable.",
    ),
    "translation_region": _paragraphs(
        "Azure Translator can require the resource region in addition to its key, especially for regional or multi-service resources. Enter the Azure region identifier associated with that resource, such as westeurope.",
        "This value is sent only on providers that use it. It is not a language or availability region and should remain blank for providers that do not require one.",
    ),
    "translation_python": _paragraphs(
        "The Local sidecar can use an isolated Python environment containing its own PyTorch, Transformers, and model dependencies. Select that environment's python executable, not the environment directory.",
        "Leave blank to use the managed/current runtime. The path is relevant to the sidecar provider only; Local in live process uses the main WhoSpeaks process.",
    ),
    "translation_port": _paragraphs(
        "This port is used by the launcher-managed local translation sidecar. It must be free and different from the live browser and Meeting Intelligence ports.",
        "The live app connects to the sidecar on this port. Cloud, LibreTranslate, Meeting Intelligence LLM, and in-process translation do not use it.",
    ),
    "translation_device": _paragraphs(
        "Automatic lets the local translation runtime choose an available accelerator. CUDA forces an NVIDIA GPU path; CPU avoids GPU allocation but can be much slower for multi-billion-parameter models.",
        "This affects local sidecar or in-process models only. It does not choose the device used by Chrome, managed APIs, a remote endpoint, ASR, or speaker embeddings.",
    ),
    "advanced_args": _paragraphs(
        "These tokens are parsed and appended to the generated live-window command after the validated profile options. Use them only for supported command-line flags that the launcher does not expose directly; inspect View command before launch.",
        "Because later duplicate flags may override earlier generated flags, do not repeat settings already represented in the form. Quoting errors or unsupported options can prevent startup, and the text is stored in the profile exactly as configuration rather than executed as a shell command.",
    ),
}

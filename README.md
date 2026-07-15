# WhoSpeaksLive

**Private, speaker-aware transcripts and meeting summaries on hardware you control.**

WhoSpeaksLive identifies speakers while a conversation is still happening, produces a stable speaker-labeled transcript, and can turn a saved session into a structured meeting report with a summary, decisions, action items, open questions, risks, and links to supporting transcript evidence.

Speaker diarization means identifying who spoke when. WhoSpeaksLive combines a fast live path for immediate speaker labels and optional draft text with a separate final path that uses more context for stable sentence-level results.

The core diarization and transcription stack can run entirely on hardware you control. Meeting reports can also use a local or self-hosted large language model (LLM), so sensitive audio, Person-owned Voice samples, transcripts, and summaries can remain inside your environment. The built-in browser server is intended for a trusted operator or protected network; it does not provide authentication or TLS by itself.

## Why WhoSpeaksLive

- **Meeting intelligence, not just transcription:** turn saved speaker-labeled transcripts into evidence-grounded summaries, decisions, action items, questions, and risks.
- **Private by design:** run locally or on self-hosted servers; no third-party cloud service is required for the core pipeline or local meeting-report generation.
- **Useful while it happens:** see live speaker labels and optional low-latency draft text while the meeting is still in progress.
- **Optional sentence-live translation:** translate stable transcript rows into one or several target languages with compact original, single-language, or multi-language views.
- **Fast now, stable when final:** a low-latency live path gives immediate feedback, while a separate final path uses more context for speaker-labeled transcripts.
- **Remember real People safely:** link meeting-local Speakers to persistent People, keep a deliberate recognition roster, and confirm suggestions instead of forcing identity.
- **Built for review and correction:** manage Person-owned Voice samples, correct assignments, and trace report evidence back to the supporting transcript rows.

Use it when the audio is sensitive, the answer is needed immediately, or both: internal meetings, research interviews, legal or medical workflows, local media analysis, and any environment where sending raw conversation audio to another company is not acceptable.

## Demo

https://github.com/user-attachments/assets/2de749e0-6c02-47de-b949-bd90b4f4efbb

For faster realtime ASR preview text like shown in the demo, WhoSpeaksLive supports two optional preview backends: Kroko/Banafo streaming models and experimental CPU-only Nemotron 3.5 through `sherpa-onnx`. Kroko Pro/private models must be installed and licensed separately; Nemotron downloads verified upstream model weights on first use and has separate NVIDIA model-license terms.

## Start Here

For a guided setup, install the lightweight CLI and let it inspect the machine:

```powershell
pip install whospeaks
whospeaks
```

The full-screen setup application opens on the Setup tab. Select the full local stack, the core/controller for remote ASR and embeddings servers, or the ASR/embeddings server packages, then choose whether to include optional realtime preview text. Kroko remains optional because its native runtime may require Python 3.12, Docker Desktop on Windows, or a prebuilt `kroko_onnx` wheel. Nemotron 3.5 is available as a manual CPU preview option while the installer flow is being integrated.

The `whospeaks` setup application keeps component readiness, diagnostics, settings, installation progress, logs, cancellation, and browser launch in one terminal interface. Run `whospeaks --classic` when the full-screen terminal interface is unavailable or you prefer the numbered menu.

The short `whospeaks` command is a setup and launcher wrapper. It saves a small profile, runs doctor checks, and expands that profile into the longer `whospeaks-window ...` browser-server command when you launch.

For a Linux container server:

```bash
docker build -t whospeaks:local .
docker run --rm --name whospeaks -p 127.0.0.1:8796:8796 -v whospeaks-data:/data -v whospeaks-models:/models whospeaks:local
```

See [Docker](docs/docker.md) for the full build/run path and volume notes.

For a manual full working setup, follow these in order:

1. [Installation](docs/installation.md): install the Windows controller.
2. [External ASR and embeddings servers](docs/external-servers.md): set up the Linux GPU services.
3. [Quickstart](docs/quickstart.md): verify a local or remote smoke run, then run the tuned provider stack.

## Current Scope

WhoSpeaksLive performs best on clean recordings where one person speaks at a time into good microphones. Diarization accuracy can degrade with background noise, background music, echo, crosstalk, overlapping speech, or low-quality microphones, and it may become less reliable as the active speaker count grows. The system assumes complete utterances can be assigned to a single speaker, so cases where one speaker starts a sentence and another finishes it are not expected to score well.

CPU-only operation is not the recommended path for the current stack. The system is GPU-heavy today; a CPU-only setup may be possible, but should be treated as a separate optimization target and will likely require engineering work, slower processing, and some accuracy or throughput tradeoffs.

## Realtime Preview Languages

Realtime preview text is the optional, low-latency draft transcript shown before final ASR results arrive. Choose one of the following three modes. Set `--language` or `WHOSPEAKS_LANGUAGE` to select the language for final ASR and sentence splitting; when preview is enabled, it also selects the preview language.

### No realtime preview text

Use `--realtime-preview-engine off` for the widest language coverage. Live speaker detection and the final speaker-labeled transcript still work; only the rapidly updating draft text is disabled.

**Languages:** Afrikaans (`af`), Albanian (`sq`), Arabic (`ar`), Armenian (`hy`), Basque (`eu`), Belarusian (`be`), Bulgarian (`bg`), Catalan (`ca`), Chinese (`zh`), Croatian (`hr`), Czech (`cs`), Danish (`da`), Dutch (`nl`), English (`en`), Estonian (`et`), Faroese (`fo`), Finnish (`fi`), French (`fr`), Galician (`gl`), Georgian (`ka`), German (`de`), Greek (`el`), Hebrew (`he` or `iw`), Hindi (`hi`), Hungarian (`hu`), Icelandic (`is`), Indonesian (`id`), Italian (`it`), Japanese (`ja`), Kazakh (`kk`), Korean (`ko`), Latin (`la`), Latvian (`lv`), Lithuanian (`lt`), Malayalam (`ml`), Maltese (`mt`), Marathi (`mr`), Myanmar/Burmese (`my`), Norwegian (`no`), Norwegian Nynorsk (`nn`), Persian (`fa`), Polish (`pl`), Portuguese (`pt`), Romanian (`ro`), Russian (`ru`), Sanskrit (`sa`), Serbian (`sr`), Sindhi (`sd`), Slovak (`sk`), Slovenian (`sl`), Spanish (`es`), Swedish (`sv`), Tamil (`ta`), Telugu (`te`), Thai (`th`), Turkish (`tr`), Ukrainian (`uk`), Urdu (`ur`), Vietnamese (`vi`), and Welsh (`cy`). These are the languages currently configured for final ASR and at least one sentence segmenter in WhoSpeaksLive.

### Kroko

Use `--realtime-preview-engine kroko_onnx` for Kroko/Banafo streaming preview text.

**Languages:** German (`de`), English (`en`), Spanish (`es`), French (`fr`), Italian (`it`), Hebrew (`he` or `iw`), Dutch (`nl`), Portuguese (`pt`), Swedish (`sv`), and Turkish (`tr`).

### Nemotron 3.5

Use `--realtime-preview-engine sherpa_onnx` for CPU-based Nemotron 3.5 preview text.

**Languages:** English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Dutch (`nl`), Portuguese (`pt`), Turkish (`tr`), and Swedish (`sv`). Swedish is available as broad coverage; the other eight are the main supported languages.

Hebrew is not supported by the Nemotron integration; use Kroko or disable realtime preview text. The underlying model may decode more languages, but WhoSpeaksLive treats only the languages above as supported until more have been validated in the realtime path.

See the [configuration guide](docs/configuration.md#language) for model selection, downloads, and sentence-segmentation details.

## License

WhoSpeaksLive's own code is licensed under the [MIT License](LICENSE).

Optional realtime preview support uses separately licensed upstream components and model files. Missing public Kroko Community preview models are downloaded automatically from Hugging Face when Kroko preview starts. Nemotron 3.5 model archives are downloaded from the upstream `k2-fsa/sherpa-onnx` release and verified with pinned SHA-256 checksums when Nemotron preview starts. This repository's MIT license does not relicense Kroko/Banafo or Nemotron assets; before downloading, bundling, or deploying them, review and respect the current terms from [Kroko by Banafo](https://kroko.ai/), the [Banafo/Kroko-ASR model card](https://huggingface.co/Banafo/Kroko-ASR), the [kroko-ai/kroko-onnx repository](https://github.com/kroko-ai/kroko-onnx), and [Third-Party Model Licenses](docs/third-party-model-licenses.md).

Optional translation weights are also downloaded separately and retain their own terms: TranslateGemma uses the Gemma terms, NLLB-200 uses CC-BY-NC-4.0, and MADLAD-400 identifies Apache-2.0. See [Live translation](docs/translation.md) and [Third-Party Model Licenses](docs/third-party-model-licenses.md).

## Documentation

| Topic | Document |
| --- | --- |
| Documentation map | [docs/index.md](docs/index.md) |
| Product overview and use cases | [docs/overview.md](docs/overview.md) |
| Installation | [docs/installation.md](docs/installation.md) |
| macOS setup | [docs/macos-setup.md](docs/macos-setup.md) |
| Quickstart | [docs/quickstart.md](docs/quickstart.md) |
| Live window workflow | [docs/live-window-workflow.md](docs/live-window-workflow.md) |
| Live translation | [docs/translation.md](docs/translation.md) |
| Meeting intelligence server | [docs/meeting-intelligence-server.md](docs/meeting-intelligence-server.md) |
| People and cross-meeting recognition | [docs/people-and-recognition.md](docs/people-and-recognition.md) |
| Security and data privacy | [docs/security-and-data-privacy.md](docs/security-and-data-privacy.md) |
| Legacy Speaker-group files | [docs/speaker-libraries.md](docs/speaker-libraries.md) |
| External ASR and embeddings servers | [docs/external-servers.md](docs/external-servers.md) |
| Docker server image | [docs/docker.md](docs/docker.md) |
| Configuration guide | [docs/configuration.md](docs/configuration.md) |
| Speaker model presets | [docs/speaker-model-presets.md](docs/speaker-model-presets.md) |
| Technical description | [docs/technical-description.md](docs/technical-description.md) |
| Technical architecture | [docs/architecture.md](docs/architecture.md) |
| Validation and scoring | [docs/validation-and-scoring.md](docs/validation-and-scoring.md) |
| Modal deployment | [docs/modal-deployment.md](docs/modal-deployment.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Development workflow | [docs/development.md](docs/development.md) |

# WhoSpeaksLive

WhoSpeaksLive is a local-first speaker diarization app for turning live or replayed media into speaker-labeled transcripts, with a browser UI for fast live speaker feedback, final sentence assignment, speaker library management, and validation.

## Demo

https://github.com/user-attachments/assets/2de749e0-6c02-47de-b949-bd90b4f4efbb

## Current Scope

WhoSpeaksLive performs best on clean recordings where one person speaks at a time into good microphones. Diarization accuracy can degrade with background noise, background music, echo, crosstalk, overlapping speech, or low-quality microphones, and it may become less reliable as the active speaker count grows. The system assumes complete utterances can be assigned to a single speaker, so cases where one speaker starts a sentence and another finishes it are not expected to score well.

All Kroko languages supported by this integration work with realtime preview text: German, English, Spanish, French, Italian, Hebrew, Dutch, Portuguese, Swedish, and Turkish. Set `--language` or `WHOSPEAKS_LANGUAGE` to keep final ASR, Kroko/Banafo preview model selection, and stream2sentence sentence splitting on the same language. See the [configuration guide](docs/configuration.md#language) for language codes and model details.

Without realtime preview text, WhoSpeaksLive can also work with additional languages. The key requirement is that the language is supported by Whisper and by at least one configured sentence segmenter, meaning NLTK or Stanza.

That currently makes these additional languages principally supported without realtime preview text: Afrikaans, Arabic, Belarusian, Bulgarian, Catalan, Czech, Welsh, Danish, Greek, Estonian, Basque, Persian, Finnish, Faroese, Galician, Hindi, Croatian, Hungarian, Armenian, Indonesian, Icelandic, Japanese, Georgian, Kazakh, Korean, Latin, Lithuanian, Latvian, Malayalam, Marathi, Maltese, Myanmar/Burmese, Norwegian, Norwegian Nynorsk, Polish, Romanian, Russian, Sanskrit, Sindhi, Slovak, Slovenian, Albanian, Serbian, Tamil, Telugu, Thai, Ukrainian, Urdu, Vietnamese, and Chinese.

CPU-only operation is not the recommended path for the current stack. The system is GPU-heavy today; a CPU-only setup may be possible, but should be treated as a separate optimization target and will likely require engineering work, slower processing, and some accuracy or throughput tradeoffs.

## License

WhoSpeaksLive's own code is licensed under the [MIT License](LICENSE).

Optional Kroko/Banafo preview support uses separately licensed upstream components and model files. Missing public Community preview models are downloaded automatically from Hugging Face when realtime preview starts. This repository's MIT license does not relicense Kroko/Banafo assets; before downloading, bundling, or deploying them, review and respect the current terms from [Kroko by Banafo](https://kroko.ai/), the [Banafo/Kroko-ASR model card](https://huggingface.co/Banafo/Kroko-ASR), and the [kroko-ai/kroko-onnx repository](https://github.com/kroko-ai/kroko-onnx).

## Start Here

For a full working setup, follow these in order:

1. [Installation](docs/installation.md): install the Windows controller.
2. [External ASR and embeddings servers](docs/external-servers.md): set up the Linux GPU services.
3. [Quickstart](docs/quickstart.md): verify the smoke provider, then run the tuned provider stack.

## Documentation

| Topic | Document |
| --- | --- |
| Documentation map | [docs/index.md](docs/index.md) |
| Product overview and use cases | [docs/overview.md](docs/overview.md) |
| Installation | [docs/installation.md](docs/installation.md) |
| macOS setup | [docs/macos-setup.md](docs/macos-setup.md) |
| Quickstart | [docs/quickstart.md](docs/quickstart.md) |
| Live window workflow | [docs/live-window-workflow.md](docs/live-window-workflow.md) |
| Speaker libraries | [docs/speaker-libraries.md](docs/speaker-libraries.md) |
| External ASR and embeddings servers | [docs/external-servers.md](docs/external-servers.md) |
| Configuration guide | [docs/configuration.md](docs/configuration.md) |
| Technical description | [docs/technical-description.md](docs/technical-description.md) |
| Technical architecture | [docs/architecture.md](docs/architecture.md) |
| Validation and scoring | [docs/validation-and-scoring.md](docs/validation-and-scoring.md) |
| Modal deployment | [docs/modal-deployment.md](docs/modal-deployment.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Development workflow | [docs/development.md](docs/development.md) |

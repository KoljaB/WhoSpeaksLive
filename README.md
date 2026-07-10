# WhoSpeaksLive

**See who is speaking while the conversation is still happening.**

WhoSpeaksLive is a private, real-time speaker diarization suite. Speaker diarization means identifying which speaker talked when. The core diarization and transcription stack can run entirely on hardware you control, so sensitive meeting audio does not need to be uploaded to a hosted transcription service.

Live speaker labels appear as audio arrives. Completed sentences then receive more stable final speaker labels once the system has enough context.

## Why WhoSpeaksLive

- **Private by design:** run locally or on self-hosted GPU servers; no third-party cloud service is required for the core pipeline.
- **Real-time by default:** see who is speaking during the meeting, not only after the recording is finished.
- **Fast now, stable when final:** a low-latency live path gives immediate feedback, while a separate final path uses more context for speaker-labeled transcripts.
- **Built for review and iteration:** manage speaker libraries, correct assignments, and validate diarization behavior from the browser UI.

Use it when the audio is sensitive, the answer is needed immediately, or both: internal meetings, research interviews, legal or medical workflows, local media analysis, and any environment where sending raw conversation audio to another company is not acceptable.

## Demo

https://github.com/user-attachments/assets/2de749e0-6c02-47de-b949-bd90b4f4efbb

For faster realtime ASR preview text like shown in the demo, use [Kroko Pro/commercial streaming models](https://docs.kroko.ai/on-premise/#2-commercial-oem-models); the public Community models work, but Pro/private models must be installed and licensed separately.

## Start Here

For a guided setup, install the lightweight CLI and let it inspect the machine:

```powershell
pip install whospeaks
whospeaks
```

The full-screen setup application opens on the Setup tab. Select the full local stack, the core/controller for remote ASR and embeddings servers, or the ASR/embeddings server packages, then choose whether to include Kroko realtime text. Kroko remains optional because its native runtime may require Python 3.12, Docker Desktop on Windows, or a prebuilt `kroko_onnx` wheel.

The `whospeaks` setup application keeps component readiness, diagnostics, settings, installation progress, logs, cancellation, and browser launch in one terminal interface. Run `whospeaks --classic` when the full-screen terminal interface is unavailable or you prefer the numbered menu.

The short `whospeaks` command is a setup and launcher wrapper. It saves a small profile, runs doctor checks, and expands that profile into the longer `whospeaks-window ...` browser-server command when you launch.

For a Linux container server:

```bash
docker build -t whospeaks:local .
docker run --rm --name whospeaks -p 8796:8796 -v whospeaks-data:/data -v whospeaks-models:/models whospeaks:local
```

See [Docker](docs/docker.md) for the full build/run path and volume notes.

For a manual full working setup, follow these in order:

1. [Installation](docs/installation.md): install the Windows controller.
2. [External ASR and embeddings servers](docs/external-servers.md): set up the Linux GPU services.
3. [Quickstart](docs/quickstart.md): verify a local or remote smoke run, then run the tuned provider stack.

## Current Scope

WhoSpeaksLive performs best on clean recordings where one person speaks at a time into good microphones. Diarization accuracy can degrade with background noise, background music, echo, crosstalk, overlapping speech, or low-quality microphones, and it may become less reliable as the active speaker count grows. The system assumes complete utterances can be assigned to a single speaker, so cases where one speaker starts a sentence and another finishes it are not expected to score well.

All Kroko languages supported by this integration work with realtime preview text: German, English, Spanish, French, Italian, Hebrew, Dutch, Portuguese, Swedish, and Turkish. Set `--language` or `WHOSPEAKS_LANGUAGE` to keep final ASR, Kroko/Banafo preview model selection, and stream2sentence sentence splitting on the same language. See the [configuration guide](docs/configuration.md#language) for language codes and model details.

Without realtime preview text, WhoSpeaksLive can also work with additional languages. The key requirement is that the language is supported by Whisper and by at least one configured sentence segmenter, meaning NLTK or Stanza.

That currently makes these additional languages principally supported without realtime preview text: Afrikaans, Arabic, Belarusian, Bulgarian, Catalan, Czech, Welsh, Danish, Greek, Estonian, Basque, Persian, Finnish, Faroese, Galician, Hindi, Croatian, Hungarian, Armenian, Indonesian, Icelandic, Japanese, Georgian, Kazakh, Korean, Latin, Lithuanian, Latvian, Malayalam, Marathi, Maltese, Myanmar/Burmese, Norwegian, Norwegian Nynorsk, Polish, Romanian, Russian, Sanskrit, Sindhi, Slovak, Slovenian, Albanian, Serbian, Tamil, Telugu, Thai, Ukrainian, Urdu, Vietnamese, and Chinese.

CPU-only operation is not the recommended path for the current stack. The system is GPU-heavy today; a CPU-only setup may be possible, but should be treated as a separate optimization target and will likely require engineering work, slower processing, and some accuracy or throughput tradeoffs.

## License

WhoSpeaksLive's own code is licensed under the [MIT License](LICENSE).

Optional Kroko/Banafo preview support uses separately licensed upstream components and model files. Missing public Community preview models are downloaded automatically from Hugging Face when realtime preview starts. This repository's MIT license does not relicense Kroko/Banafo assets; before downloading, bundling, or deploying them, review and respect the current terms from [Kroko by Banafo](https://kroko.ai/), the [Banafo/Kroko-ASR model card](https://huggingface.co/Banafo/Kroko-ASR), and the [kroko-ai/kroko-onnx repository](https://github.com/kroko-ai/kroko-onnx).

## Documentation

| Topic | Document |
| --- | --- |
| Documentation map | [docs/index.md](docs/index.md) |
| Product overview and use cases | [docs/overview.md](docs/overview.md) |
| Installation | [docs/installation.md](docs/installation.md) |
| macOS setup | [docs/macos-setup.md](docs/macos-setup.md) |
| Quickstart | [docs/quickstart.md](docs/quickstart.md) |
| Live window workflow | [docs/live-window-workflow.md](docs/live-window-workflow.md) |
| Meeting intelligence server | [docs/meeting-intelligence-server.md](docs/meeting-intelligence-server.md) |
| Speaker libraries | [docs/speaker-libraries.md](docs/speaker-libraries.md) |
| External ASR and embeddings servers | [docs/external-servers.md](docs/external-servers.md) |
| Docker server image | [docs/docker.md](docs/docker.md) |
| Configuration guide | [docs/configuration.md](docs/configuration.md) |
| Technical description | [docs/technical-description.md](docs/technical-description.md) |
| Technical architecture | [docs/architecture.md](docs/architecture.md) |
| Validation and scoring | [docs/validation-and-scoring.md](docs/validation-and-scoring.md) |
| Modal deployment | [docs/modal-deployment.md](docs/modal-deployment.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Development workflow | [docs/development.md](docs/development.md) |

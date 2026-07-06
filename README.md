# WhoSpeaksLive

WhoSpeaksLive is a local-first speaker diarization app for turning live or replayed media into speaker-labeled transcripts, with a browser UI for fast live speaker feedback, final sentence assignment, speaker library management, and validation.

## Important Notes

WhoSpeaksLive performs best on clean recordings where one person speaks at a time into good microphones. Diarization accuracy can degrade with background noise, background music, echo, crosstalk, overlapping speech, or low-quality microphones, and it may become less reliable as the active speaker count grows. The system assumes complete utterances can be assigned to a single speaker, so cases where one speaker starts a sentence and another finishes it are not expected to score well.

Language support is adaptable in principle, but the published project should be treated as English-first. The remote ASR language can be overridden with `WHOSPEAKS_ASR_LANGUAGE`, but full support for another language still needs explicit integration and validation of the transcription, sentence-boundary, and scoring behavior. If you need another language, ask your agent harness to integrate and validate that language explicitly.

CPU-only operation is not the recommended path for the current stack. The system is GPU-heavy today; a CPU-only setup may be possible, but should be treated as a separate optimization target and will likely require engineering work, slower processing, and some accuracy or throughput tradeoffs.

## License

WhoSpeaksLive's own code is licensed under the [MIT License](LICENSE).

Optional Kroko/Banafo preview support uses separately licensed upstream components and model files. This repository's MIT license does not relicense Kroko/Banafo assets; before downloading, bundling, or deploying them, review and respect the current terms from [Kroko by Banafo](https://kroko.ai/), the [Banafo/Kroko-ASR model card](https://huggingface.co/Banafo/Kroko-ASR), and the [kroko-ai/kroko-onnx repository](https://github.com/kroko-ai/kroko-onnx).

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

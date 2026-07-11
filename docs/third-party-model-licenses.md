# Third-Party Model Licenses

## NVIDIA Nemotron 3.5 ASR Streaming

WhoSpeaks can optionally download Nemotron 3.5 streaming ASR model weights through the upstream `k2-fsa/sherpa-onnx` release. The `sherpa-onnx` runtime is Apache-2.0. The Nemotron model weights are provided under NVIDIA Open Model Development and Weight License 1.1 (OpenMDW-1.1).

The optional backend downloads directly from the upstream release and verifies a pinned SHA-256 archive checksum. WhoSpeaks does not redistribute the model weights. Review the upstream model license before enabling automatic download in a deployment.

Nemotron remains an experimental realtime-preview option. It does not replace final ASR, and Kroko remains available for users who do not accept the model-license terms or need Hebrew realtime preview.

## Optional translation models

WhoSpeaks contains adapters and a sidecar server, not translation model weights. Enabling a local profile downloads the selected checkpoint from its upstream host. The repository's MIT license applies to the WhoSpeaks integration code and does not relicense any checkpoint.

### TranslateGemma 4B IT

`google/translategemma-4b-it` is distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the related prohibited-use policy. Hugging Face requires the downloading user to review and accept those terms before it grants access. The adapter reports this acceptance requirement in `/health` and the live translation status payload.

### NLLB-200 distilled 600M

`facebook/nllb-200-distilled-600M` is distributed under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/). The optional profile is intended for this project's non-commercial use. Keep the required attribution and license notice, do not use the weights commercially, and review the upstream model card's research/single-sentence/not-for-production-deployment limitations. WhoSpeaks never bundles these weights in its wheel.

### MADLAD-400 3B MT

`google/madlad400-3b-mt` identifies its checkpoint license as [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). Preserve the license and any required notices when redistributing model artifacts. Broad language-tag coverage is not a guarantee of translation quality for every language or domain.

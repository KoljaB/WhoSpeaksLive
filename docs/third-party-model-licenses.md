# Third-Party Model Licenses

## NVIDIA Nemotron 3.5 ASR Streaming

WhoSpeaks can optionally download Nemotron 3.5 streaming ASR model weights through the upstream `k2-fsa/sherpa-onnx` release. The `sherpa-onnx` runtime is Apache-2.0. The Nemotron model weights are provided under NVIDIA Open Model Development and Weight License 1.1 (OpenMDW-1.1).

The optional backend downloads directly from the upstream release and verifies a pinned SHA-256 archive checksum. WhoSpeaks does not redistribute the model weights. Review the upstream model license before enabling automatic download in a deployment.

Nemotron remains an experimental realtime-preview option. It does not replace final ASR, and Kroko remains available for users who do not accept the model-license terms or need Hebrew realtime preview.

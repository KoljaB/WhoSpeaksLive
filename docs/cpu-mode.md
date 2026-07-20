# CPU-only mode

CPU-only mode runs realtime transcription, final word alignment, speaker embeddings, VAD, and the browser controller without allocating GPU memory.

## Quality architecture

A forced aligner locates a transcript's known words in audio without generating a different transcript. WhoSpeaksLive uses a two-stage final path:

1. Kroko Community 64L produces low-latency live text and the fixed final transcript.
2. Faster-whisper `base`, running as an INT8 CPU aligner, places those exact words on the audio timeline.
3. Alignment confidence, word mapping, monotonicity, and audio bounds are checked. Unsafe results fall back to the streaming recognizer's native word starts and local energy-based word ends.
4. SpeechBrain ECAPA produces final and live speaker embeddings on CPU.

Whisper does not perform a second transcription in this mode, so it cannot replace Kroko's wording. In the local evaluation corpus, `base` alignment was more precise than `tiny`, especially at the final word boundary used for sentence splitting.

## CPU budget

The production profile uses two Kroko threads, two CTranslate2 alignment threads, and one PyTorch/OpenMP thread for the embedding helper. Final alignment runs only after an endpoint; it is not a continuous decoder. These limits target short CPU bursts below 30% on a typical 8-core/16-thread desktop and a substantially lower meeting-wide average. Actual percentages depend on logical CPU count, audio cadence, and other applications.

Do not raise `CPU alignment threads` above two merely to reduce endpoint latency. On machines with fewer than eight logical processors, select Whisper Tiny in Settings if the measured peak is too high.

## Install and launch

From the desktop launcher, choose **CPU only**, Kroko/Banafo, and the Community 64L model. The quality alignment model defaults to **Base**.

For unattended setup:

```powershell
whospeaks install --target cpu --installer uv --torch cpu --yes
whospeaks config --mode cpu --cpu-alignment-model base --cpu-alignment-threads 2 --set reports_enabled=false --set translation_enabled=false
whospeaks launch
```

The first run downloads the selected Kroko and faster-whisper weights. Kroko runs through the prebuilt `sherpa-onnx` wheels installed by the CPU plan, so the normal setup needs neither Docker nor a second Python environment. The Kroko archives are pinned by filename and SHA-256 and support English, German, Spanish, and French. All inference remains local after the downloads complete.

The explicit `whospeaks install-kroko --build` command remains available for developers who need the native `kroko_onnx` compatibility runtime. On Windows that optional source build uses Docker Desktop and CPython 3.12; it is not part of the launcher CPU installation.

## Operational checks

Run `whospeaks doctor --mode cpu --deep` before a production meeting. It checks the CPU embedding runtime, faster-whisper aligner, model cache, sherpa-onnx runtime, and managed Kroko model folder.

Watch the Activity page for `CPU forced alignment rejected`. Occasional fallback is safe; repeated fallback means the language/model does not match the audio, the transcript is badly wrong, or the audio window is unsuitable for forced alignment.

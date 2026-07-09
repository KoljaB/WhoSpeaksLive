# macOS Setup

Run the controller and both remote-style servers locally on Apple Silicon. This is an alternative to the Windows-controller + Linux-GPU-server topology in [Installation](installation.md); everything below runs on one Mac.

## Controller Setup

```bash
git clone https://github.com/KoljaB/WhoSpeaksLive.git
cd WhoSpeaksLive
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-controller.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/whospeaks-window --help
```

## ASR Server (CPU, int8)

CTranslate2 (faster-whisper's inference backend) has no MPS/Metal device, so ASR always runs on CPU on macOS. `int8` compute type keeps this fast enough for real use.

```bash
cd vendor/remote_servers/faster-whisper-asr
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ASR_MODEL=small ASR_DEVICE=cpu ASR_COMPUTE_TYPE=int8 ASR_HOST=127.0.0.1 ASR_PORT=8650 .venv/bin/python asr_server.py
```

`small` is a good default for responsiveness. If you want higher accuracy and can spend more latency, `large-v3-turbo` also runs on CPU/int8, but measured on an M3 Max it manages only about **1.2x realtime** (6.1s of audio transcribed in ~5.0s) — noticeably slower than `small`. It is not recommended as the default; if you need turbo-level accuracy, prefer the MLX server below instead of CPU turbo.

### Optional: MLX Whisper Server (faster, Apple Silicon only)

`mlx_asr_server.py` in the same directory runs `large-v3-turbo` through `mlx-whisper` on the Neural Engine/GPU instead of CPU. Measured on the same M3 Max: **~12x realtime** steady state (6.1s audio in ~0.5s), vastly faster than the CPU turbo path above. It implements the same `/health` and `/transcribe-window` contract the controller uses (`src/window/window_remote_asr.py`), so it's a drop-in alternative — just point `--remote-asr-url` at its port instead.

```bash
cd vendor/remote_servers/faster-whisper-asr
.venv/bin/python -m pip install mlx-whisper
ASR_HOST=127.0.0.1 ASR_PORT=8651 .venv/bin/python mlx_asr_server.py
```

Caveats: mlx-whisper is greedy-decode only (no beam search) — `beam_size` is accepted for API compatibility but ignored. Not usable off Apple Silicon.

## Embeddings Server (MPS)

```bash
cd vendor/remote_servers/voice-embeddings-server
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
EMBEDDINGS_HOST=127.0.0.1 EMBEDDINGS_DEVICE=auto PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python embeddings_server.py
```

Device auto-detection picks `mps` when available (falls back to `cpu`, then `cuda` if present). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op without an MPS kernel silently falls back to CPU instead of crashing.

### Provider status on MPS (M3 Max, this checkout)

Install the full provider set with `pip install -r requirements.txt` (skip `nvidia-ml-py`, it's a no-op off Nvidia hardware). Every provider below was tested with one `/embed` call (`device=auto`) against the running server.

| Provider | Device used | Result |
| --- | --- | --- |
| `speechbrain_ecapa` | mps | Works. Needs the `Pretrained.device_type` workaround already applied in `embeddings_server.py` (SpeechBrain 1.1.0 never sets this attribute for `mps`, which otherwise crashes `TorchAutocast` construction). |
| `speechbrain_resnet` | mps | Works (same SpeechBrain code path/workaround as `speechbrain_ecapa`). |
| `resemblyzer` | mps | Works out of the box. |
| `espnet_ecapa_wavlm_joint` | mps | Works. First call downloads a WavLM checkpoint (multi-GB) — expect several minutes on first load. |
| `wespeaker_campplus` | mps | Works, but needed `onnxruntime` (see below — added to `requirements.txt`). |
| `wespeaker_resnet34_lm_onnx` | — | **Broken.** Despite the name, this provider routes through `pyannote.audio`'s `PretrainedSpeakerEmbedding`, not the native wespeaker/onnx path. Blocked by the same torchaudio incompatibility below. |
| `pyannote_wespeaker_resnet34_lm` | — | **Blocked/broken.** Needs `HF_TOKEN` (gated model) *and* is broken independently by the torchaudio incompatibility below — never gets far enough to hit the auth check. |

**Known blocker: `pyannote.audio` 3.3.2 is incompatible with current torchaudio.** `pyannote.audio`'s import chain calls now-removed torchaudio APIs (`torchaudio.AudioMetaData` as a type annotation, then `torchaudio.list_audio_backends()`, and more beyond that) that don't exist in torchaudio 2.11.0 — the newest torchaudio release currently on PyPI, which no longer tracks recent torch versions (torch itself is at 2.12.1+). There is no pinned-version combination on PyPI today that satisfies both `torch>=2.x` (for MPS) and a `pyannote.audio`-compatible `torchaudio`. This affects both pyannote-backed providers above; fixing it needs either an upstream pyannote.audio release or vendoring a compatibility shim deeper than a one-line monkeypatch (multiple removed APIs, not just one). Not fixed here — flagging for follow-up.

Recommended macOS stack given the above: `speechbrain_ecapa` (already the default) or `espnet_ecapa_wavlm_joint` for higher quality; avoid the two pyannote-backed providers until the torchaudio compatibility issue is resolved upstream.

See [External Servers](external-servers.md) for the full provider list and stack recommendations; this page only tracks macOS/MPS-specific behavior.

## Live Capture: BlackHole + Multi-Output Device

macOS has no WASAPI loopback equivalent, so live system-audio capture needs a virtual audio device that mirrors output back as an input.

1. Install BlackHole 2ch:
   ```bash
   brew install blackhole-2ch
   ```
2. Open **Audio MIDI Setup** (Applications > Utilities), click **+** > **Create Multi-Output Device**, and check both your normal output (e.g. built-in speakers/headphones) and **BlackHole 2ch**. This lets you hear audio normally while it's also routed to BlackHole.
3. Set the Multi-Output Device as your system output (System Settings > Sound, or via the menu bar volume icon) while capturing.
4. Run `whospeaks-realtime`; it enumerates input devices and auto-selects the one matching "BlackHole" by name. Override with `--input-device-index N` if you have more than one candidate device.

If no BlackHole device is found, `whospeaks-realtime` raises a clear error listing available input devices and the install steps above.

`pyaudio` requires `portaudio`:

```bash
brew install portaudio
.venv/bin/python -m pip install pyaudio
```

## Known Limitations

- **CTranslate2 has no MPS backend.** ASR (faster-whisper) always runs on CPU on macOS; use `int8` compute type and/or the MLX server above for speed.
- **Gated pyannote models** (`pyannote_wespeaker_resnet34_lm`, `pyannote_embedding`) need a Hugging Face token that has accepted the model terms. Set `HF_TOKEN` before starting the embeddings server (see [External Servers](external-servers.md)); without it these providers are blocked, not broken.
- **No WASAPI-equivalent loopback API.** Live capture depends on the BlackHole virtual device and manual Multi-Output Device routing described above; there's no automatic system-audio tap on macOS.

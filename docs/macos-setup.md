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

| Provider | Device used | Result |
| --- | --- | --- |
| `speechbrain_ecapa` | mps | Works. Needs the `Pretrained.device_type` workaround already applied in `embeddings_server.py` (SpeechBrain 1.1.0 never sets this attribute for `mps`, which otherwise crashes `TorchAutocast` construction). |

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
- **`jungjee_rawnet3`** is not provisioned in the public source snapshot (its RawNet3 artifact isn't included) — same limitation as on Linux, see [External Servers](external-servers.md).
- **No WASAPI-equivalent loopback API.** Live capture depends on the BlackHole virtual device and manual Multi-Output Device routing described above; there's no automatic system-audio tap on macOS.

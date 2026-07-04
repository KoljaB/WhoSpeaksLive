# faster-whisper ASR server

GPU-backed local transcription server for short audio chunks.

Base URL from LAN:

```text
http://192.168.178.22:8650
```

Endpoints:

- `GET /health`
- `POST /transcribe` - default encoded-audio route, decoded in memory
- `POST /transcribe-memory` - explicit encoded-audio memory route
- `POST /transcribe-pcm16` - fast route for raw mono 16 kHz PCM16 or float32
- `POST /transcribe-window` - WhoSpeaks replacement route for raw mono 16 kHz float32 windows
- `POST /transcribe-file` - legacy temp-file route for comparison
- `POST /v1/audio/transcriptions` - OpenAI-style path, decoded in memory

WhoSpeaks replacement route:

```bash
curl --data-binary @window.f32le   -H 'Content-Type: application/octet-stream'   'http://192.168.178.22:8650/transcribe-window?sample_rate=16000&encoding=float32'
```

`/transcribe-window` defaults match the local faster-whisper call used by `youtube_window_diarize_gui.py`:

```text
language=en
task=transcribe
beam_size=5
word_timestamps=true
vad_filter=false
condition_on_previous_text=false
```

The response includes `segments` and a top-level `words` list with `word`, `text`, `start`, `end`, and `probability`.

Fast raw PCM16 chunk example:

```bash
curl --data-binary @chunk.s16le   -H 'Content-Type: application/octet-stream'   'http://192.168.178.22:8650/transcribe-pcm16?sample_rate=16000&encoding=pcm16&word_timestamps=true'
```

Multipart encoded audio example:

```bash
curl -F file=@chunk.wav 'http://192.168.178.22:8650/transcribe?word_timestamps=true'
```

Useful query overrides include `beam_size`, `word_timestamps`, `vad_filter`, `condition_on_previous_text`, `initial_prompt`, `hotwords`, `temperature`, threshold options, and punctuation options.

The model is `large-v2`, CUDA `float16`, and English by default.

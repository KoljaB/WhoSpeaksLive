# Repository Structure

WhoSpeaks uses a `src/` layout for first-party Python code and keeps mutable
runtime data out of importable packages.

```text
src/whospeaks/      first-party Python package
vendor/            copied third-party package sources
tests/             unit tests and small fixtures
docs/              project notes and porting history
tools/             compatibility wrappers for old script paths
runtime/           ignored local caches, models, media, outputs, speakers
```

Runtime defaults can be redirected with:

- `WHOSPEAKS_RUNTIME_DIR`
- `WHOSPEAKS_CACHE_DIR`
- `WHOSPEAKS_MODEL_DIR`
- `WHOSPEAKS_SPEAKER_LIBRARY_DIR`

Small deterministic validation data belongs under `tests/fixtures/`. Downloaded
models, user uploads, generated WAV files, traces, and benchmark outputs belong
under `runtime/`.

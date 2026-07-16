# PyPI Package Scope

The base `pip install whospeaks` or `uv tool install whospeaks` stays lightweight by installing the setup and launcher application first, then adding model runtimes only after the user chooses a deployment mode.

## Base Install

The only required third-party dependency of the base package is Textual, which powers the terminal setup application. GPU frameworks, speech models, media libraries, translation models, and server stacks remain optional dependencies or are installed by the guided setup for the selected target.

pip and uv install the same wheel and therefore have the same base-package boundary. The launcher keeps pip as the compatibility default and can use uv as an explicit backend for the later optional dependency sets. This changes resolution and installation speed, not which files belong to the published WhoSpeaks package.

The wheel contains only files needed by an installed WhoSpeaks runtime:

- Python packages under `src/` and the bundled runtime libraries under `vendor/`.
- Browser HTML, CSS, and JavaScript used by the live and report interfaces.
- Language flags, report-template presets, and the RealtimeSTT warm-up audio asset.
- The managed Apple Silicon service launcher, its two service implementations, and the macOS embeddings requirements file.
- Package metadata, the README, and required first- and third-party license notices.

## Excluded Material

Release archives do not include repository-only or machine-local material:

- Python and JavaScript test suites.
- Development tools, private notes, and the documentation tree.
- Local audio, transcripts, sessions, caches, model files, virtual environments, screenshots, logs, and diagnostics.
- Environment files, secrets, and agent/workspace instructions.

The source archive follows the same boundary. It contains the build metadata and runtime source required to build the wheel, but not the repository's tests, tools, or documentation.

## Release Check

For every release, inspect both the wheel and source archive before upload. The wheel must contain the runtime assets and licenses listed above, and neither artifact may contain repository-only paths such as `tests/`, `tests-js/`, `tools/`, `docs/`, `docs-private/`, `.env`, runtime data, or caches.

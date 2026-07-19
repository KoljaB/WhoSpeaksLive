# Development

Develop against small, repeatable checks first, then validate full media workflows only when behavior changes justify it.

## Repository Layout

- `src/`: application source.
- Console commands are declared in `pyproject.toml` and point directly at package modules.
- `tests/`: regression tests and deterministic fixtures.
- `vendor/`: copied third-party or external source snapshots used by the project.
- `vendor/remote_servers/`: ASR and embeddings server snapshots copied from the Linux GPU host.
- `runtime/`: local mutable data, ignored by Git.
- `docs/`: public documentation.

## Common Checks

Run the core regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_core_*.py"
```

Run the desktop controller and GUI checks without contacting services:

```powershell
$env:PYTHONPATH = "src"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest tests.test_whospeaks_launcher_controller tests.test_whospeaks_gui
```

Capture a deterministic launcher state and compare it with its design reference:

```powershell
.\.venv\Scripts\python.exe -m whospeaks_gui.main --demo-state ready --screenshot .tmp-ready.png
.\.venv\Scripts\python.exe tools\compare_gui_screenshots.py design-artifacts\desktop-launcher-v1\design-master-v1.png .tmp-ready.png .tmp-ready-comparison
```

Screenshot mode requires a demo state by design, so it cannot launch processes, probe ports, or write the real profile.

Check whitespace before committing:

```powershell
git diff --check
```

Check copied standalone Python files when editing server snapshots:

```powershell
.\.venv\Scripts\python.exe -m py_compile vendor\remote_servers\faster-whisper-asr\asr_server.py vendor\remote_servers\voice-embeddings-server\embeddings_server.py
```

Check the controller install path after packaging changes:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-controller.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\whospeaks-window.exe --help
```

## Documentation Rules

Public docs should explain stable workflows, concepts, and supported commands.

When adding or renaming a command-line flag, update [CLI reference](cli-reference.md) in the same change. The reference is intentionally complete so a reader can understand any parameter they see in `--help`, launch profiles, validation commands, or helper scripts.

Keep these out of committed docs:

- One-off optimization logs.
- Local screenshots and drafts.
- Machine-specific debugging transcripts.
- Experimental scores without enough context to reproduce them.

## Commit Hygiene

Before committing:

1. Inspect `git status -sb`.
2. Review `git diff --stat`.
3. Run relevant tests.
4. Stage only intentional files.
5. Confirm `git diff --cached --check`.

The repository may contain local runtime artifacts and generated caches. Do not force-add ignored files unless they are intentionally part of a source snapshot.

## Local Experiments

Some branches or local runs may include a `window.fact_lens_sidecar` module. Treat it as a very first implementation of an LLM-based transcript claim extraction experiment, not as a functional or supported fact-checking feature.

It is disabled by default and should remain opt-in only. Do not present it in public workflow docs, demos, or quickstart material until it has a validated product behavior, clear safety boundaries, and reproducible evaluation. At this stage it is extremely experimental: it may misclassify transcript fragments, does not perform reliable source verification, and should not be used as evidence that the application provides working fact checking.

If a developer needs to inspect it locally, start it explicitly with `--enable-llm` and pass the intended local OpenAI-compatible endpoint with `--llm-base-url`; otherwise it should only mirror final transcript events without creating claim cards or calling an LLM. Do not advertise it as a public feature until the evaluation and safety behavior are documented.

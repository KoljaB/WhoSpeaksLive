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
- `docs-private/`: local private/internal documentation archive, ignored by Git.

## Common Checks

Run the core regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core_regressions
```

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

Keep these out of public docs:

- One-off optimization logs.
- Private screenshots and drafts.
- Machine-specific debugging transcripts.
- Experimental scores without enough context to reproduce them.

Put private or internal notes under `docs-private/`. That directory is ignored by Git.

## Commit Hygiene

Before committing:

1. Inspect `git status -sb`.
2. Review `git diff --stat`.
3. Run relevant tests.
4. Stage only intentional files.
5. Confirm `git diff --cached --check`.

The repository may contain local runtime artifacts and generated caches. Do not force-add ignored files unless they are intentionally part of a source snapshot.

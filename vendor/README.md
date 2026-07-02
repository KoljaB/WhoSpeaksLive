# Vendored Dependencies

This directory contains copied third-party package sources that are required for
the local WhoSpeaks runtime snapshot.

- `RealtimeSTT/`: local RealtimeSTT source used by the realtime and preview flows.
- `RealtimeSTT_server/`: matching server/client utilities copied with RealtimeSTT.
- `stream2sentence/`: local sentence-boundary package used by the windowed text flow.

Do not put generated model files, virtual environments, or runtime outputs here.
When updating a vendored package, replace only that package directory, run the
unit tests, and verify the GUI `--help` command still imports successfully.

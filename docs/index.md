# Documentation

This documentation is organized by the task a reader is trying to complete.

## Full Working Install Path

Follow these in order for a new machine:

1. [Installation](installation.md): install `whospeaks`, run the starter CLI, and choose a setup mode.
2. [External servers](external-servers.md): prepare ASR and embeddings on the Linux GPU server.
3. [Quickstart](quickstart.md): run a local or remote smoke provider, then switch to higher-quality providers.
4. [Docker](docker.md): optionally build and run the packaged Linux server image.
5. [Troubleshooting](troubleshooting.md): use health checks and common failure fixes when a step fails.

## Start Here

- [Overview](overview.md): what WhoSpeaksLive does and where it fits.
- [Installation](installation.md): install the `whospeaks` starter CLI and choose local or remote setup.
- [Quickstart](quickstart.md): launch the browser app locally or after remote services are healthy.
- [Docker](docker.md): build and run the Linux containerized browser UI server.

## User Workflows

- [Live window workflow](live-window-workflow.md): run a media session, watch live speaker labels, and export results.
- [Meeting intelligence server](meeting-intelligence-server.md): generate LLM meeting reports from saved transcripts, review evidence, and recreate cached reports.
- [Custom reports](custom-reports.md): inspect predefined report templates, build reusable flat-section reports, and configure layouts, fields, language, privacy, and evidence.
- [Live translation](translation.md): configure compact multi-language transcript views and local, sidecar, or LLM translation providers.
- [Spanish executive meeting](spanish-executive-meeting.md): operate a six-person weekly leadership meeting with live Spanish transcription and an automatic Spanish report.
- [Speaker libraries](speaker-libraries.md): save, load, import, export, and reuse known speakers.
- [Public diarization events](public-events.md): subscribe to stable transcript and speaker-change events from Python tools.
- [Validation and scoring](validation-and-scoring.md): evaluate diarization and live-speaker behavior.
- [ElevenLabs Scribe baseline dataset](datasets/elevenlabs-scribe-baseline-dataset.md): local dataset map for 27 baseline videos, transcripts, sentence boundaries, and embeddings.

## System Setup

- [External servers](external-servers.md): run ASR and embeddings services on a Linux GPU machine.
- [macOS setup](macos-setup.md): run the controller and both servers locally on Apple Silicon.
- [Configuration](configuration.md): choose the most important runtime flags.
- [Speaker model presets](speaker-model-presets.md): compare the exact final/live embedding stacks exposed by the launcher.
- [Docker](docker.md): container build/run path, volumes, and validation commands.
- [CLI reference](cli-reference.md): look up every command-line parameter and environment variable.
- [Modal deployment](modal-deployment.md): deploy supported remote components on Modal.

## Technical Reference

- [CLI reference](cli-reference.md): complete parameter reference for installed commands and helper modules.
- [Meeting intelligence server](meeting-intelligence-server.md): standalone report-generation server, LLM setup, cache behavior, and evidence links.
- [Custom reports](custom-reports.md): report-template schema, predefined use-case reports, generation behavior, cache identity, and current limitations.
- [Technical description](technical-description.md): the concepts behind ASR windows, embeddings, live speakers, and final speakers.
- [Architecture](architecture.md): how audio, ASR, embeddings, memory, and the browser UI interact.
- [Troubleshooting](troubleshooting.md): common failures and concrete checks.
- [Development](development.md): tests, repo layout, and contribution workflow.

# Documentation

This documentation is organized by the task a reader is trying to complete.

## Full Working Install Path

Follow these in order for a new machine:

1. [Installation](installation.md): prepare the Windows controller and install console commands.
2. [External servers](external-servers.md): prepare ASR and embeddings on the Linux GPU server.
3. [Quickstart](quickstart.md): run the smoke provider, then switch to higher-quality providers.
4. [Troubleshooting](troubleshooting.md): use health checks and common failure fixes when a step fails.

## Start Here

- [Overview](overview.md): what WhoSpeaksLive does and where it fits.
- [Installation](installation.md): prepare the Windows controller and Python environment.
- [Quickstart](quickstart.md): launch the browser app after remote services are healthy.

## User Workflows

- [Live window workflow](live-window-workflow.md): run a media session, watch live speaker labels, and export results.
- [Speaker libraries](speaker-libraries.md): save, load, import, export, and reuse known speakers.
- [Validation and scoring](validation-and-scoring.md): evaluate diarization and live-speaker behavior.
- [ElevenLabs Scribe baseline dataset](datasets/elevenlabs-scribe-baseline-dataset.md): local dataset map for 27 baseline videos, transcripts, sentence boundaries, and embeddings.

## System Setup

- [External servers](external-servers.md): run ASR and embeddings services on a Linux GPU machine.
- [Configuration](configuration.md): choose the most important runtime flags.
- [Modal deployment](modal-deployment.md): deploy supported remote components on Modal.

## Technical Reference

- [Technical description](technical-description.md): the concepts behind ASR windows, embeddings, live speakers, and final speakers.
- [Architecture](architecture.md): how audio, ASR, embeddings, memory, and the browser UI interact.
- [Troubleshooting](troubleshooting.md): common failures and concrete checks.
- [Development](development.md): tests, repo layout, and contribution workflow.

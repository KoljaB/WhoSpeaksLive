# Security Policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository when
available. Do not include credentials, private audio, transcripts, or other
sensitive data in a public issue.

## Local credentials

WhoSpeaks reads service credentials from environment variables or an optional
local `.env` file. Local environment files are ignored by Git and are not part
of source distributions. Never commit populated credentials. If a credential is
ever exposed, revoke and rotate it immediately before removing it from history.

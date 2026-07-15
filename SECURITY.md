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

## Deployment boundary

The built-in browser server does not provide authentication, authorization roles, tenant isolation, or TLS termination. Run it on loopback or a trusted private network, or place it behind an authenticated TLS reverse proxy. The optional session lease coordinates live control and is not an authentication mechanism.

Voice recordings, voice embeddings, Person metadata, transcripts, and saved-session evidence are sensitive data stored as ordinary local files unless the surrounding operating environment encrypts and protects them. Review the full threat model, data map, backup procedure, and deletion limits in [Security And Data Privacy](docs/security-and-data-privacy.md).

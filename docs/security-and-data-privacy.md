# Security And Data Privacy

WhoSpeaks is designed for a trusted operator who controls the machine and network where meeting audio, transcripts, and voice-derived identity data are processed.

This document describes the current threat model and data lifecycle. It is operational guidance, not legal advice.

## Supported Threat Model

The built-in browser server does not provide user accounts, authentication, authorization roles, tenant isolation, or TLS termination. The optional session lease coordinates who controls a live demo session; it is not an authentication mechanism.

Use WhoSpeaks only on:

- loopback on a trusted workstation;
- a trusted private network with firewall controls;
- a VPN or SSH tunnel; or
- behind an authenticated TLS reverse proxy that you administer.

Do not expose the WhoSpeaks UI, embedding services, transcription services, or translation services directly to an untrusted LAN or the public internet. Anyone who can reach an unprotected instance may be able to view or change People, sessions, transcripts, and source media.

The Docker example publishes the UI on host loopback only. Binding `0.0.0.0:8796` on the host broadens access and requires your own authentication, TLS, and network policy.

## Voice Recognition Is Not Authentication

Voice matching is probabilistic. Recordings, microphones, noise, illness, imitation, replay, and model behavior can all change a score.

Use a suggestion to assist meeting labelling, never to:

- unlock an account or device;
- authorize a payment or legal action;
- establish physical presence;
- prove that a recording is genuine; or
- replace a human identity check.

## Operator Responsibilities

Before storing a Person or Voice sample:

- establish an appropriate purpose and legal basis for your context;
- inform participants that cross-meeting voice recognition is being used;
- disclose whether original manual-sample audio is retained;
- restrict access to the operator account, data directories, backups, and services;
- define a retention period and review it regularly; and
- provide a process for correction, access, and deletion requests where required.

WhoSpeaks does not collect consent, enforce a retention schedule, provide per-user access controls, or maintain a compliance audit log.

## Data Flow

### Local embedding provider

With a local provider, WhoSpeaks sends audio to the embedding model inside the same backend process or local worker environment. Results are voice embeddings: numeric representations used for similarity matching.

### Remote embedding provider

With a remote provider, the backend sends speech audio as PCM data to the configured embedding-server URL. The transport is only as secure as that URL and network. Plain `http://` does not encrypt audio or authenticate the server.

Use loopback, an isolated private network, a VPN, or TLS with authentication for any remote processing service. Apply the same rule to remote ASR, translation, and meeting-intelligence backends.

### Browser state

Browser/API responses omit raw embedding vectors and absolute retained-audio paths. They still include sensitive metadata such as Person names and IDs, inclusion and recognition settings, sample provenance, meeting titles and IDs, timestamps, durations, and quality summaries. Response redaction is not access control.

## Storage Map

Paths depend on command-line options and environment variables.

```text
--speaker-library-dir/
  people.json
  voice-samples/
    <person-id>/
      <sample-id>.<audio-extension>
  people.v1.<timestamp>.bak.json

--session-dir/
  <session-id>/
    manifest.json
    transcript.json
    speakers.json
    embeddings.json
    audio.wav or an external audio reference
    translations and generated reports, when enabled
```

`people.json` contains Person names, stable IDs, persistent settings, sample provenance, and embedding vectors. Manual uploads and microphone Voice samples retain their original audio under `voice-samples/`.

Saved sessions can retain transcript text, Person links, sentence-level voice embeddings, audio or an audio reference, corrections, translations, and generated reports. A session can therefore contain voice-derived evidence even after its Person link is removed.

Migration backup files preserve the pre-migration People document. They may contain identity data that was later removed from the active library.

## Protection At Rest

WhoSpeaks stores JSON and retained audio as ordinary files. It does not add application-level encryption or special permission hardening.

Protect the data with:

- a dedicated operating-system account;
- restrictive Windows ACLs or Unix permissions;
- full-disk or volume encryption;
- encrypted backups;
- a locked screen and protected credentials; and
- one trusted WhoSpeaks process per People library.

Do not run multiple processes against the same speaker-library directory. The library uses in-process locking and atomic file replacement, not a multi-process database transaction protocol.

## Shared-State And Multi-User Limits

One WhoSpeaks process owns one global People library. Every connected browser sees the same names, samples, inclusion roster, and recognition settings. The application is not multi-tenant.

Use separate processes and separate data roots for different teams, customers, legal matters, or other trust boundaries. Put authentication and TLS in front of each remotely accessible instance.

## Retention Behavior

- Inclusion, Recognition active, and sample Disable are matching controls, not deletion controls.
- Manual samples and explicitly user-confirmed meeting samples have no automatic expiry.
- Automatically derived meeting samples are capped at eight per embedding provider for each Person; weaker automatic samples may be pruned.
- Deleting a saved session removes Person-owned samples derived from that source session and suppression markers that identify it.
- Deleting a meeting-derived sample creates a suppression marker so saved-session recalculation does not silently recreate it.
- Forget voice data removes Person-owned samples and retained manual audio and removes that Person's saved identity links. Historical session embeddings and transcript labels remain unlabelled history.
- Delete person additionally removes the Person record. Historical transcript labels and unlabelled meeting evidence remain.

Review People, samples, sessions, migration backups, and external source files as separate retention categories.

## Backup And Restore

Legacy Speaker-group export is not a People backup.

For a consistent backup:

1. Stop every WhoSpeaks process using the data.
2. Copy the complete speaker-library directory, including `people.json`, `voice-samples/`, suppression state, and migration backups.
3. Copy the complete session directory at the same point in time if saved identity links and historical evidence must remain consistent.
4. Encrypt and access-control the backup.
5. Restart WhoSpeaks.

For restore:

1. Stop WhoSpeaks.
2. Restore both directory trees from the same backup point.
3. Verify ownership and permissions.
4. Start WhoSpeaks with the intended embedding provider.
5. Check that People, samples, and saved-session links are available and compatible.

Copying only sessions can leave Person IDs without corresponding People. Copying only People preserves recognition data but loses the source meetings and historical links.

Direct filesystem backup is currently the complete portability mechanism. Speaker-group files do not contain People or retained Person audio.

## Complete Erasure

Application deletion removes data from active WhoSpeaks-managed storage. It does not securely overwrite storage media or locate copies outside the configured directories.

For a complete-erasure request:

1. Use **Forget voice data** or **Delete person** as appropriate.
2. Delete saved sessions whose historical audio, embeddings, transcript names, or reports must also be removed.
3. Remove relevant v1 migration backup files after confirming they are no longer needed.
4. Remove original source audio and downloaded media outside WhoSpeaks.
5. Remove exported files and copies on other machines.
6. Expire or delete filesystem snapshots, container-volume backups, cloud backups, and offline backups under your retention policy.
7. Verify that remote processing services did not retain request audio or logs.

Secure erasure guarantees depend on the filesystem, storage device, backup system, and operating environment.

## Docker

The image stores mutable application data under the `/data` volume:

```text
/data/work
/data/output
/data/sessions
/data/speakers
```

`/data/speakers` contains the People library and retained manual Voice sample audio. Back up the whole `/data` volume consistently. The recommended `docker run` command publishes port 8796 on host loopback only.

## Internal Browser API

The `/api/people` and related routes are currently browser-internal interfaces, not a stable public API. They have no independent authentication boundary. Do not expose or automate them as though they were a secured identity service.

If you place WhoSpeaks behind a reverse proxy, protect read and mutation routes consistently, including saved-session open, rename, correction, Person link/unlink, and deletion operations.

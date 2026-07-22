# Spanish Executive Meeting

This workflow records one six-person weekly leadership meeting, shows Spanish speech as it is said, and automatically prepares a Spanish, evidence-linked report when the meeting ends.

The system uses one shared meeting microphone or mixed meeting audio. The live window transcribes it immediately and labels the current speaker; after a sentence is complete, it applies the more stable final speaker label. The meeting-intelligence server then reads the saved session and produces the Spanish executive summary, decisions, actions, questions, risks, and deadlines.

## Design

```text
Meeting-room microphone
        |
        v
WhoSpeaks live window (--language es, max 6 speakers)
        |
        +--> Immediate Spanish transcript + live speaker label
        |
        v
Saved, finalized meeting session
        |
        v
Meeting-intelligence server (--report-language es --auto-generate)
        |
        v
Spanish report with evidence links
```

`--max-speakers 6` is a guardrail, not a guarantee: it prevents a noisy room from creating unlimited meeting Speaker profiles. Create persistent People for the CEO and five managers before relying on names in the report.

## One-Time Preparation

1. Install the local stack or configure the remote GPU ASR and embeddings services following [Quickstart](quickstart.md).
2. Use a room microphone or conferencing source that captures every participant clearly. A microphone close to only the CEO cannot reliably label the other five people.
3. In the live window, create a Person for the CEO and each manager. Add a clean Voice sample or link each Person after several clean finalized sentences. In **Settings → People**, include only the six expected attendees in automatic recognition. See [People And Voice Recognition](people-and-recognition.md).
4. Obtain the required participant notice/authorization and have the company validate its retention, access, and deletion policy before recording meetings. Follow [Security And Data Privacy](security-and-data-privacy.md).

## Start The Weekly Meeting

Configure the live window once. `promoted_public` uses the currently promoted public final speaker stack and SpeechBrain ResNet for live assignment. In the `whospeaks` desktop launcher, open **Settings → Meeting Intelligence** to enable automatic report-server startup, choose Spanish or **Follow live language**, and set the LLM provider/model. Save the settings; the main **Launch WhoSpeaks** button then starts both the live window and the report server.

```powershell
whospeaks config `
  --language es `
  --provider-preset promoted_public `
  --advanced-args "--max-speakers 6"
```

Start it before the meeting:

```powershell
whospeaks launch
```

In the browser, verify that the six plausible attendees are included in automatic recognition, choose the meeting microphone/audio source, create a new session named for the weekly meeting, and press **Start**. Review every **Likely Person** suggestion with **Confirm** or **Not Person**; a suggestion is not proof of identity. At the end, let processing reach its normal finished state so the session is saved with status `Saved`; do not close the window while it is still finalizing sentences.

## Start Automatic Spanish Reporting

Start this server once and leave it running while the live window saves sessions. Use an LLM that can reliably write Spanish.

```powershell
whospeaks-meeting-intelligence `
  --host 127.0.0.1 `
  --port 8798 `
  --report-language es `
  --auto-generate `
  --llm-provider llama_cpp `
  --llm-base-url http://127.0.0.1:18081/v1 `
  --llm-model YOUR_SPANISH_CAPABLE_MODEL
```

The server checks for newly finalized saved sessions every 10 seconds, queues the report, and stores it locally. Existing saved sessions are ignored when the server starts, so only new weekly meetings are automatically processed. Open `http://127.0.0.1:8798/` to review the completed summary and evidence. The report language is stored with the cache, so an old English report is never presented as the current Spanish report.

## Expected Result

During the meeting, the CEO sees Spanish transcript rows and provisional/final speaker labels. Shortly after completion, the report page contains a Spanish executive summary plus evidence-linked decisions, action items, open questions, risks, deadlines, discussion threads, and participation. Review the report before distributing it: ambiguous speaker identification, ownership, and due dates are deliberately kept as uncertain rather than invented.

# Ask Sessions

Ask Sessions lets you question one live transcript, one saved session, or several selected sessions and receive answers linked to the exact supporting transcript rows.

## Start It

Launch the main browser together with Meeting Intelligence. In the `whospeaks` setup application, configure **Meeting Intelligence — Reports + Ask**, then launch the profile. From the command line, `--with-meeting-intelligence` enables the service; the older `--with-reports` name remains a compatibility alias.

The answer model uses the same freely configurable OpenAI-compatible provider settings as reports. Supported presets include llama.cpp, Ollama, LM Studio, OpenAI-compatible endpoints, OpenAI, and OpenRouter. API keys are read by the server and are never exposed to browser JavaScript.

## Choose The Scope

A scope is the exact set of sessions included in one conversation. The browser chooses it in this order:

1. Checked saved sessions, up to 20.
2. The currently opened saved session when no sessions are checked.
3. The running session when no saved session is selected or open.

For several saved sessions, check them in **Sessions** and click **Ask selected sessions**. The same combination of session IDs restores the same conversation history even if the selection order changes. Mixing saved and running sessions is not supported in the first version.

## Ask And Inspect Evidence

Open the **Ask** tab, enter a question, and click **Send**. The status area reports concrete stages such as indexing, retrieving, selecting evidence, and answering. Every grounded citation is displayed as `Session title · Speaker · timestamp`; clicking it opens the correct saved session and highlights the supporting transcript row.

For a running session, WhoSpeaksLive first autosaves finalized transcript rows. The answer is labeled with a cutoff such as `Live answer through 12:34`, so later speech cannot silently influence an earlier answer.

Speaker-specific questions use diarized speaker metadata. Diarization means assigning each transcript row to the person who spoke it. A trained known speaker receives an exact identity boost during retrieval, while an unknown speaker is never treated as that person merely because the words are relevant.

## Short And Long Transcripts

A single session of at most 8,000 words uses the complete transcript, so text embeddings are not required. Longer sessions and scopes containing several sessions use hybrid search: semantic search finds passages with similar meaning, while lexical search finds exact names, numbers, and phrases. The best candidates are checked by the answer model before the final response is generated.

Configure long and multi-session search with an OpenAI-compatible embeddings endpoint. Save the embedding settings in the launcher profile first, then start the live browser and Meeting Intelligence together:

```powershell
whospeaks config `
  --set text_embedding_base_url=http://127.0.0.1:18082/v1 `
  --set text_embedding_model=YOUR_EMBEDDING_MODEL

whospeaks launch `
  --with-meeting-intelligence `
  --report-llm-provider llama_cpp `
  --report-llm-base-url http://127.0.0.1:18081/v1 `
  --report-llm-model local
```

When the embedding endpoint needs authentication, also save `text_embedding_api_key_env=NAME` and place the secret in that environment variable. Text embeddings locate relevant words; they are separate from voice embeddings, which identify speakers.

## Indexing And Persistence

An index is a searchable local representation of transcript passages. New or changed finalized sessions are indexed in the background, while older sessions are indexed only when selected for Ask. The index and per-scope conversation histories are stored in runtime SQLite data, survive application restarts, and reuse unchanged transcript chunks.

Changing a speaker name or assignment invalidates the affected chunks so future answers use the correction. Changing the embedding endpoint or model rebuilds semantic vectors lazily. Deleting a session removes its indexed passages and conversations containing that session.

## Troubleshooting

- **Preparing session search does not advance:** confirm that the Meeting Intelligence service is running and inspect its launcher diagnostics. Long or multi-session scopes also need a healthy text-embedding endpoint.
- **A short session answers without embeddings:** this is expected because the complete transcript path does not need search.
- **A long or multi-session scope reports missing configuration:** set both the text-embedding URL and model; the app deliberately avoids an incomplete keyword-only fallback.
- **The answer says not established:** the selected transcript evidence did not support a grounded answer. Verify the session scope and speaker labels instead of assuming the model inferred correctly.
- **A citation opens the wrong place:** reopen the session and retry. Current citations carry canonical row IDs plus row positions so both current and older saved-session formats can be resolved.

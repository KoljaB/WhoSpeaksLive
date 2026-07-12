# Custom Reports

Custom reports let you decide what the system should look for in a finished speaker-labeled transcript and reuse that report design on later recordings.

A **report template** is an ordinary JSON document that describes a report. It contains a flat, ordered list of sections; each section states its goal, output fields, layout, item limit, sorting preference, and evidence requirement. Predefined and user-created templates use the same schema and the same generation pipeline.

An **evidence anchor** is a reference to one or more real transcript rows. Report items cite evidence anchors, and their evidence chips open and highlight the supporting transcript rows.

## Use The Report Builder

Use the **Report template** selector in the meeting-intelligence browser UI. Templates are separated into:

- **Predefined**: inspectable, read-only examples supplied with WhoSpeaks. Clone one to make an editable copy.
- **Custom**: reports created or cloned by the user and saved as local JSON files.

To create a report from scratch:

1. Click **New** to open **Report builder**.
2. Enter its name and optional description.
3. Choose whether its output language is inherited from the report server or fixed to one language.
4. Choose the inherited or local-only privacy policy.
5. Add a section and describe exactly what it should identify or summarize.
6. Configure its maximum items, evidence requirement, layout, sorting, and output fields.
7. Add, remove, or reorder more sections as needed.
8. Click **Save report template**, select a recording, and generate that report.

If the draft is invalid, the builder displays the exact validation error so the affected template or section field can be corrected before saving.

To learn from an existing design, select a Predefined report and click **Inspect**. The builder opens read-only. Close it, click **Clone**, give the copy a name, then use **Edit** under Custom. Predefined templates cannot be edited or deleted directly; Custom templates also offer **Delete**.

## Flat Sections

Reports intentionally have one structural level: a report contains sections, and each section contains report items. There are no nested subsections.

For example, a report can contain separate top-level sections named `Decisions`, `Deadlines`, and `Disagreements`. This keeps every section independently configurable, reorderable, and removable. A layout may visually group information, but it does not create another data hierarchy.

Each section supports:

| Setting | Purpose |
| --- | --- |
| Title | User-facing section name. |
| Objective | Plain-language instruction describing what to find and how to represent it. |
| Maximum items | Caps the number of returned items from 1 to 20. |
| Evidence required | Marks items without a valid evidence citation as missing required evidence. |
| Layout | `cards`, `table`, `timeline`, or `quotes`. |
| Sort order | `relevance`, `chronological`, or `severity`. |
| Output fields | Additional structured values requested for every item. |

A template can contain up to 16 sections.

## Configurable Output Fields

Every report item retains the common fields used by meeting intelligence: title, body, status, owner, due value, confidence, evidence links, and grounding status. A section can add its own fields, such as:

- severity and affected equipment for a shift handover;
- motion, proposer, and vote result for committee minutes;
- allegation status and party for mediation;
- scene, continuity issue, and production impact for film production;
- quote, speaker, and theme for qualitative research.

Supported field types are `text`, `enum`, `speaker`, `date`, `timestamp`, `boolean`, and `number`. An `enum` field also supplies its intended choices. Generated attributes whose keys are not declared in the section are discarded during normalization. In the MVP, attribute values remain strings; field types guide generation and presentation rather than providing full runtime type coercion.

## Template JSON

The builder creates the same JSON accepted by the API and used by the predefined examples:

```json
{
  "schema_version": "report_template_v1",
  "template_id": "custom.shift-safety",
  "name": "Shift safety review",
  "description": "Urgent and routine safety issues from a shift handover.",
  "version": 1,
  "builtin": false,
  "language_mode": "inherit",
  "privacy_policy": "local_only",
  "sections": [
    {
      "key": "urgent_safety",
      "title": "Urgent safety issues",
      "objective": "Identify safety issues that require immediate intervention. Do not convert routine maintenance into an emergency.",
      "max_items": 8,
      "evidence_required": true,
      "render_kind": "table",
      "sort_order": "severity",
      "output_fields": [
        {
          "key": "severity",
          "label": "Severity",
          "type": "enum",
          "description": "The supported urgency of the issue.",
          "options": ["critical", "high", "routine", "unclear"]
        },
        {
          "key": "equipment",
          "label": "Equipment",
          "type": "text",
          "description": "Affected machine or equipment when stated.",
          "options": []
        }
      ]
    }
  ]
}
```

Saving a changed custom template increments its version and calculates a deterministic revision hash. The generated report stores its template ID, revision hash, and complete template snapshot so the design used for that report remains inspectable.

## Predefined Reports

All predefined reports are ordinary JSON templates. They are not separate hard-coded generation paths. The browser exposes them as working examples that can be inspected, generated, and cloned:

| Template | Covered workflow |
| --- | --- |
| Standard Meeting Intelligence | Existing speaker map, summaries, decisions, actions, questions, risks, threads, disagreements, deadlines, participation, and grounded Q&A. |
| German works council minutes | Formal issues, positions, resolutions, responsibilities, and follow-up. |
| English podcast production | Chapters, notable quotes, summaries, and fact-check candidates. |
| French medical case conference | Case-separated findings, proposals, decisions, and next steps. |
| Italian film production | Scene decisions, schedule changes, equipment, continuity, and production risks. |
| Hebrew cybersecurity incident | Chronological events, observations, hypotheses, decisions, actions, and risks. |
| Dutch business mediation | Parties, positions, allegations, disputed claims, concessions, and tentative agreements. |
| Portuguese investigative newsroom | Story hypotheses, sources, verification tasks, and legal or ethical concerns. |
| Swedish qualitative research | Themes, participant viewpoints, recurrence, and representative quotations. |
| Turkish factory shift handover | Equipment status, workarounds, interruptions, maintenance, severity, and urgency. |
| Spanish municipal committee | Agenda items, arguments, motions, votes, decisions, and formal follow-up. |

The predefined files live in `src/window/report_template_presets`. Editing those files is a development operation; normal users should inspect and clone a preset through the Report template controls.

## Language And Privacy

`language_mode` has two behaviors:

- `inherit` uses the meeting-intelligence server's configured report language.
- A supported language code such as `de`, `fr`, or `he` fixes the generated report to that language.

Quoted transcript text remains in its original form. Changing the effective language makes an older cache stale.

`privacy_policy` can be `inherit` or `local_only`. A local-only template cannot be generated with the public OpenAI or OpenRouter providers; choose llama.cpp, Ollama, LM Studio, or mock mode. This policy controls report-model routing. It does not by itself provide user authentication, encryption, retention management, or deployment-level access control.

## How Generation Uses A Template

The generation flow is template-aware from its first model pass:

1. The final transcript is split into segments.
2. Every evidence pass receives a compact copy of all section objectives and output fields.
3. Extracted evidence anchors are tagged with the section keys they can support.
4. Each section pass receives its full definition, relevant evidence anchors, and global transcript context.
5. Returned evidence IDs are checked against real anchors, and custom attributes are filtered to configured field keys.
6. Items in evidence-required sections are omitted when no valid citation remains; the UI never presents an uncited claim as a report item.
7. The complete template snapshot and provenance are stored with the report.

The evidence budget scales with the number of sections, up to a bounded per-segment limit, so a larger template is not forced through the original eight-anchor budget.

## Multiple Reports For One Recording

Each cache is identified by both the recording session and template ID. You can therefore generate, for example, Standard Meeting Intelligence and Podcast Production reports for the same recording without one overwriting the other.

A cached report is current only when all of these still match:

- transcript and speaker-state revision;
- model provider and model ID;
- effective report language;
- template ID and template revision.

Editing a custom template makes its earlier report stale. Regenerate it to use the new definition. Deleting one cached report does not delete the recording, transcript, template, or reports made with other templates.

## Current MVP Limitations

- Templates and report caches are local JSON files; there is no account-level sharing, permissions model, or synchronized template library.
- The builder does not support nested subsections, arbitrary HTML, executable code, or unrestricted replacement system prompts.
- Evidence links prove which transcript span was cited. They do not independently prove that every conclusion logically follows from that span.
- The pipeline removes uncited items from evidence-required sections but does not perform external fact-checking or authoritative medical, legal, personnel, or incident validation.
- More sections increase model calls, latency, and token use; generation currently runs section passes sequentially.
- Background `--auto-generate` currently queues the Standard Meeting Intelligence template only; select or request other templates explicitly.
- A sampled transcript outline provides global context to each section, while detailed citations come from the section's evidence anchors. Very long or low-quality transcripts can still omit relevant material.
- High-stakes reports remain drafts that require human review.

For server startup, provider setup, browser workflow, caches, and HTTP endpoints, see [Meeting Intelligence Server](meeting-intelligence-server.md).

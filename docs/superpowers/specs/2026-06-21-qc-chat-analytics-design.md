# Agentic `/qc` Analytics and Topic Discovery Design

**Status:** Approved architecture, revised 2026-07-10

## Goal

Make `/qc` reliably answer three different classes of questions about the active Telegram chat:

1. recall questions about what people said;
2. exact, database-computable analytics such as counts, rankings, date ranges, and trends;
3. semantic topic-discovery questions such as “what have people discussed this week?”

The implementation must preserve chat isolation, clearly distinguish exact database results from LLM-assisted topic classification, and integrate the existing `feat/qc-analytics` work with the current `main` provider and authentication changes.

## Current branch findings

The feature branch has the correct foundation: a structured, read-only `chat_analytics` tool whose server-bound `chat_id` cannot be supplied by the model. It also contains good chat-isolation, aggregation, budget, and SQL-compiler tests.

Four issues prevent it from being considered complete:

- Accumulated result envelopes omit the metric, grouping, term, user filters, exclusions, and ordering that produced their rows. The separate composition call cannot reliably interpret multiple exploratory results.
- Invalid and inverted date bounds are silently discarded, widening an intended bounded query to all history.
- The approved design promises topic discovery, but the implementation exposes only `Recall` and `Analytics` lanes.
- The branch predates later changes on `main` and must be integrated without reverting current Codex/provider, access-control, or model-catalog behavior.

## Non-goals

- Raw SQL or model-selected table/column names.
- Cross-chat, global, token-usage, or administrative analytics.
- Calling a semantic topic count “exact.”
- Refactoring unrelated provider or handler code.
- Changing `/factcheck`, `/q`, `/s`, or `/tldr` behavior except where a shared helper must remain compatible.

## Architecture

`run_qc_pipeline` starts with a structured classifier that returns one of three lanes:

- `recall`
- `analytics`
- `topic_discovery`

The classifier prompt includes concise examples and treats the user text as untrusted. A malformed classifier response falls back to `recall`, which is the least expensive and least authoritative path. Questions containing attached media also stay on `recall` because the analytics and topic lanes operate on stored chat text.

Each lane has a separate evidence contract and final-composition prompt. No lane asks the final model to infer facts that its evidence contract cannot support.

## Lane 1: Recall

The existing planner → chat search → reflection → composition flow remains intact. It continues to use keyword FTS, optional web research, bounded evidence, verified Telegram message links, and the selected final model.

This lane must not gain access to `chat_analytics`; separating recall from analytics keeps ordinary questions cheap and prevents unnecessary database scans.

## Lane 2: Exact analytics

### Structured query

The model uses the existing allow-listed `QuerySpec` and `chat_analytics` tool. Rust remains authoritative for validation, SQL generation, chat scope, parameter binding, timeouts, row limits, and returned values.

Supported v1 operations remain deliberately small:

- metrics: `count`, `distinct_count`, `min_date`, `max_date`, `avg_len`;
- grouping: `user`, `day`, `hour_of_day`, `weekday`, `month`, `none`;
- filters: FTS term, literal substring, absolute date bounds, user id, username, and exclusion flags;
- ordering and a hard-capped row limit.

FTS terms must use the same normalization principles as `/s`. Empty normalized terms are rejected instead of becoming an unfiltered query.

### Date validation

The gathering prompt includes the current UTC timestamp and requires absolute `YYYY-MM-DD` or RFC3339 bounds. The tool normalizes valid values and rejects:

- a supplied value that cannot be parsed;
- `date_from >= date_to` after normalization.

Errors are returned as `invalid_arguments`, allowing the gathering model to retry. A bad bound must never be dropped and must never widen the query.

### Self-describing result envelope

Every successful result includes the normalized query that produced it:

```json
{
  "operation": "analytics",
  "query": {
    "metric": "count",
    "group_by": "user",
    "filters": {
      "term": null,
      "text_contains": null,
      "date_from": "2026-07-01T00:00:00+00:00",
      "date_to": "2026-07-08T00:00:00+00:00",
      "user_id": null,
      "username": null,
      "exclude_commands": true,
      "exclude_synthetic": true,
      "exclude_ai_asks": false
    },
    "order": "value_desc",
    "limit": 20
  },
  "coverage": {
    "chat": "active",
    "storage": "stored_text_messages",
    "timezone": "UTC",
    "anonymous_admin_and_channel_posts_excluded": true
  },
  "rows": [{"group": "alice", "value": 42}],
  "row_count": 1
}
```

The relevant analytics types derive `Serialize` as well as `Deserialize`; the envelope is created from the validated, normalized, and limit-clamped spec, not the raw model JSON.

### Gather and compose

The selected tool-capable model may refine several analytics queries. Rust accumulates successful self-describing envelopes. The model’s gather prose is discarded; a separate no-tools composition call receives the user question and newest-first result envelopes.

The composition prompt may select the relevant result, compare compatible results, and preserve database row ordering. It must not recompute values or combine incompatible filters. It must state the effective UTC range and the stored-text/exclusion coverage for numeric claims.

If the analytics lane produces no successful result, it returns a clear analytics failure. It does not fall back to the legacy monolithic loop, which cannot produce authoritative counts.

## Lane 3: Topic discovery

Topic discovery is semantic and therefore cannot be implemented as `GROUP BY text` or represented as exact SQL topic counts.

### Topic request planning

A structured planning step produces:

- absolute `date_from` and `date_to` values;
- requested topic count, clamped to `3..=10`;
- optional user filter;
- whether commands and synthetic rows are excluded.

Dates use the same strict normalization as analytics. If the user provides no range, the default is the rolling seven days ending at the current UTC timestamp. The final answer always states the effective range.

### Chat-scoped message window

The database exposes one purpose-built, read-only method that selects eligible text messages for the active `chat_id`, ordered newest first and bounded by the planned dates and configured maximum. It also performs a matching `COUNT(*)` so coverage is explicit. The eligibility count and newest-message selection run on one SQLite read transaction/snapshot, with separate timeouts, and the transaction commits only after both reads succeed.

The returned structure records:

- total eligible messages in the requested range;
- analyzed messages;
- whether the window was capped;
- the normalized user filter and exact exclusions used.

When capped, the implementation analyzes the newest configured messages. It does not extrapolate them to the omitted history. The final answer says, for example, “Analyzed the newest 2,000 of 8,431 stored text messages in this range.”

The implementation reuses `TLDR_CHUNK_SIZE` and `TLDR_MAX_MESSAGES` rather than adding duplicate size knobs. Topic map calls use a fixed concurrency of four; `/tldr` remains unchanged. `ENABLE_QC_TOPIC_DISCOVERY` is the only new topic-specific switch.

### Map phase

Messages are formatted with stable message ids, UTC timestamps, display labels, and verified Telegram links, then divided into existing TLDR-sized chunks.

Up to four map calls run concurrently. Each returns structured JSON containing a small set of candidate topics. Each candidate contains:

- a concise label;
- a one-sentence description;
- keywords found in the chunk;
- message ids assigned to that topic;
- at most two representative message ids.

The map prompt requires each substantive message to be assigned to at most one primary topic and permits routine chatter to remain unassigned. Rust removes unknown ids, duplicates ids within a candidate, and rejects empty candidates.

### Reduce phase

Rust assigns stable candidate ids before reduction. The reducer receives candidate labels/descriptions/keywords and returns clusters of existing candidate ids; it cannot supply message ids or numeric counts.

Rust validates every cluster id, prevents a candidate from appearing in multiple final topics, unions the validated message ids, and computes:

- `classified_message_count` for each final topic;
- percentage of all uniquely classified messages;
- representative examples whose message ids and links came from the selected window.

These are LLM-assisted classifications over the analyzed window. The final response labels them accordingly and never calls them exact semantic counts.

If the user explicitly asks how often a literal word or phrase appeared, that portion is delegated to `chat_analytics` through `filters.text_contains`, not the normalized FTS `term` filter. The result is the number of eligible stored-text messages containing the escaped literal substring, not the number of occurrences within those messages. Literal-substring results and semantic topic classifications are displayed separately. If an optional literal-substring query fails or times out, validated semantic topic evidence remains usable and the literal result is represented as structured unavailable metadata; composition must not invent the missing count.

### Topic composition

The final no-tools composition call receives only validated clusters, Rust-computed counts, coverage metadata, and curated examples. It summarizes the themes in the user’s language, identifies the analyzed range, discloses capped coverage, and may cite only the verified example links.

If all map calls fail or no valid candidates survive, `/qc` reports that topic discovery could not produce supported themes. It does not invent a fallback summary.

## Safety and privacy invariants

1. Every database query binds the active `chat_id` supplied by the server-side runtime.
2. Models cannot name tables, columns, SQL fragments, or another chat id.
3. Analytics and topic-window operations are read-only, parameterized, bounded, and timed out.
4. Only stored text/caption records are analyzed. Media-only, sticker, voice, service, and unrecorded edit events are absent by construction.
5. Anonymous-admin and channel-post exclusions are disclosed whenever they affect coverage.
6. Every message id returned by a map/reduce or composition step is validated against the active request’s selected rows before it can become a citation.

## Error handling

- Classifier failure: use `recall`.
- Invalid analytics/topic arguments: return a structured retryable error to the gathering/planning model.
- Search index rebuilding: analytics may retry without an FTS term; literal/topic window operations remain available.
- Query timeout: expose a bounded tool error and allow a narrower retry.
- Analytics without successful results: stop with an explicit failure; do not use legacy inference.
- Partially failed topic map: continue with successful chunks and disclose analyzed chunk/message coverage.
- Completely failed topic map or invalid reduction: stop with an explicit failure.
- Final composition failure: propagate the provider error through the existing handler path.

## Configuration

Retain the feature branch’s existing analytics settings:

- `QC_ANALYTICS_MAX_TOTAL_CALLS`
- `QC_ANALYTICS_MAX_QUERY_CALLS`
- `QC_ANALYTICS_QUERY_TIMEOUT_SECS`

Add:

- `ENABLE_QC_TOPIC_DISCOVERY` — default `true`; when false, topic requests return a clear unsupported message rather than being misrouted to exact analytics.

Topic message/chunk limits reuse `TLDR_MAX_MESSAGES` and `TLDR_CHUNK_SIZE`. Topic mapping uses a fixed concurrency of four and does not change `/tldr` execution.

## Integration with current `main`

Before feature implementation is considered complete, integrate current `main` into `feat/qc-analytics` and resolve overlaps additively:

- keep current Codex account-pinned authentication and model metadata;
- keep current Responses-provider behavior and strict completion handling;
- keep current access-control and callback protections;
- add analytics configuration fields without restoring removed configuration fields or old defaults;
- retain the current handler and audit behavior around `run_qc_pipeline`.

## Test strategy

### Analytics/compiler tests

- successful envelopes include the complete normalized query and coverage metadata;
- malformed dates and equal/inverted ranges return `invalid_arguments`;
- no supplied date silently disappears;
- FTS-only punctuation/empty normalized terms are rejected;
- all metric/group/order combinations compile to fixed SQL fragments;
- wildcard substring escaping, row limit clamping, exclusions, and UTC bucketing remain covered.

### Database/runtime tests

- two-chat sentinel tests prove analytics and topic windows cannot observe another chat;
- sentinels in audit/config tables never affect results;
- total/analyzed/capped topic coverage is correct;
- analytics and topic timeouts/budgets stop as configured;
- unsuccessful analytics calls do not create authoritative result envelopes.

### Topic pipeline tests

- planner defaults to the rolling seven-day UTC window and clamps topic count;
- unknown/duplicate map message ids are removed;
- a message cannot be counted in two map topics or two reduced topics;
- reducer clusters may reference only supplied candidate ids;
- Rust, not the model, computes final classified counts and percentages;
- capped and partial-map coverage is disclosed;
- representative citations are limited to selected active-chat message ids;
- exact keyword counts are kept separate from semantic classifications.

### Regression and delivery gates

- recall `/qc` behavior and citation verification remain unchanged;
- analytics and topic requests do not silently enter the legacy loop;
- `cargo fmt --check`;
- targeted analytics/topic tests;
- `cargo test`;
- `cargo build`;
- `cargo clippy --all-targets -- -D warnings`.

## Files expected to change

- `src/agents/qc.rs` — three-lane routing, analytics evidence composition, topic plan/map/reduce orchestration, and unit-testable validation helpers.
- `src/llm/analytics.rs` — serializable normalized query types, strict validation, and safe FTS/date handling.
- `src/llm/tool_runtime.rs` — self-describing analytics envelopes and retryable error classification.
- `src/db/database.rs` — analytics execution updates and the chat-scoped topic window/coverage query.
- `src/db/models.rs` — topic window and coverage row types if needed.
- `src/config.rs`, `.env.example`, `README.md` — topic gate and corrected capability/disclosure documentation.
- Existing adjacent test modules, following repository convention.

## Acceptance criteria

The work is complete when:

1. recall, exact analytics, and topic discovery route independently;
2. every numeric analytics answer can be traced to a self-describing normalized database result;
3. invalid date input cannot widen query scope;
4. semantic topic measurements are transparently labeled and computed from validated message assignments;
5. capped or partial topic coverage is visible in the answer;
6. cross-chat and cross-table isolation tests pass;
7. current `main` functionality remains present; and
8. the full formatting, test, build, and strict Clippy gates pass.

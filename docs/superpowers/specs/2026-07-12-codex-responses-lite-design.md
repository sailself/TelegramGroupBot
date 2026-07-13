# Codex Responses Lite Compatibility Design

## Problem

The Codex model catalog exposes `gpt-5.6-luna` as a selectable, API-supported model and marks it with `use_responses_lite: true`. The bot currently deserializes only a subset of the model metadata, so that flag is discarded. It then sends Luna through the normal Codex Responses contract, and the backend returns HTTP 404 before producing a response.

## Chosen approach

Implement Responses Lite as a catalog-driven model capability. Retain the remote flag in the selected-model record and use it to select the request contract. Do not hard-code GPT-5.6 slugs and do not hide lite-capable models.

No new dependency or configuration variable is required. Existing selected-model JSON files remain readable because missing `use_responses_lite` and metadata-version fields keep serde defaults. A missing metadata version identifies a legacy record that must be refreshed from the catalog before it can drive a request.

## Components

### Catalog and persistence

`CodexRemoteModel` deserializes `use_responses_lite` with a default of `false`. `CodexSelectedModelRecord` persists the same boolean plus a metadata schema version. Legacy records deserialize as version zero; every newly selected or catalog-refreshed record writes the current version.

This keeps the remote catalog authoritative. A future model can opt into the lite contract without a code release that recognizes its slug.

Before either public Responses entry point chooses a native tool or builds a Codex request for the selected alias, it validates the active account and selected slug and checks the metadata version. A legacy record is rehydrated from the current catalog under the existing serialized async refresh lock. The version is rechecked after acquiring the lock so concurrent requests result in one successful catalog fetch and the remaining waiters reuse the persisted result. An unchanged ETag does not suppress this schema rehydration.

### Request construction

For a selected Codex model whose persisted record matches the outgoing model slug and has `use_responses_lite: true`, request construction will mirror the official Codex client contract:

- Add `x-openai-internal-codex-responses-lite: true`.
- Remove top-level `instructions` and `tools`.
- Prepend an `additional_tools` developer input item containing the available function tools.
- Add the system instructions as a developer message immediately after `additional_tools` when the instructions are non-empty.
- Recursively remove `detail` from every `input_image` object in outgoing lite input, including images nested in structured tool outputs, before adding the developer prefix.
- Disable parallel tool calls.
- Add `reasoning.context: "all_turns"` while retaining the selected reasoning effort.
- Keep the existing `/responses` endpoint, SSE handling, authentication, session ID, and retry behavior.

Non-lite models continue using the current payload and headers unchanged.

### Web search and tool loops

Responses Lite does not accept the hosted `web_search` tool through the normal top-level tool field. The native Codex web-search builder will therefore return no hosted tool for lite models. Existing provider selection will then use the bot's function-based `web_search` tool when configured, preserving web-search behavior through the existing execution loop.

The general tool-runtime path places its existing function definitions inside the lite `additional_tools` input item. Function-call outputs and subsequent iterations continue using the current loop and turn-state handling. Each request is rebuilt from prefix-free conversation history, so repeated tool iterations retain exactly one `additional_tools` item and at most one developer-instruction item.

Redacted request summaries continue reporting names only. Normal requests read top-level `tools`; when those are absent, a lite request reads tool names from its leading `additional_tools` item. Schemas, arguments, input content, and request bodies are never included.

## Data flow

1. `/codexmodel` fetches the remote catalog, including `use_responses_lite`.
2. Selecting or refreshing a model writes the capability and current metadata version to `data/openai_codex_model.json`.
3. Runtime model loading keeps the selected slug as the inference model ID and distinguishes legacy version-zero records.
4. Before request construction, selected-alias requests validate provider, account, and slug, then serialize and recheck any required legacy catalog rehydration.
5. Request construction matches the current selected record to that slug and chooses normal or lite formatting.
6. Lite requests carry the required header, normalized image input, and developer-item contract; normal requests remain unchanged.

## Error handling and compatibility

- Old selected-model files deserialize with `use_responses_lite: false` and metadata version zero, then rehydrate from the current catalog before request routing.
- A selected record must still match both the active account and outgoing slug before its capabilities affect a request.
- Catalog refreshes update the capability and schema version together with the existing ETag and model metadata.
- Legacy refresh failures fail closed before a request is sent and return a redacted, actionable instruction to retry or reselect with `/codexmodel`.
- HTTP errors continue through the current redacted diagnostics and retry policy.
- The fix does not retry a 404 with an alternate contract because the catalog capability determines the contract before the request is sent.

## Testing

Tests will cover:

- Catalog deserialization retains `use_responses_lite: true`.
- Legacy selected-model JSON without the field defaults to `false`.
- Legacy and current selected records are distinguishable by metadata version, and newly built/refreshed records carry the current version.
- Refresh decisions require version-zero rehydration even when the ETag is unchanged, while current unchanged records skip it.
- Selected-alias provider, slug, and account validation remains fail-closed.
- Lite payloads contain the header, developer input prefix, `all_turns` reasoning context, and disabled parallel calls, with no top-level instructions or tools.
- Lite input removes image detail recursively while normal input preserves it.
- Empty lite instructions/tools, redacted lite tool summaries, retry header assembly, and subsequent-iteration prefix stability retain the official contract.
- Lite models do not select native hosted web search.
- Normal Codex payloads keep their existing structure and do not receive the lite header.

Verification will run the focused unit tests first, followed by `cargo fmt --check`, `cargo test`, `cargo build`, and `cargo clippy --all-targets -- -D warnings`.

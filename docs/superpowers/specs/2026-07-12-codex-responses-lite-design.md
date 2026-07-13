# Codex Responses Lite Compatibility Design

## Problem

The Codex model catalog exposes `gpt-5.6-luna` as a selectable, API-supported model and marks it with `use_responses_lite: true`. The bot currently deserializes only a subset of the model metadata, so that flag is discarded. It then sends Luna through the normal Codex Responses contract, and the backend returns HTTP 404 before producing a response.

## Chosen approach

Implement Responses Lite as a catalog-driven model capability. Retain the remote flag in the selected-model record and use it to select the request contract. Do not hard-code GPT-5.6 slugs and do not hide lite-capable models.

No new dependency or configuration variable is required. Existing selected-model JSON files remain valid because a missing `use_responses_lite` field defaults to `false`.

## Components

### Catalog and persistence

`CodexRemoteModel` will deserialize `use_responses_lite` with a default of `false`. `CodexSelectedModelRecord` will persist the same boolean and copy it whenever a model is selected or refreshed from the catalog.

This keeps the remote catalog authoritative. A future model can opt into the lite contract without a code release that recognizes its slug.

### Request construction

For a selected Codex model whose persisted record matches the outgoing model slug and has `use_responses_lite: true`, request construction will mirror the official Codex client contract:

- Add `x-openai-internal-codex-responses-lite: true`.
- Remove top-level `instructions` and `tools`.
- Prepend an `additional_tools` developer input item containing the available function tools.
- Add the system instructions as a developer message immediately after `additional_tools` when the instructions are non-empty.
- Disable parallel tool calls.
- Add `reasoning.context: "all_turns"` while retaining the selected reasoning effort.
- Keep the existing `/responses` endpoint, SSE handling, authentication, session ID, and retry behavior.

Non-lite models continue using the current payload and headers unchanged.

### Web search and tool loops

Responses Lite does not accept the hosted `web_search` tool through the normal top-level tool field. The native Codex web-search builder will therefore return no hosted tool for lite models. Existing provider selection will then use the bot's function-based `web_search` tool when configured, preserving web-search behavior through the existing execution loop.

The general tool-runtime path will place its existing function definitions inside the lite `additional_tools` input item. Function-call outputs and subsequent iterations continue using the current loop and turn-state handling.

## Data flow

1. `/codexmodel` fetches the remote catalog, including `use_responses_lite`.
2. Selecting or refreshing a model writes the capability to `data/openai_codex_model.json`.
3. Runtime model loading keeps the selected slug as the inference model ID.
4. Request construction matches the selected record to that slug and chooses normal or lite formatting.
5. Lite requests carry the required header and input-item contract; normal requests remain unchanged.

## Error handling and compatibility

- Old selected-model files deserialize with `use_responses_lite: false` and continue working.
- A selected record must still match both the active account and outgoing slug before its capabilities affect a request.
- Catalog refreshes update the capability together with the existing ETag and model metadata.
- HTTP errors continue through the current redacted diagnostics and retry policy.
- The fix does not retry a 404 with an alternate contract because the catalog capability determines the contract before the request is sent.

## Testing

Tests will cover:

- Catalog deserialization retains `use_responses_lite: true`.
- Legacy selected-model JSON without the field defaults to `false`.
- Selected-model record creation copies the flag.
- Lite payloads contain the header, developer input prefix, `all_turns` reasoning context, and disabled parallel calls, with no top-level instructions or tools.
- Lite models do not select native hosted web search.
- Normal Codex payloads keep their existing structure and do not receive the lite header.

Verification will run the focused unit tests first, followed by `cargo fmt --check`, `cargo test`, `cargo build`, and `cargo clippy --all-targets -- -D warnings`.

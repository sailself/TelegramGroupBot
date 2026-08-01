# Quick Codex Explicit-Slug Resolution

## Context

`DEFAULT_QUICK_TEXT_MODEL` accepts configured runtime-model IDs, but the Codex runtime currently registers only the globally selected alias `openai-codex:selected`. A value such as `openai-codex:gpt-5.6-terra` therefore resolves as “not configured” and `/qq` silently falls back to `DEFAULT_TEXT_MODEL`, even when that Codex slug is available to the authenticated account. The response label then reflects the globally selected model and reasoning level, which makes the fallback visible as Luna/max in the reported case.

## Goals

- Allow `DEFAULT_QUICK_TEXT_MODEL=openai-codex:<slug>` to select an API-supported model from the authenticated Codex catalog without changing `/codexmodel` or the global selected model.
- Preserve the selected model's catalog metadata so Responses versus Responses Lite request shaping, media capability checks, function tools, and reasoning validation remain correct.
- Keep the existing `/qq` one-search runtime, no-picker behavior, and one-time fallback to `DEFAULT_TEXT_MODEL`.
- Show the effective Quick reasoning effort in the final model label.
- Leave `/q`, public OpenAI, other third-party providers, pickers, and persisted selected-model state unchanged.

## Non-goals

- Do not make arbitrary Codex slugs selectable in `/q` or model pickers.
- Do not replace or rewrite `data/openai_codex_model.json`.
- Do not change global search budgets, `/qc`, `/s`, or database behavior.
- Do not invent capabilities when a slug is absent from the authenticated catalog.

## Design

### Quick-only catalog resolution

When `/qq` resolves its configured model, it will recognize an explicit `openai-codex:<slug>` that is not already a registered runtime model. The resolver will fetch the authenticated Codex catalog, require the returned account to match the active account, locate an exact slug with `supported_in_api=true`, and construct a runtime model configuration from that catalog entry.

The catalog-derived model will be cached in memory under its provider-qualified ID. A small async resolution lock will prevent concurrent first-use requests from issuing duplicate catalog fetches. Login/logout or the existing runtime-model reload will clear this ephemeral cache naturally.

The explicit model will be available through direct runtime lookup but will not be added to the ordinary `runtime_models()` list. Consequently, it cannot appear in `/q` pickers and cannot influence model counts or normal default-model resolution.

### Metadata and request contract

The ephemeral cache will retain the matching catalog metadata alongside the runtime model configuration. Responses-provider request construction will retrieve metadata for the exact Codex model being called:

- `openai-codex:selected` continues to use the persisted, account-bound selected record.
- A Quick-only explicit slug uses its account-bound ephemeral catalog record.
- Public OpenAI and non-Codex providers receive no Codex metadata.

This preserves per-model `use_responses_lite`, image modalities, supported reasoning levels, and tool behavior. The explicit Codex model remains tool-capable, but `/qq` continues to disable native Codex search and exposes only the existing local one-round `web_search` runtime.

### Fallback and errors

If authentication is unavailable, the catalog fetch fails, the account changes during resolution, the slug is absent, the model is not API-supported, or its media capabilities do not match the request, the explicit Quick model is treated as unavailable. `/qq` then performs its existing single fallback to the capable `DEFAULT_TEXT_MODEL`, without opening a picker. If the fallback also fails, the existing user-facing error path remains in effect.

Failures will be logged with the configured slug and a non-sensitive reason; credentials and response bodies will not be logged.

### Reasoning and labels

`QUICK_REASONING_EFFORT` remains Codex-only. The same reasoning-resolution logic used to build the Codex request will determine the final `/qq` model label, so a supported `low` override renders as `gpt-5.6-terra low`. If the requested effort is unsupported, both the request and label use the catalog-derived fallback effort. Ordinary `/q` labels continue to reflect the globally selected reasoning level.

## Testing

- A failing regression will prove that an available explicit Codex Quick slug resolves to its own provider-qualified ID instead of the selected/default model.
- Catalog mapping tests will cover exact slug matching, API-support rejection, account mismatch, media capabilities, and preservation of Responses Lite/tool metadata.
- Runtime lookup tests will prove ephemeral explicit models remain absent from the picker/catalog list and do not replace the selected record.
- Responses-provider tests will prove the matching explicit record controls Lite shaping and reasoning validation while public OpenAI remains unchanged.
- QA tests will cover fallback when explicit resolution fails and the effective Quick reasoning label.
- Run focused tests, `cargo test`, `cargo build`, `cargo fmt --check`, and `cargo clippy --all-targets -- -D warnings` before completion.

## Documentation

Update `.env.example` and `README.md` to state that `DEFAULT_QUICK_TEXT_MODEL` accepts configured runtime IDs and authenticated `openai-codex:<slug>` values, while `openai-codex:selected` continues to mean the globally selected model.

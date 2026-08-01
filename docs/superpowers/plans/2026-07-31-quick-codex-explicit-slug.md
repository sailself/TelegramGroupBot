# Quick Codex Explicit-Slug Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `DEFAULT_QUICK_TEXT_MODEL=openai-codex:<slug>` use that authenticated Codex catalog model for `/qq`, with its exact request metadata and Quick reasoning, while leaving `/q` and the globally selected Codex model unchanged.

**Architecture:** Add a Quick-only, in-memory catalog cache to `runtime_models.rs`: explicit Codex entries are available through direct lookup but excluded from `runtime_models()` and therefore from pickers. Resolve the explicit slug asynchronously before the existing Quick fallback logic, and let the Responses provider retrieve metadata for the exact cached model so Lite shaping and reasoning validation use the correct record.

**Tech Stack:** Rust, Tokio, parking_lot, reqwest-backed Codex catalog client, teloxide, existing unit tests beside source modules.

## Global Constraints

- Do not change `/q`, `/qc`, `/s`, model-picker contents, global search budgets, or database behavior.
- Do not overwrite `data/openai_codex_model.json` or alter the global `/codexmodel` selection.
- Keep native Codex search disabled for `/qq`; retain the existing local one-round `web_search` runtime.
- Treat absent, unsupported, account-mismatched, or unreachable explicit Codex models as unavailable and perform the existing single fallback to `DEFAULT_TEXT_MODEL`.
- Apply `QUICK_REASONING_EFFORT` only to OpenAI Codex; public OpenAI and other providers remain unchanged.
- Never log credentials, tokens, or remote response bodies.

---

### Task 1: Cache catalog-resolved Codex models outside the picker list

**Files:**
- Modify: `src/llm/runtime_models.rs:48-305`
- Test: `src/llm/runtime_models.rs` existing `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `openai_codex::fetch_models() -> Result<CodexModelList>` and `parse_third_party_model_id(&str)`.
- Produces: `fn build_explicit_codex_runtime_entry(model_id: &str, model: &CodexRemoteModel, etag: Option<String>, account_id: &str) -> Result<(ThirdPartyModelConfig, CodexSelectedModelRecord)>`, `fn insert_explicit_codex_runtime_entry(state: &mut RuntimeModelsState, config: ThirdPartyModelConfig, record: CodexSelectedModelRecord)`, `pub async fn ensure_explicit_codex_model(model_id: &str) -> Result<ThirdPartyModelConfig>`, and the explicit record cache consumed by Task 2.

- [ ] **Step 1: Write failing pure catalog-mapping tests**

Add fixtures that construct a complete `CodexRemoteModel` and assert these behaviors:

```rust
#[test]
fn explicit_codex_catalog_entry_preserves_contract_metadata() {
    let model = remote_model("gpt-5.6-terra", true, true, &[CodexInputModality::Text]);
    let (config, record) = build_explicit_codex_runtime_entry(
        "openai-codex:gpt-5.6-terra",
        &model,
        Some("etag-1".to_string()),
        "acct-1",
    )
    .expect("catalog entry should map");

    assert_eq!(config.id, "openai-codex:gpt-5.6-terra");
    assert_eq!(config.model, "gpt-5.6-terra");
    assert_eq!(config.provider, ThirdPartyProvider::OpenAICodex);
    assert!(config.tools);
    assert!(!config.image);
    assert!(record.use_responses_lite);
    assert!(record.supports_search_tool);
    assert_eq!(record.account_id.as_deref(), Some("acct-1"));
}

#[test]
fn explicit_codex_catalog_entry_rejects_non_api_model() {
    let mut model = remote_model("gpt-5.6-terra", false, false, &[CodexInputModality::Text]);
    model.supported_in_api = false;
    assert!(build_explicit_codex_runtime_entry(
        "openai-codex:gpt-5.6-terra",
        &model,
        None,
        "acct-1",
    )
    .is_err());
}
```

The production change that makes these tests pass is a catalog-to-runtime mapper that retains the remote model's API and request-contract fields instead of synthesizing defaults.

- [ ] **Step 2: Run the mapping tests and verify RED**

Run:

```powershell
cargo test explicit_codex_catalog_entry -- --nocapture
```

Expected: compilation fails because `build_explicit_codex_runtime_entry` and the complete `remote_model` fixture do not exist.

- [ ] **Step 3: Implement catalog mapping and isolated cache state**

Extend `RuntimeModelsState` with an explicit-record map while keeping `models` unchanged:

```rust
struct RuntimeModelsState {
    models: Vec<ThirdPartyModelConfig>,
    models_by_id: HashMap<String, ThirdPartyModelConfig>,
    codex_selected_model: Option<CodexSelectedModelRecord>,
    explicit_codex_records_by_id: HashMap<String, CodexSelectedModelRecord>,
}
```

Add a single-flight lock:

```rust
static EXPLICIT_CODEX_MODEL_RESOLUTION_LOCK: Lazy<AsyncMutex<()>> =
    Lazy::new(|| AsyncMutex::new(()));
```

Implement `build_explicit_codex_runtime_entry` by requiring an `OpenAICodex` qualified ID whose slug exactly matches an API-supported catalog model. Build the record using the catalog metadata and `selected_reasoning_level: None`; build a config with `tools: true`, image support derived from `input_modalities`, and no video/audio support.

Implement `ensure_explicit_codex_model` with this sequence:

```rust
pub async fn ensure_explicit_codex_model(model_id: &str) -> Result<ThirdPartyModelConfig> {
    if let Some(config) = runtime_model_config(model_id) {
        return Ok(config);
    }
    let _guard = EXPLICIT_CODEX_MODEL_RESOLUTION_LOCK.lock().await;
    if let Some(config) = runtime_model_config(model_id) {
        return Ok(config);
    }

    let (provider, slug) = parse_third_party_model_id(model_id)
        .ok_or_else(|| anyhow!("Invalid explicit Codex model id"))?;
    if provider != ThirdPartyProvider::OpenAICodex || slug == "selected" {
        return Err(anyhow!("Model is not an explicit Codex slug"));
    }

    let list = openai_codex::fetch_models().await?;
    let active_account = current_codex_account_id()
        .ok_or_else(|| anyhow!("Codex auth token does not include a ChatGPT account id"))?;
    if list.account_id != active_account {
        return Err(anyhow!("The active ChatGPT account changed during model resolution"));
    }
    let model = list.models.iter().find(|model| model.slug == slug)
        .ok_or_else(|| anyhow!("Codex model '{}' is unavailable", slug))?;
    let (config, record) = build_explicit_codex_runtime_entry(
        model_id,
        model,
        list.etag,
        &active_account,
    )?;
    let mut state = RUNTIME_MODELS.write();
    state.models_by_id.insert(config.id.clone(), config.clone());
    state.explicit_codex_records_by_id.insert(config.id.clone(), record);
    Ok(config)
}
```

- [ ] **Step 4: Add and run cache-isolation tests**

Add a state-level test which calls `insert_explicit_codex_runtime_entry(&mut state, config, record)`, then asserts `state.models_by_id` finds the explicit ID while `state.models` remains unchanged and `state.codex_selected_model` still points to the original selected slug. Use the same helper inside `ensure_explicit_codex_model` instead of inserting into both maps inline. Run:

```powershell
cargo test explicit_codex_ -- --nocapture
```

Expected: all explicit Codex runtime tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/llm/runtime_models.rs
git commit -m "feat: cache explicit Codex runtime models"
```

---

### Task 2: Use metadata belonging to the exact Codex request model

**Files:**
- Modify: `src/llm/runtime_models.rs:110-130,302-305`
- Modify: `src/llm/responses_provider.rs:18-20,654-656,690-740,846-875`
- Test: `src/llm/runtime_models.rs` and `src/llm/responses_provider.rs` existing test modules

**Interfaces:**
- Consumes: Task 1's `explicit_codex_records_by_id` cache and runtime config.
- Produces: `fn codex_model_record_for_request_with_state(state: &RuntimeModelsState, config: &ThirdPartyModelConfig, current_account_id: Option<&str>) -> Result<Option<CodexSelectedModelRecord>>`, the public wrapper `pub fn codex_model_record_for_request(config: &ThirdPartyModelConfig) -> Result<Option<CodexSelectedModelRecord>>`, and `pub(crate) fn reasoning_effort_for_request(...)` for the QA label in Task 3.

- [ ] **Step 1: Write failing exact-record tests**

Add tests which build a selected Luna record and an explicit Terra record and assert:

```rust
assert_eq!(codex_model_record_for_request_with_state(&state, &selected_config, Some("acct-1"))?.unwrap().slug, "gpt-5.6-luna");
assert_eq!(codex_model_record_for_request_with_state(&state, &terra_config, Some("acct-1"))?.unwrap().slug, "gpt-5.6-terra");
assert!(codex_model_record_for_request_with_state(&state, &terra_config, Some("acct-2")).is_err());
```

Add a Responses-provider test with selected Luna `use_responses_lite=false` and explicit Terra `use_responses_lite=true`; assert `build_responses_payload` returns `actual_use_lite=true` only when passed the Terra record.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
cargo test codex_model_record_for_request_with_state -- --nocapture
cargo test explicit_codex_metadata_controls_responses_lite -- --nocapture
```

Expected: compilation fails because the exact-record selector is not implemented.

- [ ] **Step 3: Implement exact metadata selection**

Refactor selected-model validation into a state-backed selector:

```rust
pub fn codex_model_record_for_request(
    config: &ThirdPartyModelConfig,
) -> Result<Option<CodexSelectedModelRecord>>
```

For non-Codex providers return `Ok(None)`. For `openai-codex:selected`, require the selected record to match the config slug and active account. For explicit IDs, require `explicit_codex_records_by_id[config.id]` to match both the config slug and active account. Return an error on stale or mismatched metadata rather than borrowing the selected model's record.

In `responses_provider.rs`, replace direct `selected_codex_model_record()` reads in request construction and native-tool metadata lookup with `codex_model_record_for_request(model_config)?`. Make `reasoning_effort_for_request` `pub(crate)` without changing its public-OpenAI guard.

- [ ] **Step 4: Run focused metadata and Responses tests**

```powershell
cargo test codex_model_record_for_request_with_state -- --nocapture
cargo test explicit_codex_metadata_controls_responses_lite -- --nocapture
cargo test selected_metadata_guard_leaves_public_openai_requests_unchanged -- --nocapture
cargo test reasoning_ -- --nocapture
```

Expected: all commands pass; public OpenAI payload behavior remains unchanged.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/llm/runtime_models.rs src/llm/responses_provider.rs
git commit -m "fix: use exact Codex request metadata"
```

---

### Task 3: Resolve explicit Codex slugs only for `/qq` and report effective reasoning

**Files:**
- Modify: `src/handlers/qa.rs:28-41,300-335,822-906,1736-1805,1929-1934,3180-3190`
- Test: `src/handlers/qa.rs` existing `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `ensure_explicit_codex_model`, `codex_model_record_for_request`, and `reasoning_effort_for_request` from Tasks 1-2.
- Produces: `fn add_explicit_quick_model(models: &mut Vec<ThirdPartyModelConfig>, explicit: Option<ThirdPartyModelConfig>)`, an async `resolve_quick_text_model_for_request(...) -> Result<String>`, and `fn codex_quick_result_label(config: &ThirdPartyModelConfig, record: Option<&CodexSelectedModelRecord>, requested_effort: Option<&str>) -> String`.

- [ ] **Step 1: Write the failing resolver regression**

Add a regression proving the exact explicit ID is added to the Quick candidate set and wins instead of the selected default:

```rust
#[test]
fn explicit_codex_quick_model_does_not_fall_back_to_selected_default() {
    let terra = model(ThirdPartyProvider::OpenAICodex, "GPT-5.6-Terra", "gpt-5.6-terra");
    let luna = model(ThirdPartyProvider::OpenAICodex, "GPT-5.6-Luna", "selected");
    let mut models = vec![luna];
    add_explicit_quick_model(&mut models, Some(terra));
    let result = resolve_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        "openai-codex:selected",
        &models,
        &[ThirdPartyProvider::OpenAICodex],
        false,
        ModelRequestCapabilities::default(),
    );
    assert_eq!(result.as_deref(), Ok("openai-codex:gpt-5.6-terra"));
}
```

Add a second regression in which Terra is absent and assert the result remains `openai-codex:selected`, preserving the one-time fallback.

- [ ] **Step 2: Run the resolver tests and verify RED**

Run:

```powershell
cargo test explicit_codex_quick_model -- --nocapture
```

Expected: compilation fails because `add_explicit_quick_model` does not exist.

- [ ] **Step 3: Make Quick request preparation catalog-aware**

Change `resolve_quick_text_model_for_request` to `async`. Before building the local model slice, inspect `CONFIG.default_quick_text_model`; when it is an explicit `OpenAICodex` slug other than `selected`, call `ensure_explicit_codex_model`. On success append the returned config to the local slice only if absent. On failure log a sanitized warning containing the configured ID and error chain, then continue into the existing resolver so it performs exactly one fallback.

Update the direct Quick call site:

```rust
resolve_quick_text_model_for_request(has_images, has_video, has_audio, has_documents)
    .await
    .map(|model| (model, "default_quick_text_model"))
```

No standard-mode call site should invoke the new catalog resolver.

- [ ] **Step 4: Write and verify the effective-label regression**

Add a label helper test using a literal explicit config and catalog record:

```rust
assert_eq!(
    codex_quick_result_label(
        &terra_config,
        Some(&terra_record),
        Some("low"),
    ),
    "gpt-5.6-terra low"
);
```

Implement `codex_quick_result_label` so it calls the exported `reasoning_effort_for_request` with the exact cached record and requested effort. Make the mode-aware response label call it only for Quick Codex requests using `CONFIG.quick_reasoning_effort`; Standard mode continues to call the existing selected-model label path.

Run:

```powershell
cargo test quick_ -- --nocapture
```

Expected: all Quick tests pass, including explicit Terra selection, selected fallback, Codex-only low reasoning, one-search enforcement, and labels.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/handlers/qa.rs
git commit -m "fix: honor explicit Codex quick models"
```

---

### Task 4: Document configuration and complete repository verification

**Files:**
- Modify: `.env.example:19-22`
- Modify: `README.md:182-185,295-297`
- Modify: `agent_logs/20260731_222955_quick_codex_explicit_slug.md` (ignored local execution record)

**Interfaces:**
- Consumes: completed runtime and QA behavior from Tasks 1-3.
- Produces: user-facing configuration guidance and final validation evidence.

- [ ] **Step 1: Update configuration documentation**

Document these exact cases:

```env
# Empty inherits DEFAULT_TEXT_MODEL. Accepts configured runtime IDs,
# openai-codex:selected, or an authenticated catalog slug such as:
# DEFAULT_QUICK_TEXT_MODEL=openai-codex:gpt-5.6-terra
DEFAULT_QUICK_TEXT_MODEL=
QUICK_REASONING_EFFORT=low
```

In `README.md`, state that explicit Codex Quick slugs are resolved from the authenticated catalog without changing `/codexmodel`, while unavailable/incompatible slugs fall back once.

- [ ] **Step 2: Run formatting and focused tests**

```powershell
cargo fmt
cargo fmt --check
cargo test explicit_codex_ -- --nocapture
cargo test quick_ -- --nocapture
cargo test selected_metadata_guard_leaves_public_openai_requests_unchanged -- --nocapture
```

Expected: every command exits 0.

- [ ] **Step 3: Run the full verification gate**

```powershell
cargo test
cargo build
cargo clippy --all-targets -- -D warnings
git diff --check
```

Expected: all tests pass, build succeeds, Clippy reports no warnings, and the diff check is clean.

- [ ] **Step 4: Update the agent log and re-read the originating request**

Record files touched, RED/GREEN evidence, catalog/cache decisions, all command results, and review findings. Confirm literal compliance: Terra is used only by `/qq`; `/q` still uses the global selected model; low reasoning is applied only to Codex; unrelated SQLite `-shm`/`-wal` files remain untouched.

- [ ] **Step 5: Commit documentation**

```powershell
git add .env.example README.md
git commit -m "docs: explain explicit Codex quick models"
```

- [ ] **Step 6: Request review and finish the branch**

Use `superpowers:requesting-code-review`, correct every Critical or Important finding with a regression test, rerun the full verification gate, then use `superpowers:finishing-a-development-branch` to offer local merge, push/PR, or branch retention.

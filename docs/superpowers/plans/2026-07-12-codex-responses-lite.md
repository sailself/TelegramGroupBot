# Codex Responses Lite Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make catalog-selected Codex Responses Lite models such as `gpt-5.6-luna` use the transport contract advertised by their model metadata.

**Architecture:** Carry `use_responses_lite` from remote catalog deserialization into the account-bound selected-model record. Build either the existing normal Responses payload or the official lite payload from that flag, and route lite web search through the bot's existing function-tool loop rather than the unsupported hosted tool.

**Tech Stack:** Rust, Serde/serde_json, reqwest headers, Tokio, existing inline unit-test modules.

## Global Constraints

- Do not hard-code GPT-5.6 model slugs; the remote catalog flag is authoritative.
- Missing `use_responses_lite` fields must deserialize as `false` for backward compatibility.
- Normal Codex and public OpenAI Responses requests must remain structurally unchanged.
- Add no dependency or environment variable.
- Never log request bodies, prompts, access tokens, account identifiers, or remote error text.
- Before delivery, run `cargo clippy --all-targets -- -D warnings` and fix every warning.

---

### Task 1: Preserve the Responses Lite catalog capability

**Files:**
- Modify: `src/llm/openai_codex.rs:149`
- Modify: `src/llm/runtime_models.rs:25`
- Modify: `src/handlers/codex_admin.rs:1010`
- Modify: `src/handlers/qa.rs:2190`
- Test: `src/llm/openai_codex.rs:1404`
- Test: `src/llm/runtime_models.rs:541`

**Interfaces:**
- Consumes: remote `/models` JSON field `use_responses_lite: bool`.
- Produces: `CodexRemoteModel::use_responses_lite` and `CodexSelectedModelRecord::use_responses_lite`, defaulting to `false`.

- [ ] **Step 1: Write failing catalog and persistence tests**

Add to the existing `openai_codex.rs` test module:

```rust
#[test]
fn codex_remote_model_deserializes_responses_lite_capability() {
    let model: CodexRemoteModel = serde_json::from_value(json!({
        "slug": "gpt-5.6-luna",
        "display_name": "GPT-5.6-Luna",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 3,
        "use_responses_lite": true
    }))
    .expect("catalog model should deserialize");

    assert!(model.use_responses_lite);
}

#[test]
fn codex_remote_model_defaults_responses_lite_to_false() {
    let model: CodexRemoteModel = serde_json::from_value(json!({
        "slug": "gpt-5.5",
        "display_name": "GPT-5.5",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 7
    }))
    .expect("legacy catalog model should deserialize");

    assert!(!model.use_responses_lite);
}
```

Add to the existing `runtime_models.rs` test module:

```rust
#[test]
fn selected_model_record_copies_responses_lite_capability() {
    let model = CodexRemoteModel {
        slug: "gpt-5.6-luna".to_string(),
        display_name: "GPT-5.6-Luna".to_string(),
        description: None,
        default_reasoning_level: Some("medium".to_string()),
        supported_reasoning_levels: vec![],
        visibility: crate::llm::openai_codex::CodexModelVisibility::List,
        supported_in_api: true,
        priority: 3,
        web_search_tool_type: CodexWebSearchToolType::TextAndImage,
        input_modalities: vec![CodexInputModality::Text],
        supports_search_tool: true,
        use_responses_lite: true,
    };

    let record = build_codex_selected_model_record(&model, None, "acct-1", None);

    assert!(record.use_responses_lite);
}
```

Extend `legacy_selected_model_record_deserializes_but_remains_unbound`:

```rust
assert!(!record.use_responses_lite);
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cargo test codex_remote_model_deserializes_responses_lite_capability -- --nocapture`

Expected: compilation fails because `CodexRemoteModel` has no `use_responses_lite` field.

- [ ] **Step 3: Add the catalog and persisted-record fields**

Add to both model structs:

```rust
#[serde(default)]
pub use_responses_lite: bool,
```

Copy the field in `build_codex_selected_model_record`:

```rust
use_responses_lite: model.use_responses_lite,
```

Add `use_responses_lite: false` to existing literal initializers in `src/handlers/codex_admin.rs`, `src/handlers/qa.rs`, `src/llm/responses_provider.rs`, and `src/llm/runtime_models.rs`.

- [ ] **Step 4: Run focused metadata tests and verify GREEN**

Run:

```powershell
cargo test codex_remote_model_ -- --nocapture
cargo test selected_model_record_copies_responses_lite_capability -- --nocapture
cargo test legacy_selected_model_record_deserializes_but_remains_unbound -- --nocapture
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the metadata change**

```powershell
git add src/llm/openai_codex.rs src/llm/runtime_models.rs src/handlers/codex_admin.rs src/handlers/qa.rs src/llm/responses_provider.rs
git commit -m "fix: preserve Codex Responses Lite metadata"
```

---

### Task 2: Emit the official Responses Lite request contract

**Files:**
- Modify: `src/llm/responses_provider.rs:31`
- Modify: `src/llm/responses_provider.rs:641`
- Modify: `src/llm/responses_provider.rs:734`
- Test: `src/llm/responses_provider.rs:1798`

**Interfaces:**
- Consumes: a slug-matching `CodexSelectedModelRecord` and its lite capability.
- Produces: `(Value, bool)` from `build_responses_payload`; the boolean controls the lite header.
- Produces: function-based web search for lite models by disabling their native hosted tool.

- [ ] **Step 1: Write failing payload, header, and web-search tests**

Change `codex_record` to accept `use_responses_lite: bool`, store the value, and update existing calls to pass `false`. Add:

```rust
#[test]
fn responses_lite_payload_uses_developer_input_contract() {
    let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-luna");
    let record = codex_record("gpt-5.6-luna", &["medium", "max"], Some("max"), true);
    let tools = vec![json!({
        "type": "function",
        "name": "web_search",
        "parameters": {"type": "object"}
    })];

    let (payload, use_lite) = build_responses_payload(
        &config,
        "System instructions",
        vec![json!({"type": "message", "role": "user", "content": []})],
        Some(tools.clone()),
        "session-1",
        None,
        true,
        Some(&record),
    );

    assert!(use_lite);
    assert!(payload.get("instructions").is_none());
    assert!(payload.get("tools").is_none());
    assert_eq!(payload["parallel_tool_calls"], false);
    assert_eq!(payload["reasoning"]["effort"], "max");
    assert_eq!(payload["reasoning"]["context"], "all_turns");
    assert_eq!(payload["input"][0]["type"], "additional_tools");
    assert_eq!(payload["input"][0]["tools"], Value::Array(tools));
    assert_eq!(payload["input"][1]["role"], "developer");
    assert_eq!(payload["input"][2]["role"], "user");
}

#[test]
fn normal_responses_payload_keeps_top_level_contract() {
    let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.5");
    let record = codex_record("gpt-5.5", &["medium"], Some("medium"), false);
    let tools = vec![json!({"type": "web_search"})];

    let (payload, use_lite) = build_responses_payload(
        &config,
        "System instructions",
        vec![json!({"type": "message", "role": "user", "content": []})],
        Some(tools.clone()),
        "session-1",
        None,
        true,
        Some(&record),
    );

    assert!(!use_lite);
    assert_eq!(payload["instructions"], "System instructions");
    assert_eq!(payload["tools"], Value::Array(tools));
    assert_eq!(payload["parallel_tool_calls"], true);
    assert!(payload["reasoning"].get("context").is_none());
    assert_eq!(payload["input"][0]["role"], "user");
}

#[test]
fn responses_lite_header_is_added_only_for_lite_requests() {
    let mut lite_headers = Vec::new();
    add_codex_responses_lite_header(&mut lite_headers, true);
    assert_eq!(
        lite_headers,
        vec![(
            CODEX_RESPONSES_LITE_HEADER.to_string(),
            "true".to_string()
        )]
    );

    let mut normal_headers = Vec::new();
    add_codex_responses_lite_header(&mut normal_headers, false);
    assert!(normal_headers.is_empty());
}

#[test]
fn responses_lite_does_not_use_native_hosted_web_search() {
    let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-luna");
    let mut record = codex_record("gpt-5.6-luna", &["medium"], Some("medium"), true);
    record.supports_search_tool = true;

    assert!(build_native_codex_web_search_tool_from_record(&config, &record).is_none());
}
```

- [ ] **Step 2: Run the payload test and verify RED**

Run: `cargo test responses_lite_payload_uses_developer_input_contract -- --nocapture`

Expected: compilation fails because `build_responses_payload` and the lite header helper do not exist.

- [ ] **Step 3: Add capability and header helpers**

```rust
const CODEX_RESPONSES_LITE_HEADER: &str =
    "x-openai-internal-codex-responses-lite";

fn selected_model_uses_responses_lite(
    model_config: &ThirdPartyModelConfig,
    selected_record: Option<&CodexSelectedModelRecord>,
) -> bool {
    model_config.provider == ThirdPartyProvider::OpenAICodex
        && selected_record.is_some_and(|record| {
            record.slug == model_config.model && record.use_responses_lite
        })
}

fn add_codex_responses_lite_header(
    headers: &mut Vec<(String, String)>,
    use_responses_lite: bool,
) {
    if use_responses_lite {
        headers.push((
            CODEX_RESPONSES_LITE_HEADER.to_string(),
            "true".to_string(),
        ));
    }
}
```

- [ ] **Step 4: Add normal/lite payload construction**

```rust
#[allow(clippy::too_many_arguments)]
fn build_responses_payload(
    model_config: &ThirdPartyModelConfig,
    instructions: &str,
    mut input_items: Vec<Value>,
    tools: Option<Vec<Value>>,
    session_id: &str,
    reasoning_override: Option<&str>,
    streaming_sse: bool,
    selected_record: Option<&CodexSelectedModelRecord>,
) -> (Value, bool) {
    let use_lite = selected_model_uses_responses_lite(model_config, selected_record);
    let mut payload = if use_lite {
        let mut prefix = vec![json!({
            "type": "additional_tools",
            "role": "developer",
            "tools": tools.unwrap_or_default(),
        })];
        if !instructions.is_empty() {
            prefix.push(json!({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            }));
        }
        input_items.splice(0..0, prefix);
        json!({
            "model": model_config.model,
            "input": input_items,
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "store": false,
            "stream": streaming_sse,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": session_id,
            "text": {"verbosity": "medium"},
        })
    } else {
        let mut payload = json!({
            "model": model_config.model,
            "instructions": instructions,
            "input": input_items,
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "store": false,
            "stream": streaming_sse,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": session_id,
            "text": {"verbosity": "medium"},
        });
        if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
            payload["tools"] = Value::Array(tools);
        }
        payload
    };

    let effort = reasoning_effort_for_request(
        model_config.provider,
        &model_config.model,
        reasoning_override,
        selected_record,
    );
    if use_lite || effort.is_some() {
        let mut reasoning = json!({});
        if let Some(effort) = effort {
            reasoning["effort"] = Value::String(effort);
        }
        if use_lite {
            reasoning["context"] = Value::String("all_turns".to_string());
        }
        payload["reasoning"] = reasoning;
    }

    (payload, use_lite)
}
```

Replace inline payload construction in `build_request_details`:

```rust
let (payload, use_responses_lite) = build_responses_payload(
    model_config,
    instructions,
    input_items,
    tools,
    session_id,
    reasoning_override,
    streaming_sse,
    selected_record.as_ref(),
);
add_codex_responses_lite_header(&mut headers, use_responses_lite);
```

- [ ] **Step 5: Route lite models away from native hosted web search**

```rust
fn build_native_codex_web_search_tool_from_record(
    model_config: &ThirdPartyModelConfig,
    record: &CodexSelectedModelRecord,
) -> Option<Value> {
    if model_config.provider != ThirdPartyProvider::OpenAICodex
        || record.slug != model_config.model
        || record.use_responses_lite
    {
        return None;
    }

    openai_codex::build_native_web_search_tool_from_record(
        record.supports_search_tool,
        record.web_search_tool_type,
        openai_codex::native_web_search_mode(),
        &CONFIG.openai_codex_web_search_allowed_domains,
        Some(&CONFIG.openai_codex_web_search_context_size),
    )
}

fn build_native_codex_web_search_tool(
    model_config: &ThirdPartyModelConfig,
) -> Option<Value> {
    let record = selected_codex_model_record()?;
    build_native_codex_web_search_tool_from_record(model_config, &record)
}
```

- [ ] **Step 6: Run focused request tests and verify GREEN**

```powershell
cargo test responses_lite_ -- --nocapture
cargo test normal_responses_payload_keeps_top_level_contract -- --nocapture
cargo test reasoning_ -- --nocapture
```

Expected: all selected tests pass, including the unchanged reasoning tests.

- [ ] **Step 7: Commit the request-contract fix**

```powershell
git add src/llm/responses_provider.rs
git commit -m "fix: support Codex Responses Lite requests"
```

---

### Task 3: Verify the complete fix and update the execution log

**Files:**
- Modify: `agent_logs/20260712_210244_codex_luna_404.md` (ignored execution record)

**Interfaces:**
- Consumes: completed metadata and request-contract changes.
- Produces: fresh targeted, suite, build, format, and lint evidence.

- [ ] **Step 1: Format and check formatting**

Run `cargo fmt` and `cargo fmt --check`. Expected: exit 0.

- [ ] **Step 2: Run the full test suite**

Run `cargo test`. Expected: exit 0 with zero failed tests.

- [ ] **Step 3: Build the bot**

Run `cargo build`. Expected: exit 0.

- [ ] **Step 4: Run repository-required Clippy**

Run `cargo clippy --all-targets -- -D warnings`. Expected: exit 0 with no warnings.

- [ ] **Step 5: Inspect the final diff and state**

```powershell
git diff --check HEAD~2..HEAD
git status --short
git log -3 --oneline
```

Expected: no whitespace errors; only the user's pre-existing untracked database WAL/SHM files remain outside committed changes.

- [ ] **Step 6: Update the required execution log**

Append changed files, red/green evidence, verification results, design decisions, deviations, and remaining live-integration risk to `agent_logs/20260712_210244_codex_luna_404.md`. Do not include auth tokens, account IDs, prompts, or full provider error bodies.

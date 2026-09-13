//! Building the Responses API request payload and its logging summaries.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::openai_codex;
use crate::llm::runtime_models::CodexSelectedModelRecord;

use super::codex_identity::CodexRequestIdentity;

const CODEX_FREEFORM_STYLE_GUIDANCE: &str = r#"Keep the answer substantive: retain requested facts, supporting evidence, important qualifications, and next actions. If shortening, remove preambles, repetition, empty encouragement, optional background, and routine sign-offs first.
Task-specific format and length requirements take precedence."#;
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) fn generate_session_id() -> String {
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = chrono::Utc::now().timestamp_millis();
    format!("tg-codex-{now}-{counter}")
}

pub(super) fn summarize_responses_payload(payload: &Value) -> String {
    let model = payload
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let input_items = payload
        .get("input")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let input_images = payload
        .get("input")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("content").and_then(|value| value.as_array()))
                .flatten()
                .filter(|item| {
                    item.get("type").and_then(|value| value.as_str()) == Some("input_image")
                })
                .count()
        })
        .unwrap_or(0);
    let tools = payload.get("tools").and_then(Value::as_array).or_else(|| {
        payload
            .get("input")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"))
            .and_then(|item| item.get("tools"))
            .and_then(Value::as_array)
    });
    let tool_names = tools
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    tool.get("name")
                        .and_then(|value| value.as_str())
                        .or_else(|| tool.get("type").and_then(|value| value.as_str()))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let reasoning = payload
        .pointer("/reasoning/effort")
        .and_then(|value| value.as_str())
        .unwrap_or("default");
    let session_id = payload
        .get("prompt_cache_key")
        .and_then(|value| value.as_str())
        .unwrap_or("none");
    let stream = payload
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    format!(
        "model={model}, session_id={session_id}, input_items={input_items}, input_images={input_images}, tools={}, tool_names=[{}], stream={stream}, reasoning={reasoning}",
        tool_names.len(),
        tool_names.join(",")
    )
}

pub(super) fn summarize_output_items(output_items: &[Value]) -> String {
    let mut counts = BTreeMap::new();
    for item in output_items {
        let item_type = item
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown");
        *counts.entry(item_type.to_string()).or_insert(0usize) += 1;
    }

    counts
        .into_iter()
        .map(|(item_type, count)| format!("{item_type}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn debug_model_label(model_config: &ThirdPartyModelConfig) -> &str {
    if model_config.provider == ThirdPartyProvider::OpenAICodex {
        model_config.name.as_str()
    } else {
        model_config.id.as_str()
    }
}

pub(super) fn build_responses_system_prompt(
    system_prompt: &str,
    model_config: &ThirdPartyModelConfig,
    codex_prompt_style: crate::llm::CodexPromptStyle,
    extra_guidance: Option<&str>,
) -> String {
    let mut sections = vec![system_prompt.to_string()];

    if model_config.provider == ThirdPartyProvider::OpenAICodex
        && codex_prompt_style == crate::llm::CodexPromptStyle::FreeformAnswer
    {
        sections.push(CODEX_FREEFORM_STYLE_GUIDANCE.to_string());
    }

    if let Some(guidance) = extra_guidance
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        sections.push(guidance.to_string());
    }

    sections.join("\n\n")
}

pub(super) fn build_responses_user_input(
    user_content: &str,
    image_data_list: &[Vec<u8>],
) -> Vec<Value> {
    let mut content = vec![json!({
        "type": "input_text",
        "text": user_content.to_string(),
    })];

    for image_data in image_data_list {
        let mime_type = crate::llm::media::detect_mime_type(image_data)
            .unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        content.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": data_url,
        }));
    }

    vec![json!({
        "type": "message",
        "role": "user",
        "content": content,
    })]
}

pub(super) fn build_native_codex_web_search_tool_from_record(
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

fn remove_input_image_details(value: &mut Value) {
    match value {
        Value::Array(items) => {
            for item in items {
                remove_input_image_details(item);
            }
        }
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("input_image") {
                object.remove("detail");
            }
            for child in object.values_mut() {
                remove_input_image_details(child);
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_responses_payload(
    model_config: &ThirdPartyModelConfig,
    instructions: &str,
    mut input_items: Vec<Value>,
    tools: Option<Vec<Value>>,
    session_id: &str,
    identity: Option<&CodexRequestIdentity>,
    streaming_sse: bool,
) -> (Value, bool) {
    let use_lite = identity.is_some_and(|identity| identity.use_responses_lite);
    let mut payload = if use_lite {
        for input_item in &mut input_items {
            remove_input_image_details(input_item);
        }
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

    let effort = identity.and_then(|identity| identity.reasoning_effort.clone());
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

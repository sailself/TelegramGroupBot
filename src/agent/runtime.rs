use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::agent::policy::evaluate_agent_tool_call;
use crate::agent::prompt_scaffold::build_agent_system_prompt;
use crate::agent::tools::{
    build_gemini_tool_definitions, build_openrouter_tool_definitions, execute_memory_tool,
    execute_tool, is_memory_tool, requires_confirmation,
};
use crate::agent::types::{AgentProvider, AgentRunOutcome, PendingAgentAction};
use crate::agent::workspace::ensure_chat_workspace;
use crate::config::CONFIG;
use crate::db::models::{AgentMemoryInsert, AgentMemorySearchRow};
use crate::llm::gemini::build_gemini_user_parts_with_media;
use crate::llm::media::{detect_mime_type, MediaFile, MediaKind};
use crate::skills::index::{build_selected_skill_context, build_skill_index};
use crate::skills::loader::load_skills;
use crate::skills::select::select_active_skills;
use crate::state::AppState;
use crate::utils::http::get_http_client;

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn resolve_agent_provider() -> AgentProvider {
    AgentProvider::from_str(&CONFIG.agent_provider)
}

fn resolve_openrouter_model() -> Result<String> {
    if !CONFIG.agent_model.trim().is_empty() {
        return Ok(CONFIG.agent_model.clone());
    }
    if !CONFIG.gpt_model.trim().is_empty() {
        return Ok(CONFIG.gpt_model.clone());
    }
    if let Some(model) = CONFIG
        .iter_openrouter_models()
        .iter()
        .find(|model| model.tools)
        .map(|model| model.model.clone())
    {
        return Ok(model);
    }

    Err(anyhow!(
        "No OpenRouter agent model configured. Set AGENT_MODEL or configure OpenRouter models."
    ))
}

fn resolve_gemini_model() -> Result<String> {
    if !CONFIG.agent_model.trim().is_empty() {
        return Ok(CONFIG.agent_model.clone());
    }
    if !CONFIG.gemini_pro_model.trim().is_empty() {
        return Ok(CONFIG.gemini_pro_model.clone());
    }
    if !CONFIG.gemini_model.trim().is_empty() {
        return Ok(CONFIG.gemini_model.clone());
    }

    Err(anyhow!(
        "No Gemini agent model configured. Set AGENT_MODEL or GEMINI_MODEL."
    ))
}

fn resolve_agent_runtime() -> Result<(AgentProvider, String)> {
    let provider = resolve_agent_provider();
    let model = match provider {
        AgentProvider::OpenRouter => resolve_openrouter_model()?,
        AgentProvider::Gemini => resolve_gemini_model()?,
    };
    Ok((provider, model))
}

fn resolve_agent_runtime_for_media(media_files: &[MediaFile]) -> Result<(AgentProvider, String)> {
    let (provider, model) = resolve_agent_runtime()?;
    if media_files.is_empty() || provider != AgentProvider::OpenRouter {
        return Ok((provider, model));
    }

    if CONFIG.gemini_api_key.trim().is_empty() {
        warn!(
            "Agent received media attachments but GEMINI_API_KEY is not set; keeping OpenRouter with image-only media support"
        );
        return Ok((provider, model));
    }

    match resolve_gemini_model() {
        Ok(gemini_model) => {
            info!("Agent received media attachments; switching provider from OpenRouter to Gemini");
            Ok((AgentProvider::Gemini, gemini_model))
        }
        Err(err) => {
            warn!(
                "Agent received media attachments but Gemini model resolution failed ({}); keeping OpenRouter with image-only media support",
                err
            );
            Ok((provider, model))
        }
    }
}

fn build_openrouter_user_content(prompt: &str, media_files: &[MediaFile]) -> Value {
    if media_files.is_empty() {
        return Value::String(prompt.to_string());
    }

    let mut parts = Vec::new();
    let non_image_count = media_files
        .iter()
        .filter(|file| file.kind != MediaKind::Image)
        .count();
    let text = if non_image_count > 0 {
        format!(
            "{prompt}\n\n[Note: {non_image_count} non-image attachment(s) were omitted because OpenRouter in this runtime only accepts inline images.]"
        )
    } else {
        prompt.to_string()
    };
    parts.push(json!({
        "type": "text",
        "text": text,
    }));

    for file in media_files {
        if file.kind != MediaKind::Image || file.bytes.is_empty() {
            continue;
        }
        let mime_type = detect_mime_type(&file.bytes)
            .or_else(|| {
                let mime = file.mime_type.trim();
                if mime.is_empty() {
                    None
                } else {
                    Some(mime.to_string())
                }
            })
            .unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(&file.bytes);
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        parts.push(json!({
            "type": "image_url",
            "image_url": { "url": data_url },
        }));
    }

    Value::Array(parts)
}

#[derive(Debug, Clone, Copy, Default)]
struct MediaKindSummary {
    total: usize,
    images: usize,
    videos: usize,
    audios: usize,
    documents: usize,
}

fn summarize_media_kinds(media_files: &[MediaFile]) -> MediaKindSummary {
    let mut summary = MediaKindSummary {
        total: media_files.len(),
        ..MediaKindSummary::default()
    };
    for file in media_files {
        match file.kind {
            MediaKind::Image => summary.images += 1,
            MediaKind::Video => summary.videos += 1,
            MediaKind::Audio => summary.audios += 1,
            MediaKind::Document => summary.documents += 1,
        }
    }
    summary
}

fn summarize_error_body(raw: &str) -> String {
    if let Ok(json) = serde_json::from_str::<Value>(raw) {
        if let Some(message) = json
            .pointer("/error/message")
            .and_then(|value| value.as_str())
            .or_else(|| json.get("message").and_then(|value| value.as_str()))
        {
            return message.to_string();
        }
        return json.to_string();
    }
    if raw.trim().is_empty() {
        "empty response body".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn extract_assistant_content(message: &Value) -> String {
    let content = message
        .get("content")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if !content.is_empty() {
        return content;
    }

    if let Some(reasoning) = message.get("reasoning").and_then(|value| value.as_str()) {
        let trimmed = reasoning.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(details) = message
        .get("reasoning_details")
        .and_then(|value| value.as_array())
    {
        let mut parts = Vec::new();
        for detail in details {
            if let Some(text) = detail.get("text").and_then(|value| value.as_str()) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    parts.push(trimmed.to_string());
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    String::new()
}

async fn call_openrouter_chat(model: &str, messages: &[Value], tools: &[Value]) -> Result<Value> {
    if !CONFIG.enable_openrouter || CONFIG.openrouter_api_key.trim().is_empty() {
        return Err(anyhow!(
            "OpenRouter is not configured. Enable OPENROUTER and set OPENROUTER_API_KEY."
        ));
    }

    let mut payload = json!({
        "model": model,
        "messages": messages,
        "temperature": CONFIG.openrouter_temperature,
        "top_p": CONFIG.openrouter_top_p,
        "top_k": CONFIG.openrouter_top_k,
    });
    if !tools.is_empty() {
        payload["tools"] = Value::Array(tools.to_vec());
        payload["tool_choice"] = Value::String("auto".to_string());
    }

    let client = get_http_client();
    let response = client
        .post(format!(
            "{}/chat/completions",
            CONFIG.openrouter_base_url.trim_end_matches('/')
        ))
        .header(
            "Authorization",
            format!("Bearer {}", CONFIG.openrouter_api_key),
        )
        .header(
            "HTTP-Referer",
            "https://github.com/sailself/TelegramGroupHelperBot",
        )
        .header("X-Title", "TelegramGroupHelperBot")
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let detail = summarize_error_body(&body);
        return Err(anyhow!(
            "OpenRouter request failed with status {}: {}",
            status,
            detail
        ));
    }

    let json = response
        .json::<Value>()
        .await
        .map_err(|err| anyhow!("Failed to decode OpenRouter response: {}", err))?;
    Ok(json)
}

async fn call_gemini_chat(
    model: &str,
    system_prompt: &str,
    contents: &[Value],
    tools: &[Value],
) -> Result<Value> {
    if CONFIG.gemini_api_key.trim().is_empty() {
        return Err(anyhow!(
            "Gemini is not configured. Set GEMINI_API_KEY to use AGENT_PROVIDER=gemini."
        ));
    }

    let mut payload = json!({
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "contents": contents,
        "generationConfig": {
            "temperature": CONFIG.gemini_temperature,
            "topK": CONFIG.gemini_top_k,
            "topP": CONFIG.gemini_top_p,
            "maxOutputTokens": CONFIG.gemini_max_output_tokens,
        },
    });

    if !tools.is_empty() {
        payload["tools"] = Value::Array(tools.to_vec());
    }

    let client = get_http_client();
    let response = client
        .post(format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, CONFIG.gemini_api_key
        ))
        .timeout(std::time::Duration::from_secs(90))
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let detail = summarize_error_body(&body);
        return Err(anyhow!(
            "Gemini request failed with status {}: {}",
            status,
            detail
        ));
    }

    response
        .json::<Value>()
        .await
        .map_err(|err| anyhow!("Failed to decode Gemini response: {}", err))
}

fn parse_gemini_assistant_parts(
    response: &Value,
) -> Result<(Vec<Value>, String, Vec<(String, String, Value)>)> {
    let parts = response
        .get("candidates")
        .and_then(|value| value.get(0))
        .and_then(|value| value.get("content"))
        .and_then(|value| value.get("parts"))
        .and_then(|value| value.as_array())
        .cloned()
        .ok_or_else(|| anyhow!("Gemini response is missing candidates[0].content.parts"))?;

    let mut text_parts = Vec::new();
    let mut function_calls = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                text_parts.push(trimmed.to_string());
            }
        }

        if let Some(call) = part.get("functionCall") {
            let name = call
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if name.is_empty() {
                continue;
            }
            let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
            let call_id = call
                .get("id")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string())
                .unwrap_or_else(|| format!("gemini_call_{}_{}", index, now_nanos()));
            function_calls.push((call_id, name, args));
        }
    }

    Ok((parts, text_parts.join("\n"), function_calls))
}

fn build_gemini_function_response_message(tool_name: &str, payload: &Value) -> Value {
    json!({
        "role": "user",
        "parts": [{
            "functionResponse": {
                "name": tool_name,
                "response": {
                    "name": tool_name,
                    "content": payload
                }
            }
        }]
    })
}

fn trim_to_chars(value: &str, max_chars: usize) -> String {
    let mut trimmed = String::new();
    for ch in value.chars().take(max_chars) {
        trimmed.push(ch);
    }
    trimmed
}

fn summarize_for_memory(value: &str) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= CONFIG.agent_memory_save_summary_chars {
        return normalized;
    }
    let mut summary = trim_to_chars(&normalized, CONFIG.agent_memory_save_summary_chars);
    summary.push_str("...");
    summary
}

async fn save_memory_entry(
    state: &AppState,
    chat_id: i64,
    user_id: Option<i64>,
    session_id: i64,
    source_role: &str,
    category: &str,
    content: &str,
    importance: f64,
) {
    if !CONFIG.agent_memory_enabled {
        return;
    }

    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }

    let summary = summarize_for_memory(trimmed);
    if let Err(err) = state
        .db
        .insert_agent_memory(AgentMemoryInsert {
            chat_id,
            user_id,
            session_id: Some(session_id),
            source_role,
            category,
            content: trimmed,
            summary: Some(summary.as_str()),
            importance,
        })
        .await
    {
        warn!(
            "Failed to save agent memory for session {} (role={}): {}",
            session_id, source_role, err
        );
    }
}

fn build_memory_context_block(rows: Vec<AgentMemorySearchRow>) -> String {
    if rows.is_empty() {
        return String::new();
    }

    let mut scored = Vec::new();
    for row in rows {
        // bm25 scores are lower-is-better; convert to an inverse relevance score.
        let lexical = 1.0 / (1.0 + row.lexical_score.max(0.0));
        let recency = 1.0 / (1.0 + row.recency_days.max(0.0));
        let score = 0.75 * lexical + 0.25 * recency;
        if score >= CONFIG.agent_memory_min_relevance {
            scored.push((score, row));
        }
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if scored.is_empty() {
        return String::new();
    }

    let mut lines = vec!["[Memory context]".to_string()];
    let mut used_chars = lines[0].chars().count();
    for (index, (_score, row)) in scored
        .into_iter()
        .take(CONFIG.agent_memory_recall_limit)
        .enumerate()
    {
        let source = row.memory.source_role;
        let summary = row
            .memory
            .summary
            .unwrap_or_else(|| summarize_for_memory(&row.memory.content));
        let line = format!("{}. [{}] {}", index + 1, source, summary);
        let next_len = used_chars + 1 + line.chars().count();
        if next_len > CONFIG.agent_memory_max_context_chars {
            break;
        }
        used_chars = next_len;
        lines.push(line);
    }

    if lines.len() == 1 {
        return String::new();
    }
    lines.join("\n")
}

async fn build_augmented_prompt(
    state: &AppState,
    chat_id: i64,
    prompt: &str,
) -> Result<(String, String)> {
    if !CONFIG.agent_memory_enabled {
        return Ok((prompt.to_string(), String::new()));
    }

    let mut recalls = state
        .db
        .search_agent_memories(
            chat_id,
            prompt,
            CONFIG.agent_memory_recall_limit.saturating_mul(3),
        )
        .await
        .unwrap_or_default();

    if recalls.is_empty() {
        let recent = state
            .db
            .recent_agent_memories(chat_id, CONFIG.agent_memory_recall_limit)
            .await
            .unwrap_or_default();
        recalls = recent
            .into_iter()
            .map(|memory| AgentMemorySearchRow {
                memory,
                lexical_score: 0.0,
                recency_days: 0.0,
            })
            .collect();
    }

    let memory_block = build_memory_context_block(recalls);
    if memory_block.is_empty() {
        return Ok((prompt.to_string(), String::new()));
    }

    let augmented = format!("{}\n\nUser request:\n{}", memory_block, prompt);
    Ok((augmented, memory_block))
}

fn new_confirmation_key(session_id: i64, tool_name: &str) -> String {
    format!(
        "s{}_{}_{}",
        session_id,
        tool_name.replace([':', '|', ' '], "_"),
        now_nanos()
    )
}

async fn append_step(
    state: &AppState,
    session_id: i64,
    role: &str,
    content: &str,
    raw_json: &Value,
) -> Result<i64> {
    state
        .db
        .insert_agent_step(session_id, role, content, raw_json)
        .await
}

async fn execute_runtime_tool(
    state: &AppState,
    workspace_root: &Path,
    session_id: i64,
    chat_id: i64,
    user_id: i64,
    tool_name: &str,
    args_value: &Value,
) -> Result<String> {
    if is_memory_tool(tool_name) {
        execute_memory_tool(
            &state.db, session_id, chat_id, user_id, tool_name, args_value,
        )
        .await
    } else {
        execute_tool(tool_name, args_value, workspace_root).await
    }
}

async fn run_openrouter_loop(
    state: &AppState,
    session_id: i64,
    user_id: i64,
    chat_id: i64,
    processing_message_id: i64,
    workspace_root: &Path,
    model: &str,
    selected_skill_names: &[String],
    allowed_tools: &[String],
    mut messages: Vec<Value>,
) -> Result<AgentRunOutcome> {
    let tool_defs = build_openrouter_tool_definitions(allowed_tools);

    for _iteration in 0..CONFIG.agent_max_tool_iterations {
        let response = call_openrouter_chat(model, &messages, &tool_defs).await?;
        let assistant_message = response
            .get("choices")
            .and_then(|value| value.get(0))
            .and_then(|value| value.get("message"))
            .cloned()
            .ok_or_else(|| anyhow!("OpenRouter response is missing assistant message"))?;

        let assistant_content = extract_assistant_content(&assistant_message);
        let tool_calls = assistant_message
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();

        let assistant_json = if tool_calls.is_empty() {
            json!({
                "role": "assistant",
                "content": assistant_content
            })
        } else {
            json!({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            })
        };
        messages.push(assistant_json.clone());
        let assistant_step_id = append_step(
            state,
            session_id,
            "assistant",
            &assistant_content,
            &assistant_json,
        )
        .await?;

        if tool_calls.is_empty() {
            save_memory_entry(
                state,
                chat_id,
                Some(user_id),
                session_id,
                "assistant",
                "conversation",
                &assistant_content,
                0.6,
            )
            .await;
            state
                .db
                .complete_agent_session(session_id, "completed", Some(&assistant_content))
                .await?;
            return Ok(AgentRunOutcome::Completed {
                session_id,
                response_text: assistant_content,
                selected_skills: selected_skill_names.to_vec(),
            });
        }

        for tool_call in tool_calls {
            let tool_call_id = tool_call
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            let tool_name = tool_call
                .get("function")
                .and_then(|value| value.get("name"))
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            let args_raw = tool_call
                .get("function")
                .and_then(|value| value.get("arguments"))
                .and_then(|value| value.as_str())
                .unwrap_or("{}")
                .to_string();
            let args_value = serde_json::from_str::<Value>(&args_raw)
                .unwrap_or_else(|_| json!({ "_raw": args_raw }));

            let requires_user_confirmation = requires_confirmation(&tool_name);
            let mut tool_call_status = "requested".to_string();
            if requires_user_confirmation {
                tool_call_status = "awaiting_confirmation".to_string();
            }
            let tool_call_record_id = state
                .db
                .insert_agent_tool_call(
                    session_id,
                    assistant_step_id,
                    &tool_call_id,
                    &tool_name,
                    &args_value,
                    &tool_call_status,
                    requires_user_confirmation,
                )
                .await?;

            if let Err(reason) = evaluate_agent_tool_call(&tool_name, &args_value, allowed_tools) {
                warn!(
                    "Denied tool call in session {}: tool='{}' reason='{}'",
                    session_id, tool_name, reason
                );
                state
                    .db
                    .update_agent_tool_call_status(
                        tool_call_record_id,
                        "denied",
                        Some(&json!({ "error": reason })),
                        None,
                    )
                    .await?;
                let tool_msg = json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": reason
                });
                messages.push(tool_msg.clone());
                append_step(state, session_id, "tool", &reason, &tool_msg).await?;
                continue;
            }

            if requires_user_confirmation {
                let confirmation_key = new_confirmation_key(session_id, &tool_name);
                let pending = PendingAgentAction {
                    provider: AgentProvider::OpenRouter,
                    system_prompt: String::new(),
                    user_id,
                    chat_id,
                    session_id,
                    processing_message_id,
                    tool_call_record_id,
                    tool_call_id,
                    tool_name: tool_name.clone(),
                    tool_args: args_value.clone(),
                    workspace_root: workspace_root.to_path_buf(),
                    model_name: model.to_string(),
                    allowed_tools: allowed_tools.to_vec(),
                    selected_skills: selected_skill_names.to_vec(),
                    messages,
                };
                state
                    .pending_agent_actions
                    .lock()
                    .insert(confirmation_key.clone(), pending);
                state
                    .db
                    .complete_agent_session(
                        session_id,
                        "awaiting_confirmation",
                        Some("Awaiting side-effect tool confirmation"),
                    )
                    .await?;

                let notice_text = format!(
                    "Tool `{}` is ready to run and requires confirmation.\nArguments:\n```json\n{}\n```",
                    tool_name,
                    serde_json::to_string_pretty(&args_value).unwrap_or_else(|_| "{}".to_string())
                );

                return Ok(AgentRunOutcome::AwaitingConfirmation {
                    confirmation_key,
                    notice_text,
                });
            }

            let execution_result = execute_runtime_tool(
                state,
                &workspace_root,
                session_id,
                chat_id,
                user_id,
                &tool_name,
                &args_value,
            )
            .await;
            let tool_result = match execution_result {
                Ok(content) => {
                    state
                        .db
                        .update_agent_tool_call_status(
                            tool_call_record_id,
                            "completed",
                            Some(&json!({ "output": content })),
                            None,
                        )
                        .await?;
                    content
                }
                Err(err) => {
                    let err_text = format!("Tool '{}' failed: {}", tool_name, err);
                    state
                        .db
                        .update_agent_tool_call_status(
                            tool_call_record_id,
                            "failed",
                            Some(&json!({ "error": err_text })),
                            None,
                        )
                        .await?;
                    err_text
                }
            };

            let tool_message = json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result
            });
            messages.push(tool_message.clone());
            append_step(state, session_id, "tool", &tool_result, &tool_message).await?;
        }
    }

    let final_text =
        "Tool call limit reached. Please refine your request or confirm required actions.";
    save_memory_entry(
        state,
        chat_id,
        Some(user_id),
        session_id,
        "assistant",
        "conversation",
        final_text,
        0.4,
    )
    .await;
    state
        .db
        .complete_agent_session(session_id, "limit_reached", Some(final_text))
        .await?;
    Ok(AgentRunOutcome::Completed {
        session_id,
        response_text: final_text.to_string(),
        selected_skills: selected_skill_names.to_vec(),
    })
}

async fn run_gemini_loop(
    state: &AppState,
    session_id: i64,
    user_id: i64,
    chat_id: i64,
    processing_message_id: i64,
    workspace_root: &Path,
    model: &str,
    system_prompt: &str,
    selected_skill_names: &[String],
    allowed_tools: &[String],
    mut messages: Vec<Value>,
) -> Result<AgentRunOutcome> {
    let tool_defs = build_gemini_tool_definitions(allowed_tools);

    for _iteration in 0..CONFIG.agent_max_tool_iterations {
        let response = call_gemini_chat(model, system_prompt, &messages, &tool_defs).await?;
        let (parts, assistant_content, function_calls) = parse_gemini_assistant_parts(&response)?;

        let model_message = json!({
            "role": "model",
            "parts": parts,
        });
        messages.push(model_message.clone());
        let assistant_step_id = append_step(
            state,
            session_id,
            "assistant",
            &assistant_content,
            &model_message,
        )
        .await?;

        if function_calls.is_empty() {
            let final_text = if assistant_content.trim().is_empty() {
                "No response content returned by Gemini.".to_string()
            } else {
                assistant_content
            };
            save_memory_entry(
                state,
                chat_id,
                Some(user_id),
                session_id,
                "assistant",
                "conversation",
                &final_text,
                0.6,
            )
            .await;
            state
                .db
                .complete_agent_session(session_id, "completed", Some(&final_text))
                .await?;
            return Ok(AgentRunOutcome::Completed {
                session_id,
                response_text: final_text,
                selected_skills: selected_skill_names.to_vec(),
            });
        }

        for (tool_call_id, tool_name, args_value) in function_calls {
            let requires_user_confirmation = requires_confirmation(&tool_name);
            let mut tool_call_status = "requested".to_string();
            if requires_user_confirmation {
                tool_call_status = "awaiting_confirmation".to_string();
            }
            let tool_call_record_id = state
                .db
                .insert_agent_tool_call(
                    session_id,
                    assistant_step_id,
                    &tool_call_id,
                    &tool_name,
                    &args_value,
                    &tool_call_status,
                    requires_user_confirmation,
                )
                .await?;

            if let Err(reason) = evaluate_agent_tool_call(&tool_name, &args_value, allowed_tools) {
                warn!(
                    "Denied tool call in session {}: tool='{}' reason='{}'",
                    session_id, tool_name, reason
                );
                state
                    .db
                    .update_agent_tool_call_status(
                        tool_call_record_id,
                        "denied",
                        Some(&json!({ "error": reason })),
                        None,
                    )
                    .await?;
                let tool_payload = json!({ "error": reason.clone() });
                let function_response =
                    build_gemini_function_response_message(&tool_name, &tool_payload);
                messages.push(function_response.clone());
                append_step(state, session_id, "tool", &reason, &function_response).await?;
                continue;
            }

            if requires_user_confirmation {
                let confirmation_key = new_confirmation_key(session_id, &tool_name);
                let pending = PendingAgentAction {
                    provider: AgentProvider::Gemini,
                    system_prompt: system_prompt.to_string(),
                    user_id,
                    chat_id,
                    session_id,
                    processing_message_id,
                    tool_call_record_id,
                    tool_call_id,
                    tool_name: tool_name.clone(),
                    tool_args: args_value.clone(),
                    workspace_root: workspace_root.to_path_buf(),
                    model_name: model.to_string(),
                    allowed_tools: allowed_tools.to_vec(),
                    selected_skills: selected_skill_names.to_vec(),
                    messages,
                };
                state
                    .pending_agent_actions
                    .lock()
                    .insert(confirmation_key.clone(), pending);
                state
                    .db
                    .complete_agent_session(
                        session_id,
                        "awaiting_confirmation",
                        Some("Awaiting side-effect tool confirmation"),
                    )
                    .await?;

                let notice_text = format!(
                    "Tool `{}` is ready to run and requires confirmation.\nArguments:\n```json\n{}\n```",
                    tool_name,
                    serde_json::to_string_pretty(&args_value).unwrap_or_else(|_| "{}".to_string())
                );

                return Ok(AgentRunOutcome::AwaitingConfirmation {
                    confirmation_key,
                    notice_text,
                });
            }

            let execution_result = execute_runtime_tool(
                state,
                &workspace_root,
                session_id,
                chat_id,
                user_id,
                &tool_name,
                &args_value,
            )
            .await;
            let (tool_payload, tool_result_text, status) = match execution_result {
                Ok(content) => (json!({ "output": content.clone() }), content, "completed"),
                Err(err) => {
                    let err_text = format!("Tool '{}' failed: {}", tool_name, err);
                    (json!({ "error": err_text.clone() }), err_text, "failed")
                }
            };

            state
                .db
                .update_agent_tool_call_status(
                    tool_call_record_id,
                    status,
                    Some(&tool_payload),
                    None,
                )
                .await?;

            let function_response =
                build_gemini_function_response_message(&tool_name, &tool_payload);
            messages.push(function_response.clone());
            append_step(
                state,
                session_id,
                "tool",
                &tool_result_text,
                &function_response,
            )
            .await?;
        }
    }

    let final_text =
        "Tool call limit reached. Please refine your request or confirm required actions.";
    save_memory_entry(
        state,
        chat_id,
        Some(user_id),
        session_id,
        "assistant",
        "conversation",
        final_text,
        0.4,
    )
    .await;
    state
        .db
        .complete_agent_session(session_id, "limit_reached", Some(final_text))
        .await?;
    Ok(AgentRunOutcome::Completed {
        session_id,
        response_text: final_text.to_string(),
        selected_skills: selected_skill_names.to_vec(),
    })
}

pub async fn start_agent_run(
    state: &AppState,
    user_id: i64,
    chat_id: i64,
    processing_message_id: i64,
    prompt: &str,
    initial_media_files: Vec<MediaFile>,
) -> Result<AgentRunOutcome> {
    let (provider, model) = resolve_agent_runtime_for_media(&initial_media_files)?;
    let workspace_root = ensure_chat_workspace(chat_id)
        .map_err(|err| anyhow!("Failed to initialize agent workspace: {}", err))?;
    let skills_dir = std::path::Path::new(&CONFIG.skills_dir);
    let loaded_skills = load_skills(skills_dir);
    let active_skills = select_active_skills(
        prompt,
        &loaded_skills,
        CONFIG.agent_skill_candidate_limit,
        CONFIG.agent_max_active_skills,
    )
    .await;

    let skill_index = build_skill_index(&loaded_skills);
    let selected_skill_context = build_selected_skill_context(&active_skills.selected);
    let system_prompt =
        build_agent_system_prompt(&workspace_root, &skill_index, &selected_skill_context);
    let selected_skills_json =
        serde_json::to_string(&active_skills.selected_names).unwrap_or_else(|_| "[]".to_string());

    let session_model = format!("{}:{}", provider.as_str(), model);
    let session_id = state
        .db
        .create_agent_session(
            chat_id,
            user_id,
            &session_model,
            prompt,
            &selected_skills_json,
        )
        .await?;
    state
        .db
        .record_agent_session_skills(
            session_id,
            &active_skills.selected_names,
            "heuristic_then_llm",
        )
        .await?;

    let (model_prompt, recalled_memory_context) =
        build_augmented_prompt(state, chat_id, prompt).await?;

    save_memory_entry(
        state,
        chat_id,
        Some(user_id),
        session_id,
        "user",
        "conversation",
        prompt,
        0.7,
    )
    .await;

    let system_msg = json!({ "role": "system", "content": system_prompt });
    append_step(
        state,
        session_id,
        "system",
        system_prompt.as_str(),
        &system_msg,
    )
    .await?;

    info!(
        "Starting agent session {} with provider='{}' model='{}' skills [{}] workspace='{}' media_files={}",
        session_id,
        provider.as_str(),
        model,
        active_skills.selected_names.join(", "),
        workspace_root.display(),
        initial_media_files.len()
    );
    let input_media_summary = summarize_media_kinds(&initial_media_files);
    if input_media_summary.total > 0 {
        debug!(
            "Session {} input media summary: total={}, images={}, videos={}, audios={}, documents={}",
            session_id,
            input_media_summary.total,
            input_media_summary.images,
            input_media_summary.videos,
            input_media_summary.audios,
            input_media_summary.documents
        );
    }
    if !recalled_memory_context.is_empty() {
        debug!(
            "Session {} recalled memory context ({} chars)",
            session_id,
            recalled_memory_context.chars().count()
        );
    }

    match provider {
        AgentProvider::OpenRouter => {
            let mut messages = Vec::new();
            messages.push(system_msg);
            if input_media_summary.total > 0 {
                let sent_images = initial_media_files
                    .iter()
                    .filter(|file| file.kind == MediaKind::Image && !file.bytes.is_empty())
                    .count();
                let omitted_empty_images = input_media_summary.images.saturating_sub(sent_images);
                let omitted_non_images = input_media_summary.videos
                    + input_media_summary.audios
                    + input_media_summary.documents;
                debug!(
                    "Session {} OpenRouter media dispatch: sent_images={}, omitted_non_images={}, omitted_empty_images={}",
                    session_id,
                    sent_images,
                    omitted_non_images,
                    omitted_empty_images
                );
            }
            let user_content = build_openrouter_user_content(&model_prompt, &initial_media_files);
            let user_msg = json!({ "role": "user", "content": user_content });
            messages.push(user_msg.clone());
            append_step(state, session_id, "user", &model_prompt, &user_msg).await?;
            run_openrouter_loop(
                state,
                session_id,
                user_id,
                chat_id,
                processing_message_id,
                &workspace_root,
                &model,
                &active_skills.selected_names,
                &active_skills.allowed_tools,
                messages,
            )
            .await
        }
        AgentProvider::Gemini => {
            let mut messages = Vec::new();
            let (user_parts, media_dispatch) =
                build_gemini_user_parts_with_media(&model_prompt, &initial_media_files).await?;
            if input_media_summary.total > 0 {
                let dropped = input_media_summary
                    .total
                    .saturating_sub(media_dispatch.uploaded_total);
                debug!(
                    "Session {} Gemini media dispatch: uploaded_total={}, uploaded_images={}, uploaded_videos={}, uploaded_audios={}, uploaded_documents={}, dropped={}",
                    session_id,
                    media_dispatch.uploaded_total,
                    media_dispatch.uploaded_images,
                    media_dispatch.uploaded_videos,
                    media_dispatch.uploaded_audios,
                    media_dispatch.uploaded_documents,
                    dropped
                );
            }
            let user_msg = json!({ "role": "user", "parts": user_parts });
            messages.push(user_msg.clone());
            append_step(state, session_id, "user", &model_prompt, &user_msg).await?;
            run_gemini_loop(
                state,
                session_id,
                user_id,
                chat_id,
                processing_message_id,
                &workspace_root,
                &model,
                &system_prompt,
                &active_skills.selected_names,
                &active_skills.allowed_tools,
                messages,
            )
            .await
        }
    }
}

pub async fn continue_after_confirmation(
    state: &AppState,
    pending: PendingAgentAction,
    confirmed_by: i64,
) -> Result<AgentRunOutcome> {
    if let Err(reason) = evaluate_agent_tool_call(
        &pending.tool_name,
        &pending.tool_args,
        &pending.allowed_tools,
    ) {
        warn!(
            "Denied confirmed tool call in session {}: tool='{}' reason='{}'",
            pending.session_id, pending.tool_name, reason
        );
        state
            .db
            .update_agent_tool_call_status(
                pending.tool_call_record_id,
                "denied",
                Some(&json!({ "error": reason })),
                Some(confirmed_by),
            )
            .await?;
        state
            .db
            .complete_agent_session(
                pending.session_id,
                "denied",
                Some("Tool execution denied by policy."),
            )
            .await?;
        save_memory_entry(
            state,
            pending.chat_id,
            Some(pending.user_id),
            pending.session_id,
            "assistant",
            "conversation",
            "Tool execution denied by policy.",
            0.5,
        )
        .await;
        return Ok(AgentRunOutcome::Completed {
            session_id: pending.session_id,
            response_text: "Tool execution denied by policy.".to_string(),
            selected_skills: pending.selected_skills,
        });
    }

    let execution_result = execute_runtime_tool(
        state,
        &pending.workspace_root,
        pending.session_id,
        pending.chat_id,
        pending.user_id,
        &pending.tool_name,
        &pending.tool_args,
    )
    .await;

    let (tool_payload, tool_result, status) = match execution_result {
        Ok(content) => (json!({ "output": content.clone() }), content, "completed"),
        Err(err) => {
            let err_text = format!("Tool '{}' failed: {}", pending.tool_name, err);
            (json!({ "error": err_text.clone() }), err_text, "failed")
        }
    };

    state
        .db
        .update_agent_tool_call_status(
            pending.tool_call_record_id,
            status,
            Some(&tool_payload),
            Some(confirmed_by),
        )
        .await?;

    let mut messages = pending.messages.clone();
    let tool_message = match pending.provider {
        AgentProvider::OpenRouter => json!({
            "role": "tool",
            "tool_call_id": pending.tool_call_id,
            "content": tool_result
        }),
        AgentProvider::Gemini => {
            build_gemini_function_response_message(&pending.tool_name, &tool_payload)
        }
    };
    messages.push(tool_message.clone());
    append_step(
        state,
        pending.session_id,
        "tool",
        &tool_result,
        &tool_message,
    )
    .await?;

    debug!(
        "Continuing agent session {} after confirming tool '{}'",
        pending.session_id, pending.tool_name
    );

    match pending.provider {
        AgentProvider::OpenRouter => {
            run_openrouter_loop(
                state,
                pending.session_id,
                pending.user_id,
                pending.chat_id,
                pending.processing_message_id,
                &pending.workspace_root,
                &pending.model_name,
                &pending.selected_skills,
                &pending.allowed_tools,
                messages,
            )
            .await
        }
        AgentProvider::Gemini => {
            run_gemini_loop(
                state,
                pending.session_id,
                pending.user_id,
                pending.chat_id,
                pending.processing_message_id,
                &pending.workspace_root,
                &pending.model_name,
                &pending.system_prompt,
                &pending.selected_skills,
                &pending.allowed_tools,
                messages,
            )
            .await
        }
    }
}

pub async fn cancel_pending_action(state: &AppState, pending: &PendingAgentAction) -> Result<()> {
    state
        .db
        .update_agent_tool_call_status(
            pending.tool_call_record_id,
            "cancelled",
            Some(&json!({ "status": "cancelled_by_user" })),
            None,
        )
        .await?;
    state
        .db
        .complete_agent_session(
            pending.session_id,
            "cancelled",
            Some("User cancelled side-effect tool execution."),
        )
        .await?;
    warn!(
        "Agent session {} cancelled pending tool '{}'",
        pending.session_id, pending.tool_name
    );
    Ok(())
}

use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use regex::Regex;
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::config::CONFIG;
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::http::get_http_client;
use crate::utils::timing::log_llm_timing;

const MAX_TOOL_CALL_ITERATIONS: usize = 3;
const TOOL_LIMIT_SYSTEM_PROMPT: &str = "Tool call limit reached. Provide the best possible answer using the available information without requesting more tool calls.";

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn summarize_payload(payload: &Value) -> String {
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let message_count = payload
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|messages| messages.len())
        .unwrap_or(0);
    let tool_names = payload
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tool_choice = payload
        .get("tool_choice")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");

    format!(
        "model={}, messages={}, tools={}, tool_choice={}, tool_names=[{}]",
        model,
        message_count,
        tool_names.len(),
        tool_choice,
        tool_names.join(",")
    )
}

fn summarize_error_body(body: &str) -> (Option<String>, String) {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return (None, "empty response body".to_string());
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let message = value
            .pointer("/error/message")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string())
            .or_else(|| {
                value
                    .get("message")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string())
            });
        return (message, truncate_for_log(&value.to_string(), 2000));
    }

    (None, truncate_for_log(trimmed, 2000))
}

fn parse_gpt_content(content: &str) -> String {
    if let Some(last_pos) = content.rfind("<|message|>") {
        let analysis = &content[..last_pos];
        let final_text = &content[last_pos + "<|message|>".len()..];
        let cleanup = Regex::new(r"<\|.*?\|>").unwrap();
        let final_clean = cleanup.replace_all(final_text, "").trim().to_string();
        if !final_clean.is_empty() {
            return final_clean;
        }
        let analysis_clean = cleanup.replace_all(analysis, "").trim().to_string();
        return analysis_clean;
    }
    content.to_string()
}

fn parse_qwen_content(content: &str) -> String {
    let re = Regex::new(r"<think>(.*?)</think>(.*)").unwrap();
    if let Some(caps) = re.captures(content) {
        let final_text = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let final_text = final_text.trim();
        if !final_text.is_empty() {
            return final_text.to_string();
        }
        let analysis = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        return analysis.trim().to_string();
    }
    content.trim().to_string()
}

fn parse_openrouter_response(model_name: &str, content: &str) -> String {
    if model_name == CONFIG.gpt_model {
        return parse_gpt_content(content);
    }
    if model_name == CONFIG.qwen_model {
        return parse_qwen_content(content);
    }
    content.to_string()
}

fn extract_reasoning_text(message: &Value) -> Option<String> {
    if let Some(reasoning) = message.get("reasoning").and_then(|v| v.as_str()) {
        let trimmed = reasoning.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let details = message.get("reasoning_details").and_then(|v| v.as_array())?;
    let mut parts = Vec::new();
    for detail in details {
        let text = detail.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let text = text.trim();
        if !text.is_empty() {
            parts.push(text.to_string());
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_openrouter_content(message: &Value) -> String {
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if !content.is_empty() {
        return content;
    }

    extract_reasoning_text(message).unwrap_or_default()
}

fn build_function_tools() -> Vec<Value> {
    if !web_search::is_search_enabled() {
        return Vec::new();
    }

    vec![json!({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using the configured providers (Brave, Exa, Jina) and return a concise Markdown summary of the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to look up."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    })]
}

fn build_message_content(user_content: &str, image_data_list: &[Vec<u8>]) -> Value {
    if image_data_list.is_empty() {
        return Value::String(user_content.to_string());
    }

    let mut parts = Vec::new();
    parts.push(json!({
        "type": "text",
        "text": user_content
    }));

    for image_data in image_data_list {
        let mime_type = crate::llm::media::detect_mime_type(image_data)
            .unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        parts.push(json!({
            "type": "image_url",
            "image_url": { "url": data_url }
        }));
    }

    Value::Array(parts)
}

async fn call_openrouter_api(payload: &Value) -> Result<Value> {
    debug!("OpenRouter request: {}", summarize_payload(payload));

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
        .timeout(Duration::from_secs(60))
        .json(payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let (message, body_summary) = summarize_error_body(&body);
        warn!(
            "OpenRouter API error: status={}, body={}",
            status, body_summary
        );
        let detail = message.unwrap_or(body_summary);
        return Err(anyhow!(
            "OpenRouter request failed with status {}: {}",
            status,
            detail
        ));
    }

    let value = response.json::<Value>().await?;
    debug!(
        "OpenRouter response received for model={}",
        payload
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    );
    Ok(value)
}

async fn execute_function_tool(name: &str, arguments: &Value) -> Result<String> {
    match name {
        "web_search" => {
            let query = arguments
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let max_results = arguments
                .get("max_results")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            debug!(
                "Executing tool call web_search: query='{}', max_results={:?}",
                truncate_for_log(query, 200),
                max_results
            );
            match web_search_tool(query, max_results).await {
                Ok(result) => {
                    debug!(
                        "web_search returned {} chars",
                        result.chars().count()
                    );
                    Ok(result)
                }
                Err(err) => {
                    warn!("web_search tool failed: {}", err);
                    Err(err)
                }
            }
        }
        _ => Ok(String::from("Unsupported tool call")),
    }
}

async fn chat_completion_with_tools(
    mut messages: Vec<Value>,
    model_name: &str,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<String> {
    let tools = build_function_tools();

    for iteration in 0..MAX_TOOL_CALL_ITERATIONS {
        debug!(
            "OpenRouter tool iteration {}/{}",
            iteration + 1,
            MAX_TOOL_CALL_ITERATIONS
        );
        let payload = json!({
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "tools": tools,
            "tool_choice": "auto"
        });

        let response = call_openrouter_api(&payload).await?;
        let message = response
            .get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .cloned()
            .unwrap_or(Value::Null);

        let content = extract_openrouter_content(&message);
        let tool_calls = message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            if content.trim().is_empty() {
                warn!(
                    "OpenRouter response had empty content and no tool calls: {}",
                    truncate_for_log(&response.to_string(), 2000)
                );
            }
            return Ok(parse_openrouter_response(model_name, &content));
        }

        messages.push(message.clone());

        for tool_call in tool_calls {
            let tool_name = tool_call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args_text = tool_call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let args_value: Value = serde_json::from_str(args_text).unwrap_or(Value::Null);
            let result = execute_function_tool(tool_name, &args_value)
                .await
                .unwrap_or_else(|err| err.to_string());
            if result.trim().is_empty() {
                warn!("Tool call '{}' returned empty content", tool_name);
            }

            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_call.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                "content": result
            }));
        }

        if iteration + 1 == MAX_TOOL_CALL_ITERATIONS {
            messages.push(json!({
                "role": "system",
                "content": TOOL_LIMIT_SYSTEM_PROMPT
            }));
        }
    }

    Ok("".to_string())
}

pub async fn call_openrouter(
    system_prompt: &str,
    user_content: &str,
    model_identifier: &str,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    supports_tools: bool,
) -> Result<String> {
    if model_identifier.trim().is_empty() {
        return Err(anyhow!("Model identifier is required"));
    }

    let message_content = build_message_content(user_content, image_data_list);

    let messages = vec![
        json!({ "role": "system", "content": system_prompt }),
        json!({ "role": "user", "content": message_content }),
    ];

    let operation = format!("openrouter:{}", response_title);
    let model_name = model_identifier.to_string();

    if supports_tools && web_search::is_search_enabled() {
        return log_llm_timing("openrouter", &model_name, &operation, None, || async {
            chat_completion_with_tools(
                messages,
                &model_name,
                CONFIG.openrouter_temperature,
                CONFIG.openrouter_top_p,
                CONFIG.openrouter_top_k,
            )
            .await
            .map_err(|err| anyhow!(err))
        })
        .await;
    }

    let payload = json!({
        "model": model_name,
        "messages": messages,
        "temperature": CONFIG.openrouter_temperature,
        "top_p": CONFIG.openrouter_top_p,
        "top_k": CONFIG.openrouter_top_k,
    });

    let result = log_llm_timing("openrouter", model_identifier, &operation, None, || async {
        let response = call_openrouter_api(&payload).await?;
        let content = response
            .get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Ok(parse_openrouter_response(model_identifier, content))
    })
    .await?;

    Ok(result)
}

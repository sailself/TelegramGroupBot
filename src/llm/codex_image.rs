use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use reqwest::StatusCode;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::config::{ThirdPartyProvider, CONFIG};
use crate::llm::audit::{
    log_llm_request_started, record_llm_request_success, LlmAuditContext, LlmUsageRecord,
};
use crate::llm::gemini::ImageGenerationError;
use crate::llm::media::{detect_mime_type, download_media};
use crate::llm::openai_codex;
use crate::utils::http::get_http_client_no_compression;

pub const CODEX_IMAGE_RESPONSES_MODEL: &str = "gpt-5.5";
pub const CODEX_IMAGE_TOOL_MODEL: &str = "gpt-image-2";
pub const CODEX_IMAGE_INSTRUCTIONS: &str = "You are an image generation assistant.";
pub const CODEX_IMAGE_MAX_INPUT_IMAGES: usize = 5;

const CODEX_IMAGE_MAX_ATTEMPTS: usize = 3;
const CODEX_IMAGE_RETRY_BASE_DELAY_MS: u64 = 1_000;
const MAX_CODEX_IMAGE_SSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CODEX_IMAGE_BASE64_CHARS: usize = 64 * 1024 * 1024;

pub const CODEX_IMAGE_SUPPORTED_SIZES: [&str; 7] = [
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
];

#[derive(Debug, Clone)]
pub struct CodexImageConfig {
    pub size: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone)]
struct CodexImageGenerationResult {
    images: Vec<Vec<u8>>,
    usage_model: Option<String>,
    usage: LlmUsageRecord,
}

pub fn codex_image_display_model() -> String {
    let model = CONFIG.openai_codex_image_model.trim();
    if model.is_empty() {
        CODEX_IMAGE_TOOL_MODEL.to_string()
    } else {
        model.to_string()
    }
}

fn codex_image_responses_model() -> String {
    let model = CONFIG.openai_codex_image_responses_model.trim();
    if model.is_empty() {
        CODEX_IMAGE_RESPONSES_MODEL.to_string()
    } else {
        model.to_string()
    }
}

pub fn codex_image_available() -> bool {
    CONFIG.enable_openai_codex
        && openai_codex::is_auth_ready()
        && crate::llm::runtime_models::selected_codex_model_record().is_some()
}

pub fn is_supported_codex_image_size(size: &str) -> bool {
    CODEX_IMAGE_SUPPORTED_SIZES.contains(&size)
}

pub fn canonicalize_codex_responses_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.eq_ignore_ascii_case("https://chatgpt.com/backend-api")
        || trimmed.eq_ignore_ascii_case("https://chatgpt.com/backend-api/v1")
        || trimmed.eq_ignore_ascii_case("https://chatgpt.com/backend-api/codex")
        || trimmed.eq_ignore_ascii_case("https://chatgpt.com/backend-api/codex/v1")
    {
        "https://chatgpt.com/backend-api/codex".to_string()
    } else {
        trimmed.to_string()
    }
}

fn codex_image_response_url() -> String {
    format!(
        "{}/responses",
        canonicalize_codex_responses_base_url(&CONFIG.openai_codex_base_url)
    )
}

pub fn build_codex_image_generation_payload(
    prompt: &str,
    input_images: &[ImageInput],
    size: Option<&str>,
) -> Value {
    let mut content = vec![json!({
        "type": "input_text",
        "text": prompt,
    })];

    for image in input_images {
        let encoded = general_purpose::STANDARD.encode(&image.bytes);
        content.push(json!({
            "type": "input_image",
            "image_url": format!("data:{};base64,{}", image.mime_type, encoded),
            "detail": "auto",
        }));
    }

    let mut tool = json!({
        "type": "image_generation",
        "model": codex_image_display_model(),
    });
    if let Some(size) = size {
        tool["size"] = json!(size);
    }

    json!({
        "model": codex_image_responses_model(),
        "input": [{
            "role": "user",
            "content": content,
        }],
        "instructions": CODEX_IMAGE_INSTRUCTIONS,
        "tools": [tool],
        "tool_choice": { "type": "image_generation" },
        "stream": true,
        "store": false,
    })
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn parse_sse_events(body: &str) -> Result<Vec<Value>> {
    let mut events = Vec::new();
    let mut current_data_lines = Vec::new();

    let flush_event = |lines: &mut Vec<String>, events: &mut Vec<Value>| -> Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let payload = lines.join("\n");
        lines.clear();
        if payload.trim().is_empty() || payload.trim() == "[DONE]" {
            return Ok(());
        }
        let value = serde_json::from_str::<Value>(&payload).with_context(|| {
            format!(
                "Failed to parse Codex image SSE event payload: {}",
                truncate_for_log(&payload, 500)
            )
        })?;
        events.push(value);
        Ok(())
    };

    for line in body.lines() {
        if line.trim().is_empty() {
            flush_event(&mut current_data_lines, &mut events)?;
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            current_data_lines.push(data.trim_start().to_string());
        }
    }
    flush_event(&mut current_data_lines, &mut events)?;

    Ok(events)
}

fn failure_message(event: &Value) -> Option<String> {
    event
        .pointer("/response/error/message")
        .and_then(|value| value.as_str())
        .or_else(|| {
            event
                .pointer("/error/message")
                .and_then(|value| value.as_str())
        })
        .or_else(|| event.get("message").and_then(|value| value.as_str()))
        .map(|value| value.to_string())
        .or_else(|| {
            event
                .pointer("/error/code")
                .and_then(|value| value.as_str())
                .map(|code| format!("OpenAI Codex image generation failed ({code})"))
        })
}

fn collect_image_generation_results_from_item(item: &Value, encoded_images: &mut Vec<String>) {
    if item.get("type").and_then(|value| value.as_str()) != Some("image_generation_call") {
        return;
    }
    if let Some(result) = item
        .get("result")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
    {
        encoded_images.push(result.to_string());
    }
}

fn usage_has_counts(usage: &LlmUsageRecord) -> bool {
    usage.input_tokens.is_some() || usage.output_tokens.is_some() || usage.total_tokens.is_some()
}

fn usage_record_from_value(usage: &Value, response_id: Option<String>) -> LlmUsageRecord {
    let input_tokens = usage.get("input_tokens").and_then(|value| value.as_i64());
    let output_tokens = usage.get("output_tokens").and_then(|value| value.as_i64());
    let total_tokens = usage
        .get("total_tokens")
        .and_then(|value| value.as_i64())
        .or_else(|| match (input_tokens, output_tokens) {
            (Some(input_tokens), Some(output_tokens)) => Some(input_tokens + output_tokens),
            _ => None,
        });
    let reasoning_tokens = usage
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(|value| value.as_i64());
    let cached_input_tokens = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|value| value.as_i64());

    LlmUsageRecord {
        response_id,
        input_tokens,
        output_tokens,
        total_tokens,
        reasoning_tokens,
        cached_input_tokens,
        raw_usage_json: Some(usage.to_string()),
    }
}

fn usage_value(value: &Value) -> Option<&Value> {
    value.get("usage").filter(|usage| !usage.is_null())
}

fn extract_codex_image_generation_result(
    body: &str,
) -> Result<CodexImageGenerationResult, ImageGenerationError> {
    let events = parse_sse_events(body).map_err(|err| ImageGenerationError(err.to_string()))?;
    let mut output_item_images = Vec::new();
    let mut completed_output_images = Vec::new();
    let mut response_id = None;
    let mut response_usage = LlmUsageRecord::default();
    let mut image_usage = LlmUsageRecord::default();

    for event in &events {
        match event.get("type").and_then(|value| value.as_str()) {
            Some("response.failed") | Some("error") => {
                let message = failure_message(event)
                    .unwrap_or_else(|| "OpenAI Codex image generation failed".to_string());
                return Err(ImageGenerationError(message));
            }
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    collect_image_generation_results_from_item(item, &mut output_item_images);
                    if item.get("type").and_then(|value| value.as_str())
                        == Some("image_generation_call")
                    {
                        if let Some(usage) = usage_value(item) {
                            image_usage = usage_record_from_value(usage, response_id.clone());
                        }
                    }
                }
            }
            Some("response.completed") => {
                response_id = event
                    .pointer("/response/id")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string());
                if let Some(usage) = event
                    .pointer("/response/usage")
                    .filter(|usage| !usage.is_null())
                {
                    response_usage = usage_record_from_value(usage, response_id.clone());
                }
                if let Some(output) = event
                    .pointer("/response/output")
                    .and_then(|value| value.as_array())
                {
                    for item in output {
                        collect_image_generation_results_from_item(
                            item,
                            &mut completed_output_images,
                        );
                        if item.get("type").and_then(|value| value.as_str())
                            == Some("image_generation_call")
                        {
                            if let Some(usage) = usage_value(item) {
                                image_usage = usage_record_from_value(usage, response_id.clone());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let encoded_images = if output_item_images.is_empty() {
        completed_output_images
    } else {
        output_item_images
    };

    let mut images = Vec::new();
    for encoded in encoded_images {
        if encoded.len() > MAX_CODEX_IMAGE_BASE64_CHARS {
            return Err(ImageGenerationError(
                "OpenAI Codex image result exceeded the maximum size".to_string(),
            ));
        }
        let bytes = general_purpose::STANDARD
            .decode(encoded)
            .map_err(|err| ImageGenerationError(format!("Invalid Codex image payload: {err}")))?;
        images.push(bytes);
    }

    if images.is_empty() {
        return Err(ImageGenerationError(
            "No images returned by OpenAI Codex".to_string(),
        ));
    }

    let (usage_model, usage) = if usage_has_counts(&image_usage) {
        (Some(codex_image_display_model()), image_usage)
    } else if usage_has_counts(&response_usage) {
        (Some(codex_image_display_model()), response_usage)
    } else {
        (None, LlmUsageRecord::default())
    };

    Ok(CodexImageGenerationResult {
        images,
        usage_model,
        usage,
    })
}

async fn read_response_body_limited(mut response: reqwest::Response) -> Result<String> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        body.extend_from_slice(&chunk);
        if body.len() > MAX_CODEX_IMAGE_SSE_BYTES {
            return Err(anyhow!(
                "OpenAI Codex image response exceeded the maximum size"
            ));
        }
    }
    String::from_utf8(body).context("OpenAI Codex image response was not valid UTF-8")
}

fn should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

fn should_retry_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

fn retry_delay(attempt: usize) -> Duration {
    Duration::from_millis(CODEX_IMAGE_RETRY_BASE_DELAY_MS.saturating_mul(attempt as u64))
}

async fn call_codex_image_api(
    payload: &Value,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    let url = codex_image_response_url();
    let model = codex_image_display_model();
    let started_at = chrono::Utc::now();
    let metadata = json!({
        "responses_model": payload.get("model").cloned().unwrap_or(Value::Null),
        "timeout_secs": CONFIG.openai_codex_request_timeout_secs,
        "streaming_sse": true,
    });
    log_llm_request_started(
        ThirdPartyProvider::OpenAICodex.as_str(),
        &model,
        "generate_image_with_codex",
        started_at,
        Some(&metadata),
    );

    let client = get_http_client_no_compression();
    for attempt in 1..=CODEX_IMAGE_MAX_ATTEMPTS {
        let auth = openai_codex::get_valid_auth_context()
            .await
            .map_err(|err| ImageGenerationError(err.to_string()))?;
        let mut request = client
            .post(&url)
            .timeout(Duration::from_secs(
                CONFIG.openai_codex_request_timeout_secs,
            ))
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .header(reqwest::header::ACCEPT_ENCODING, "identity")
            .header(reqwest::header::CONTENT_TYPE, "application/json");
        for (name, value) in openai_codex::codex_headers(&auth, None) {
            request = request.header(name, value);
        }

        debug!(
            "OpenAI Codex image request starting: model={}, responses_model={}, size={}, attempt={}/{}",
            model,
            payload.get("model").and_then(|value| value.as_str()).unwrap_or("unknown"),
            payload.pointer("/tools/0/size").and_then(|value| value.as_str()).unwrap_or("unknown"),
            attempt,
            CODEX_IMAGE_MAX_ATTEMPTS
        );

        let response = match request.json(payload).send().await {
            Ok(response) => response,
            Err(err) => {
                let retrying = should_retry_error(&err) && attempt < CODEX_IMAGE_MAX_ATTEMPTS;
                warn!(
                    "OpenAI Codex image request failed to send: model={}, error={}, timeout={}, connect={}, status={:?}, attempt={}/{}, retrying={}",
                    model,
                    err,
                    err.is_timeout(),
                    err.is_connect(),
                    err.status(),
                    attempt,
                    CODEX_IMAGE_MAX_ATTEMPTS,
                    retrying
                );
                if retrying {
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(ImageGenerationError(format!(
                    "OpenAI Codex image request failed: {err}"
                )));
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status == StatusCode::UNAUTHORIZED && attempt < CODEX_IMAGE_MAX_ATTEMPTS {
                warn!(
                    "OpenAI Codex image request unauthorized; refreshing auth and retrying (attempt={}/{})",
                    attempt,
                    CODEX_IMAGE_MAX_ATTEMPTS
                );
                openai_codex::force_refresh_auth_tokens()
                    .await
                    .map_err(|err| ImageGenerationError(err.to_string()))?;
                continue;
            }
            let retrying = should_retry_status(status) && attempt < CODEX_IMAGE_MAX_ATTEMPTS;
            warn!(
                "OpenAI Codex image API error: model={}, status={}, body={}, attempt={}/{}, retrying={}",
                model,
                status,
                truncate_for_log(&body, 1000),
                attempt,
                CODEX_IMAGE_MAX_ATTEMPTS,
                retrying
            );
            if retrying {
                tokio::time::sleep(retry_delay(attempt)).await;
                continue;
            }
            return Err(ImageGenerationError(format!(
                "OpenAI Codex image request failed with status {}: {}",
                status,
                truncate_for_log(&body, 1000)
            )));
        }

        let body = read_response_body_limited(response)
            .await
            .map_err(|err| ImageGenerationError(err.to_string()))?;
        let result = extract_codex_image_generation_result(&body)?;
        let images = result.images;
        info!(
            "OpenAI Codex image request completed: model={}, images={}",
            model,
            images.len()
        );
        let usage_model = result.usage_model.unwrap_or_else(|| model.clone());
        record_llm_request_success(
            audit_context,
            ThirdPartyProvider::OpenAICodex.as_str(),
            &usage_model,
            "generate_image_with_codex",
            started_at,
            chrono::Utc::now(),
            result.usage,
        )
        .await;
        return Ok(images);
    }

    Err(ImageGenerationError(
        "OpenAI Codex image request exhausted retries".to_string(),
    ))
}

pub async fn generate_image_with_codex(
    prompt: &str,
    image_urls: &[String],
    image_config: Option<CodexImageConfig>,
    upload_to_cwd: bool,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    if image_urls.len() > CODEX_IMAGE_MAX_INPUT_IMAGES {
        return Err(ImageGenerationError(format!(
            "OpenAI Codex image generation supports at most {} input images",
            CODEX_IMAGE_MAX_INPUT_IMAGES
        )));
    }

    let mut input_images = Vec::new();
    for url in image_urls {
        if let Some(bytes) = download_media(url).await {
            let mime_type = detect_mime_type(&bytes).unwrap_or_else(|| "image/png".to_string());
            input_images.push(ImageInput { bytes, mime_type });
        }
    }

    let size = image_config
        .as_ref()
        .and_then(|config| config.size.as_deref())
        .filter(|size| is_supported_codex_image_size(size));
    let payload = build_codex_image_generation_payload(prompt, &input_images, size);
    let images = call_codex_image_api(&payload, audit_context).await?;

    if upload_to_cwd && !CONFIG.cwd_pw_api_key.trim().is_empty() {
        let model = codex_image_display_model();
        for image in &images {
            let mime_type = detect_mime_type(image).unwrap_or_else(|| "image/png".to_string());
            let _ = crate::tools::cwd_uploader::upload_image_bytes_to_cwd(
                image,
                &CONFIG.cwd_pw_api_key,
                &mime_type,
                Some(model.as_str()),
                Some(prompt),
            )
            .await;
        }
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose;

    #[test]
    fn canonicalizes_chatgpt_codex_responses_base_urls() {
        assert_eq!(
            canonicalize_codex_responses_base_url("https://chatgpt.com/backend-api"),
            "https://chatgpt.com/backend-api/codex"
        );
        assert_eq!(
            canonicalize_codex_responses_base_url("https://chatgpt.com/backend-api/v1"),
            "https://chatgpt.com/backend-api/codex"
        );
        assert_eq!(
            canonicalize_codex_responses_base_url("https://chatgpt.com/backend-api/codex/v1"),
            "https://chatgpt.com/backend-api/codex"
        );
        assert_eq!(
            canonicalize_codex_responses_base_url("https://proxy.example.com/codex"),
            "https://proxy.example.com/codex"
        );
    }

    #[test]
    fn builds_codex_image_generation_payload_with_data_url_inputs() {
        let payload = build_codex_image_generation_payload(
            "draw a lighthouse",
            &[ImageInput {
                bytes: b"image-bytes".to_vec(),
                mime_type: "image/png".to_string(),
            }],
            Some("2048x1152"),
        );

        assert_eq!(payload["model"], codex_image_responses_model());
        assert_eq!(payload["instructions"], CODEX_IMAGE_INSTRUCTIONS);
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["store"], false);
        assert_eq!(payload["tool_choice"]["type"], "image_generation");
        assert_eq!(payload["tools"][0]["type"], "image_generation");
        assert_eq!(payload["tools"][0]["model"], codex_image_display_model());
        assert_eq!(payload["tools"][0]["size"], "2048x1152");

        let content = payload["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "draw a lighthouse");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["detail"], "auto");
        assert_eq!(
            content[1]["image_url"],
            format!(
                "data:image/png;base64,{}",
                general_purpose::STANDARD.encode(b"image-bytes")
            )
        );
    }

    #[test]
    fn omits_codex_image_size_when_size_is_unspecified() {
        let payload = build_codex_image_generation_payload("draw a poster", &[], None);

        assert_eq!(payload["tools"][0]["model"], codex_image_display_model());
        assert!(payload["tools"][0].get("size").is_none());
    }

    #[test]
    fn extracts_images_from_output_item_done_sse_events() {
        let body = format!(
            "event: response.output_item.done\ndata: {}\n\n",
            serde_json::json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "image_generation_call",
                    "id": "ig_1",
                    "status": "completed",
                    "result": general_purpose::STANDARD.encode(b"png-bytes")
                }
            })
        );

        let images = extract_codex_image_generation_result(&body).unwrap().images;
        assert_eq!(images, vec![b"png-bytes".to_vec()]);
    }

    #[test]
    fn extracts_images_from_completed_response_sse_events() {
        let body = format!(
            "event: response.completed\ndata: {}\n\n",
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "output": [{
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "status": "completed",
                        "result": general_purpose::STANDARD.encode(b"completed-image")
                    }]
                }
            })
        );

        let images = extract_codex_image_generation_result(&body).unwrap().images;
        assert_eq!(images, vec![b"completed-image".to_vec()]);
    }

    #[test]
    fn attributes_top_level_response_usage_to_image_model() {
        let body = format!(
            "event: response.completed\ndata: {}\n\n",
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5.5",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "input_tokens_details": {
                            "cached_tokens": 3
                        },
                        "output_tokens_details": {
                            "reasoning_tokens": 7
                        }
                    },
                    "output": [{
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "status": "completed",
                        "result": general_purpose::STANDARD.encode(b"completed-image")
                    }]
                }
            })
        );

        let result = extract_codex_image_generation_result(&body).unwrap();

        assert_eq!(result.images, vec![b"completed-image".to_vec()]);
        assert_eq!(
            result.usage_model.as_deref(),
            Some(codex_image_display_model().as_str())
        );
        assert_eq!(result.usage.response_id.as_deref(), Some("resp_123"));
        assert_eq!(result.usage.input_tokens, Some(10));
        assert_eq!(result.usage.output_tokens, Some(20));
        assert_eq!(result.usage.total_tokens, Some(30));
        assert_eq!(result.usage.cached_input_tokens, Some(3));
        assert_eq!(result.usage.reasoning_tokens, Some(7));
    }

    #[test]
    fn prefers_image_generation_item_usage_for_image_model() {
        let body = format!(
            "event: response.completed\ndata: {}\n\n",
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5.5",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30
                    },
                    "output": [{
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 200,
                            "total_tokens": 300,
                            "input_tokens_details": {
                                "image_tokens": 80,
                                "text_tokens": 20
                            }
                        },
                        "result": general_purpose::STANDARD.encode(b"completed-image")
                    }]
                }
            })
        );

        let result = extract_codex_image_generation_result(&body).unwrap();

        assert_eq!(
            result.usage_model.as_deref(),
            Some(codex_image_display_model().as_str())
        );
        assert_eq!(result.usage.response_id.as_deref(), Some("resp_123"));
        assert_eq!(result.usage.input_tokens, Some(100));
        assert_eq!(result.usage.output_tokens, Some(200));
        assert_eq!(result.usage.total_tokens, Some(300));
    }

    #[test]
    fn returns_failure_message_from_failed_sse_event() {
        let body = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"quota exceeded\"}}}\n\n";

        let err = extract_codex_image_generation_result(body).unwrap_err();
        assert!(err.0.contains("quota exceeded"));
    }
}

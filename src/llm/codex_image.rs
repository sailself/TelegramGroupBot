use std::time::Duration;

use base64::{engine::general_purpose, Engine as _};
use reqwest::header::{ACCEPT, ACCEPT_ENCODING, CONTENT_TYPE};
use serde_json::{json, Value};
use tracing::{debug, info};

use crate::config::{ThirdPartyProvider, CONFIG};
use crate::llm::audit::{LlmAuditContext, LlmUsageRecord};
use crate::llm::gemini::ImageGenerationError;
use crate::llm::media::{detect_mime_type, download_media};
use crate::llm::openai_codex;
use crate::llm::transport::sse::parse_sse_data_events;
use crate::llm::transport::{
    call_with_retry, read_body_limited, usage, LlmCall, ProviderError, RetryPolicy,
};
use crate::utils::http::get_http_client_no_compression;

pub const CODEX_IMAGE_RESPONSES_MODEL: &str = "gpt-5.5";
pub const CODEX_IMAGE_TOOL_MODEL: &str = "gpt-image-2";
pub const CODEX_IMAGE_INSTRUCTIONS: &str = "You are an image generation assistant.";
pub const CODEX_IMAGE_MAX_INPUT_IMAGES: usize = 5;

const CODEX_IMAGE_PROVIDER: &str = "OpenAI Codex";
/// Three attempts a second apart; a 401 triggers a token refresh first.
const CODEX_IMAGE_RETRY_POLICY: RetryPolicy = RetryPolicy {
    refresh_auth_on_unauthorized: true,
    ..RetryPolicy::linear(3, Duration::from_millis(1_000))
};
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
    usage: LlmUsageRecord,
}

pub fn codex_image_display_model() -> String {
    let model = CONFIG.codex.image_model.trim();
    if model.is_empty() {
        CODEX_IMAGE_TOOL_MODEL.to_string()
    } else {
        model.to_string()
    }
}

fn codex_image_responses_model() -> String {
    let model = CONFIG.codex.image_responses_model.trim();
    if model.is_empty() {
        CODEX_IMAGE_RESPONSES_MODEL.to_string()
    } else {
        model.to_string()
    }
}

pub fn codex_image_available() -> bool {
    CONFIG.codex.enabled
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
        canonicalize_codex_responses_base_url(&CONFIG.codex.base_url)
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

fn usage_value(value: &Value) -> Option<&Value> {
    value.get("usage").filter(|usage| !usage.is_null())
}

fn extract_codex_image_generation_result(
    body: &str,
) -> Result<CodexImageGenerationResult, ImageGenerationError> {
    let events = parse_sse_data_events(body).map_err(|err| {
        ImageGenerationError(format!("Failed to parse Codex image SSE stream: {err}"))
    })?;
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
                            image_usage =
                                usage::from_responses_usage_object(usage, response_id.clone());
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
                    response_usage = usage::from_responses_usage_object(usage, response_id.clone());
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
                                image_usage =
                                    usage::from_responses_usage_object(usage, response_id.clone());
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

    // Prefer the image tool's own usage; fall back to the response total.
    let usage = if usage_has_counts(&image_usage) {
        image_usage
    } else if usage_has_counts(&response_usage) {
        response_usage
    } else {
        LlmUsageRecord::default()
    };

    Ok(CodexImageGenerationResult { images, usage })
}

async fn call_codex_image_api(
    payload: &Value,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    let url = codex_image_response_url();
    let url = url.as_str();
    let model = codex_image_display_model();
    let model_ref = model.as_str();
    let metadata = json!({
        "responses_model": payload.get("model").cloned().unwrap_or(Value::Null),
        "timeout_secs": CONFIG.codex.request_timeout_secs,
        "streaming_sse": true,
    });
    let call = LlmCall::begin(
        ThirdPartyProvider::OpenAICodex.as_str(),
        model_ref,
        "generate_image_with_codex",
        audit_context,
        Some(&metadata),
    )
    .with_label(CODEX_IMAGE_PROVIDER);
    let timeout = Duration::from_secs(CONFIG.codex.request_timeout_secs);

    let result = call_with_retry(
        &call,
        &CODEX_IMAGE_RETRY_POLICY,
        |attempt| async move {
            if attempt.previous_unauthorized {
                openai_codex::force_refresh_auth_tokens()
                    .await
                    .map_err(|err| ProviderError::rejected(err.to_string()))?;
            }
            let auth = openai_codex::get_valid_auth_context()
                .await
                .map_err(|err| ProviderError::rejected(err.to_string()))?;
            let mut request = get_http_client_no_compression()
                .post(url)
                .timeout(timeout)
                .header(ACCEPT, "text/event-stream")
                .header(ACCEPT_ENCODING, "identity")
                .header(CONTENT_TYPE, "application/json");
            for (name, value) in openai_codex::codex_headers(&auth, None) {
                request = request.header(name, value);
            }
            debug!(
                "OpenAI Codex image request starting: model={}, responses_model={}, size={}, attempt={}/{}",
                model_ref,
                payload
                    .get("model")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
                payload
                    .pointer("/tools/0/size")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
                attempt.number,
                attempt.max_attempts
            );
            Ok(request.json(payload))
        },
        |_| {},
        |response| async move {
            // A stream interrupted mid-body surfaces as a retryable transport
            // error and is re-issued; `store=false` leaves no server state.
            let bytes =
                read_body_limited(response, CODEX_IMAGE_PROVIDER, MAX_CODEX_IMAGE_SSE_BYTES)
                    .await?;
            let body = String::from_utf8(bytes).map_err(|_| {
                ProviderError::decode(
                    CODEX_IMAGE_PROVIDER,
                    "OpenAI Codex image response was not valid UTF-8",
                    false,
                )
            })?;
            extract_codex_image_generation_result(&body)
                .map_err(|err| ProviderError::decode(CODEX_IMAGE_PROVIDER, err.0, false))
        },
        |result: &CodexImageGenerationResult| result.usage.clone(),
    )
    .await;

    match result {
        Ok(result) => {
            info!(
                "OpenAI Codex image request completed: model={}, images={}",
                model,
                result.images.len()
            );
            Ok(result.images)
        }
        Err(err) => Err(codex_image_error(err)),
    }
}

/// Decode failures already carry the user-facing message (`response.failed`
/// reasons, payload limits); other failures keep the transport's wording.
fn codex_image_error(err: ProviderError) -> ImageGenerationError {
    match err {
        ProviderError::Decode { detail, .. } => ImageGenerationError(detail),
        other => ImageGenerationError(other.to_string()),
    }
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

    if upload_to_cwd && !CONFIG.cwd_pw.api_key.trim().is_empty() {
        let model = codex_image_display_model();
        for image in &images {
            let mime_type = detect_mime_type(image).unwrap_or_else(|| "image/png".to_string());
            let _ = crate::tools::cwd_uploader::upload_image_bytes_to_cwd(
                image,
                &CONFIG.cwd_pw.api_key,
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
                            "cached_tokens": 3,
                            "cache_write_tokens": 4
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
        assert_eq!(result.usage.response_id.as_deref(), Some("resp_123"));
        assert_eq!(result.usage.input_tokens, Some(10));
        assert_eq!(result.usage.output_tokens, Some(20));
        assert_eq!(result.usage.total_tokens, Some(30));
        assert_eq!(result.usage.cached_input_tokens, Some(3));
        assert_eq!(result.usage.cache_write_tokens, Some(4));
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

        assert_eq!(result.usage.response_id.as_deref(), Some("resp_123"));
        assert_eq!(result.usage.input_tokens, Some(100));
        assert_eq!(result.usage.output_tokens, Some(200));
        assert_eq!(result.usage.total_tokens, Some(300));
    }

    #[test]
    fn invalid_sse_payloads_are_reported_by_the_shared_decoder() {
        let err = extract_codex_image_generation_result(
            "data: {not json

",
        )
        .expect_err("invalid payload");
        assert!(
            err.0.contains("9-byte SSE event payload"),
            "expected the shared SSE decoder's message, got: {}",
            err.0
        );
    }

    #[test]
    fn returns_failure_message_from_failed_sse_event() {
        let body = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"quota exceeded\"}}}\n\n";

        let err = extract_codex_image_generation_result(body).unwrap_err();
        assert!(err.0.contains("quota exceeded"));
    }
}

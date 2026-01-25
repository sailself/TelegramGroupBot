use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tracing::{debug, warn};

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, download_media};
use crate::utils::http::get_http_client;
use crate::utils::timing::log_llm_timing;

#[derive(Debug, thiserror::Error)]
#[error("Image generation failed: {0}")]
pub struct ImageGenerationError(pub String);

#[derive(Debug, Clone)]
pub struct GeminiImageConfig {
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { #[serde(rename = "inlineData")] inline_data: GeminiInlineData },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

fn build_safety_settings() -> Vec<serde_json::Value> {
    let profile = CONFIG.gemini_safety_settings.as_str();
    let threshold = match profile {
        "standard" => "BLOCK_MEDIUM_AND_ABOVE",
        "permissive" => "OFF",
        _ => {
            warn!("Unknown GEMINI_SAFETY_SETTINGS value '{}', using permissive defaults.", profile);
            "OFF"
        }
    };

    vec![
        json!({ "category": "HARM_CATEGORY_HARASSMENT", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": threshold }),
    ]
}

fn build_image_config(config: Option<&GeminiImageConfig>) -> Option<Value> {
    let config = config?;
    let mut map = Map::new();

    if let Some(aspect_ratio) = config.aspect_ratio.as_deref() {
        let trimmed = aspect_ratio.trim();
        if !trimmed.is_empty() {
            map.insert("aspectRatio".to_string(), json!(trimmed));
        }
    }

    if let Some(image_size) = config.image_size.as_deref() {
        let trimmed = image_size.trim();
        if !trimmed.is_empty() {
            map.insert("imageSize".to_string(), json!(trimmed));
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(Value::Object(map))
    }
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn summarize_gemini_parts(parts: &[Value]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| {
            if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                json!({ "text": truncate_for_log(text, 200) })
            } else if let Some(inline_data) = part.get("inlineData") {
                let mime_type = inline_data
                    .get("mimeType")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown");
                let data_len = inline_data
                    .get("data")
                    .and_then(|value| value.as_str())
                    .map(|value| value.len())
                    .unwrap_or(0);
                json!({ "inlineData": { "mimeType": mime_type, "dataLen": data_len } })
            } else if let Some(file_data) = part.get("fileData") {
                let file_uri = file_data
                    .get("fileUri")
                    .and_then(|value| value.as_str())
                    .map(|value| truncate_for_log(value, 200));
                json!({ "fileData": { "fileUri": file_uri } })
            } else {
                json!({ "unknownPart": true })
            }
        })
        .collect()
}

fn summarize_gemini_payload(payload: &Value, system_prompt_label: Option<&str>) -> Value {
    let mut summary = Map::new();

    if payload.pointer("/systemInstruction").is_some() {
        let label = system_prompt_label.unwrap_or("inline_system_prompt");
        summary.insert(
            "systemInstruction".to_string(),
            Value::String(label.to_string()),
        );
    }

    if let Some(contents) = payload.get("contents").and_then(|value| value.as_array()) {
        let mut summarized_contents = Vec::new();
        for content in contents {
            let role = content
                .get("role")
                .and_then(|value| value.as_str())
                .unwrap_or("user");
            let parts = content
                .get("parts")
                .and_then(|value| value.as_array())
                .map(|parts| summarize_gemini_parts(parts))
                .unwrap_or_default();
            summarized_contents.push(json!({ "role": role, "parts": parts }));
        }
        summary.insert("contents".to_string(), Value::Array(summarized_contents));
    }

    if let Some(config) = payload.get("generationConfig") {
        summary.insert("generationConfig".to_string(), config.clone());
    }

    if let Some(tools) = payload.get("tools") {
        summary.insert("tools".to_string(), tools.clone());
    }

    if let Some(safety) = payload.get("safetySettings").and_then(|value| value.as_array()) {
        summary.insert("safetySettingsCount".to_string(), json!(safety.len()));
    }

    Value::Object(summary)
}

fn summarize_gemini_response(response: &GeminiResponse) -> Value {
    let mut text_parts = 0usize;
    let mut image_parts = 0usize;
    let mut text_preview = None;

    let candidates = response.candidates.as_deref().unwrap_or(&[]);
    for candidate in candidates {
        if let Some(content) = &candidate.content {
            if let Some(parts) = &content.parts {
                for part in parts {
                    match part {
                        GeminiPart::Text { text } => {
                            text_parts += 1;
                            if text_preview.is_none() && !text.trim().is_empty() {
                                text_preview = Some(truncate_for_log(text, 200));
                            }
                        }
                        GeminiPart::InlineData { inline_data } => {
                            if inline_data.mime_type.starts_with("image/") {
                                image_parts += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    json!({
        "candidates": response.candidates.as_ref().map(|candidates| candidates.len()).unwrap_or(0),
        "textParts": text_parts,
        "imageParts": image_parts,
        "textPreview": text_preview
    })
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

fn build_gemini_parts(
    user_content: &str,
    image_data_list: &[Vec<u8>],
    video_data: Option<&[u8]>,
    audio_data: Option<&[u8]>,
    youtube_urls: &[String],
    text_after_media: bool,
) -> Vec<serde_json::Value> {
    let mut parts = Vec::new();
    let text_part = json!({ "text": user_content });

    if !text_after_media {
        parts.push(text_part.clone());
    }

    for url in youtube_urls {
        parts.push(json!({
            "fileData": {
                "fileUri": url
            }
        }));
    }

    for image_data in image_data_list {
        let mime_type = detect_mime_type(image_data).unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        parts.push(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }));
    }

    if let Some(video_bytes) = video_data {
        let mime_type = detect_mime_type(video_bytes).unwrap_or_else(|| "video/mp4".to_string());
        let encoded = general_purpose::STANDARD.encode(video_bytes);
        parts.push(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }));
    }

    if let Some(audio_bytes) = audio_data {
        let mime_type = detect_mime_type(audio_bytes).unwrap_or_else(|| "audio/mpeg".to_string());
        let encoded = general_purpose::STANDARD.encode(audio_bytes);
        parts.push(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }));
    }

    if text_after_media {
        parts.push(text_part);
    }

    parts
}

fn extract_text_from_response(response: GeminiResponse) -> String {
    let mut text_parts = Vec::new();
    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    if let GeminiPart::Text { text } = part {
                        if !text.trim().is_empty() {
                            text_parts.push(text);
                        }
                    }
                }
            }
        }
    }
    text_parts.join("\n")
}

fn extract_images_from_response(response: GeminiResponse) -> Vec<Vec<u8>> {
    let mut images = Vec::new();
    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    if let GeminiPart::InlineData { inline_data } = part {
                        if inline_data.mime_type.starts_with("image/") {
                            if let Ok(bytes) = general_purpose::STANDARD.decode(inline_data.data) {
                                images.push(bytes);
                            }
                        }
                    }
                }
            }
        }
    }
    images
}

async fn call_gemini_api(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
) -> Result<GeminiResponse> {
    let client = get_http_client();
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, CONFIG.gemini_api_key
    );

    if tracing::enabled!(tracing::Level::DEBUG) {
        let payload_summary = summarize_gemini_payload(&payload, system_prompt_label);
        debug!(target: "llm.gemini", model = model, payload = %payload_summary);
    }

    let response = match client
        .post(url)
        .timeout(Duration::from_secs(90))
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => response,
        Err(err) => {
            warn!(
                "Gemini request failed to send: {err} (timeout={}, connect={}, status={:?}, url={:?})",
                err.is_timeout(),
                err.is_connect(),
                err.status(),
                err.url(),
            );
            return Err(anyhow!("Gemini request failed: {}", err));
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let (message, body_summary) = summarize_error_body(&body);
        warn!("Gemini API error: status={}, body={}", status, body_summary);
        if tracing::enabled!(tracing::Level::DEBUG) {
            debug!(
                target: "llm.gemini",
                status = %status,
                body = %truncate_for_log(&body, 4000)
            );
        }
        let detail = message.unwrap_or(body_summary);
        return Err(anyhow!("Gemini request failed with status {}: {}", status, detail));
    }

    let value = response.json::<GeminiResponse>().await?;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let response_summary = summarize_gemini_response(&value);
        debug!(target: "llm.gemini", model = model, response = %response_summary);
    }
    Ok(value)
}

pub async fn call_gemini(
    system_prompt: &str,
    user_content: &str,
    response_language: Option<&str>,
    use_search_grounding: bool,
    _use_url_context: bool,
    _thinking_level: Option<&str>,
    image_url: Option<&str>,
    use_pro_model: bool,
    image_data_list: Option<Vec<Vec<u8>>>,
    video_data: Option<Vec<u8>>,
    audio_data: Option<Vec<u8>>,
    youtube_urls: Option<Vec<String>>,
    system_prompt_label: Option<&str>,
) -> Result<String> {
    let mut content = user_content.to_string();
    if let Some(lang) = response_language {
        content.push_str(&format!("\n\nPlease reply in {}.", lang));
    }

    let youtube_urls = youtube_urls.unwrap_or_default();

    let mut images = image_data_list.unwrap_or_default();
    if images.is_empty() {
        if let Some(url) = image_url {
            if let Some(data) = download_media(url).await {
                images.push(data);
            }
        }
    }

    let text_after_media = video_data.is_some() || audio_data.is_some() || !youtube_urls.is_empty();
    let parts = build_gemini_parts(
        &content,
        &images,
        video_data.as_deref(),
        audio_data.as_deref(),
        &youtube_urls,
        text_after_media,
    );
    let payload = json!({
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": {
            "temperature": CONFIG.gemini_temperature,
            "topK": CONFIG.gemini_top_k,
            "topP": CONFIG.gemini_top_p,
            "maxOutputTokens": CONFIG.gemini_max_output_tokens,
        },
        "safetySettings": build_safety_settings(),
        "tools": if use_search_grounding { vec![json!({ "google_search": {} })] } else { vec![] },
    });

    let model = if use_pro_model { &CONFIG.gemini_pro_model } else { &CONFIG.gemini_model };
    let operation = if use_pro_model { "call_gemini_pro" } else { "call_gemini" };

    log_llm_timing("gemini", model, operation, None, || async {
        let response = call_gemini_api(model, payload, system_prompt_label).await?;
        Ok(extract_text_from_response(response))
    })
    .await
}

pub async fn generate_image_with_gemini(
    prompt: &str,
    image_urls: &[String],
    image_config: Option<GeminiImageConfig>,
    upload_to_cwd: bool,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    let mut images = Vec::new();
    for url in image_urls {
        if let Some(data) = download_media(url).await {
            images.push(data);
        }
    }

    let base_instruction = if images.is_empty() {
        "Generate an image based on the prompt. CRITICAL: response be an image, NOT TEXT."
    } else {
        "Edit the images based on the prompt. CRITICAL: response be an image, NOT TEXT."
    };

    let system_instruction = base_instruction.to_string();
    let parts = build_gemini_parts(prompt, &images, None, None, &[], false);
    let mut generation_config = json!({
        "responseModalities": ["TEXT", "IMAGE"]
    });
    if let Some(image_config) = build_image_config(image_config.as_ref()) {
        if let Some(config_object) = generation_config.as_object_mut() {
            config_object.insert("imageConfig".to_string(), image_config);
        }
    }

    let payload = json!({
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": generation_config,
        "safetySettings": build_safety_settings(),
        "tools": [{ "google_search": {} }],
    });

    let model = &CONFIG.gemini_image_model;
    let response = call_gemini_api(model, payload, Some("image_generation_system_prompt"))
        .await
        .map_err(|err| ImageGenerationError(err.to_string()))?;

    let images = extract_images_from_response(response);
    if images.is_empty() {
        return Err(ImageGenerationError("No images returned by Gemini".to_string()));
    }

    if upload_to_cwd && !CONFIG.cwd_pw_api_key.trim().is_empty() {
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

pub async fn generate_image_with_vertex(
    prompt: &str,
    image_urls: &[String],
    _model_hint: Option<&str>,
    image_config: Option<GeminiImageConfig>,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    warn!("Vertex image generation is not implemented; falling back to Gemini image generation.");
    generate_image_with_gemini(
        prompt,
        image_urls,
        image_config,
        !CONFIG.cwd_pw_api_key.is_empty(),
    )
    .await
}

pub async fn generate_video_with_veo(
    user_prompt: &str,
    image_data: Option<Vec<u8>>,
) -> Result<(Option<Vec<u8>>, Option<String>), anyhow::Error> {
    warn!("Video generation via VEO is not implemented in the Rust port.");
    let _ = (user_prompt, image_data);
    Ok((None, None))
}

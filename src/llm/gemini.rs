use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use reqwest::header::CONTENT_TYPE;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::config::CONFIG;
use crate::llm::audit::{LlmAuditContext, LlmUsageRecord};
use crate::llm::media::{detect_mime_type, download_media, MediaFile, MediaKind};
use crate::llm::tool_loop::{
    run_tool_loop, BoxFuture, ModelTurn, ToolCall, ToolProtocol, TurnDeadline,
};
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::transport::retry::is_retryable_status;
use crate::llm::transport::{
    call_with_retry, read_body_limited, read_json, usage, LlmCall, ProviderError, RetryPolicy,
};
use crate::utils::http::get_http_client;
use crate::utils::text::truncate_for_log;

#[derive(Debug, thiserror::Error)]
#[error("Image generation failed: {0}")]
pub struct ImageGenerationError(pub String);

#[derive(Debug, Clone)]
pub struct GeminiImageConfig {
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GeminiCallResult {
    pub text: String,
    pub model_used: String,
}

#[derive(Debug, Clone)]
pub struct GeminiMusicGenerationResult {
    pub lyrics_text: String,
    pub notes_text: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub model_used: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<Value>,
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
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    ExecutableCode {
        #[serde(rename = "executableCode")]
        executable_code: GeminiExecutableCode,
    },
    CodeExecutionResult {
        #[serde(rename = "codeExecutionResult")]
        code_execution_result: GeminiCodeExecutionResult,
    },
    /// Anything else the API may add (function calls, thought signatures,
    /// file data). Kept so one unfamiliar part never fails the whole response.
    Other(Value),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiExecutableCode {
    code: Option<String>,
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCodeExecutionResult {
    output: Option<String>,
    outcome: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFileInfo {
    name: String,
    uri: String,
    mime_type: Option<String>,
    state: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiFileResponse {
    file: GeminiFileInfo,
}

#[derive(Debug, Clone)]
struct UploadedFileRef {
    uri: String,
}

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
/// generateContent: two attempts, 900ms apart (plus jitter / Retry-After).
const GEMINI_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(2, Duration::from_millis(900));
/// File API, operation polls and video download: cheap to repeat.
const GEMINI_FILE_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(2, Duration::from_millis(500));
const GEMINI_FILE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Uploading a Telegram video or audio attachment can take a while.
const GEMINI_UPLOAD_TIMEOUT: Duration = Duration::from_secs(180);
const GEMINI_LITE_FALLBACK_MAX_ATTEMPTS: usize = 3;
const LYRIA_GENERATION_TIMEOUT_SECS: u64 = 240;
const VEO_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const VEO_VIDEO_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);
/// Eight seconds of 1080p video is tens of MiB; refuse runaway bodies.
const VEO_VIDEO_MAX_BYTES: usize = 256 * 1024 * 1024;
const VEO_DEFAULT_RESOLUTION: &str = "1080p";
const VEO_DEFAULT_DURATION_SECONDS: u32 = 8;
const VEO_DEFAULT_ASPECT_RATIO: &str = "16:9";
const VEO_POLL_INTERVAL_SECS: u64 = 20;
const VEO_MAX_POLL_ATTEMPTS: usize = 30;

fn redact_gemini_api_key(text: &str) -> String {
    let key = CONFIG.gemini_api_key.trim();
    if key.is_empty() {
        return text.to_string();
    }
    text.replace(key, "[redacted]")
}

/// Whether an error from one Gemini model justifies retrying the same
/// payload on a fallback model. Only capacity/availability failures do;
/// a 400/401/403 or a decode failure will fail identically elsewhere.
fn gemini_error_allows_model_fallback(err: &anyhow::Error) -> bool {
    match err.downcast_ref::<ProviderError>() {
        Some(ProviderError::Http { status, .. }) => {
            is_retryable_status(*status) || *status == StatusCode::NOT_FOUND
        }
        Some(other) => other.is_retryable(),
        None => false,
    }
}

/// Total time the fallback chain may spend after the primary model failed.
fn gemini_fallback_budget() -> Duration {
    gemini_generate_content_timeout()
}

fn gemini_generate_content_timeout() -> Duration {
    Duration::from_secs(CONFIG.gemini_request_timeout_secs)
}

fn gemini_generate_content_url(model: &str) -> String {
    format!("{GEMINI_API_BASE}/v1beta/models/{model}:generateContent")
}

fn ensure_gemini_api_available() -> Result<()> {
    if CONFIG.gemini_api_available() {
        Ok(())
    } else {
        Err(anyhow!(
            "Gemini is disabled or GEMINI_API_KEY is not configured"
        ))
    }
}

fn gemini_image_generation_timeout() -> Duration {
    Duration::from_secs(CONFIG.gemini_image_request_timeout_secs)
}

fn build_safety_settings() -> Vec<serde_json::Value> {
    let profile = CONFIG.gemini_safety_settings.as_str();
    let threshold = match profile {
        "standard" => "BLOCK_MEDIUM_AND_ABOVE",
        "permissive" => "OFF",
        _ => {
            warn!(
                "Unknown GEMINI_SAFETY_SETTINGS value '{}', using permissive defaults.",
                profile
            );
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

fn decode_file_info_from_value(value: serde_json::Value, context: &str) -> Result<GeminiFileInfo> {
    if let Some(file_value) = value.get("file").cloned() {
        serde_json::from_value::<GeminiFileInfo>(file_value).map_err(|err| {
            anyhow!(
                "{} failed to decode file metadata wrapper: {}",
                context,
                err
            )
        })
    } else {
        serde_json::from_value::<GeminiFileInfo>(value)
            .map_err(|err| anyhow!("{} failed to decode file metadata: {}", context, err))
    }
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
                let mime_type = file_data
                    .get("mimeType")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string());
                json!({ "fileData": { "fileUri": file_uri, "mimeType": mime_type } })
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

    if let Some(safety) = payload
        .get("safetySettings")
        .and_then(|value| value.as_array())
    {
        summary.insert("safetySettingsCount".to_string(), json!(safety.len()));
    }

    Value::Object(summary)
}

fn summarize_gemini_response(response: &GeminiResponse) -> Value {
    let mut text_parts = 0usize;
    let mut image_parts = 0usize;
    let mut other_parts: Vec<String> = Vec::new();
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
                        GeminiPart::ExecutableCode { executable_code } => {
                            if text_preview.is_none() {
                                if let Some(code) = executable_code.code.as_deref() {
                                    if !code.trim().is_empty() {
                                        text_preview = Some(truncate_for_log(code, 200));
                                    }
                                }
                            }
                        }
                        GeminiPart::CodeExecutionResult {
                            code_execution_result,
                        } => {
                            if text_preview.is_none() {
                                if let Some(output) = code_execution_result.output.as_deref() {
                                    if !output.trim().is_empty() {
                                        text_preview = Some(truncate_for_log(output, 200));
                                    }
                                }
                            }
                        }
                        GeminiPart::Other(part) => {
                            // Record which unfamiliar keys the API sent (e.g.
                            // "functionCall", "thoughtSignature") without the payload.
                            let keys = part
                                .as_object()
                                .map(|object| object.keys().cloned().collect::<Vec<_>>())
                                .unwrap_or_default();
                            other_parts.push(keys.join("+"));
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
        "otherParts": other_parts,
        "textPreview": text_preview,
        "hasUsageMetadata": response.usage_metadata.is_some()
    })
}

fn kind_label(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Image => "image",
        MediaKind::Video => "video",
        MediaKind::Audio => "audio",
        MediaKind::Document => "document",
    }
}

fn normalize_gemini_mime_type(mime_type: &str) -> String {
    let lowered = mime_type.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "image/jpg" => "image/jpeg".to_string(),
        "audio/mpeg" => "audio/mp3".to_string(),
        "audio/x-wav" => "audio/wav".to_string(),
        "video/quicktime" => "video/mov".to_string(),
        "video/x-msvideo" => "video/avi".to_string(),
        "video/x-ms-wmv" => "video/wmv".to_string(),
        _ => lowered,
    }
}

fn gemini_supports_mime(kind: MediaKind, mime_type: &str) -> bool {
    match kind {
        MediaKind::Image => matches!(
            mime_type,
            "image/png" | "image/jpeg" | "image/webp" | "image/heic" | "image/heif"
        ),
        MediaKind::Video => matches!(
            mime_type,
            "video/mp4"
                | "video/mpeg"
                | "video/mov"
                | "video/avi"
                | "video/x-flv"
                | "video/mpg"
                | "video/webm"
                | "video/wmv"
                | "video/3gpp"
        ),
        MediaKind::Audio => matches!(
            mime_type,
            "audio/wav" | "audio/mp3" | "audio/aiff" | "audio/aac" | "audio/ogg" | "audio/flac"
        ),
        MediaKind::Document => mime_type == "application/pdf",
    }
}

fn gemini_mime_for_file(file: &MediaFile) -> Option<String> {
    let mut candidates = Vec::new();
    if !file.mime_type.trim().is_empty() {
        candidates.push(file.mime_type.clone());
    }
    if let Some(detected) = detect_mime_type(file.bytes()) {
        candidates.push(detected);
    }

    for candidate in candidates {
        let normalized = normalize_gemini_mime_type(&candidate);
        if gemini_supports_mime(file.kind, &normalized) {
            return Some(normalized);
        }
    }

    None
}

async fn upload_file_bytes(
    display_name: &str,
    mime_type: &str,
    bytes: &[u8],
) -> Result<GeminiFileInfo> {
    ensure_gemini_api_available()?;
    upload_file_bytes_at(
        GEMINI_API_BASE,
        &CONFIG.gemini_api_key,
        display_name,
        mime_type,
        bytes,
    )
    .await
}

/// Resumable upload in two steps: `start` yields the upload URL, `finalize`
/// sends the bytes. Both run on the shared transport with explicit timeouts.
async fn upload_file_bytes_at(
    base: &str,
    api_key: &str,
    display_name: &str,
    mime_type: &str,
    bytes: &[u8],
) -> Result<GeminiFileInfo> {
    let start_url = format!("{base}/upload/v1beta/files");
    let start_url = start_url.as_str();
    let content_length = bytes.len().to_string();
    let content_length = content_length.as_str();
    let start_call =
        LlmCall::untracked("gemini", "file-upload-start").with_redaction(redact_gemini_api_key);
    let upload_url: String = call_with_retry(
        &start_call,
        &GEMINI_FILE_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .post(start_url)
                .header("x-goog-api-key", api_key)
                .header("X-Goog-Upload-Protocol", "resumable")
                .header("X-Goog-Upload-Command", "start")
                .header("X-Goog-Upload-Header-Content-Length", content_length)
                .header("X-Goog-Upload-Header-Content-Type", mime_type)
                .timeout(GEMINI_FILE_REQUEST_TIMEOUT)
                .json(&json!({ "file": { "display_name": display_name } })))
        },
        |_| {},
        |response| async move {
            let upload_url = response
                .headers()
                .get("x-goog-upload-url")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            // Drain the (normally empty) body so the connection can be reused.
            let _ = read_body_limited(response, "gemini", 64 * 1024).await;
            upload_url.ok_or_else(|| {
                ProviderError::decode(
                    "gemini",
                    "file upload start did not return an upload URL",
                    false,
                )
            })
        },
        |_| LlmUsageRecord::default(),
    )
    .await?;

    let upload_url = upload_url.as_str();
    let finalize_call =
        LlmCall::untracked("gemini", "file-upload-finalize").with_redaction(redact_gemini_api_key);
    let payload: GeminiFileResponse = call_with_retry(
        &finalize_call,
        &GEMINI_FILE_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .post(upload_url)
                .header("X-Goog-Upload-Command", "upload, finalize")
                .header("X-Goog-Upload-Offset", "0")
                .header("Content-Length", content_length)
                .timeout(GEMINI_UPLOAD_TIMEOUT)
                .body(bytes.to_vec()))
        },
        |_| {},
        |response| read_json::<GeminiFileResponse>(response, "gemini"),
        |_| LlmUsageRecord::default(),
    )
    .await?;
    Ok(payload.file)
}

/// Authenticated GET returning JSON (file metadata, Veo operation polls).
async fn gemini_get_json(
    url: &str,
    api_key: &str,
    label: &str,
    timeout: Duration,
) -> Result<Value> {
    let call = LlmCall::untracked("gemini", label).with_redaction(redact_gemini_api_key);
    let value = call_with_retry(
        &call,
        &GEMINI_FILE_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .get(url)
                .header("x-goog-api-key", api_key)
                .timeout(timeout))
        },
        |_| {},
        |response| read_json::<Value>(response, "gemini"),
        |_| LlmUsageRecord::default(),
    )
    .await?;
    Ok(value)
}

async fn get_file_metadata(name: &str) -> Result<GeminiFileInfo> {
    get_file_metadata_at(
        GEMINI_API_BASE,
        &CONFIG.gemini_api_key,
        name,
        GEMINI_FILE_REQUEST_TIMEOUT,
    )
    .await
}

async fn get_file_metadata_at(
    base: &str,
    api_key: &str,
    name: &str,
    timeout: Duration,
) -> Result<GeminiFileInfo> {
    let name = name.trim();
    let name = name.strip_prefix("files/").unwrap_or(name);
    let url = format!("{base}/v1beta/files/{name}");
    let payload = gemini_get_json(&url, api_key, "file-metadata", timeout).await?;
    decode_file_info_from_value(payload, "Gemini file metadata")
}

async fn wait_for_file_active(file: GeminiFileInfo) -> Result<GeminiFileInfo> {
    let name = file.name.clone();
    let mut latest = file;

    for _ in 0..15 {
        match latest.state.as_deref().unwrap_or("PROCESSING") {
            "ACTIVE" => return Ok(latest),
            "FAILED" => return Err(anyhow!("Gemini file processing failed for {}", latest.uri)),
            _ => {}
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
        latest = get_file_metadata(&name).await?;
    }

    Err(anyhow!(
        "Timed out waiting for Gemini file processing for {}",
        name
    ))
}

async fn upload_media_files(files: &[MediaFile]) -> Result<Vec<UploadedFileRef>> {
    let semaphore = Arc::new(Semaphore::new(CONFIG.gemini_upload_fanout));
    let mut join_set = JoinSet::new();

    for (index, file) in files.iter().cloned().enumerate() {
        let semaphore = semaphore.clone();
        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .expect("gemini upload semaphore should remain open");
            let display_name = file
                .display_name
                .clone()
                .unwrap_or_else(|| format!("{}-{}", kind_label(file.kind), index + 1));
            if file.bytes().is_empty() {
                warn!("Skipping empty media file {}", display_name);
                return Ok::<_, anyhow::Error>((index, None));
            }
            let Some(mime_type) = gemini_mime_for_file(&file) else {
                warn!(
                    "Skipping unsupported Gemini media {} (kind={}, mime={})",
                    display_name,
                    kind_label(file.kind),
                    file.mime_type
                );
                return Ok((index, None));
            };
            let info = upload_file_bytes(&display_name, &mime_type, file.bytes()).await?;
            let info = wait_for_file_active(info).await?;
            if let Some(uploaded_mime_type) = info
                .mime_type
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                debug!(
                    display_name = %display_name,
                    mime_type = %uploaded_mime_type,
                    "Gemini file upload became active"
                );
            }
            let uri = if !info.uri.trim().is_empty() {
                info.uri
            } else if !info.name.trim().is_empty() {
                format!(
                    "https://generativelanguage.googleapis.com/files/{}",
                    info.name.trim_start_matches("files/")
                )
            } else {
                warn!(
                    "Gemini file upload response missing uri/name for {}",
                    display_name
                );
                return Ok((index, None));
            };
            Ok((index, Some(UploadedFileRef { uri })))
        });
    }

    let mut uploaded = Vec::new();
    while let Some(result) = join_set.join_next().await {
        let (index, file_ref) = result??;
        if let Some(file_ref) = file_ref {
            uploaded.push((index, file_ref));
        }
    }
    uploaded.sort_by_key(|(index, _)| *index);
    Ok(uploaded.into_iter().map(|(_, file_ref)| file_ref).collect())
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

fn build_gemini_file_parts(
    user_content: &str,
    uploaded_files: &[UploadedFileRef],
    youtube_urls: &[String],
    text_after_media: bool,
) -> Vec<serde_json::Value> {
    let mut parts = Vec::new();
    let text_part = json!({ "text": user_content });

    if !text_after_media {
        parts.push(text_part.clone());
    }

    for file in uploaded_files {
        parts.push(json!({
            "fileData": {
                "fileUri": file.uri
            }
        }));
    }

    for url in youtube_urls {
        parts.push(json!({
            "fileData": {
                "fileUri": url
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
    let mut fallback_parts = Vec::new();
    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    match part {
                        GeminiPart::Text { text } if !text.trim().is_empty() => {
                            text_parts.push(text);
                        }
                        GeminiPart::ExecutableCode { executable_code } => {
                            if let Some(language) = executable_code
                                .language
                                .as_deref()
                                .map(str::trim)
                                .filter(|value| !value.is_empty())
                            {
                                debug!(%language, "Gemini response included executable code");
                            }
                            if let Some(code) = executable_code.code.as_deref() {
                                if !code.trim().is_empty() {
                                    fallback_parts.push(code.to_string());
                                }
                            }
                        }
                        GeminiPart::CodeExecutionResult {
                            code_execution_result,
                        } => {
                            if let Some(outcome) = code_execution_result
                                .outcome
                                .as_deref()
                                .map(str::trim)
                                .filter(|value| !value.is_empty())
                            {
                                debug!(%outcome, "Gemini response included code execution result");
                            }
                            if let Some(output) = code_execution_result.output.as_deref() {
                                if !output.trim().is_empty() {
                                    fallback_parts.push(output.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    if text_parts.is_empty() {
        fallback_parts.join("\n")
    } else {
        text_parts.join("\n")
    }
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
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<GeminiResponse> {
    let value = call_gemini_api_value(
        model,
        payload,
        system_prompt_label,
        audit_context,
        operation,
    )
    .await?;
    let parsed = serde_json::from_value::<GeminiResponse>(value)
        .map_err(|err| anyhow!("Gemini generateContent response decode failed: {}", err))?;
    Ok(parsed)
}

async fn call_gemini_api_with_timeout(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
    timeout: Duration,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<GeminiResponse> {
    let value = call_gemini_api_value_with_timeout(
        model,
        payload,
        system_prompt_label,
        timeout,
        audit_context,
        operation,
    )
    .await?;
    let parsed = serde_json::from_value::<GeminiResponse>(value)
        .map_err(|err| anyhow!("Gemini generateContent response decode failed: {}", err))?;
    Ok(parsed)
}

async fn call_gemini_api_value(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<Value> {
    call_gemini_api_value_with_timeout(
        model,
        payload,
        system_prompt_label,
        gemini_generate_content_timeout(),
        audit_context,
        operation,
    )
    .await
}

async fn call_gemini_api_value_with_timeout(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
    timeout: Duration,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<Value> {
    ensure_gemini_api_available()?;
    let url = gemini_generate_content_url(model);
    let url = url.as_str();
    let metadata = json!({
        "system_prompt_label": system_prompt_label.unwrap_or(""),
        "timeout_secs": timeout.as_secs()
    });
    let call = LlmCall::begin("gemini", model, operation, audit_context, Some(&metadata))
        .with_redaction(redact_gemini_api_key);

    if tracing::enabled!(tracing::Level::DEBUG) {
        let payload_summary = summarize_gemini_payload(&payload, system_prompt_label);
        debug!(target: "llm.gemini", model = model, payload = %payload_summary);
    }

    let payload = &payload;
    let value = call_with_retry(
        &call,
        &GEMINI_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .post(url)
                .header("x-goog-api-key", CONFIG.gemini_api_key.as_str())
                .timeout(timeout)
                .json(payload))
        },
        |_| {},
        |response| read_json::<Value>(response, "gemini"),
        usage::from_gemini,
    )
    .await?;

    if tracing::enabled!(tracing::Level::DEBUG) {
        let parsed = serde_json::from_value::<GeminiResponse>(value.clone()).ok();
        let response_summary = parsed
            .as_ref()
            .map(summarize_gemini_response)
            .unwrap_or_else(|| {
                json!({
                    "rawResponsePreview": truncate_for_log(&value.to_string(), 400)
                })
            });
        debug!(target: "llm.gemini", model = model, response = %response_summary);
    }

    Ok(value)
}

fn text_part_looks_like_music_metadata(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.starts_with("Caption:") {
        return true;
    }

    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.contains('{'))
    {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    lower.contains("\nbpm:")
        || lower.starts_with("bpm:")
        || lower.contains("\nmusic:")
        || lower.starts_with("music:")
        || lower.contains("\ncaption:")
}

fn extract_music_generation_result(
    response: GeminiResponse,
    model: &str,
) -> Result<GeminiMusicGenerationResult> {
    let mut lyric_parts = Vec::new();
    let mut note_parts = Vec::new();
    let mut audio_bytes = None;
    let mut audio_mime_type = None;

    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    match part {
                        GeminiPart::Text { text } => {
                            let trimmed = text.trim();
                            if trimmed.is_empty() {
                                continue;
                            }

                            if text_part_looks_like_music_metadata(trimmed) {
                                note_parts.push(trimmed.to_string());
                            } else {
                                lyric_parts.push(trimmed.to_string());
                            }
                        }
                        GeminiPart::InlineData { inline_data }
                            if inline_data.mime_type.starts_with("audio/") =>
                        {
                            let bytes = general_purpose::STANDARD
                                .decode(&inline_data.data)
                                .map_err(|err| {
                                    anyhow!("Failed to decode Lyria audio payload: {}", err)
                                })?;
                            audio_mime_type = Some(inline_data.mime_type);
                            audio_bytes = Some(bytes);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    if lyric_parts.is_empty() && !note_parts.is_empty() {
        lyric_parts = note_parts.clone();
        note_parts.clear();
    }

    let audio_bytes =
        audio_bytes.ok_or_else(|| anyhow!("No audio returned by Lyria (model: {})", model))?;

    Ok(GeminiMusicGenerationResult {
        lyrics_text: lyric_parts.join("\n\n"),
        notes_text: if note_parts.is_empty() {
            None
        } else {
            Some(note_parts.join("\n\n"))
        },
        audio_bytes,
        audio_mime_type: audio_mime_type.unwrap_or_else(|| "audio/mpeg".to_string()),
        model_used: model.to_string(),
    })
}

fn extract_text_from_response_value(response: &Value) -> String {
    let mut text_parts = Vec::new();
    let mut fallback_parts = Vec::new();
    let candidates = response
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    for candidate in candidates {
        let parts = candidate
            .get("content")
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.trim().is_empty() {
                    text_parts.push(text.to_string());
                }
            } else if let Some(code) = part
                .get("executableCode")
                .and_then(|value| value.get("code"))
                .and_then(Value::as_str)
            {
                if !code.trim().is_empty() {
                    fallback_parts.push(code.to_string());
                }
            } else if let Some(output) = part
                .get("codeExecutionResult")
                .and_then(|value| value.get("output"))
                .and_then(Value::as_str)
            {
                if !output.trim().is_empty() {
                    fallback_parts.push(output.to_string());
                }
            }
        }
    }

    if text_parts.is_empty() {
        fallback_parts.join("\n")
    } else {
        text_parts.join("\n")
    }
}

fn extract_candidate_content(response: &Value) -> Option<Value> {
    response
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| candidate.get("content"))
        .cloned()
}

fn extract_function_calls(content: &Value) -> Vec<Value> {
    content
        .get("parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("functionCall").cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn base_generation_config() -> Value {
    json!({
        "temperature": CONFIG.gemini_temperature,
        "topK": CONFIG.gemini_top_k,
        "topP": CONFIG.gemini_top_p,
        "maxOutputTokens": CONFIG.gemini_max_output_tokens,
    })
}

/// `thinkingConfig` for `GEMINI_THINKING_LEVEL`. Only Gemini 3 models accept
/// `thinkingLevel`; the 2.5 generation uses token budgets and rejects it, so
/// the setting is ignored there.
fn thinking_config_for(model: &str, level: &str) -> Option<Value> {
    let level = level.trim();
    if level.is_empty() || !model.trim().to_ascii_lowercase().starts_with("gemini-3") {
        return None;
    }
    Some(json!({ "thinkingLevel": level }))
}

/// Sampling config for text calls, with the thinking level when `model`
/// supports it.
fn generation_config_for(model: &str, thinking_level: &str) -> Value {
    let mut config = base_generation_config();
    if let Some(thinking) = thinking_config_for(model, thinking_level) {
        if let Some(object) = config.as_object_mut() {
            object.insert("thinkingConfig".to_string(), thinking);
        }
    }
    config
}

/// The same request retargeted at `model`, with `thinkingConfig` re-derived
/// for that model: a fallback chain may cross generations, and Gemini 2.5
/// rejects the `thinkingLevel` that Gemini 3 accepts.
fn payload_for_model(payload: &Value, model: &str, thinking_level: &str) -> Value {
    let mut payload = payload.clone();
    if let Some(config) = payload
        .get_mut("generationConfig")
        .and_then(Value::as_object_mut)
    {
        config.remove("thinkingConfig");
        if let Some(thinking) = thinking_config_for(model, thinking_level) {
            config.insert("thinkingConfig".to_string(), thinking);
        }
    }
    payload
}

fn with_response_json_schema(config: Value, response_json_schema: Option<&Value>) -> Value {
    let Some(schema) = response_json_schema else {
        return config;
    };

    let mut config_object = config.as_object().cloned().unwrap_or_default();
    config_object.insert(
        "responseMimeType".to_string(),
        Value::String("application/json".to_string()),
    );
    config_object.insert("responseJsonSchema".to_string(), schema.clone());
    Value::Object(config_object)
}

/// Gemini `functionResponse` part for one tool result. The fenced result is
/// text, and Gemini wants an object, so it travels under `result`.
fn build_function_response_part(call: &ToolCall, result: &str) -> Value {
    let mut function_response = json!({
        "name": call.name,
        "response": { "result": result },
    });
    if !call.id.is_empty() {
        function_response["id"] = Value::String(call.id.clone());
    }
    json!({ "functionResponse": function_response })
}

/// A Gemini `functionCall` part as a [`ToolCall`].
fn gemini_tool_call(function_call: &Value) -> ToolCall {
    ToolCall {
        id: function_call
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        name: function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        arguments: function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({})),
    }
}

/// Gemini half of the shared tool loop.
struct GeminiProtocol<'a> {
    model: &'a str,
    system_prompt: &'a str,
    system_prompt_label: Option<&'a str>,
    /// Applied on the final pass only; its presence forces that pass.
    final_response_json_schema: Option<Value>,
    audit_context: Option<&'a LlmAuditContext>,
    final_pass: bool,
}

impl ToolProtocol for GeminiProtocol<'_> {
    type Item = Value;

    fn tool_declarations(&self, runtime: &ToolRuntime) -> Vec<Value> {
        runtime.build_gemini_tools()
    }

    fn complete<'a>(
        &'a mut self,
        transcript: &'a [Value],
        tools: Option<&'a [Value]>,
        request_timeout: Duration,
    ) -> BoxFuture<'a, Result<ModelTurn<Value>>> {
        Box::pin(async move {
            let mut generation_config =
                generation_config_for(self.model, &CONFIG.gemini_thinking_level);
            if self.final_pass {
                generation_config = with_response_json_schema(
                    generation_config,
                    self.final_response_json_schema.as_ref(),
                );
            }
            let mut payload = json!({
                "systemInstruction": { "parts": [{ "text": self.system_prompt }] },
                "contents": transcript,
                "generationConfig": generation_config,
                "safetySettings": build_safety_settings(),
            });
            if let Some(tools) = tools {
                payload["tools"] = Value::Array(tools.to_vec());
            }

            let response = call_gemini_api_value_with_timeout(
                self.model,
                payload,
                self.system_prompt_label,
                request_timeout,
                self.audit_context,
                "call_gemini_with_tool_runtime",
            )
            .await?;
            let content = extract_candidate_content(&response)
                .ok_or_else(|| anyhow!("Gemini tool response did not include candidate content"))?;
            let tool_calls = extract_function_calls(&content)
                .iter()
                .map(gemini_tool_call)
                .collect();
            Ok(ModelTurn {
                text: extract_text_from_response_value(&response),
                tool_calls,
                transcript: vec![content],
            })
        })
    }

    fn tool_results(&self, results: Vec<(ToolCall, String)>) -> Vec<Value> {
        // Gemini takes every functionResponse of a turn in one user content.
        let parts = results
            .iter()
            .map(|(call, output)| build_function_response_part(call, output))
            .collect::<Vec<_>>();
        vec![json!({ "role": "user", "parts": parts })]
    }

    fn requires_final_pass(&self) -> bool {
        self.final_response_json_schema.is_some()
    }

    fn begin_final_pass(&mut self) {
        self.final_pass = true;
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn call_gemini_with_tool_runtime(
    system_prompt: &str,
    user_content: &str,
    runtime: &mut ToolRuntime,
    use_pro_model: bool,
    media_files: Option<Vec<MediaFile>>,
    youtube_urls: Option<Vec<String>>,
    system_prompt_label: Option<&str>,
    final_response_json_schema: Option<Value>,
    audit_context: Option<&LlmAuditContext>,
) -> Result<GeminiCallResult> {
    ensure_gemini_api_available()?;
    let youtube_urls = youtube_urls.unwrap_or_default();
    let files = media_files.unwrap_or_default();
    let uploaded_files = if files.is_empty() {
        Vec::new()
    } else {
        upload_media_files(&files).await?
    };
    let text_after_media = !uploaded_files.is_empty() || !youtube_urls.is_empty();
    let parts = build_gemini_file_parts(
        user_content,
        &uploaded_files,
        &youtube_urls,
        text_after_media,
    );
    let contents = vec![json!({ "role": "user", "parts": parts })];

    let model = if use_pro_model {
        CONFIG.gemini_pro_model.as_str()
    } else {
        CONFIG.gemini_model.as_str()
    };
    let deadline = TurnDeadline::for_runtime(gemini_generate_content_timeout(), runtime);
    let mut protocol = GeminiProtocol {
        model,
        system_prompt,
        system_prompt_label,
        final_response_json_schema,
        audit_context,
        final_pass: false,
    };
    let text = run_tool_loop(&mut protocol, runtime, contents, &deadline).await?;
    Ok(GeminiCallResult {
        text,
        model_used: model.to_string(),
    })
}

/// Single Gemini call against a specific model, with optional media and an
/// optional JSON response schema, and no tools or pro/lite fallback chain.
/// This is the cheap-step primitive used by the agentic pipelines (typically
/// with `CONFIG.gemini_lite_model`).
#[allow(clippy::too_many_arguments)]
pub async fn call_gemini_model_simple(
    model: &str,
    system_prompt: &str,
    user_content: &str,
    media_files: Option<Vec<MediaFile>>,
    response_json_schema: Option<&Value>,
    system_prompt_label: Option<&str>,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<GeminiCallResult> {
    ensure_gemini_api_available()?;
    let files = media_files.unwrap_or_default();
    let uploaded_files = if files.is_empty() {
        Vec::new()
    } else {
        upload_media_files(&files).await?
    };
    let text_after_media = !uploaded_files.is_empty();
    let parts = build_gemini_file_parts(user_content, &uploaded_files, &[], text_after_media);

    let payload = json!({
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "contents": [json!({ "role": "user", "parts": parts })],
        "generationConfig": with_response_json_schema(
            generation_config_for(model, &CONFIG.gemini_thinking_level),
            response_json_schema
        ),
        "safetySettings": build_safety_settings(),
    });

    let response = call_gemini_api_value(
        model,
        payload,
        system_prompt_label,
        audit_context,
        operation,
    )
    .await?;
    Ok(GeminiCallResult {
        text: extract_text_from_response_value(&response),
        model_used: model.to_string(),
    })
}

async fn call_gemini_lite_fallback(
    payload: &serde_json::Value,
    system_prompt_label: Option<&str>,
    previous_model: &str,
    previous_err: &anyhow::Error,
    audit_context: Option<&LlmAuditContext>,
) -> Result<GeminiCallResult> {
    let lite_model = CONFIG.gemini_lite_model.trim();
    if lite_model.is_empty() {
        return Err(anyhow!(
            "Gemini request failed on model '{}' and GEMINI_LITE_MODEL is not configured. Previous error: {}",
            previous_model,
            previous_err
        ));
    }

    if lite_model.eq_ignore_ascii_case(previous_model) {
        return Err(anyhow!(
            "Gemini request failed on model '{}' and GEMINI_LITE_MODEL points to the same model. Previous error: {}",
            previous_model,
            previous_err
        ));
    }

    warn!(
        "Gemini model '{}' failed after retries; trying lite fallback model '{}' for up to {} attempts: {}",
        previous_model,
        lite_model,
        GEMINI_LITE_FALLBACK_MAX_ATTEMPTS,
        previous_err
    );

    let mut last_lite_err = None;
    for attempt in 1..=GEMINI_LITE_FALLBACK_MAX_ATTEMPTS {
        let result = async {
            let response = call_gemini_api(
                lite_model,
                payload_for_model(payload, lite_model, &CONFIG.gemini_thinking_level),
                system_prompt_label,
                audit_context,
                "call_gemini_lite_fallback",
            )
            .await?;
            Ok::<_, anyhow::Error>(extract_text_from_response(response))
        }
        .await;

        match result {
            Ok(text) => {
                return Ok(GeminiCallResult {
                    text,
                    model_used: lite_model.to_string(),
                });
            }
            Err(err) => {
                warn!(
                    "Gemini lite fallback attempt {}/{} failed on model '{}': {}",
                    attempt, GEMINI_LITE_FALLBACK_MAX_ATTEMPTS, lite_model, err
                );
                let worth_retrying = gemini_error_allows_model_fallback(&err);
                last_lite_err = Some(err);
                if !worth_retrying {
                    break;
                }
            }
        }
    }

    let lite_err = last_lite_err.unwrap_or_else(|| anyhow!("Unknown Gemini lite fallback failure"));
    Err(anyhow!(
        "Gemini request failed on model '{}' and lite fallback model '{}' after {} attempts. Previous error: {}. Lite fallback error: {}",
        previous_model,
        lite_model,
        GEMINI_LITE_FALLBACK_MAX_ATTEMPTS,
        previous_err,
        lite_err
    ))
}

/// One Gemini text call (default or pro model, with the lite fallback chain).
#[derive(Default)]
pub struct GeminiCallRequest<'a> {
    pub system_prompt: &'a str,
    pub user_content: &'a str,
    /// Attach the `google_search` grounding tool.
    pub use_search_grounding: bool,
    pub use_pro_model: bool,
    pub media_files: Vec<MediaFile>,
    pub youtube_urls: Vec<String>,
    /// Name used in place of the system prompt text in logs.
    pub system_prompt_label: Option<&'a str>,
    pub audit_context: Option<&'a LlmAuditContext>,
}

pub async fn call_gemini(request: GeminiCallRequest<'_>) -> Result<GeminiCallResult> {
    ensure_gemini_api_available()?;
    let GeminiCallRequest {
        system_prompt,
        user_content,
        use_search_grounding,
        use_pro_model,
        media_files: files,
        youtube_urls,
        system_prompt_label,
        audit_context,
    } = request;
    let content = user_content.to_string();

    let has_video_or_audio = files
        .iter()
        .any(|file| matches!(file.kind, MediaKind::Video | MediaKind::Audio));

    let uploaded_files = if files.is_empty() {
        Vec::new()
    } else {
        upload_media_files(&files).await?
    };

    let text_after_media = !uploaded_files.is_empty() || !youtube_urls.is_empty();
    let parts = build_gemini_file_parts(&content, &uploaded_files, &youtube_urls, text_after_media);
    let tools = {
        let mut tools = Vec::new();
        if !has_video_or_audio && youtube_urls.is_empty() {
            tools.push(json!({ "code_execution": {} }));
        }
        if use_search_grounding {
            tools.push(json!({ "google_search": {} }));
        }
        tools
    };

    let primary_model = if use_pro_model {
        &CONFIG.gemini_pro_model
    } else {
        &CONFIG.gemini_model
    };
    // Fallback models reuse this payload with thinkingConfig re-derived per
    // model (payload_for_model): the chain may cross generations.
    let payload = json!({
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": generation_config_for(primary_model, &CONFIG.gemini_thinking_level),
        "safetySettings": build_safety_settings(),
        "tools": tools,
    });
    let primary_operation = if use_pro_model {
        "call_gemini_pro"
    } else {
        "call_gemini"
    };

    let primary_attempt = async {
        let response = call_gemini_api(
            primary_model,
            payload.clone(),
            system_prompt_label,
            audit_context,
            primary_operation,
        )
        .await?;
        Ok::<_, anyhow::Error>(extract_text_from_response(response))
    }
    .await;

    match primary_attempt {
        Ok(text) => Ok(GeminiCallResult {
            text,
            model_used: primary_model.to_string(),
        }),
        Err(primary_err) => {
            // A 400/401/403 or a decode failure fails identically on every
            // model; only capacity/availability errors justify the extra
            // round-trips.
            if !gemini_error_allows_model_fallback(&primary_err) {
                return Err(primary_err);
            }

            let budget = gemini_fallback_budget();
            let primary_err_text = primary_err.to_string();
            let fallbacks = run_gemini_model_fallbacks(
                &payload,
                system_prompt_label,
                primary_model,
                use_pro_model,
                primary_err,
                audit_context,
            );
            match tokio::time::timeout(budget, fallbacks).await {
                Ok(result) => result,
                Err(_) => Err(anyhow!(
                    "Gemini fallback chain exceeded its {}s budget after primary model '{}' failed: {}",
                    budget.as_secs(),
                    primary_model,
                    primary_err_text
                )),
            }
        }
    }
}

/// Try the remaining models after the primary one failed with a fallback-worthy
/// error: pro -> default -> lite, or default -> lite.
async fn run_gemini_model_fallbacks(
    payload: &serde_json::Value,
    system_prompt_label: Option<&str>,
    primary_model: &str,
    use_pro_model: bool,
    primary_err: anyhow::Error,
    audit_context: Option<&LlmAuditContext>,
) -> Result<GeminiCallResult> {
    if !use_pro_model {
        return call_gemini_lite_fallback(
            payload,
            system_prompt_label,
            primary_model,
            &primary_err,
            audit_context,
        )
        .await;
    }

    let fallback_model = CONFIG.gemini_model.as_str();
    warn!(
        "Gemini Pro model '{}' failed after retries; falling back to default model '{}': {}",
        primary_model, fallback_model, primary_err
    );

    let fallback_text = async {
        let response = call_gemini_api(
            fallback_model,
            payload_for_model(payload, fallback_model, &CONFIG.gemini_thinking_level),
            system_prompt_label,
            audit_context,
            "call_gemini_fallback",
        )
        .await?;
        Ok::<_, anyhow::Error>(extract_text_from_response(response))
    }
    .await;

    match fallback_text {
        Ok(text) => Ok(GeminiCallResult {
            text,
            model_used: fallback_model.to_string(),
        }),
        Err(fallback_err) => {
            if !gemini_error_allows_model_fallback(&fallback_err) {
                return Err(anyhow!(
                    "Gemini request failed on primary model '{}' and fallback model '{}'. \
Primary error: {}. Fallback error: {}",
                    primary_model,
                    fallback_model,
                    primary_err,
                    fallback_err
                ));
            }
            call_gemini_lite_fallback(
                payload,
                system_prompt_label,
                fallback_model,
                &fallback_err,
                audit_context,
            )
            .await
            .map_err(|lite_err| {
                anyhow!(
                    "Gemini request failed on primary model '{}' and fallback model '{}'. \
Primary error: {}. Fallback error: {}. Lite fallback error: {}",
                    primary_model,
                    fallback_model,
                    primary_err,
                    fallback_err,
                    lite_err
                )
            })
        }
    }
}

pub async fn generate_image_with_gemini(
    prompt: &str,
    image_urls: &[String],
    image_config: Option<GeminiImageConfig>,
    upload_to_cwd: bool,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    if let Err(err) = ensure_gemini_api_available() {
        return Err(ImageGenerationError(err.to_string()));
    }
    let mut images = Vec::new();
    for url in image_urls {
        if let Some(data) = download_media(url).await {
            images.push(data);
        }
    }

    let base_instruction = if images.is_empty() {
        "Generate an image from the prompt. Respond with an image, not text."
    } else {
        "Edit the provided images according to the prompt. Respond with an image, not text."
    };

    let system_instruction = base_instruction.to_string();
    let parts = build_gemini_parts(prompt, &images, None, None, &[], false);
    let mut generation_config = json!({
        "responseModalities": ["IMAGE"]
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
        "tools": [{ "google_search": {"searchTypes": {"webSearch": {}, "imageSearch": {}}} }],
    });

    let model = &CONFIG.gemini_image_model;
    let response = call_gemini_api_with_timeout(
        model,
        payload,
        Some("image_generation_system_prompt"),
        gemini_image_generation_timeout(),
        audit_context,
        "generate_image_with_gemini",
    )
    .await
    .map_err(|err| ImageGenerationError(err.to_string()))?;

    let images = extract_images_from_response(response);
    if images.is_empty() {
        return Err(ImageGenerationError(format!(
            "No images returned by Gemini (model: {})",
            model
        )));
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

pub async fn generate_music_with_lyria(
    prompt: &str,
    audit_context: Option<&LlmAuditContext>,
) -> Result<GeminiMusicGenerationResult, anyhow::Error> {
    ensure_gemini_api_available()?;
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(anyhow!("Music prompt is empty"));
    }

    let model = CONFIG.gemini_music_model.trim();
    if model.is_empty() {
        return Err(anyhow!("GEMINI_MUSIC_MODEL is not configured"));
    }

    let payload = json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO", "TEXT"]
        },
        "safetySettings": build_safety_settings(),
    });

    let response = call_gemini_api_with_timeout(
        model,
        payload,
        Some("lyria_music_generation"),
        Duration::from_secs(LYRIA_GENERATION_TIMEOUT_SECS),
        audit_context,
        "lyria_generate_content",
    )
    .await?;

    extract_music_generation_result(response, model)
}

pub async fn generate_video_with_veo(
    user_prompt: &str,
    audit_context: Option<&LlmAuditContext>,
) -> Result<(Option<Vec<u8>>, Option<String>), anyhow::Error> {
    ensure_gemini_api_available()?;
    let prompt = user_prompt.trim();
    if prompt.is_empty() {
        return Ok((None, None));
    }

    let model = CONFIG.gemini_video_model.trim();
    if model.is_empty() {
        return Err(anyhow!("GEMINI_VIDEO_MODEL is not configured"));
    }

    let mut instance = Map::new();
    instance.insert("prompt".to_string(), json!(prompt));

    let mut parameters = Map::new();
    parameters.insert("resolution".to_string(), json!(VEO_DEFAULT_RESOLUTION));
    parameters.insert(
        "durationSeconds".to_string(),
        json!(VEO_DEFAULT_DURATION_SECONDS),
    );
    parameters.insert("aspectRatio".to_string(), json!(VEO_DEFAULT_ASPECT_RATIO));

    let payload = json!({
        "instances": [Value::Object(instance)],
        "parameters": Value::Object(parameters),
    });

    let url = format!("{GEMINI_API_BASE}/v1beta/models/{model}:predictLongRunning");
    let metadata = json!({
        "resolution": VEO_DEFAULT_RESOLUTION,
        "duration_seconds": VEO_DEFAULT_DURATION_SECONDS,
    });
    let operation = veo_start_operation(&url, &payload, model, &metadata, audit_context).await?;

    let operation_name = operation
        .get("name")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("Veo operation response missing name"))?
        .to_string();
    let operation_url = format!("{GEMINI_API_BASE}/v1beta/{operation_name}");

    let mut current_operation = operation;
    for attempt in 0..VEO_MAX_POLL_ATTEMPTS {
        let done = current_operation
            .get("done")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        if done {
            if let Some(error) = current_operation.get("error") {
                let message = error
                    .get("message")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error");
                warn!("Veo operation failed: {}", message);
                return Ok((None, None));
            }

            let video = current_operation
                .pointer("/response/generateVideoResponse/generatedSamples/0/video");
            let video_uri = video
                .and_then(|value| value.get("uri"))
                .and_then(|value| value.as_str());
            let declared_mime_type = video
                .and_then(|value| value.get("mimeType"))
                .and_then(|value| value.as_str())
                .map(|value| value.to_string());

            let Some(video_uri) = video_uri else {
                warn!("Veo operation completed without a video uri");
                return Ok((None, None));
            };

            let (bytes, mime_type) = veo_download_video(video_uri, declared_mime_type).await?;
            info!(
                "Veo video download completed (bytes={}, mime={:?})",
                bytes.len(),
                mime_type
            );
            return Ok((Some(bytes), mime_type));
        }

        if attempt + 1 < VEO_MAX_POLL_ATTEMPTS {
            info!(
                "Polling Veo operation (attempt {}/{})",
                attempt + 1,
                VEO_MAX_POLL_ATTEMPTS
            );
            tokio::time::sleep(Duration::from_secs(VEO_POLL_INTERVAL_SECS)).await;
            current_operation = gemini_get_json(
                &operation_url,
                &CONFIG.gemini_api_key,
                "veo-operation-poll",
                VEO_REQUEST_TIMEOUT,
            )
            .await?;
        }
    }

    warn!("Veo operation timed out after polling");
    Ok((None, None))
}

/// Start the long-running Veo generation. The audit row records the
/// operation name as its response id.
async fn veo_start_operation(
    url: &str,
    payload: &Value,
    model: &str,
    metadata: &Value,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Value> {
    let call = LlmCall::begin(
        "gemini",
        model,
        "veo_predict_long_running",
        audit_context,
        Some(metadata),
    )
    .with_redaction(redact_gemini_api_key);
    let operation = call_with_retry(
        &call,
        &GEMINI_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .post(url)
                .header("x-goog-api-key", CONFIG.gemini_api_key.as_str())
                .timeout(VEO_REQUEST_TIMEOUT)
                .json(payload))
        },
        |_| {},
        |response| read_json::<Value>(response, "gemini"),
        |operation: &Value| LlmUsageRecord {
            response_id: operation
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..LlmUsageRecord::default()
        },
    )
    .await?;
    Ok(operation)
}

/// Fetch the finished video, preferring the operation's declared MIME type
/// over the download's `Content-Type`.
async fn veo_download_video(
    video_uri: &str,
    declared_mime_type: Option<String>,
) -> Result<(Vec<u8>, Option<String>)> {
    let call =
        LlmCall::untracked("gemini", "veo-video-download").with_redaction(redact_gemini_api_key);
    let (bytes, header_mime_type) = call_with_retry(
        &call,
        &GEMINI_FILE_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .get(video_uri)
                .header("x-goog-api-key", CONFIG.gemini_api_key.as_str())
                .timeout(VEO_VIDEO_DOWNLOAD_TIMEOUT))
        },
        |_| {},
        |response| async move {
            let mime_type = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let bytes = read_body_limited(response, "gemini", VEO_VIDEO_MAX_BYTES).await?;
            Ok((bytes, mime_type))
        },
        |_| LlmUsageRecord::default(),
    )
    .await?;
    Ok((bytes, declared_mime_type.or(header_mime_type)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tools::twitter_extractor::test_support::{
        response_with_headers, ExpectedRequest, TestServer,
    };

    fn http_error(status: u16) -> anyhow::Error {
        ProviderError::http(
            "gemini",
            StatusCode::from_u16(status).unwrap(),
            None,
            "detail".to_string(),
            None,
        )
        .into()
    }

    fn transport_error(retryable: bool) -> anyhow::Error {
        ProviderError::Transport {
            provider: "gemini".to_string(),
            message: "boom".to_string(),
            retryable,
        }
        .into()
    }

    #[tokio::test]
    async fn file_metadata_fetch_decodes_the_wrapped_file_object() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/v1beta/files/abc123",
            response_with_headers(
                200,
                &[],
                br#"{"file":{"name":"files/abc123","uri":"https://files/abc123","state":"ACTIVE"}}"#
                    .to_vec(),
            ),
        )
        .with_header("x-goog-api-key", "test-key")]);
        let base = server.base_url().to_string();

        let info = get_file_metadata_at(
            base.trim_end_matches('/'),
            "test-key",
            "files/abc123",
            Duration::from_secs(5),
        )
        .await
        .expect("metadata decodes");

        assert_eq!(info.name, "files/abc123");
        assert_eq!(info.state.as_deref(), Some("ACTIVE"));
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn file_metadata_fetch_applies_its_explicit_timeout_and_retries() {
        // Without the explicit timeout the first (400ms-delayed) response
        // would be awaited and the second expectation would go unmet.
        let body = br#"{"name":"files/slow","uri":"u","state":"ACTIVE"}"#.to_vec();
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "GET",
                "/v1beta/files/slow",
                response_with_headers(200, &[], body.clone()),
            )
            .delayed(Duration::from_millis(400)),
            ExpectedRequest::new(
                "GET",
                "/v1beta/files/slow",
                response_with_headers(200, &[], body),
            ),
        ]);
        let base = server.base_url().to_string();

        let info = get_file_metadata_at(
            base.trim_end_matches('/'),
            "test-key",
            "slow",
            Duration::from_millis(50),
        )
        .await
        .expect("the retry after the timed-out attempt succeeds");

        assert_eq!(info.name, "files/slow");
        server
            .join_allowing_client_disconnect()
            .expect("both requests were served; the first client gave up");
    }

    #[test]
    fn unknown_response_parts_do_not_break_text_extraction() {
        let response: GeminiResponse = serde_json::from_value(json!({
            "candidates": [{ "content": { "parts": [
                { "functionCall": { "name": "web_search", "args": {} } },
                { "text": "visible answer" }
            ] } }]
        }))
        .expect("unknown part kinds must deserialize");

        assert_eq!(extract_text_from_response(response), "visible answer");
    }

    #[test]
    fn fallback_payloads_carry_the_thinking_config_of_their_own_model() {
        let mut primary = json!({
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
            "generationConfig": generation_config_for("gemini-3-pro-preview", "high"),
        });
        primary["generationConfig"]["temperature"] = json!(0.25);
        assert_eq!(
            primary["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "high"
        );

        let for_flash_25 = payload_for_model(&primary, "gemini-2.5-flash", "high");
        assert!(
            for_flash_25["generationConfig"]
                .get("thinkingConfig")
                .is_none(),
            "2.5 models reject thinkingLevel: {for_flash_25}"
        );
        assert_eq!(
            for_flash_25["generationConfig"]["temperature"], 0.25,
            "other generation settings survive"
        );
        assert_eq!(for_flash_25["contents"], primary["contents"]);

        let back_to_3 = payload_for_model(&for_flash_25, "gemini-3-flash", "high");
        assert_eq!(
            back_to_3["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "high"
        );
    }

    #[test]
    fn thinking_level_is_only_sent_to_gemini_3_models() {
        assert_eq!(
            thinking_config_for("gemini-3-pro-preview", "high"),
            Some(json!({ "thinkingLevel": "high" }))
        );
        assert_eq!(
            thinking_config_for("gemini-3-flash", "low"),
            Some(json!({ "thinkingLevel": "low" }))
        );
        assert_eq!(thinking_config_for("gemini-2.5-flash", "high"), None);
        assert_eq!(thinking_config_for("gemini-3-pro", "  "), None);
    }

    #[test]
    fn generation_config_carries_the_thinking_level_for_gemini_3() {
        let config = generation_config_for("gemini-3-pro", "high");
        assert_eq!(config["thinkingConfig"]["thinkingLevel"], "high");
        assert_eq!(config["temperature"], json!(CONFIG.gemini_temperature));
        assert!(generation_config_for("gemini-2.5-flash", "high")
            .get("thinkingConfig")
            .is_none());
    }

    #[test]
    fn model_fallback_allowed_for_capacity_and_availability_failures() {
        assert!(gemini_error_allows_model_fallback(&http_error(429)));
        assert!(gemini_error_allows_model_fallback(&http_error(503)));
        assert!(gemini_error_allows_model_fallback(&http_error(404)));
        assert!(gemini_error_allows_model_fallback(&transport_error(true)));
    }

    #[test]
    fn model_fallback_denied_for_request_errors_that_would_repeat() {
        assert!(!gemini_error_allows_model_fallback(&http_error(400)));
        assert!(!gemini_error_allows_model_fallback(&http_error(401)));
        assert!(!gemini_error_allows_model_fallback(&http_error(403)));
        assert!(!gemini_error_allows_model_fallback(&transport_error(false)));
        assert!(!gemini_error_allows_model_fallback(&anyhow!(
            "Gemini generateContent response decode failed"
        )));
    }

    #[test]
    fn fallback_budget_matches_one_request_timeout() {
        assert_eq!(
            gemini_fallback_budget().as_secs(),
            CONFIG.gemini_request_timeout_secs
        );
    }

    #[test]
    fn gemini_generate_content_url_does_not_embed_api_key() {
        let url = gemini_generate_content_url("gemini-test-model");

        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test-model:generateContent"
        );
        assert!(!url.contains("key="));
        if !CONFIG.gemini_api_key.is_empty() {
            assert!(!url.contains(&CONFIG.gemini_api_key));
        }
    }

    #[test]
    fn extract_music_generation_result_collects_lyrics_audio_and_notes() {
        let response: GeminiResponse = serde_json::from_value(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "[Intro]\nA bright beginning" },
                        { "text": "Caption: upbeat indie pop with layered harmonies\nBPM: 112" },
                        {
                            "inlineData": {
                                "mimeType": "audio/mpeg",
                                "data": general_purpose::STANDARD.encode(b"song-bytes")
                            }
                        }
                    ]
                }
            }]
        }))
        .expect("valid music response");

        let result =
            extract_music_generation_result(response, "lyria-3-pro-preview").expect("music");

        assert_eq!(result.lyrics_text, "[Intro]\nA bright beginning");
        assert_eq!(
            result.notes_text.as_deref(),
            Some("Caption: upbeat indie pop with layered harmonies\nBPM: 112")
        );
        assert_eq!(result.audio_bytes, b"song-bytes");
        assert_eq!(result.audio_mime_type, "audio/mpeg");
        assert_eq!(result.model_used, "lyria-3-pro-preview");
    }

    #[test]
    fn extract_music_generation_result_errors_when_audio_is_missing() {
        let response: GeminiResponse = serde_json::from_value(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "[Verse]\nNo audio returned" }
                    ]
                }
            }]
        }))
        .expect("valid music response");

        let err = extract_music_generation_result(response, "lyria-3-pro-preview").unwrap_err();

        assert!(err
            .to_string()
            .contains("No audio returned by Lyria (model: lyria-3-pro-preview)"));
    }

    #[test]
    fn gemini_timeout_helpers_use_general_and_image_specific_config() {
        assert_eq!(
            gemini_generate_content_timeout().as_secs(),
            CONFIG.gemini_request_timeout_secs
        );
        assert_eq!(
            gemini_image_generation_timeout().as_secs(),
            CONFIG.gemini_image_request_timeout_secs
        );
    }
}

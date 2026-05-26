use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::multipart::{Form, Part};
use reqwest::StatusCode;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::config::CONFIG;
use crate::llm::audit::{
    log_llm_request_started, record_llm_request_success, LlmAuditContext, LlmUsageRecord,
};
use crate::llm::gemini::ImageGenerationError;
use crate::llm::media::{detect_mime_type, download_media};
use crate::utils::http::get_http_client;

const IMG2_PROVIDER: &str = "img2";
const IMG2_MODEL: &str = "img2";
const IMG2_OPERATION: &str = "generate_image_with_img2";
const IMG2_ERROR_BODY_LIMIT: usize = 1200;

static IMG2_SAVE_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Img2RequestOptions {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
}

impl Img2RequestOptions {
    pub fn from_config() -> Self {
        Self {
            width: CONFIG.img2_width,
            height: CONFIG.img2_height,
            steps: CONFIG.img2_steps,
        }
    }

    pub(crate) fn optional_form_fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = Vec::new();
        if let Some(width) = self.width {
            fields.push(("width", width.to_string()));
        }
        if let Some(height) = self.height {
            fields.push(("height", height.to_string()));
        }
        if let Some(steps) = self.steps {
            fields.push(("steps", steps.to_string()));
        }
        fields
    }
}

#[derive(Debug, Clone)]
pub struct Img2GeneratedImage {
    pub path: PathBuf,
    pub request_id: Option<String>,
    pub content_type: Option<String>,
    pub byte_len: usize,
}

#[derive(Debug, Clone)]
struct SourceImage {
    bytes: Vec<u8>,
    mime_type: String,
}

pub fn img2_available() -> bool {
    CONFIG.img2_api_available()
}

pub fn img2_generate_url() -> String {
    join_base_and_path(&CONFIG.img2_base_url, &CONFIG.img2_generate_path)
}

pub fn img2_health_url() -> String {
    join_base_and_path(&CONFIG.img2_base_url, &CONFIG.img2_health_path)
}

fn join_base_and_path(base_url: &str, path: &str) -> String {
    let trimmed_path = path.trim();
    if trimmed_path.starts_with("http://") || trimmed_path.starts_with("https://") {
        return trimmed_path.to_string();
    }

    format!(
        "{}/{}",
        base_url.trim().trim_end_matches('/'),
        trimmed_path.trim_start_matches('/')
    )
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated = value.chars().take(limit).collect::<String>();
    format!("{truncated}... (truncated)")
}

fn sanitize_path_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .chars()
        .take(64)
        .collect::<String>();

    if sanitized.is_empty() {
        "no_request_id".to_string()
    } else {
        sanitized
    }
}

fn file_name_for_mime(mime_type: &str) -> &'static str {
    match mime_type {
        "image/jpeg" | "image/jpg" => "input.jpg",
        "image/webp" => "input.webp",
        "image/gif" => "input.gif",
        _ => "input.png",
    }
}

pub(crate) fn build_img2_output_path(
    media_dir: &Path,
    chat_id: i64,
    message_id: i64,
    request_id: Option<&str>,
    sequence: u64,
) -> PathBuf {
    let now = chrono::Utc::now();
    let timestamp = format!(
        "{}_{:03}",
        now.format("%Y%m%d_%H%M%S"),
        now.timestamp_subsec_millis()
    );
    let request_id = request_id
        .map(sanitize_path_component)
        .unwrap_or_else(|| "no_request_id".to_string());
    let filename =
        format!("img2_{timestamp}_chat{chat_id}_msg{message_id}_{request_id}_{sequence}.png");
    media_dir.join(filename)
}

async fn first_source_image(image_urls: &[String]) -> Option<SourceImage> {
    let url = image_urls.first()?;
    if image_urls.len() > 1 {
        warn!(
            "Img2 image editing received {} source image URLs; only the first image will be sent",
            image_urls.len()
        );
    }

    let Some(bytes) = download_media(url).await else {
        warn!("Img2 source image download failed; continuing as text-to-image");
        return None;
    };
    let mime_type = detect_mime_type(&bytes).unwrap_or_else(|| "image/png".to_string());
    Some(SourceImage { bytes, mime_type })
}

fn build_form(
    prompt: &str,
    source_image: Option<SourceImage>,
    options: &Img2RequestOptions,
) -> Result<Form, ImageGenerationError> {
    let mut form = Form::new().text("prompt", prompt.to_string());
    if let Some(source_image) = source_image {
        let file_name = file_name_for_mime(&source_image.mime_type);
        let part = Part::bytes(source_image.bytes)
            .file_name(file_name.to_string())
            .mime_str(&source_image.mime_type)
            .map_err(|err| {
                ImageGenerationError(format!("Img2 source image multipart setup failed: {err}"))
            })?;
        form = form.part("image", part);
    }

    for (name, value) in options.optional_form_fields() {
        form = form.text(name, value);
    }

    Ok(form)
}

async fn save_image_bytes(
    bytes: &[u8],
    request_id: Option<&str>,
    chat_id: i64,
    message_id: i64,
) -> Result<PathBuf, ImageGenerationError> {
    let media_dir = PathBuf::from(&CONFIG.img2_media_dir);
    tokio::fs::create_dir_all(&media_dir).await.map_err(|err| {
        ImageGenerationError(format!(
            "Failed to create Img2 media directory {}: {err}",
            media_dir.display()
        ))
    })?;

    let sequence = IMG2_SAVE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = build_img2_output_path(&media_dir, chat_id, message_id, request_id, sequence);
    tokio::fs::write(&path, bytes).await.map_err(|err| {
        ImageGenerationError(format!(
            "Failed to save Img2 image to {}: {err}",
            path.display()
        ))
    })?;
    Ok(path)
}

fn header_string(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string())
}

fn status_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

pub async fn generate_image_with_img2(
    prompt: &str,
    image_urls: &[String],
    chat_id: i64,
    message_id: i64,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Img2GeneratedImage, ImageGenerationError> {
    if !CONFIG.enable_img2 {
        return Err(ImageGenerationError(
            "Img2 image generation is disabled. Set ENABLE_IMG2=true to enable it.".to_string(),
        ));
    }
    let api_key = CONFIG.img2_api_key.trim();
    if api_key.is_empty() {
        return Err(ImageGenerationError(
            "Img2 image generation requires IMG2_API_KEY.".to_string(),
        ));
    }
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(ImageGenerationError(
            "Img2 image generation requires a prompt.".to_string(),
        ));
    }

    let options = Img2RequestOptions::from_config();
    let source_image = first_source_image(image_urls).await;
    let source_image_present = source_image.is_some();
    let form = build_form(prompt, source_image, &options)?;
    let url = img2_generate_url();
    let started_at = chrono::Utc::now();
    let metadata = json!({
        "url": url,
        "timeout_secs": CONFIG.img2_request_timeout_secs,
        "source_image": source_image_present,
        "width": options.width,
        "height": options.height,
        "steps": options.steps,
        "media_dir": CONFIG.img2_media_dir,
    });
    log_llm_request_started(
        IMG2_PROVIDER,
        IMG2_MODEL,
        IMG2_OPERATION,
        started_at,
        Some(&metadata),
    );

    debug!(
        "Img2 image request starting: source_image={}, width={:?}, height={:?}, steps={:?}, timeout_secs={}, url={}",
        source_image_present,
        options.width,
        options.height,
        options.steps,
        CONFIG.img2_request_timeout_secs,
        url
    );

    let client = get_http_client();
    let response = client
        .post(&url)
        .timeout(Duration::from_secs(CONFIG.img2_request_timeout_secs))
        .header("X-API-Key", api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|err| {
            ImageGenerationError(format!(
                "Img2 image request failed to send: {err} (timeout={}, connect={}, status={:?})",
                err.is_timeout(),
                err.is_connect(),
                err.status()
            ))
        })?;

    let status = response.status();
    let headers = response.headers().clone();
    let request_id = header_string(&headers, "x-request-id");
    let content_type = header_string(&headers, "content-type");

    if !status.is_success() {
        let retryable = status_retryable(status);
        let body = response
            .text()
            .await
            .unwrap_or_else(|err| format!("<failed to read error body: {err}>"));
        warn!(
            "Img2 image request failed: status={}, request_id={:?}, retryable={}, body={}",
            status,
            request_id,
            retryable,
            truncate_for_log(&body, IMG2_ERROR_BODY_LIMIT)
        );
        return Err(ImageGenerationError(format!(
            "Img2 image request failed with status {} (request_id={})",
            status,
            request_id.as_deref().unwrap_or("unknown")
        )));
    }

    if !content_type
        .as_deref()
        .map(|value| value.to_ascii_lowercase().contains("image/png"))
        .unwrap_or(false)
    {
        warn!(
            "Img2 image response content type was not image/png: request_id={:?}, content_type={:?}",
            request_id, content_type
        );
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|err| ImageGenerationError(format!("Failed to read Img2 image bytes: {err}")))?;
    if bytes.is_empty() {
        return Err(ImageGenerationError(format!(
            "Img2 image response was empty (request_id={})",
            request_id.as_deref().unwrap_or("unknown")
        )));
    }

    let path = save_image_bytes(&bytes, request_id.as_deref(), chat_id, message_id).await?;
    let completed_at = chrono::Utc::now();
    let byte_len = bytes.len();
    info!(
        "Img2 image request completed: request_id={:?}, bytes={}, content_type={:?}, saved_path={}",
        request_id,
        byte_len,
        content_type,
        path.display()
    );

    record_llm_request_success(
        audit_context,
        IMG2_PROVIDER,
        IMG2_MODEL,
        IMG2_OPERATION,
        started_at,
        completed_at,
        LlmUsageRecord {
            response_id: request_id.clone(),
            raw_usage_json: Some(
                json!({
                    "request_id": request_id.clone(),
                    "bytes": byte_len,
                    "content_type": content_type.clone(),
                    "path": path.to_string_lossy(),
                })
                .to_string(),
            ),
            ..LlmUsageRecord::default()
        },
    )
    .await;

    Ok(Img2GeneratedImage {
        path,
        request_id,
        content_type,
        byte_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_form_fields_omit_unset_values() {
        let options = Img2RequestOptions {
            width: Some(1024),
            height: None,
            steps: Some(4),
        };

        assert_eq!(
            options.optional_form_fields(),
            vec![("width", "1024".to_string()), ("steps", "4".to_string())]
        );
    }

    #[test]
    fn output_path_stays_under_media_dir_and_omits_prompt_text() {
        let media_dir = PathBuf::from("data/media/img2");
        let path =
            build_img2_output_path(&media_dir, -100123, 42, Some("../req/id with spaces"), 7);
        let path_text = path.to_string_lossy();

        assert!(path.starts_with(&media_dir));
        assert!(path_text.ends_with(".png"));
        assert!(path_text.contains("chat-100123"));
        assert!(path_text.contains("msg42"));
        assert!(!path_text.contains(".."));
        assert!(!path_text.contains("id with spaces"));
    }

    #[test]
    fn endpoint_joining_accepts_relative_and_absolute_paths() {
        assert_eq!(
            join_base_and_path("https://example.com/", "/v1/images/generate"),
            "https://example.com/v1/images/generate"
        );
        assert_eq!(
            join_base_and_path("https://example.com", "https://override.local/generate"),
            "https://override.local/generate"
        );
    }
}

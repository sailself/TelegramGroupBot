use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::multipart::{Form, Part};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::config::CONFIG;
use crate::llm::audit::{LlmAuditContext, LlmUsageRecord};
use crate::llm::gemini::ImageGenerationError;
use crate::llm::media::{detect_mime_type, download_media};
use crate::llm::transport::{
    call_with_retry, read_body_limited, LlmCall, ProviderError, RetryPolicy,
};
use crate::utils::http::get_http_client;

const IMG2_PROVIDER: &str = "img2";
const IMG2_MODEL: &str = "img2";
const IMG2_OPERATION: &str = "generate_image_with_img2";
const IMG2_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(2, Duration::from_millis(500));
/// Generated PNGs are a few MiB at most; anything larger is a misbehaving endpoint.
const IMG2_MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;

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

/// Build the multipart body. Called once per attempt, so the source image is
/// borrowed and its bytes copied into the part.
fn build_form(
    prompt: &str,
    source_image: Option<&SourceImage>,
    options: &Img2RequestOptions,
) -> Result<Form, ImageGenerationError> {
    let mut form = Form::new().text("prompt", prompt.to_string());
    if let Some(source_image) = source_image {
        let file_name = file_name_for_mime(&source_image.mime_type);
        let part = Part::bytes(source_image.bytes.clone())
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

/// Raw result of one Img2 generation request, before it is written to disk.
#[derive(Debug)]
pub(crate) struct Img2FetchedImage {
    pub bytes: Vec<u8>,
    pub request_id: Option<String>,
    pub content_type: Option<String>,
}

/// Send the multipart generation request, rebuilding the form for each
/// attempt, and return the PNG bytes. Transient failures are retried by the
/// shared transport, which also writes the audit row.
async fn fetch_img2_image<F>(
    url: &str,
    api_key: &str,
    timeout: Duration,
    build_form: F,
    audit_context: Option<&LlmAuditContext>,
    metadata: &Value,
) -> Result<Img2FetchedImage, ImageGenerationError>
where
    F: Fn() -> Result<Form, ImageGenerationError>,
{
    let call = LlmCall::begin(
        IMG2_PROVIDER,
        IMG2_MODEL,
        IMG2_OPERATION,
        audit_context,
        Some(metadata),
    );
    let last_request_id = parking_lot::Mutex::new(None::<String>);

    let result = call_with_retry(
        &call,
        &IMG2_RETRY_POLICY,
        |attempt| {
            let form = build_form().map_err(|err| ProviderError::rejected(err.0));
            async move {
                debug!(
                    "Img2 image request attempt {}/{}: timeout_secs={}",
                    attempt.number,
                    attempt.max_attempts,
                    timeout.as_secs()
                );
                Ok(get_http_client()
                    .post(url)
                    .timeout(timeout)
                    .header("X-API-Key", api_key)
                    .multipart(form?))
            }
        },
        |response| {
            *last_request_id.lock() = header_string(response.headers(), "x-request-id");
        },
        |response| async move {
            let request_id = header_string(response.headers(), "x-request-id");
            let content_type = header_string(response.headers(), "content-type");
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
            let bytes = read_body_limited(response, IMG2_PROVIDER, IMG2_MAX_IMAGE_BYTES).await?;
            if bytes.is_empty() {
                return Err(ProviderError::decode(
                    IMG2_PROVIDER,
                    format!(
                        "Img2 image response was empty (request_id={})",
                        request_id.as_deref().unwrap_or("unknown")
                    ),
                    false,
                ));
            }
            Ok(Img2FetchedImage {
                bytes,
                request_id,
                content_type,
            })
        },
        |fetched| LlmUsageRecord {
            response_id: fetched.request_id.clone(),
            raw_usage_json: Some(
                json!({
                    "request_id": fetched.request_id,
                    "bytes": fetched.bytes.len(),
                    "content_type": fetched.content_type,
                })
                .to_string(),
            ),
            ..LlmUsageRecord::default()
        },
    )
    .await;

    result.map_err(|err| img2_error(err, last_request_id.lock().clone()))
}

/// Keep the user-facing message free of provider error bodies; the transport
/// already logged the details.
fn img2_error(err: ProviderError, request_id: Option<String>) -> ImageGenerationError {
    match err.status() {
        Some(status) => ImageGenerationError(format!(
            "Img2 image request failed with status {status} (request_id={})",
            request_id.as_deref().unwrap_or("unknown")
        )),
        None => ImageGenerationError(err.to_string()),
    }
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
    let url = img2_generate_url();
    let timeout = Duration::from_secs(CONFIG.img2_request_timeout_secs);
    let metadata = json!({
        "url": url,
        "timeout_secs": CONFIG.img2_request_timeout_secs,
        "source_image": source_image_present,
        "width": options.width,
        "height": options.height,
        "steps": options.steps,
        "media_dir": CONFIG.img2_media_dir,
    });

    debug!(
        "Img2 image request starting: source_image={}, width={:?}, height={:?}, steps={:?}, timeout_secs={}, url={}",
        source_image_present,
        options.width,
        options.height,
        options.steps,
        CONFIG.img2_request_timeout_secs,
        url
    );

    let fetched = fetch_img2_image(
        &url,
        api_key,
        timeout,
        || build_form(prompt, source_image.as_ref(), &options),
        audit_context,
        &metadata,
    )
    .await?;

    let path = save_image_bytes(
        &fetched.bytes,
        fetched.request_id.as_deref(),
        chat_id,
        message_id,
    )
    .await?;
    let byte_len = fetched.bytes.len();
    info!(
        "Img2 image request completed: request_id={:?}, bytes={}, content_type={:?}, saved_path={}",
        fetched.request_id,
        byte_len,
        fetched.content_type,
        path.display()
    );

    Ok(Img2GeneratedImage {
        path,
        request_id: fetched.request_id,
        content_type: fetched.content_type,
        byte_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::test_support::{
        response_with_headers, ExpectedRequest, TestServer,
    };

    #[tokio::test]
    async fn fetch_img2_image_retries_a_transient_failure_and_returns_the_png() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/generate",
                response_with_headers(503, &[], b"busy".to_vec()),
            ),
            ExpectedRequest::new(
                "POST",
                "/generate",
                response_with_headers(
                    200,
                    &[("content-type", "image/png"), ("x-request-id", "req_1")],
                    b"PNGDATA".to_vec(),
                ),
            )
            .with_header("x-api-key", "k"),
        ]);
        let url = server.url("/generate").to_string();

        let fetched = fetch_img2_image(
            &url,
            "k",
            Duration::from_secs(5),
            || build_form("a cat", None, &Img2RequestOptions::default()),
            None,
            &json!({}),
        )
        .await
        .expect("second attempt succeeds");

        assert_eq!(fetched.bytes, b"PNGDATA");
        assert_eq!(fetched.request_id.as_deref(), Some("req_1"));
        assert_eq!(fetched.content_type.as_deref(), Some("image/png"));
        server.join().expect("both requests were served");
    }

    #[tokio::test]
    async fn fetch_img2_image_reports_permanent_failures_without_echoing_the_body() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/generate",
            response_with_headers(400, &[("x-request-id", "req_2")], b"secret detail".to_vec()),
        )]);
        let url = server.url("/generate").to_string();

        let err = fetch_img2_image(
            &url,
            "k",
            Duration::from_secs(5),
            || build_form("a cat", None, &Img2RequestOptions::default()),
            None,
            &json!({}),
        )
        .await
        .expect_err("400 is permanent");

        assert!(err.0.contains("status 400"), "{}", err.0);
        assert!(err.0.contains("req_2"), "{}", err.0);
        assert!(!err.0.contains("secret detail"), "{}", err.0);
        server.join().expect("exactly one request was served");
    }

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

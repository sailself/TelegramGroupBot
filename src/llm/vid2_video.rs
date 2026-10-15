//! The asynchronous Vid2 job API. Only GET operations may be retried.

use std::future::Future;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use reqwest::multipart::{Form, Part};
use serde_json::{json, Value};
use tracing::{error, info};
use url::Url;

use crate::config::{Vid2Config, CONFIG};
use crate::llm::audit::{LlmAuditContext, LlmUsageRecord};
use crate::llm::transport::{
    call_with_retry, read_body_limited, LlmCall, ProviderError, RetryPolicy,
};
use crate::utils::http::get_http_client_no_redirect;
use crate::utils::text::truncate_to_chars;

const PROVIDER: &str = "vid2";
const JSON_MAX_BYTES: usize = 64 * 1024;
const VIDEO_MAX_BYTES: usize = 50_000_000;
const SOURCE_MAX_BYTES: usize = 32 * 1024 * 1024;
const REJECTION_REASON_MAX_CHARS: usize = 512;
const GET_POLICY: RetryPolicy = RetryPolicy::exponential(3, Duration::from_millis(400));
static SAVE_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, thiserror::Error)]
pub enum Vid2Error {
    #[error("Vid2 generation failed: {0}")]
    Generation(String),
    #[error("Vid2 generation reached its time limit")]
    Timeout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vid2Phase {
    PreparingImage,
    Submitting,
    Queued,
    Running,
    Downloading,
}

pub struct Vid2GenerationRequest<'a> {
    pub prompt: &'a str,
    pub chat_id: i64,
    pub message_id: i64,
}

#[derive(Debug)]
pub struct Vid2GeneratedVideo {
    pub path: PathBuf,
    pub job_id: String,
    pub byte_len: usize,
}

#[derive(Debug)]
pub struct Vid2SourceImage {
    bytes: Vec<u8>,
    mime_type: &'static str,
    file_name: &'static str,
    resolution: VideoResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoResolution {
    width: u32,
    height: u32,
}

fn video_resolution(defaults: VideoResolution, source: Option<VideoResolution>) -> VideoResolution {
    let Some(source) = source else {
        return defaults;
    };
    let cap = defaults.width.max(defaults.height);
    let long_edge = source.width.max(source.height);
    if long_edge <= cap {
        return source;
    }
    // Integer rounding keeps the long edge exact and the other edge within
    // half a pixel of the source ratio. A very narrow image still needs 1px.
    let scaled = |edge: u32| {
        ((u64::from(edge) * u64::from(cap) + u64::from(long_edge) / 2) / u64::from(long_edge))
            .max(1) as u32
    };
    VideoResolution {
        width: scaled(source.width),
        height: scaled(source.height),
    }
}

fn generation_error(message: impl Into<String>) -> Vid2Error {
    Vid2Error::Generation(message.into())
}

// The server may echo its credential in arbitrary error bodies. Keep such
// bodies out of shared-transport logs as well as the user-visible error.
fn omit_transport_detail(_: &str) -> String {
    "external request details omitted".to_string()
}

fn structured_rejection_reason(value: &Value) -> Option<String> {
    match value {
        Value::String(message) if !message.trim().is_empty() => Some(message.trim().to_string()),
        Value::Object(object) => {
            let reason = ["detail", "error", "message", "msg"]
                .into_iter()
                .find_map(|field| object.get(field).and_then(structured_rejection_reason))?;
            // Keep useful FastAPI field locations, but never arbitrary input
            // values or caller-provided strings from the validation object.
            let field = object
                .get("loc")
                .and_then(Value::as_array)
                .and_then(|location| location.last())
                .and_then(Value::as_str)
                .filter(|field| {
                    matches!(
                        *field,
                        "width" | "height" | "seconds" | "prompt" | "first_frame"
                    )
                });
            Some(field.map_or(reason.clone(), |field| format!("{field}: {reason}")))
        }
        Value::Array(entries) => {
            let messages = entries
                .iter()
                .take(4)
                .filter_map(structured_rejection_reason)
                .collect::<Vec<_>>();
            (!messages.is_empty()).then(|| messages.join("; "))
        }
        _ => None,
    }
}

fn redact_rejection_reason(reason: &str, active_key: &str, prompt: &str) -> String {
    // Parse/select JSON messages before this boundary, never validation input
    // objects. URLs/data URIs may contain credentials or source-media content.
    static PRIVATE_REFERENCES: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?i)\b(?:https?://|data:image/)(?:<REDACTED>|[^\s<>"'])+"#)
            .expect("valid private reference regex")
    });
    let mut safe = reason.to_string();
    let mut secrets = CONFIG.secret_values();
    secrets.extend([active_key.trim(), prompt]);
    secrets.retain(|secret| !secret.is_empty());
    secrets.sort_unstable_by_key(|secret| std::cmp::Reverse(secret.len()));
    for secret in secrets {
        safe = safe.replace(secret, "<REDACTED>");
    }
    // Whole prompts can themselves contain URLs: remove them before altering
    // URL substrings, then scrub remaining URLs including redacted tokens.
    let safe = PRIVATE_REFERENCES.replace_all(&safe, "<REDACTED_URL>");
    // Scrub complete credentials BEFORE bounding output, including a key
    // whose raw value straddles the cutoff. Keep messages single-line.
    let safe = safe
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect::<String>();
    truncate_to_chars(&safe, REJECTION_REASON_MAX_CHARS).to_string()
}

async fn rejection_reason(response: reqwest::Response, active_key: &str, prompt: &str) -> String {
    let reason = match read_body_limited(response, PROVIDER, JSON_MAX_BYTES).await {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|value| structured_rejection_reason(&value))
            .unwrap_or_else(|| "server returned no structured error reason".to_string()),
        Err(ProviderError::BodyTooLarge { .. }) => {
            "error response exceeded the 65536-byte diagnostic limit".to_string()
        }
        Err(_) => "error response could not be read".to_string(),
    };
    redact_rejection_reason(&reason, active_key, prompt)
}

fn transport_error(error: ProviderError, phase: &str) -> Vid2Error {
    let detail = match error {
        ProviderError::Http { status, .. } => format!("{phase}: HTTP {status}"),
        ProviderError::BodyTooLarge { limit, .. } => {
            format!("{phase}: response exceeded the {limit}-byte limit")
        }
        ProviderError::Decode { .. } => format!("{phase}: invalid response"),
        _ => format!("{phase}: request failed"),
    };
    generation_error(detail)
}

fn endpoint(base: &str, job_id: Option<&str>, file: bool) -> Result<Url, Vid2Error> {
    let mut url = Url::parse(base).map_err(|_| generation_error("invalid base URL"))?;
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| generation_error("invalid base URL path"))?;
    segments.pop_if_empty().extend(["v1", "videos"]);
    if let Some(job_id) = job_id {
        // URL's path-segment API deliberately ignores literal dot segments.
        if job_id.trim().is_empty()
            || matches!(job_id, "." | "..")
            || job_id.len() > 512
            || job_id.chars().any(char::is_control)
        {
            return Err(generation_error("invalid job ID"));
        }
        segments.push(job_id);
        if file {
            segments.push("file");
        }
    } else {
        segments.push("generate");
    }
    drop(segments);
    Ok(url)
}

fn build_form(
    prompt: &str,
    seconds: u64,
    resolution: VideoResolution,
    source: Option<Vid2SourceImage>,
) -> Result<Form, Vid2Error> {
    let mut form = Form::new()
        .text("prompt", prompt.to_string())
        .text("seconds", seconds.to_string())
        .text("width", resolution.width.to_string())
        .text("height", resolution.height.to_string());
    if let Some(source) = source {
        let part = Part::bytes(source.bytes)
            .file_name(source.file_name)
            .mime_str(source.mime_type)
            .map_err(|_| generation_error("could not prepare the first frame"))?;
        form = form.part("first_frame", part);
    }
    Ok(form)
}

async fn read_json(response: reqwest::Response) -> Result<Value, ProviderError> {
    let bytes = read_body_limited(response, PROVIDER, JSON_MAX_BYTES).await?;
    serde_json::from_slice(&bytes)
        .map_err(|_| ProviderError::decode(PROVIDER, "invalid JSON", false))
}

async fn submit(
    settings: &Vid2Config,
    prompt: &str,
    resolution: VideoResolution,
    source: Option<Vid2SourceImage>,
) -> Result<String, Vid2Error> {
    let url = endpoint(&settings.base_url, None, false)?;
    let source_image = source.is_some();
    let source_mime = source.as_ref().map(|image| image.mime_type);
    let source_bytes = source.as_ref().map(|image| image.bytes.len());
    let source_width = source.as_ref().map(|image| image.resolution.width);
    let source_height = source.as_ref().map(|image| image.resolution.height);
    info!(
        source_image,
        source_mime,
        source_bytes,
        source_width,
        source_height,
        target_width = resolution.width,
        target_height = resolution.height,
        seconds = settings.seconds,
        "Submitting Vid2 video"
    );
    let form = build_form(prompt, settings.seconds, resolution, source)?;
    // Send exactly once. The API has no idempotency key, so even a lost
    // response must never trigger a second submission or fallback request.
    let response = get_http_client_no_redirect()
        .post(url)
        .timeout(Duration::from_secs(settings.request_timeout_secs))
        .header("X-API-Key", settings.api_key.trim())
        .multipart(form)
        .send()
        .await
        .map_err(|error| {
            error!(
                timeout = error.is_timeout(),
                connect = error.is_connect(),
                "Vid2 submission transport failed; request details omitted"
            );
            generation_error("submission: request failed")
        })?;
    let status = response.status();
    if !status.is_success() {
        let reason = rejection_reason(response, &settings.api_key, prompt).await;
        error!(
            status = status.as_u16(),
            reason,
            source_image,
            source_mime,
            source_bytes,
            source_width,
            source_height,
            target_width = resolution.width,
            target_height = resolution.height,
            seconds = settings.seconds,
            "Vid2 submission rejected"
        );
        return Err(generation_error(format!("submission: HTTP {status}")));
    }
    let value = read_json(response).await.map_err(|error| {
        error!("Vid2 submission returned an invalid job response");
        transport_error(error, "submission")
    })?;
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| generation_error("submission returned no job ID"))?;
    endpoint(&settings.base_url, Some(id), false)?;
    Ok(id.to_string())
}

async fn get<T, F, R>(settings: &Vid2Config, url: Url, phase: &str, read: F) -> Result<T, Vid2Error>
where
    F: FnMut(reqwest::Response) -> R,
    R: Future<Output = Result<T, ProviderError>>,
{
    let call = LlmCall::untracked(PROVIDER, phase).with_redaction(omit_transport_detail);
    call_with_retry(
        &call,
        &GET_POLICY,
        |_| {
            let url = url.clone();
            async move {
                Ok(get_http_client_no_redirect()
                    .get(url)
                    .timeout(Duration::from_secs(settings.request_timeout_secs))
                    .header("X-API-Key", settings.api_key.trim()))
            }
        },
        |_| {},
        read,
        |_| LlmUsageRecord::default(),
    )
    .await
    .map_err(|error| transport_error(error, phase))
}

fn validate_mp4(bytes: &[u8]) -> Result<(), ProviderError> {
    // Validate the ISO BMFF file-type box and an MP4-compatible brand rather
    // than trusting Content-Type (an error page may be served as video/mp4).
    let invalid = || ProviderError::decode(PROVIDER, "empty or invalid MP4", false);
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" {
        return Err(invalid());
    }
    let box_len = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    if box_len < 16 || box_len > bytes.len() || !(box_len - 16).is_multiple_of(4) {
        return Err(invalid());
    }
    let compatible = |brand: &[u8]| {
        matches!(
            brand,
            b"isom"
                | b"iso2"
                | b"iso3"
                | b"iso4"
                | b"iso5"
                | b"iso6"
                | b"mp41"
                | b"mp42"
                | b"avc1"
                | b"M4V "
        )
    };
    if !compatible(&bytes[8..12])
        && !bytes[16..box_len]
            .as_chunks::<4>()
            .0
            .iter()
            .any(|brand| compatible(brand))
    {
        return Err(invalid());
    }
    let mut offset = box_len;
    let mut has_movie = false;
    let mut has_media = false;
    while offset < bytes.len() {
        let remaining = &bytes[offset..];
        if remaining.len() < 8 {
            return Err(invalid());
        }
        let length = u32::from_be_bytes(remaining[..4].try_into().unwrap());
        let (length, header_len) = match length {
            0 => (remaining.len(), 8),
            1 if remaining.len() >= 16 => {
                let length = u64::from_be_bytes(remaining[8..16].try_into().unwrap());
                (usize::try_from(length).map_err(|_| invalid())?, 16)
            }
            1 => return Err(invalid()),
            length => (length as usize, 8),
        };
        if length < header_len || length > remaining.len() {
            return Err(invalid());
        }
        has_movie |= &remaining[4..8] == b"moov";
        has_media |= &remaining[4..8] == b"mdat" && length > header_len;
        offset += length;
    }
    if !has_movie || !has_media {
        return Err(invalid());
    }
    Ok(())
}

fn output_path(media_dir: &Path, request: &Vid2GenerationRequest<'_>, sequence: u64) -> PathBuf {
    // Never derive filenames from server IDs, prompts, or uploaded filenames.
    media_dir.join(format!(
        "vid2_{}_chat{}_msg{}_{}_{}.mp4",
        chrono::Utc::now().format("%Y%m%d_%H%M%S_%f"),
        request.chat_id,
        request.message_id,
        std::process::id(),
        sequence,
    ))
}

async fn save_video(
    settings: &Vid2Config,
    request: &Vid2GenerationRequest<'_>,
    bytes: &[u8],
) -> Result<PathBuf, Vid2Error> {
    use tokio::io::AsyncWriteExt;
    let dir = Path::new(&settings.media_dir);
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|_| generation_error("could not create the video directory"))?;
    let path = output_path(dir, request, SAVE_COUNTER.fetch_add(1, Ordering::Relaxed));
    // Persist outside the cancellation boundary below: a completed file is
    // never exposed until write+flush finishes, and failures remove partials.
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .await
        .map_err(|_| generation_error("could not create the video file"))?;
    let write = async {
        file.write_all(bytes).await?;
        file.flush().await
    }
    .await;
    drop(file);
    if write.is_err() {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(generation_error("could not save the generated video"));
    }
    Ok(path)
}

pub async fn download_source_image(
    url: &str,
    timeout_secs: u64,
) -> Result<Vid2SourceImage, Vid2Error> {
    let call = LlmCall::untracked(PROVIDER, "first frame").with_redaction(omit_transport_detail);
    let bytes = call_with_retry(
        &call,
        &GET_POLICY,
        |_| async {
            Ok(get_http_client_no_redirect()
                .get(url)
                .timeout(Duration::from_secs(timeout_secs)))
        },
        |_| {},
        |response| read_body_limited(response, PROVIDER, SOURCE_MAX_BYTES),
        |_| LlmUsageRecord::default(),
    )
    .await
    .map_err(|error| transport_error(error, "first frame download"))?;
    tokio::task::spawn_blocking(move || validate_source_image(bytes))
        .await
        .map_err(|_| generation_error("first frame validation failed"))?
}

fn validate_source_image(bytes: Vec<u8>) -> Result<Vid2SourceImage, Vid2Error> {
    let mut reader = image::ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|_| generation_error("invalid first frame"))?;
    let (mime_type, file_name) = match reader.format() {
        Some(image::ImageFormat::Jpeg) => ("image/jpeg", "first_frame.jpg"),
        Some(image::ImageFormat::Png) => ("image/png", "first_frame.png"),
        Some(image::ImageFormat::WebP) => ("image/webp", "first_frame.webp"),
        _ => return Err(generation_error("first frame must be JPEG, PNG, or WebP")),
    };
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let decoded = reader
        .decode()
        .map_err(|_| generation_error("first frame is invalid or too large"))?;
    Ok(Vid2SourceImage {
        bytes,
        mime_type,
        file_name,
        resolution: VideoResolution {
            width: decoded.width(),
            height: decoded.height(),
        },
    })
}

/// Image lookup/download belongs to the same deadline as submission, polling,
/// and download. Phase notifications must be synchronous and nonblocking.
pub async fn generate_video_with_vid2(
    request: Vid2GenerationRequest<'_>,
    load_source: impl Future<Output = Result<Option<Vid2SourceImage>, Vid2Error>>,
    on_phase: impl FnMut(Vid2Phase),
    audit: Option<&LlmAuditContext>,
) -> Result<Vid2GeneratedVideo, Vid2Error> {
    generate_with_settings(&CONFIG.vid2, request, load_source, on_phase, audit).await
}

async fn generate_with_settings(
    settings: &Vid2Config,
    request: Vid2GenerationRequest<'_>,
    load_source: impl Future<Output = Result<Option<Vid2SourceImage>, Vid2Error>>,
    mut on_phase: impl FnMut(Vid2Phase),
    audit: Option<&LlmAuditContext>,
) -> Result<Vid2GeneratedVideo, Vid2Error> {
    if !settings.enabled || settings.base_url.is_empty() || settings.api_key.trim().is_empty() {
        return Err(generation_error("provider is disabled or unconfigured"));
    }
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(generation_error("a prompt is required"));
    }
    let call = LlmCall::begin(
        PROVIDER,
        PROVIDER,
        "generate_video_with_vid2",
        audit,
        Some(
            &json!({"seconds": settings.seconds, "default_width": settings.width, "default_height": settings.height, "timeout_secs": settings.generation_timeout_secs}),
        ),
    );
    let mut job_id = None;
    let mut target_resolution = None;
    let started = std::time::Instant::now();
    let work = async {
        let source = load_source.await?;
        let resolution = video_resolution(
            VideoResolution {
                width: settings.width,
                height: settings.height,
            },
            source.as_ref().map(|image| image.resolution),
        );
        target_resolution = Some(resolution);
        on_phase(Vid2Phase::Submitting);
        let id = submit(settings, prompt, resolution, source).await?;
        job_id = Some(id.clone());
        let status_url = endpoint(&settings.base_url, Some(&id), false)?;
        let mut last_phase = None;
        loop {
            let status: Value = get(settings, status_url.clone(), "status", read_json).await?;
            let phase = match status.get("status").and_then(Value::as_str) {
                Some("queued") => Vid2Phase::Queued,
                Some("running") => Vid2Phase::Running,
                Some("success") => break,
                Some("error") => return Err(generation_error("the server reported a failed job")),
                _ => {
                    return Err(generation_error(
                        "the server returned an invalid job status",
                    ))
                }
            };
            if last_phase != Some(phase) {
                on_phase(phase);
                last_phase = Some(phase);
            }
            tokio::time::sleep(Duration::from_secs(settings.poll_interval_secs)).await;
        }
        on_phase(Vid2Phase::Downloading);
        get(
            settings,
            endpoint(&settings.base_url, Some(&id), true)?,
            "video download",
            |response| async {
                let bytes = read_body_limited(response, PROVIDER, VIDEO_MAX_BYTES).await?;
                validate_mp4(&bytes)?;
                Ok(bytes)
            },
        )
        .await
    };
    let fetched =
        match tokio::time::timeout(Duration::from_secs(settings.generation_timeout_secs), work)
            .await
        {
            Ok(result) => result,
            Err(_) => Err(Vid2Error::Timeout),
        };
    // Disk persistence finishes after the network deadline so cancelling a
    // write cannot leave a partial MP4. All input/network work was budgeted.
    let result = match fetched {
        Ok(bytes) => save_video(settings, &request, &bytes)
            .await
            .map(|path| Vid2GeneratedVideo {
                path,
                job_id: job_id.clone().expect("download follows submission"),
                byte_len: bytes.len(),
            }),
        Err(error) => Err(error),
    };
    let safe_id = job_id
        .as_deref()
        .map(|id| id.replace(settings.api_key.trim(), "[REDACTED]"));
    match &result {
        Ok(video) => {
            call.succeed(LlmUsageRecord {
                response_id: safe_id.clone(),
                raw_usage_json: Some(json!({"job_id": safe_id, "seconds": settings.seconds, "width": target_resolution.map(|size| size.width), "height": target_resolution.map(|size| size.height), "bytes": video.byte_len}).to_string()),
                ..LlmUsageRecord::default()
            }).await;
            info!(
                "Vid2 generation completed: job_id={safe_id:?}, bytes={}, duration_s={}",
                video.byte_len,
                started.elapsed().as_secs_f64()
            );
        }
        Err(error) => {
            call.fail(&ProviderError::rejected(format!(
                "{error} (job_id={})",
                safe_id.as_deref().unwrap_or("unknown")
            )))
            .await;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::models::LlmInvocationInsert;
    use crate::db::test_support::init_test_db;
    use crate::test_support::{
        redirect_response, response_with_content_length, response_with_headers,
        response_with_status, ExpectedRequest, TestServer,
    };
    type AuditRow = (String, Option<String>, Option<i64>, i64, Option<String>);

    fn settings(server: &TestServer) -> Vid2Config {
        Vid2Config {
            enabled: true,
            base_url: server.base_url().to_string(),
            api_key: "synthetic-vid2-key".into(),
            width: 1344,
            height: 768,
            seconds: 5,
            request_timeout_secs: 5,
            generation_timeout_secs: 5,
            poll_interval_secs: 0,
            media_dir: "target/test-vid2-media".into(),
        }
    }

    fn request() -> Vid2GenerationRequest<'static> {
        Vid2GenerationRequest {
            prompt: "a fox in snow",
            chat_id: -100123,
            message_id: 42,
        }
    }

    fn expected(method: &str, path: &str, status: u16, body: &[u8]) -> ExpectedRequest {
        ExpectedRequest::new(method, path, response_with_status(status, body.to_vec()))
            .with_header("x-api-key", "synthetic-vid2-key")
    }

    fn submission() -> ExpectedRequest {
        submission_at_resolution(1344, 768)
    }

    fn submission_at_resolution(width: u32, height: u32) -> ExpectedRequest {
        expected("POST", "/v1/videos/generate", 200, br#"{"id":"job-1"}"#)
            .with_body_fragment(b"name=\"prompt\"")
            .with_body_fragment(b"a fox in snow")
            .with_body_fragment(b"name=\"seconds\"\r\n\r\n5")
            .with_body_fragment(format!("name=\"width\"\r\n\r\n{width}\r\n").as_bytes())
            .with_body_fragment(format!("name=\"height\"\r\n\r\n{height}\r\n").as_bytes())
    }

    fn mp4() -> Vec<u8> {
        b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42\x00\x00\x00\x08moov\x00\x00\x00\x0cmdatDATA".to_vec()
    }

    fn png() -> Vec<u8> {
        png_at_resolution(1, 1)
    }

    fn png_at_resolution(width: u32, height: u32) -> Vec<u8> {
        let mut data = Cursor::new(Vec::new());
        image::DynamicImage::new_luma8(width, height)
            .write_to(&mut data, image::ImageFormat::Png)
            .unwrap();
        data.into_inner()
    }

    #[tokio::test]
    async fn text_video_authenticates_all_requests_and_tracks_status_progression() {
        let server = TestServer::new(vec![
            submission(),
            expected(
                "GET",
                "/v1/videos/job-1",
                200,
                br#"{"status":"queued","extra":true}"#,
            ),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"running"}"#),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
            expected("GET", "/v1/videos/job-1/file", 200, &mp4()),
        ]);
        let mut phases = Vec::new();
        let video = generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(None) },
            |phase| phases.push(phase),
            None,
        )
        .await
        .unwrap();
        assert_eq!(video.job_id, "job-1");
        assert_eq!(video.byte_len, mp4().len());
        assert_eq!(tokio::fs::read(&video.path).await.unwrap(), mp4());
        assert_eq!(
            phases,
            [
                Vid2Phase::Submitting,
                Vid2Phase::Queued,
                Vid2Phase::Running,
                Vid2Phase::Downloading
            ]
        );
        tokio::fs::remove_file(video.path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn user_confirmed_job_responses_ignore_params_duration_and_nullable_error() {
        // Real submission/running/success JSON supplied by the user. Server
        // dimensions, quality, seed and duration are informational fields.
        const SUBMITTED: &[u8] = br#"{"id":"cbb9ac6f-80c7-4c96-ad55-00e0dadefa5d","status":"queued","params":{"width":1344,"height":768,"seconds":5.0,"seed":962578975330414183,"quality":"fast","steps":null}}"#;
        const RUNNING: &[u8] = br#"{"id":"cbb9ac6f-80c7-4c96-ad55-00e0dadefa5d","status":"running","params":{"width":1344,"height":768,"seconds":5.0,"seed":962578975330414183,"quality":"fast","steps":null},"duration":null,"error":null}"#;
        const SUCCEEDED: &[u8] = br#"{"id":"cbb9ac6f-80c7-4c96-ad55-00e0dadefa5d","status":"success","params":{"width":1344,"height":768,"seconds":5.0,"seed":962578975330414183,"quality":"fast","steps":null},"duration":406.79,"error":null}"#;
        const JOB_ID: &str = "cbb9ac6f-80c7-4c96-ad55-00e0dadefa5d";
        let status_path = format!("/v1/videos/{JOB_ID}");
        let server = TestServer::new(vec![
            expected("POST", "/v1/videos/generate", 200, SUBMITTED)
                .with_body_fragment(b"name=\"prompt\"")
                .with_body_fragment(b"name=\"seconds\"\r\n\r\n5")
                .with_body_fragment(b"name=\"width\"\r\n\r\n1344\r\n")
                .with_body_fragment(b"name=\"height\"\r\n\r\n768\r\n"),
            expected("GET", &status_path, 200, RUNNING),
            expected("GET", &status_path, 200, SUCCEEDED),
            expected("GET", &format!("{status_path}/file"), 200, &mp4()),
        ]);
        let video = generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(None) },
            |_| {},
            None,
        )
        .await
        .unwrap();
        assert_eq!(video.job_id, JOB_ID);
        assert_eq!(tokio::fs::read(&video.path).await.unwrap(), mp4());
        tokio::fs::remove_file(video.path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn image_video_uses_the_first_frame_multipart_field() {
        let bytes = png_at_resolution(320, 200);
        let server = TestServer::new(vec![
            submission_at_resolution(320, 200)
                .with_body_fragment(b"name=\"first_frame\"; filename=\"first_frame.png\"")
                .with_body_fragment(b"Content-Type: image/png")
                .with_body_fragment(&bytes),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
            expected("GET", "/v1/videos/job-1/file", 200, &mp4()),
        ]);
        let source = validate_source_image(bytes).unwrap();
        let video = generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(Some(source)) },
            |_| {},
            None,
        )
        .await
        .unwrap();
        tokio::fs::remove_file(video.path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn image_video_uses_decoded_dimensions_and_preserves_original_first_frame_bytes() {
        for (width, height, target_width, target_height) in [
            (4000, 3000, 1344, 1008),
            (3000, 4000, 1008, 1344),
            (1200, 1200, 1200, 1200),
            (640, 360, 640, 360),
            (1344, 1000, 1344, 1000),
            (1500, 1001, 1344, 897),
        ] {
            let bytes = png_at_resolution(width, height);
            let server = TestServer::new(vec![
                submission_at_resolution(target_width, target_height).with_body_fragment(&bytes),
                expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
                expected("GET", "/v1/videos/job-1/file", 200, &mp4()),
            ]);
            let source = validate_source_image(bytes).unwrap();
            let video = generate_with_settings(
                &settings(&server),
                request(),
                async { Ok(Some(source)) },
                |_| {},
                None,
            )
            .await
            .unwrap();
            tokio::fs::remove_file(video.path).await.unwrap();
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn text_video_sends_custom_portrait_default_dimensions() {
        let server = TestServer::new(vec![
            submission_at_resolution(720, 1280),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
            expected("GET", "/v1/videos/job-1/file", 200, &mp4()),
        ]);
        let mut config = settings(&server);
        config.width = 720;
        config.height = 1280;
        let video = generate_with_settings(&config, request(), async { Ok(None) }, |_| {}, None)
            .await
            .unwrap();
        tokio::fs::remove_file(video.path).await.unwrap();
        server.join().unwrap();
    }

    #[test]
    fn image_video_resolution_caps_long_edge_in_either_default_orientation() {
        for defaults in [
            VideoResolution {
                width: 1344,
                height: 768,
            },
            VideoResolution {
                width: 768,
                height: 1344,
            },
        ] {
            assert_eq!(video_resolution(defaults, None), defaults);
            for ((width, height), (target_width, target_height)) in [
                ((4000, 3000), (1344, 1008)),
                ((3000, 4000), (1008, 1344)),
                ((2000, 2000), (1344, 1344)),
                ((1200, 1200), (1200, 1200)),
                ((1344, 1000), (1344, 1000)),
                ((1000, 1344), (1000, 1344)),
                ((1500, 1001), (1344, 897)),
                ((1001, 1500), (897, 1344)),
                ((8192, 1), (1344, 1)),
                ((1, 8192), (1, 1344)),
            ] {
                assert_eq!(
                    video_resolution(defaults, Some(VideoResolution { width, height })),
                    VideoResolution {
                        width: target_width,
                        height: target_height
                    }
                );
            }
        }
        // Exactly half a pixel rounds upwards, rather than truncating.
        assert_eq!(
            video_resolution(
                VideoResolution {
                    width: 3,
                    height: 2
                },
                Some(VideoResolution {
                    width: 4,
                    height: 2
                })
            ),
            VideoResolution {
                width: 3,
                height: 2
            }
        );
    }

    #[tokio::test]
    async fn submission_is_never_retried_even_on_a_transient_failure() {
        for status in [401, 503] {
            let server = TestServer::new(vec![expected(
                "POST",
                "/v1/videos/generate",
                status,
                b"synthetic-vid2-key private detail",
            )]);
            let (result, events) =
                crate::utils::log_capture::capture_json_events_async(generate_with_settings(
                    &settings(&server),
                    request(),
                    async { Ok(None) },
                    |_| {},
                    None,
                ))
                .await;
            let error = result.unwrap_err();
            assert!(error.to_string().contains(&status.to_string()));
            assert!(!error.to_string().contains("synthetic-vid2-key"));
            assert!(!error.to_string().contains("private detail"));
            let logs = serde_json::to_string(&events).unwrap();
            assert!(!logs.contains("synthetic-vid2-key"));
            assert!(!logs.contains("private detail"));
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn submit_rejection_logs_structured_reason_with_secrets_redacted_and_bounded() {
        const ACTIONABLE: &str = "Synthetic first_frame validation failure";
        const PROMPT: &str = "PRIVATE_TEST_PROMPT_DO_NOT_LOG https://private.example.test/input KEEP_THIS_PROMPT_TAIL_PRIVATE";
        const ACTIVE_KEY: &str = "active-submit-test-key";
        let configured_secret = CONFIG.telegram.bot_token.trim();
        let private_url =
            format!("https://api.telegram.org/file/bot{configured_secret}/private.jpg");
        let mut message = format!(
            "{ACTIONABLE}; configured={configured_secret}; prompt={PROMPT}; source={private_url}; "
        );
        message.push_str(&"x".repeat(500usize.saturating_sub(message.chars().count())));
        message.push_str(ACTIVE_KEY);
        message.push_str(&"z".repeat(2048));
        let source_bytes = png_at_resolution(1600, 1000);
        let private_image =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &source_bytes);
        for body in [
            json!({"detail": message, "input": {"prompt": PROMPT, "first_frame": private_image}}),
            json!({"error": {"message": message, "input": private_image}}),
            json!({"message": message, "prompt": PROMPT}),
            json!({"detail": [{"loc": ["body", "first_frame"], "msg": message, "input": {"prompt": PROMPT, "first_frame": private_image}}]}),
        ] {
            let server = TestServer::new(vec![ExpectedRequest::new(
                "POST",
                "/v1/videos/generate",
                response_with_status(400, serde_json::to_vec(&body).unwrap()),
            )
            .with_header("x-api-key", ACTIVE_KEY)]);
            let mut config = settings(&server);
            config.api_key = format!(" {ACTIVE_KEY} ");
            let source = validate_source_image(source_bytes.clone()).unwrap();
            let (result, events) = crate::utils::log_capture::capture_json_events_async(submit(
                &config,
                PROMPT,
                VideoResolution {
                    width: 1344,
                    height: 840,
                },
                Some(source),
            ))
            .await;
            server.join().unwrap();
            let error = result.unwrap_err().to_string();
            assert!(error.contains("400"));
            assert!(!error.contains(ACTIONABLE), "user error stays generic");
            let reason = events
                .iter()
                .find_map(|event| event["fields"]["reason"].as_str());
            assert!(
                reason.is_some_and(|reason| reason.contains(ACTIONABLE)),
                "submission400 must expose its structured operator reason"
            );
            assert!(reason.unwrap().chars().count() <= 512);
            assert!(events.iter().any(|event| event["level"] == "ERROR"
                && event["fields"]["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains(ACTIONABLE))));
            assert!(events
                .iter()
                .any(|event| event["fields"]["source_width"] == 1600
                    && event["fields"]["source_height"] == 1000
                    && event["fields"]["target_width"] == 1344
                    && event["fields"]["target_height"] == 840
                    && event["fields"]["source_mime"] == "image/png"
                    && event["fields"]["source_bytes"] == source_bytes.len()));
            let logs = serde_json::to_string(&events).unwrap();
            assert!(!logs.contains(ACTIVE_KEY));
            assert!(
                !logs.contains("active-submit"),
                "partial credentials must not leak across truncation"
            );
            assert!(!logs.contains(configured_secret));
            assert!(!logs.contains(PROMPT));
            assert!(!logs.contains("PRIVATE_TEST_PROMPT_DO_NOT_LOG"));
            assert!(!logs.contains("KEEP_THIS_PROMPT_TAIL_PRIVATE"));
            assert!(!logs.contains("api.telegram.org/file"));
            assert!(!logs.contains("private.jpg"));
            assert!(!logs.contains(&private_image));
        }
    }

    #[tokio::test]
    async fn submit_rejections_bound_body_reads_and_never_log_unstructured_input() {
        let oversized = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            JSON_MAX_BYTES + 1
        )
        .into_bytes();
        for (response, expected_reason) in [
            (oversized, "diagnostic limit"),
            (
                response_with_status(400, b"PRIVATE_UNSTRUCTURED_INPUT".to_vec()),
                "no structured error reason",
            ),
            (
                response_with_status(
                    400,
                    br#"{"detail":{"input":"PRIVATE_VALIDATION_INPUT"}}"#.to_vec(),
                ),
                "no structured error reason",
            ),
        ] {
            let server = TestServer::new(vec![ExpectedRequest::new(
                "POST",
                "/v1/videos/generate",
                response,
            )]);
            let config = settings(&server);
            let (result, events) = crate::utils::log_capture::capture_json_events_async(submit(
                &config,
                "private prompt",
                VideoResolution {
                    width: 1344,
                    height: 768,
                },
                None,
            ))
            .await;
            server.join().unwrap();
            assert!(result.unwrap_err().to_string().contains("400"));
            let reason = events
                .iter()
                .find_map(|event| event["fields"]["reason"].as_str())
                .unwrap();
            assert!(reason.contains(expected_reason));
            assert!(reason.chars().count() <= 512);
            let logs = serde_json::to_string(&events).unwrap();
            assert!(!logs.contains("PRIVATE_UNSTRUCTURED_INPUT"));
            assert!(!logs.contains("PRIVATE_VALIDATION_INPUT"));
        }
    }

    #[tokio::test]
    async fn submit_timeout_never_repeats_the_creation_request() {
        let server = TestServer::new(vec![submission().delayed(Duration::from_millis(1200))]);
        let mut config = settings(&server);
        config.request_timeout_secs = 1;
        let error = submit(
            &config,
            request().prompt,
            VideoResolution {
                width: 1344,
                height: 768,
            },
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("submission: request failed"));
        server.join_allowing_client_disconnect().unwrap();
    }

    #[tokio::test]
    async fn transient_status_and_download_errors_retry_gets_without_resubmitting() {
        let server = TestServer::new(vec![
            submission(),
            ExpectedRequest::new(
                "GET",
                "/v1/videos/job-1",
                response_with_headers(503, &[("retry-after", "0")], b"busy".to_vec()),
            )
            .with_header("x-api-key", "synthetic-vid2-key"),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
            ExpectedRequest::new(
                "GET",
                "/v1/videos/job-1/file",
                response_with_headers(503, &[("retry-after", "0")], b"busy".to_vec()),
            )
            .with_header("x-api-key", "synthetic-vid2-key"),
            expected("GET", "/v1/videos/job-1/file", 200, &mp4()),
        ]);
        let video = generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(None) },
            |_| {},
            None,
        )
        .await
        .unwrap();
        tokio::fs::remove_file(video.path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn get_retry_budget_is_three_and_authentication_failure_is_permanent() {
        for status in [401, 503] {
            let attempts = if status == 401 { 1 } else { 3 };
            let mut requests = vec![submission()];
            requests.extend((0..attempts).map(|_| {
                ExpectedRequest::new(
                    "GET",
                    "/v1/videos/job-1",
                    response_with_headers(status, &[("retry-after", "0")], b"failed".to_vec()),
                )
                .with_header("x-api-key", "synthetic-vid2-key")
            }));
            let server = TestServer::new(requests);
            assert!(generate_with_settings(
                &settings(&server),
                request(),
                async { Ok(None) },
                |_| {},
                None
            )
            .await
            .is_err());
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn invalid_submission_and_status_responses_fail_without_downloading() {
        for body in [
            b"invalid JSON".as_slice(),
            br#"{}"#,
            br#"{"id":1}"#,
            br#"{"id":""}"#,
            br#"{"id":".."}"#,
        ] {
            let server = TestServer::new(vec![expected("POST", "/v1/videos/generate", 200, body)]);
            assert!(generate_with_settings(
                &settings(&server),
                request(),
                async { Ok(None) },
                |_| {},
                None
            )
            .await
            .is_err());
            server.join().unwrap();
        }
        for body in [
            b"invalid JSON".as_slice(),
            br#"{}"#,
            br#"{"status":"unexpected"}"#,
            br#"{"status":"error","error":"synthetic-vid2-key"}"#,
        ] {
            let server = TestServer::new(vec![
                submission(),
                expected("GET", "/v1/videos/job-1", 200, body),
            ]);
            let error = generate_with_settings(
                &settings(&server),
                request(),
                async { Ok(None) },
                |_| {},
                None,
            )
            .await
            .unwrap_err();
            assert!(!error.to_string().contains("synthetic-vid2-key"));
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn redirects_are_not_followed_with_credentials() {
        let destination = TestServer::expect_no_requests();
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1/videos/generate",
            redirect_response(destination.url("/steal").as_str()),
        )
        .with_header("x-api-key", "synthetic-vid2-key")]);
        assert!(generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(None) },
            |_| {},
            None
        )
        .await
        .is_err());
        server.join().unwrap();
        destination.join().unwrap();
    }

    #[tokio::test]
    async fn source_download_failure_prevents_submission() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/frame",
            response_with_status(404, b"missing".to_vec()),
        )
        .without_header("x-api-key")]);
        let url = server.url("/frame").to_string();
        let error = generate_with_settings(
            &settings(&server),
            request(),
            async { download_source_image(&url, 5).await.map(Some) },
            |_| {},
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("first frame download"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn generation_deadline_covers_source_retrieval() {
        let server = TestServer::expect_no_requests();
        let mut config = settings(&server);
        config.generation_timeout_secs = 1;
        let error = generate_with_settings(
            &config,
            request(),
            async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(None)
            },
            |_| {},
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Vid2Error::Timeout));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn generation_deadline_covers_polling_waits() {
        let server = TestServer::new(vec![
            submission(),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"queued"}"#),
        ]);
        let mut config = settings(&server);
        config.generation_timeout_secs = 1;
        config.poll_interval_secs = 2;
        assert!(matches!(
            generate_with_settings(&config, request(), async { Ok(None) }, |_| {}, None).await,
            Err(Vid2Error::Timeout)
        ));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn generation_deadline_interrupts_a_slow_download() {
        let server = TestServer::new(vec![
            submission(),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
            expected("GET", "/v1/videos/job-1/file", 200, &mp4())
                .delayed(Duration::from_millis(1200)),
        ]);
        let mut config = settings(&server);
        config.generation_timeout_secs = 1;
        assert!(matches!(
            generate_with_settings(&config, request(), async { Ok(None) }, |_| {}, None).await,
            Err(Vid2Error::Timeout)
        ));
        server.join_allowing_client_disconnect().unwrap();
    }

    #[tokio::test]
    async fn oversized_json_and_video_bodies_are_rejected() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1/videos/generate",
            response_with_content_length(JSON_MAX_BYTES + 1, vec![]),
        )]);
        let error = generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(None) },
            |_| {},
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("65536-byte"));
        server.join().unwrap();
        let server = TestServer::new(vec![
            submission(),
            expected("GET", "/v1/videos/job-1", 200, br#"{"status":"success"}"#),
            ExpectedRequest::new(
                "GET",
                "/v1/videos/job-1/file",
                response_with_content_length(VIDEO_MAX_BYTES + 1, vec![]),
            ),
        ]);
        let error = generate_with_settings(
            &settings(&server),
            request(),
            async { Ok(None) },
            |_| {},
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("50000000-byte"));
        server.join().unwrap();
    }

    #[test]
    fn source_images_must_decode_and_use_supported_formats() {
        assert!(validate_source_image(png()).is_ok());
        assert!(validate_source_image(b"\x89PNG\r\n\x1a\ntruncated".to_vec()).is_err());
        assert!(validate_source_image(b"GIF89a".to_vec()).is_err());
        assert!(validate_source_image(vec![]).is_err());
    }

    #[test]
    fn mp4_validation_rejects_empty_other_formats_and_truncated_boxes() {
        assert!(validate_mp4(&mp4()).is_ok());
        for data in [
            vec![],
            b"<html>error</html>".to_vec(),
            mp4()[..24].to_vec(),
            mp4()[..35].to_vec(),
        ] {
            assert!(validate_mp4(&data).is_err());
        }
    }

    #[test]
    fn endpoints_preserve_base_paths_and_encode_untrusted_job_ids() {
        assert_eq!(
            endpoint("https://example.test:8443/api/", None, false)
                .unwrap()
                .as_str(),
            "https://example.test:8443/api/v1/videos/generate"
        );
        assert_eq!(
            endpoint("https://example.test", Some("../a?b#c"), true)
                .unwrap()
                .as_str(),
            "https://example.test/v1/videos/..%2Fa%3Fb%23c/file"
        );
        assert_eq!(
            endpoint("https://example.test", Some("%2e%2e"), false)
                .unwrap()
                .as_str(),
            "https://example.test/v1/videos/%252e%252e"
        );
        for id in ["", " ", ".", "..", "bad\njob"] {
            assert!(endpoint("https://example.test", Some(id), false).is_err());
        }
        let path = output_path(Path::new("data/media/vid2"), &request(), 1);
        assert!(path.starts_with("data/media/vid2"));
        assert_eq!(path.extension().unwrap(), "mp4");
        assert!(!path.to_string_lossy().contains("a fox"));
    }

    #[tokio::test]
    async fn a_generation_writes_one_audit_outcome_without_token_counts() {
        let db = init_test_db("vid2-audit").await;
        for success in [true, false] {
            let invocation = db
                .insert_llm_invocation(LlmInvocationInsert {
                    trigger_kind: "command".into(),
                    trigger_name: "vid2".into(),
                    chat_id: -100123,
                    user_id: Some(1),
                    username: None,
                    message_id: 42,
                    reply_to_message_id: None,
                    message_text: Some("/vid2 a fox in snow".into()),
                    created_at: chrono::Utc::now(),
                })
                .await
                .unwrap();
            let mut requests = vec![
                if success {
                    submission_at_resolution(1344, 672)
                } else {
                    submission()
                },
                expected(
                    "GET",
                    "/v1/videos/job-1",
                    200,
                    if success {
                        br#"{"status":"success"}"#
                    } else {
                        br#"{"status":"error"}"#
                    },
                ),
            ];
            if success {
                requests.push(expected("GET", "/v1/videos/job-1/file", 200, &mp4()));
            }
            let server = TestServer::new(requests);
            let audit = LlmAuditContext::new(db.clone(), invocation);
            let source =
                success.then(|| validate_source_image(png_at_resolution(2000, 1000)).unwrap());
            let result = generate_with_settings(
                &settings(&server),
                request(),
                async { Ok(source) },
                |_| {},
                Some(&audit),
            )
            .await;
            if let Ok(video) = result {
                tokio::fs::remove_file(video.path).await.unwrap();
            }
            let rows: Vec<AuditRow> = sqlx::query_as("SELECT status, response_id, total_tokens, duration_ms, error_summary FROM llm_requests WHERE invocation_id = ?")
                .bind(invocation).fetch_all(db.pool()).await.unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].0, if success { "success" } else { "error" });
            assert_eq!(rows[0].2, None);
            assert!(rows[0].3 >= 0);
            if success {
                assert_eq!(rows[0].1.as_deref(), Some("job-1"));
                let raw: String = sqlx::query_scalar(
                    "SELECT raw_usage_json FROM llm_requests WHERE invocation_id = ?",
                )
                .bind(invocation)
                .fetch_one(db.pool())
                .await
                .unwrap();
                let usage: Value = serde_json::from_str(&raw).unwrap();
                assert_eq!(usage["width"], 1344);
                assert_eq!(usage["height"], 672);
            } else {
                assert!(rows[0].4.as_deref().unwrap().contains("job-1"));
            }
            server.join().unwrap();
        }
        db.shutdown().await;
    }
}

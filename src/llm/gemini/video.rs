//! `/vid` video generation: Gemini Omni through the Interactions API.
//!
//! One synchronous `POST /v1beta/interactions` renders the clip. The request
//! asks for URI delivery, because a clip at the default size is larger than an
//! inline response may be. URI delivery requires the interaction to be stored,
//! so the stored copy is deleted again once the request settles.

use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};
use tracing::{info, warn};

use super::{
    ensure_gemini_api_available, gemini_mime_for_file, get_file_metadata_at, kind_label,
    redact_gemini_api_key, upload_file_bytes_at, wait_for_file_active_at, GEMINI_API_BASE,
    GEMINI_FILE_REQUEST_TIMEOUT, GEMINI_FILE_RETRY_POLICY, GEMINI_RETRY_POLICY,
};
use crate::config::CONFIG;
use crate::llm::audit::{LlmAuditContext, LlmUsageRecord};
use crate::llm::media::{MediaFile, MediaKind};
use crate::llm::transport::retry::is_retryable_status;
use crate::llm::transport::{
    call_with_retry, read_body_limited, read_json, usage, LlmCall, ProviderError,
};
use crate::utils::http::get_http_client;
use crate::utils::text::truncate_with_ellipsis;

/// Reference images one request may carry.
pub const VIDEO_MAX_INPUT_IMAGES: usize = 6;
/// Reference videos one request may carry; editing and extension take one.
pub const VIDEO_MAX_INPUT_VIDEOS: usize = 3;

/// Rendering is synchronous: the response arrives when the clip is ready,
/// about 45 seconds for the default ten seconds of 720p.
const VIDEO_GENERATION_TIMEOUT: Duration = Duration::from_secs(300);
const VIDEO_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);
/// Fifteen seconds of 720p is tens of MiB at most; refuse runaway bodies.
const VIDEO_MAX_BYTES: usize = 256 * 1024 * 1024;
const VIDEO_ERROR_DETAIL_CHARS: usize = 400;

#[derive(Debug, Clone, Copy)]
pub struct VideoGenerationRequest<'a> {
    pub prompt: &'a str,
    /// Images and videos in the order the model should read them.
    pub media: &'a [MediaFile],
}

/// A failure whose detail is safe and useful to show the requester: a
/// rejected request, an unusable input, or a response without a video.
#[derive(Debug, thiserror::Error)]
#[error("Video generation failed: {0}")]
pub struct VideoGenerationError(pub String);

#[derive(Debug, PartialEq, Eq)]
enum VideoSource {
    Uri(String),
    Inline(String),
}

pub fn check_video_input_counts(images: usize, videos: usize) -> Result<(), VideoGenerationError> {
    if images > VIDEO_MAX_INPUT_IMAGES {
        return Err(VideoGenerationError(format!(
            "at most {VIDEO_MAX_INPUT_IMAGES} images can be used, but {images} were given"
        )));
    }
    if videos > VIDEO_MAX_INPUT_VIDEOS {
        return Err(VideoGenerationError(format!(
            "at most {VIDEO_MAX_INPUT_VIDEOS} videos can be used, but {videos} were given"
        )));
    }
    Ok(())
}

/// Generate a clip with `GEMINI_VIDEO_MODEL` from a prompt and optional
/// images and videos.
pub async fn generate_video_with_omni(
    request: VideoGenerationRequest<'_>,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<u8>> {
    ensure_gemini_api_available()?;
    let model = normalize_video_model(&CONFIG.gemini.video_model)?;
    generate_video_at(
        GEMINI_API_BASE,
        &CONFIG.gemini.api_key,
        &model,
        request,
        audit_context,
    )
    .await
}

/// The API takes the bare model code; AI Studio exports it as `models/<code>`.
fn normalize_video_model(configured: &str) -> Result<String> {
    let model = configured.trim();
    let model = model.strip_prefix("models/").unwrap_or(model);
    if model.is_empty() {
        return Err(anyhow!("GEMINI_VIDEO_MODEL is not configured"));
    }
    Ok(model.to_string())
}

async fn generate_video_at(
    base: &str,
    api_key: &str,
    model: &str,
    request: VideoGenerationRequest<'_>,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<u8>> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(VideoGenerationError("a text prompt is required".to_string()).into());
    }
    let images = count_kind(request.media, MediaKind::Image);
    let videos = count_kind(request.media, MediaKind::Video);
    check_video_input_counts(images, videos)?;

    let mut media_parts = Vec::with_capacity(request.media.len());
    for (index, file) in request.media.iter().enumerate() {
        media_parts.push(match file.kind {
            MediaKind::Image => image_part(file)?,
            MediaKind::Video => video_part_at(base, api_key, index, file).await?,
            MediaKind::Audio | MediaKind::Document => {
                return Err(VideoGenerationError(
                    "only images and videos can be used as inputs".to_string(),
                )
                .into())
            }
        });
    }

    let payload = build_interaction_payload(model, prompt, media_parts);
    let metadata = json!({
        "input_images": images,
        "input_videos": videos,
    });
    let interaction =
        create_interaction_at(base, api_key, model, &payload, &metadata, audit_context).await?;

    let video = collect_video_at(base, api_key, &interaction).await;
    // URI delivery forces the interaction to be stored. Nothing here reads it
    // back, so do not leave the prompt and input media behind.
    if let Some(id) = interaction.get("id").and_then(Value::as_str) {
        delete_interaction_at(base, api_key, id).await;
    }
    video
}

fn count_kind(media: &[MediaFile], kind: MediaKind) -> usize {
    media.iter().filter(|file| file.kind == kind).count()
}

fn supported_mime_type(file: &MediaFile) -> Result<String, VideoGenerationError> {
    gemini_mime_for_file(file).ok_or_else(|| {
        VideoGenerationError(format!(
            "unsupported {} format ({})",
            kind_label(file.kind),
            file.mime_type
        ))
    })
}

/// Images travel inline, as the API's own image examples do.
fn image_part(file: &MediaFile) -> Result<Value, VideoGenerationError> {
    let mime_type = supported_mime_type(file)?;
    Ok(json!({
        "type": "image",
        "data": general_purpose::STANDARD.encode(file.bytes()),
        "mime_type": mime_type,
    }))
}

/// Videos go through the Files API and are referenced by URI.
async fn video_part_at(base: &str, api_key: &str, index: usize, file: &MediaFile) -> Result<Value> {
    let mime_type = supported_mime_type(file)?;
    let display_name = format!("vid-input-{}", index + 1);
    let uploaded =
        upload_file_bytes_at(base, api_key, &display_name, &mime_type, file.bytes()).await?;
    let uploaded = wait_for_file_active_at(base, api_key, uploaded).await?;
    Ok(json!({ "type": "video", "uri": uploaded.uri }))
}

fn build_interaction_payload(model: &str, prompt: &str, mut media_parts: Vec<Value>) -> Value {
    let input = if media_parts.is_empty() {
        Value::String(prompt.to_string())
    } else {
        media_parts.push(json!({ "type": "text", "text": prompt }));
        Value::Array(media_parts)
    };
    json!({
        "model": model,
        "input": input,
        "response_format": { "type": "video", "delivery": "uri" },
    })
}

async fn create_interaction_at(
    base: &str,
    api_key: &str,
    model: &str,
    payload: &Value,
    metadata: &Value,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Value> {
    let url = format!("{base}/v1beta/interactions");
    let url = url.as_str();
    let call = LlmCall::begin(
        "gemini",
        model,
        "omni_video_generation",
        audit_context,
        Some(metadata),
    )
    .with_redaction(redact_gemini_api_key);
    call_with_retry(
        &call,
        &GEMINI_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .post(url)
                .header("x-goog-api-key", api_key)
                .timeout(VIDEO_GENERATION_TIMEOUT)
                .json(payload))
        },
        |_| {},
        |response| read_json::<Value>(response, "gemini"),
        usage::from_gemini_interaction,
    )
    .await
    .map_err(requester_facing)
}

/// A request the API refused outright says why in terms the requester can act
/// on (a wrong input for the task, a blocked prompt, an unknown model).
/// Transient and server-side failures stay generic.
fn requester_facing(error: ProviderError) -> anyhow::Error {
    match &error {
        ProviderError::Http {
            status,
            message: Some(message),
            ..
        } if status.is_client_error() && !is_retryable_status(*status) => {
            VideoGenerationError(truncate_with_ellipsis(message, VIDEO_ERROR_DETAIL_CHARS)).into()
        }
        _ => error.into(),
    }
}

async fn collect_video_at(base: &str, api_key: &str, interaction: &Value) -> Result<Vec<u8>> {
    match extract_video_output(interaction)? {
        VideoSource::Uri(uri) => download_generated_video_at(base, api_key, &uri).await,
        VideoSource::Inline(data) => Ok(general_purpose::STANDARD.decode(data).map_err(|err| {
            ProviderError::decode("gemini", format!("invalid inline video data: {err}"), false)
        })?),
    }
}

fn model_output_parts(interaction: &Value) -> impl Iterator<Item = &Value> {
    interaction
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|step| step.get("type").and_then(Value::as_str) == Some("model_output"))
        .filter_map(|step| step.get("content").and_then(Value::as_array))
        .flatten()
}

fn non_empty_str<'a>(part: &'a Value, field: &str) -> Option<&'a str> {
    part.get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

/// The clip is the video part of a `model_output` step. Stored interactions
/// echo input videos in `user_input` steps, which must not be mistaken for it.
fn extract_video_output(interaction: &Value) -> Result<VideoSource, VideoGenerationError> {
    for part in model_output_parts(interaction) {
        if part.get("type").and_then(Value::as_str) != Some("video") {
            continue;
        }
        let source = if let Some(uri) = non_empty_str(part, "uri") {
            VideoSource::Uri(uri.to_string())
        } else if let Some(data) = non_empty_str(part, "data") {
            VideoSource::Inline(data.to_string())
        } else {
            continue;
        };
        return Ok(source);
    }

    let said = model_output_parts(interaction)
        .filter_map(|part| non_empty_str(part, "text"))
        .collect::<Vec<_>>()
        .join(" ");
    let said = said.trim();
    Err(VideoGenerationError(if said.is_empty() {
        format!(
            "the model returned no video (status: {})",
            non_empty_str(interaction, "status").unwrap_or("unknown")
        )
    } else {
        format!(
            "the model returned no video: {}",
            truncate_with_ellipsis(said, VIDEO_ERROR_DETAIL_CHARS)
        )
    }))
}

/// The Files API name inside a generated-video URI, which arrives as
/// `…/v1beta/files/<name>:download?alt=media` (or, per the docs, `files/<name>`).
fn generated_file_name(uri: &str) -> Option<&str> {
    let (_, rest) = uri.rsplit_once("files/")?;
    let name = rest.split([':', '?', '/', '#']).next()?;
    (!name.is_empty()).then_some(name)
}

async fn download_generated_video_at(base: &str, api_key: &str, uri: &str) -> Result<Vec<u8>> {
    // The request carries the API key, so it only ever goes back to the API.
    if !uri.starts_with(&format!("{base}/")) {
        return Err(anyhow!("unexpected generated video location"));
    }

    // The docs ask callers to wait for the file to become ACTIVE. A failed
    // lookup is not fatal: the clip is already paid for, so try the download.
    if let Some(name) = generated_file_name(uri) {
        match get_file_metadata_at(base, api_key, name, GEMINI_FILE_REQUEST_TIMEOUT).await {
            Ok(file) => {
                wait_for_file_active_at(base, api_key, file).await?;
            }
            Err(err) => warn!("Generated video file lookup failed, downloading anyway: {err}"),
        }
    }

    let call = LlmCall::untracked("gemini", "generated-video-download")
        .with_redaction(redact_gemini_api_key);
    let bytes = call_with_retry(
        &call,
        &GEMINI_FILE_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .get(uri)
                .header("x-goog-api-key", api_key)
                .timeout(VIDEO_DOWNLOAD_TIMEOUT))
        },
        |_| {},
        |response| read_body_limited(response, "gemini", VIDEO_MAX_BYTES),
        |_| LlmUsageRecord::default(),
    )
    .await?;
    info!("Generated video downloaded (bytes={})", bytes.len());
    Ok(bytes)
}

/// Best effort: a leftover interaction expires on its own, so a failed
/// deletion must not cost the requester a finished clip.
async fn delete_interaction_at(base: &str, api_key: &str, id: &str) {
    // The id is a path segment taken from a response, so let `Url` encode it.
    let url = reqwest::Url::parse(&format!("{base}/v1beta/interactions"))
        .ok()
        .and_then(|mut url| {
            url.path_segments_mut().ok()?.push(id);
            Some(url)
        });
    let Some(url) = url else {
        warn!("Stored video interaction was not deleted: unusable API base URL");
        return;
    };
    let url = &url;
    let call =
        LlmCall::untracked("gemini", "interaction-delete").with_redaction(redact_gemini_api_key);
    let deleted = call_with_retry(
        &call,
        &GEMINI_FILE_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .delete(url.clone())
                .header("x-goog-api-key", api_key)
                .timeout(GEMINI_FILE_REQUEST_TIMEOUT))
        },
        |_| {},
        |response| async move {
            // Drain the small `{}` body so the connection can be reused.
            let _ = read_body_limited(response, "gemini", 64 * 1024).await;
            Ok(())
        },
        |_| LlmUsageRecord::default(),
    )
    .await;
    if let Err(err) = deleted {
        warn!("Stored video interaction was not deleted: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        response_with_headers, response_with_status, ExpectedRequest, TestServer,
    };

    fn image(mime_type: &str, bytes: &[u8]) -> MediaFile {
        MediaFile::new(
            bytes.to_vec(),
            mime_type.to_string(),
            MediaKind::Image,
            None,
        )
    }

    fn text_request(prompt: &str) -> VideoGenerationRequest<'_> {
        VideoGenerationRequest { prompt, media: &[] }
    }

    fn base_of(server: &TestServer) -> String {
        server
            .base_url()
            .to_string()
            .trim_end_matches('/')
            .to_string()
    }

    fn json_response(status: u16, body: Value) -> Vec<u8> {
        response_with_status(status, body.to_string().into_bytes())
    }

    fn delete_interaction(id: &str) -> ExpectedRequest {
        ExpectedRequest::new(
            "DELETE",
            &format!("/v1beta/interactions/{id}"),
            json_response(200, json!({})),
        )
        .with_header("x-goog-api-key", "test-key")
    }

    #[test]
    fn input_counts_are_capped_at_the_documented_reference_limits() {
        assert!(check_video_input_counts(0, 0).is_ok());
        assert!(check_video_input_counts(6, 3).is_ok());
        let too_many_images = check_video_input_counts(7, 0).unwrap_err();
        assert!(too_many_images.0.contains("6 images"), "{too_many_images}");
        let too_many_videos = check_video_input_counts(1, 4).unwrap_err();
        assert!(too_many_videos.0.contains("3 videos"), "{too_many_videos}");
    }

    #[test]
    fn configured_model_drops_the_models_prefix_and_must_not_be_empty() {
        assert_eq!(
            normalize_video_model(" models/gemini-omni-1.2-flash ").unwrap(),
            "gemini-omni-1.2-flash"
        );
        assert_eq!(
            normalize_video_model("gemini-omni-1.2-flash").unwrap(),
            "gemini-omni-1.2-flash"
        );
        assert!(normalize_video_model("  ").is_err());
        assert!(normalize_video_model("models/").is_err());
    }

    #[test]
    fn a_text_only_prompt_is_sent_as_a_string_with_uri_delivery() {
        let payload = build_interaction_payload("gemini-omni-1.2-flash", "a fox", vec![]);
        assert_eq!(
            payload,
            json!({
                "model": "gemini-omni-1.2-flash",
                "input": "a fox",
                "response_format": { "type": "video", "delivery": "uri" },
            })
        );
    }

    #[test]
    fn media_precedes_the_prompt() {
        let parts = vec![
            json!({ "type": "video", "uri": "https://files/v1" }),
            image_part(&image("image/jpg", b"\xFF\xD8\xFFjpeg")).unwrap(),
        ];
        let payload = build_interaction_payload("m", "extend it", parts);
        assert_eq!(
            payload["input"],
            json!([
                { "type": "video", "uri": "https://files/v1" },
                { "type": "image", "data": "/9j/anBlZw==", "mime_type": "image/jpeg" },
                { "type": "text", "text": "extend it" },
            ])
        );
    }

    #[test]
    fn an_unsupported_image_format_is_refused_rather_than_dropped() {
        let error = image_part(&image("image/svg+xml", b"<svg/>")).unwrap_err();
        assert!(error.0.contains("image/svg+xml"), "{error}");
    }

    #[test]
    fn the_video_is_read_from_model_output_only() {
        let by_uri = json!({
            "status": "completed",
            "steps": [
                { "type": "user_input", "content": [
                    { "type": "video", "uri": "https://files/input" },
                    { "type": "text", "text": "extend it" }
                ] },
                { "type": "thought", "signature": "sig" },
                { "type": "model_output", "content": [
                    { "type": "video", "mime_type": "video/mp4", "uri": "https://files/out" }
                ] }
            ]
        });
        assert_eq!(
            extract_video_output(&by_uri).unwrap(),
            VideoSource::Uri("https://files/out".to_string())
        );

        let inline = json!({
            "status": "completed",
            "steps": [{ "type": "model_output", "content": [
                { "type": "video", "mime_type": "video/mp4", "data": "QUJD" }
            ] }]
        });
        assert_eq!(
            extract_video_output(&inline).unwrap(),
            VideoSource::Inline("QUJD".to_string())
        );
    }

    #[test]
    fn a_response_without_a_video_reports_the_model_text_or_the_status() {
        let refused = json!({
            "status": "completed",
            "steps": [{ "type": "model_output", "content": [
                { "type": "text", "text": "I can't create that video." }
            ] }]
        });
        let error = extract_video_output(&refused).unwrap_err();
        assert!(error.0.contains("I can't create that video."), "{error}");

        let failed = json!({ "status": "failed", "steps": [] });
        let error = extract_video_output(&failed).unwrap_err();
        assert!(error.0.contains("failed"), "{error}");
    }

    #[test]
    fn the_generated_file_name_comes_from_either_uri_form() {
        assert_eq!(
            generated_file_name(
                "https://generativelanguage.googleapis.com/v1beta/files/sd4mgbhzh0fi:download?alt=media"
            ),
            Some("sd4mgbhzh0fi")
        );
        assert_eq!(generated_file_name("files/abc-123"), Some("abc-123"));
        assert_eq!(generated_file_name("https://example.com/video.mp4"), None);
        assert_eq!(generated_file_name("https://example.com/files/"), None);
    }

    #[tokio::test]
    async fn generation_returns_the_clip_and_deletes_the_stored_interaction() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/v1beta/interactions",
                json_response(
                    200,
                    json!({
                        "id": "v1_abc",
                        "status": "completed",
                        "steps": [
                            { "type": "thought", "signature": "sig" },
                            { "type": "model_output", "content": [
                                { "type": "video", "mime_type": "video/mp4", "data": "TVA0QllURVM=" }
                            ] }
                        ],
                        "usage": { "total_input_tokens": 14, "total_output_tokens": 5793 }
                    }),
                ),
            )
            .with_header("x-goog-api-key", "test-key")
            .with_body_fragment(br#""input":"a fox""#)
            .with_body_fragment(br#""delivery":"uri""#)
            .with_body_fragment(br#""model":"gemini-omni-1.2-flash""#),
            delete_interaction("v1_abc"),
        ]);

        let video = generate_video_at(
            &base_of(&server),
            "test-key",
            "gemini-omni-1.2-flash",
            text_request("  a fox  "),
            None,
        )
        .await
        .expect("the clip is returned");

        assert_eq!(video, b"MP4BYTES");
        server.join().expect("generation then deletion");
    }

    #[tokio::test]
    async fn a_stored_interaction_is_deleted_even_when_it_holds_no_video() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/v1beta/interactions",
                json_response(
                    200,
                    json!({
                        "id": "v1_refused",
                        "status": "completed",
                        "steps": [{ "type": "model_output", "content": [
                            { "type": "text", "text": "I can't create that video." }
                        ] }]
                    }),
                ),
            ),
            delete_interaction("v1_refused"),
        ]);

        let error = generate_video_at(&base_of(&server), "test-key", "m", text_request("x"), None)
            .await
            .unwrap_err();

        let error = error
            .downcast_ref::<VideoGenerationError>()
            .expect("a user-facing error");
        assert!(error.0.contains("I can't create that video."), "{error}");
        server.join().expect("generation then deletion");
    }

    #[tokio::test]
    async fn a_rejected_request_surfaces_the_provider_message_without_retrying() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1beta/interactions",
            json_response(
                400,
                json!({ "error": {
                    "message": "Exactly one input video is required for edit task.",
                    "code": "invalid_request"
                } }),
            ),
        )]);

        let error = generate_video_at(&base_of(&server), "test-key", "m", text_request("x"), None)
            .await
            .unwrap_err();

        assert_eq!(
            error
                .downcast_ref::<VideoGenerationError>()
                .expect("a user-facing error")
                .0,
            "Exactly one input video is required for edit task."
        );
        server.join().expect("one request only");
    }

    #[tokio::test]
    async fn an_empty_prompt_or_too_many_inputs_never_reaches_the_api() {
        let unreachable = "http://127.0.0.1:9";
        let empty = generate_video_at(unreachable, "k", "m", text_request("   "), None)
            .await
            .unwrap_err();
        assert!(empty.is::<VideoGenerationError>(), "{empty}");

        let images: Vec<MediaFile> = (0..7).map(|_| image("image/png", b"png")).collect();
        let crowded = generate_video_at(
            unreachable,
            "k",
            "m",
            VideoGenerationRequest {
                prompt: "x",
                media: &images,
            },
            None,
        )
        .await
        .unwrap_err();
        assert!(crowded.is::<VideoGenerationError>(), "{crowded}");
    }

    #[tokio::test]
    async fn a_uri_clip_is_downloaded_once_its_file_is_active() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "GET",
                "/v1beta/files/gen1",
                json_response(
                    200,
                    json!({ "name": "files/gen1", "uri": "u", "state": "ACTIVE" }),
                ),
            )
            .with_header("x-goog-api-key", "test-key"),
            ExpectedRequest::new(
                "GET",
                "/v1beta/files/gen1:download?alt=media",
                response_with_headers(200, &[("Content-Type", "video/mp4")], b"MP4BYTES".to_vec()),
            )
            .with_header("x-goog-api-key", "test-key"),
        ]);
        let base = base_of(&server);

        let bytes = download_generated_video_at(
            &base,
            "test-key",
            &format!("{base}/v1beta/files/gen1:download?alt=media"),
        )
        .await
        .expect("the clip downloads");

        assert_eq!(bytes, b"MP4BYTES");
        server.join().expect("metadata then download");
    }

    #[tokio::test]
    async fn the_api_key_is_never_sent_to_another_host() {
        for uri in [
            "https://evil.example/v1beta/files/x:download?alt=media",
            "https://generativelanguage.googleapis.com.evil.example/v1beta/files/x",
            "http://generativelanguage.googleapis.com/v1beta/files/x",
        ] {
            let error = download_generated_video_at(GEMINI_API_BASE, "secret-key", uri)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("unexpected"), "{uri}: {error}");
        }
    }
}

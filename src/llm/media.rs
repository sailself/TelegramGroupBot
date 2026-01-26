use std::time::Duration;

use reqwest::StatusCode;
use tracing::{error, warn};

use crate::utils::http::get_http_client;

pub fn detect_mime_type(data: &[u8]) -> Option<String> {
    if data.len() > 12 {
        let ftyp = &data[4..12];
        if ftyp.starts_with(b"ftyp") {
            let brand = &ftyp[4..8];
            if brand == b"heic" || brand == b"heif" || brand == b"hevc" {
                return Some("image/heic".to_string());
            }
        }
    }

    infer::get(data).map(|kind| kind.mime_type().to_string())
}

const MEDIA_DOWNLOAD_MAX_ATTEMPTS: usize = 3;
const MEDIA_DOWNLOAD_BASE_DELAY_MS: u64 = 400;
const MEDIA_DOWNLOAD_ERROR_BODY_LIMIT: usize = 800;

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn should_retry_status(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
}

fn should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

pub async fn download_media(url: &str) -> Option<Vec<u8>> {
    let client = get_http_client();
    for attempt in 0..MEDIA_DOWNLOAD_MAX_ATTEMPTS {
        let response = match client.get(url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                warn!(
                    "Failed to fetch media {url}: {err} (timeout={}, connect={}, status={:?}, attempt={}/{})",
                    err.is_timeout(),
                    err.is_connect(),
                    err.status(),
                    attempt + 1,
                    MEDIA_DOWNLOAD_MAX_ATTEMPTS
                );
                if !should_retry_error(&err) || attempt + 1 == MEDIA_DOWNLOAD_MAX_ATTEMPTS {
                    return None;
                }
                let delay = Duration::from_millis(MEDIA_DOWNLOAD_BASE_DELAY_MS << attempt);
                tokio::time::sleep(delay).await;
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!(
                "Media download failed for {url} with status {}: {}",
                status,
                truncate_for_log(&body, MEDIA_DOWNLOAD_ERROR_BODY_LIMIT)
            );
            if !should_retry_status(status) || attempt + 1 == MEDIA_DOWNLOAD_MAX_ATTEMPTS {
                return None;
            }
            let delay = Duration::from_millis(MEDIA_DOWNLOAD_BASE_DELAY_MS << attempt);
            tokio::time::sleep(delay).await;
            continue;
        }

        return match response.bytes().await {
            Ok(bytes) => Some(bytes.to_vec()),
            Err(err) => {
                error!(
                    "Failed to read media bytes {url}: {err} (attempt={}/{})",
                    attempt + 1,
                    MEDIA_DOWNLOAD_MAX_ATTEMPTS
                );
                if attempt + 1 == MEDIA_DOWNLOAD_MAX_ATTEMPTS {
                    None
                } else {
                    let delay = Duration::from_millis(MEDIA_DOWNLOAD_BASE_DELAY_MS << attempt);
                    tokio::time::sleep(delay).await;
                    continue;
                }
            }
        };
    }

    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
    Document,
}

#[derive(Debug, Clone)]
pub struct MediaFile {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub kind: MediaKind,
    pub display_name: Option<String>,
}

impl MediaFile {
    pub fn new(
        bytes: Vec<u8>,
        mime_type: String,
        kind: MediaKind,
        display_name: Option<String>,
    ) -> Self {
        Self {
            bytes,
            mime_type,
            kind,
            display_name,
        }
    }
}

pub fn kind_for_mime(mime_type: &str) -> MediaKind {
    if mime_type.starts_with("image/") {
        MediaKind::Image
    } else if mime_type.starts_with("video/") {
        MediaKind::Video
    } else if mime_type.starts_with("audio/") {
        MediaKind::Audio
    } else {
        MediaKind::Document
    }
}

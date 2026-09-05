use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::Lazy;
use regex::Regex;
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

/// Mask the bot token embedded in Telegram file-download URLs so the URL can
/// be logged safely. Other URLs pass through unchanged.
pub fn redact_url_for_log(url: &str) -> String {
    static TELEGRAM_FILE_TOKEN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(https?://api\.telegram\.org/file/bot)[^/]+/")
            .expect("valid telegram file url regex")
    });
    TELEGRAM_FILE_TOKEN
        .replace(url, "${1}<redacted>/")
        .into_owned()
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
    // Telegram file URLs embed the bot token; never log the raw URL.
    let log_url = redact_url_for_log(url);
    for attempt in 0..MEDIA_DOWNLOAD_MAX_ATTEMPTS {
        let response = match client.get(url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                warn!(
                    "Failed to fetch media {log_url}: {err} (timeout={}, connect={}, status={:?}, attempt={}/{})",
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
                "Media download failed for {log_url} with status {}: {}",
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
                    "Failed to read media bytes {log_url}: {err} (attempt={}/{})",
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
    pub bytes: Arc<Vec<u8>>,
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
            bytes: Arc::new(bytes),
            mime_type,
            kind,
            display_name,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_bot_token_in_telegram_file_urls() {
        let url = "https://api.telegram.org/file/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/photos/file_1.jpg";
        assert_eq!(
            redact_url_for_log(url),
            "https://api.telegram.org/file/bot<redacted>/photos/file_1.jpg"
        );
    }

    #[test]
    fn leaves_urls_without_a_bot_token_unchanged() {
        let url = "https://pbs.twimg.com/media/abc.jpg?name=orig";
        assert_eq!(redact_url_for_log(url), url);
    }
}

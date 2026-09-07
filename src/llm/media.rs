use std::sync::Arc;
use std::time::Duration;

use regex::Regex;
use std::sync::LazyLock;
use tracing::error;

use crate::llm::audit::LlmUsageRecord;
use crate::llm::transport::{call_with_retry, read_body_limited, LlmCall, RetryPolicy};
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

/// Mask the bot token embedded in Telegram file-download URLs so the URL can
/// be logged safely. Other URLs pass through unchanged.
pub fn redact_url_for_log(url: &str) -> String {
    static TELEGRAM_FILE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(https?://api\.telegram\.org/file/bot)[^/]+/")
            .expect("valid telegram file url regex")
    });
    TELEGRAM_FILE_TOKEN
        .replace(url, "${1}<redacted>/")
        .into_owned()
}

/// Largest media payload accepted from Telegram or an external host. Telegram's
/// bot API caps file downloads at 20 MiB; the headroom covers external images.
pub const MEDIA_DOWNLOAD_MAX_BYTES: usize = 32 * 1024 * 1024;
const MEDIA_DOWNLOAD_POLICY: RetryPolicy = RetryPolicy::exponential(3, Duration::from_millis(400));

pub async fn download_media(url: &str) -> Option<Vec<u8>> {
    download_media_limited(url, MEDIA_DOWNLOAD_MAX_BYTES).await
}

/// Download `url`, retrying transient failures and refusing bodies above
/// `max_bytes`. Returns `None` after logging the cause; callers treat a
/// missing download as "skip this attachment".
pub async fn download_media_limited(url: &str, max_bytes: usize) -> Option<Vec<u8>> {
    // Telegram file URLs embed the bot token; never log the raw URL (reqwest
    // error text repeats it, hence the redaction on the call).
    let log_url = redact_url_for_log(url);
    let call = LlmCall::untracked("media", &log_url).with_redaction(redact_url_for_log);
    let result = call_with_retry(
        &call,
        &MEDIA_DOWNLOAD_POLICY,
        |_| async move { Ok(get_http_client().get(url)) },
        |_| {},
        |response| read_body_limited(response, "media", max_bytes),
        |_| LlmUsageRecord::default(),
    )
    .await;

    match result {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            error!("Media download failed for {log_url}: {err}");
            None
        }
    }
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
    use crate::tools::twitter_extractor::test_support::{
        response_with_status, ExpectedRequest, TestServer,
    };

    #[tokio::test]
    async fn download_media_limited_refuses_bodies_over_the_cap() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/f",
            response_with_status(200, vec![b'x'; 100]),
        )]);
        let url = server.url("/f").to_string();

        assert!(download_media_limited(&url, 10).await.is_none());
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn download_media_retries_transient_server_errors() {
        let server = TestServer::new(vec![
            ExpectedRequest::new("GET", "/f", response_with_status(503, b"busy".to_vec())),
            ExpectedRequest::new("GET", "/f", response_with_status(200, b"bytes".to_vec())),
        ]);
        let url = server.url("/f").to_string();

        assert_eq!(
            download_media_limited(&url, 1024).await.as_deref(),
            Some(b"bytes".as_slice())
        );
        server.join().expect("both requests were served");
    }

    #[tokio::test]
    async fn download_media_gives_up_on_permanent_errors_without_retrying() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/f",
            response_with_status(404, b"gone".to_vec()),
        )]);
        let url = server.url("/f").to_string();

        assert!(download_media_limited(&url, 1024).await.is_none());
        server.join().expect("exactly one request was served");
    }

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

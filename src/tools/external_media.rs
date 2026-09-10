use std::future::Future;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use reqwest::header::CONTENT_TYPE;
use reqwest::Response;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, warn};
use url::Url;

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, redact_url_for_log, MediaFile, MediaKind};
use crate::tools::telegraph_extractor::TelegraphContent;
use crate::tools::twitter_extractor::{parse_allowed_media_url, TwitterAttachment, TwitterContent};
use crate::utils::http::{get_http_client_no_redirect, parse_https_allowlisted};

#[derive(Clone, Debug)]
pub struct ExternalMediaBudget {
    remaining: Arc<AtomicUsize>,
}

impl ExternalMediaBudget {
    pub fn new(total: usize) -> Self {
        Self {
            remaining: Arc::new(AtomicUsize::new(total)),
        }
    }

    #[cfg(test)]
    pub(crate) fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Acquire)
    }

    pub(crate) fn reservation(&self) -> ExternalMediaReservation {
        ExternalMediaReservation {
            budget: self.clone(),
            reserved: 0,
            committed: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_commit(&self, amount: usize) -> Result<()> {
        let mut reservation = self.reservation();
        if !reservation.try_reserve(amount) {
            bail!("external media budget exhausted")
        }
        reservation.commit()
    }

    fn try_reserve(&self, amount: usize) -> bool {
        if amount == 0 {
            return true;
        }
        let mut current = self.remaining.load(Ordering::Acquire);
        loop {
            if current < amount {
                return false;
            }
            match self.remaining.compare_exchange_weak(
                current,
                current - amount,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }
}

pub(crate) struct ExternalMediaReservation {
    budget: ExternalMediaBudget,
    reserved: usize,
    committed: bool,
}

impl ExternalMediaReservation {
    pub(crate) fn try_reserve(&mut self, amount: usize) -> bool {
        if self.committed || !self.budget.try_reserve(amount) {
            return false;
        }
        self.reserved = self.reserved.saturating_add(amount);
        true
    }

    pub(crate) fn commit(&mut self) -> Result<()> {
        if self.committed {
            return Err(anyhow!("external media reservation already committed"));
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for ExternalMediaReservation {
    fn drop(&mut self) {
        if !self.committed && self.reserved > 0 {
            self.budget
                .remaining
                .fetch_add(self.reserved, Ordering::AcqRel);
        }
    }
}

pub(crate) async fn read_external_media_response(
    mut response: Response,
    per_file_max: usize,
    budget: &ExternalMediaBudget,
) -> Result<Vec<u8>> {
    if response.status().is_redirection() {
        bail!("external media redirects are not allowed")
    }
    if !response.status().is_success() {
        bail!(
            "external media request failed with status {}",
            response.status()
        )
    }
    if per_file_max == 0 {
        bail!("external media per-file limit must be positive")
    }
    if response
        .content_length()
        .is_some_and(|length| length > per_file_max as u64)
    {
        bail!("external media Content-Length exceeds per-file limit")
    }

    let mut reservation = budget.reservation();
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len().saturating_add(chunk.len()) > per_file_max {
            bail!("external media body exceeds per-file limit")
        }
        if !reservation.try_reserve(chunk.len()) {
            bail!("external media command budget exhausted")
        }
        body.extend_from_slice(&chunk);
    }
    reservation.commit()?;
    Ok(body)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExternalMediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaSource {
    Telegraph,
    Twitter,
}

impl MediaSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Telegraph => "telegraph",
            Self::Twitter => "twitter",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExternalMediaRequest {
    pub(crate) index: usize,
    pub(crate) url: String,
    pub(crate) kind: ExternalMediaKind,
    source: MediaSource,
    thumbnail_url: Option<String>,
}

impl ExternalMediaRequest {
    pub(crate) fn thumbnail_fallback(&self) -> Option<Self> {
        self.thumbnail_url.as_ref().map(|url| Self {
            index: self.index,
            url: url.clone(),
            kind: ExternalMediaKind::Image,
            source: self.source,
            thumbnail_url: None,
        })
    }
}

pub(crate) fn twitter_video_request(
    index: usize,
    video_url: &str,
    thumbnail_url: Option<&str>,
) -> ExternalMediaRequest {
    ExternalMediaRequest {
        index,
        url: video_url.to_string(),
        kind: ExternalMediaKind::Video,
        source: MediaSource::Twitter,
        thumbnail_url: thumbnail_url.map(str::to_string),
    }
}

fn telegraph_media_url(raw_url: &str) -> Result<Url> {
    parse_https_allowlisted(
        "telegraph media",
        raw_url,
        Some(&["telegra.ph", "graph.org"]),
    )
}

fn validate_media_url(url: &str, source: MediaSource) -> Result<Url> {
    match source {
        MediaSource::Twitter => parse_allowed_media_url(url),
        MediaSource::Telegraph => telegraph_media_url(url),
    }
}

fn display_name_from_url(url: &str) -> Option<String> {
    let trimmed = url.split('?').next().unwrap_or(url);
    trimmed
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn image_mime_from_url(url: &str) -> Option<&'static str> {
    let lowered = url.to_ascii_lowercase();
    if lowered.contains("format=png") || lowered.ends_with(".png") {
        Some("image/png")
    } else if lowered.contains("format=jpg")
        || lowered.contains("format=jpeg")
        || lowered.ends_with(".jpg")
        || lowered.ends_with(".jpeg")
    {
        Some("image/jpeg")
    } else if lowered.contains("format=webp") || lowered.ends_with(".webp") {
        Some("image/webp")
    } else if lowered.ends_with(".heic") {
        Some("image/heic")
    } else if lowered.ends_with(".heif") {
        Some("image/heif")
    } else {
        None
    }
}

fn video_mime_from_url(url: &str) -> Option<&'static str> {
    let lowered = url.to_ascii_lowercase();
    if lowered.ends_with(".m3u8") {
        Some("application/x-mpegURL")
    } else if lowered.ends_with(".mpd") {
        Some("application/dash+xml")
    } else if lowered.ends_with(".webm") {
        Some("video/webm")
    } else if lowered.ends_with(".mp4") {
        Some("video/mp4")
    } else {
        None
    }
}

fn resolve_video_mime_type(url: &str, _content_type: Option<&str>, bytes: &[u8]) -> String {
    video_mime_from_url(url)
        .map(ToString::to_string)
        .or_else(|| detect_mime_type(bytes))
        .unwrap_or_else(|| "video/mp4".to_string())
}

async fn start_request(request: &ExternalMediaRequest) -> Result<Response> {
    let parsed_url = validate_media_url(&request.url, request.source)?;
    if matches!(request.kind, ExternalMediaKind::Image)
        && request.url.to_ascii_lowercase().contains(".svg")
    {
        bail!("external media URL looks like an SVG image");
    }
    Ok(get_http_client_no_redirect().get(parsed_url).send().await?)
}

/// Runs `start` and turns a transport/validation failure into a logged
/// `warn!` (with the URL redacted) plus `None`, so callers can treat a
/// failed fetch the same way as a rejected response.
async fn start_logged<S, Fut>(start: S, request: ExternalMediaRequest) -> Option<Response>
where
    S: FnOnce(ExternalMediaRequest) -> Fut,
    Fut: Future<Output = Result<Response>>,
{
    let source = request.source.as_str();
    let redacted_url = redact_url_for_log(&request.url);
    match start(request).await {
        Ok(response) => Some(response),
        Err(err) => {
            warn!(source, media_url = %redacted_url, error = %err, "External media download failed");
            None
        }
    }
}

async fn process_response(
    request: &ExternalMediaRequest,
    response: Response,
    budget: &ExternalMediaBudget,
) -> Option<MediaFile> {
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or(value)
                .trim()
                .to_ascii_lowercase()
        });
    if let ExternalMediaKind::Image = request.kind {
        let source = request.source.as_str();
        let content_type =
            content_type.or_else(|| image_mime_from_url(&request.url).map(ToString::to_string));
        let Some(content_type) = content_type else {
            warn!(source, media_url = %redact_url_for_log(&request.url), "Skipping image without Content-Type or URL MIME hint");
            return None;
        };
        if !content_type.starts_with("image/") || content_type == "image/svg+xml" {
            warn!(source, media_url = %redact_url_for_log(&request.url), content_type = %content_type, "Skipping non-image external media");
            return None;
        }
        let bytes = match read_external_media_response(
            response,
            CONFIG.external_media_max_bytes,
            budget,
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(err) => {
                warn!(source, media_url = %redact_url_for_log(&request.url), error = %err, "Skipping image that failed to download");
                return None;
            }
        };
        return Some(MediaFile::new(
            bytes,
            content_type,
            MediaKind::Image,
            display_name_from_url(&request.url),
        ));
    }

    let bytes = match read_external_media_response(
        response,
        CONFIG.external_media_max_bytes,
        budget,
    )
    .await
    {
        Ok(bytes) => bytes,
        Err(err) => {
            debug!(media_url = %redact_url_for_log(&request.url), error = %err, "External video download failed");
            return None;
        }
    };
    let mime_type = resolve_video_mime_type(&request.url, content_type.as_deref(), &bytes);
    Some(MediaFile::new(
        bytes,
        mime_type,
        MediaKind::Video,
        display_name_from_url(&request.url),
    ))
}

async fn process_response_with_fallback<F, Fut>(
    request: ExternalMediaRequest,
    response: Option<Response>,
    budget: &ExternalMediaBudget,
    fallback_start: F,
) -> Option<(usize, MediaFile)>
where
    F: FnOnce(ExternalMediaRequest) -> Fut,
    Fut: std::future::Future<Output = Option<Response>>,
{
    let mut media = match response {
        Some(response) => process_response(&request, response, budget).await,
        None => None,
    };
    if media.is_none() {
        if let Some(fallback) = request.thumbnail_fallback() {
            if let Some(response) = fallback_start(fallback.clone()).await {
                media = process_response(&fallback, response, budget).await;
            }
        }
    }
    media.map(|file| (request.index, file))
}

/// Upper bound on how long a single `collect_external_media` call may take
/// across every request combined, regardless of `fanout`. A caller-side
/// cancellation is handled the same way: dropping this function's `JoinSet`
/// (on timeout, or because our own future got dropped) aborts every worker
/// still in flight.
pub(crate) const EXTERNAL_MEDIA_COLLECTION_TIMEOUT: Duration = Duration::from_secs(60);

/// Downloads every request, bounded to at most `fanout` concurrent fetches
/// and `EXTERNAL_MEDIA_COLLECTION_TIMEOUT` total, returning the successful
/// files in input order (results race independently, so an index travels
/// with each one to restore that order at the end).
async fn collect_external_media<S, Fut>(
    requests: Vec<ExternalMediaRequest>,
    budget: &ExternalMediaBudget,
    fanout: usize,
    start: S,
) -> Vec<MediaFile>
where
    S: Fn(ExternalMediaRequest) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<Response>> + Send + 'static,
{
    if requests.is_empty() {
        return Vec::new();
    }
    let semaphore = Arc::new(Semaphore::new(fanout.max(1)));
    let mut workers = JoinSet::new();
    for request in requests {
        let semaphore = semaphore.clone();
        let start = start.clone();
        let budget = budget.clone();
        workers.spawn(async move {
            let index = request.index;
            let permit = match semaphore.acquire_owned().await {
                Ok(permit) => permit,
                // The semaphore is only ever closed if it were `close()`d,
                // which nothing here does; treat it as "this request did
                // not produce a file" rather than panicking a worker task.
                Err(_) => return (index, None),
            };
            let response = start_logged(start.clone(), request.clone()).await;
            let fallback_start = {
                let start = start.clone();
                move |fallback: ExternalMediaRequest| start_logged(start, fallback)
            };
            let file = process_response_with_fallback(request, response, &budget, fallback_start)
                .await
                .map(|(_, file)| file);
            drop(permit);
            (index, file)
        });
    }

    let mut collected = Vec::new();
    let drain = async {
        while let Some(joined) = workers.join_next().await {
            if let Ok((index, Some(file))) = joined {
                collected.push((index, file));
            }
        }
    };
    // On timeout `workers` (still owned here) is dropped at the end of this
    // function, which aborts every worker still running; the same happens if
    // our own caller cancels this future mid-poll.
    let _ = tokio::time::timeout(EXTERNAL_MEDIA_COLLECTION_TIMEOUT, drain).await;

    collected.sort_by_key(|(index, _)| *index);
    collected.into_iter().map(|(_, file)| file).collect()
}

pub async fn download_telegraph_media(
    contents: &[TelegraphContent],
    max_files: usize,
    budget: &ExternalMediaBudget,
) -> Vec<MediaFile> {
    let mut requests = Vec::new();
    let mut index = 0;
    for content in contents {
        for url in &content.image_urls {
            if requests.len() >= max_files {
                break;
            }
            requests.push(ExternalMediaRequest {
                index,
                url: url.clone(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            });
            index += 1;
        }
        for url in &content.video_urls {
            if requests.len() >= max_files {
                break;
            }
            requests.push(ExternalMediaRequest {
                index,
                url: url.clone(),
                kind: ExternalMediaKind::Video,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            });
            index += 1;
        }
    }
    collect_external_media(
        requests,
        budget,
        CONFIG.external_enrich_fanout,
        |request| async move { start_request(&request).await },
    )
    .await
}

pub async fn download_twitter_media(
    contents: &[TwitterContent],
    max_files: usize,
    budget: &ExternalMediaBudget,
) -> Vec<MediaFile> {
    collect_external_media(
        twitter_media_requests(contents, max_files),
        budget,
        CONFIG.external_enrich_fanout,
        |request| async move { start_request(&request).await },
    )
    .await
}

fn twitter_media_requests(
    contents: &[TwitterContent],
    max_files: usize,
) -> Vec<ExternalMediaRequest> {
    let mut requests = Vec::new();
    let mut index = 0;
    for content in contents {
        for attachment in &content.attachment_plan {
            if requests.len() >= max_files {
                return requests;
            }
            match attachment {
                TwitterAttachment::Image { url } => requests.push(ExternalMediaRequest {
                    index,
                    url: url.clone(),
                    kind: ExternalMediaKind::Image,
                    source: MediaSource::Twitter,
                    thumbnail_url: None,
                }),
                TwitterAttachment::Video { url, thumbnail_url } => {
                    requests.push(twitter_video_request(index, url, thumbnail_url.as_deref()))
                }
            }
            index += 1;
        }
    }
    requests
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread::JoinHandle;

    use super::*;
    use crate::tools::twitter_extractor::model::{build_twitter_content, XMedia, XPost};
    use crate::tools::twitter_extractor::test_support::{redirect_response, TestServer};
    use crate::tools::twitter_extractor::url::XStatusIdentity;
    use crate::utils::http::get_http_client_no_redirect;

    const SIGNAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    fn twitter_content_with_video(thumbnail_url: Option<&str>) -> TwitterContent {
        build_twitter_content(
            &XStatusIdentity {
                id: "123".to_owned(),
                canonical_url: Url::parse("https://x.com/i/status/123").unwrap(),
            },
            XPost {
                id: "123".to_owned(),
                author: None,
                text: "root video".to_owned(),
                created_at: None,
                media: vec![XMedia::Video {
                    url: Some(Url::parse("https://video.twimg.com/video/root.mp4").unwrap()),
                    thumbnail_url: thumbnail_url.map(|url| Url::parse(url).unwrap()),
                    alt_text: None,
                    bitrate: Some(1_000),
                }],
                quote: None,
            },
        )
        .unwrap()
    }

    fn controlled_thirty_chunk_server() -> (
        url::Url,
        tokio::sync::oneshot::Receiver<()>,
        Sender<()>,
        JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let url = url::Url::parse(&format!("http://{address}/media")).unwrap();
        let (ready_sender, ready_receiver) = tokio::sync::oneshot::channel();
        let (release_sender, release_receiver): (Sender<()>, Receiver<()>) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n1E\r\n",
                )
                .unwrap();
            stream.write_all(&[b'a'; 30]).unwrap();
            stream.write_all(b"\r\n").unwrap();
            stream.flush().unwrap();
            let _ = ready_sender.send(());
            let _ = release_receiver.recv_timeout(std::time::Duration::from_secs(2));
            stream.write_all(b"1E\r\n").unwrap();
            stream.write_all(&[b'b'; 30]).unwrap();
            stream.write_all(b"\r\n0\r\n\r\n").unwrap();
            stream.flush().unwrap();
        });
        (url, ready_receiver, release_sender, worker)
    }

    #[tokio::test]
    async fn twitter_media_fetch_rejects_redirect_without_following_location() {
        let server = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/media",
                redirect_response("/private"),
            ),
        ]);
        let budget = ExternalMediaBudget::new(1_024);
        let response = get_http_client_no_redirect()
            .get(server.url("/media"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        let result = read_external_media_response(response, 1_024, &budget).await;
        assert!(result.is_err());
        assert_eq!(budget.remaining(), 1_024);
        server.join().unwrap();
    }

    #[test]
    fn failed_budget_reservation_is_refunded() {
        let budget = ExternalMediaBudget::new(100);
        {
            let mut reservation = budget.reservation();
            assert!(reservation.try_reserve(60));
        }
        assert_eq!(budget.remaining(), 100);
    }

    #[test]
    fn failed_video_request_selects_thumbnail_at_the_same_index() {
        let request = twitter_video_request(
            3,
            "https://video.twimg.com/video/a.mp4",
            Some("https://pbs.twimg.com/media/a-thumb.jpg"),
        );
        let fallback = request.thumbnail_fallback().unwrap();
        assert_eq!(fallback.index, 3);
        assert!(matches!(fallback.kind, ExternalMediaKind::Image));
        assert_eq!(fallback.source, MediaSource::Twitter);
    }

    #[test]
    fn committed_bytes_carry_between_command_stages() {
        let budget = ExternalMediaBudget::new(100);
        budget.test_commit(60).unwrap();
        assert_eq!(budget.remaining(), 40);
        assert!(budget.test_commit(41).is_err());
    }

    #[tokio::test]
    async fn declared_per_file_overflow_is_rejected_before_read_and_keeps_budget() {
        let server = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                101,
                vec![b'x'; 101],
            ),
        );
        let budget = ExternalMediaBudget::new(200);
        let response = get_http_client_no_redirect()
            .get(server.url("/media"))
            .send()
            .await
            .unwrap();
        assert!(read_external_media_response(response, 100, &budget)
            .await
            .is_err());
        assert_eq!(budget.remaining(), 200);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn streamed_per_file_overflow_refunds_partial_reservation() {
        let server = TestServer::single(
            crate::tools::twitter_extractor::test_support::chunked_response(vec![
                vec![b'a'; 60],
                vec![b'b'; 60],
            ]),
        );
        let budget = ExternalMediaBudget::new(200);
        let response = get_http_client_no_redirect()
            .get(server.url("/media"))
            .send()
            .await
            .unwrap();
        assert!(read_external_media_response(response, 100, &budget)
            .await
            .is_err());
        assert_eq!(budget.remaining(), 200);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn total_budget_exhaustion_refunds_and_allows_a_later_request() {
        let (first_url, first_ready, release_first, first_worker) =
            controlled_thirty_chunk_server();
        let budget = ExternalMediaBudget::new(50);
        let response = get_http_client_no_redirect()
            .get(first_url)
            .send()
            .await
            .unwrap();
        let read = read_external_media_response(response, 100, &budget);
        tokio::pin!(read);
        tokio::time::timeout(SIGNAL_TIMEOUT, first_ready)
            .await
            .expect("controlled first chunk should become ready")
            .unwrap();
        let first_reserved = async {
            loop {
                if budget.remaining() == 20 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            tokio::select! {
                _ = first_reserved => {}
                result = &mut read => panic!("stream completed before second chunk gate: {result:?}"),
            }
        })
        .await
        .expect("first 30-byte chunk should reserve before the gate");
        release_first.send(()).unwrap();
        assert!(tokio::time::timeout(SIGNAL_TIMEOUT, read)
            .await
            .expect("controlled read should finish after release")
            .is_err());
        assert_eq!(budget.remaining(), 50);
        first_worker.join().unwrap();

        let second = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                50,
                vec![b'b'; 50],
            ),
        );
        let response = get_http_client_no_redirect()
            .get(second.url("/media"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            read_external_media_response(response, 100, &budget)
                .await
                .unwrap()
                .len(),
            50
        );
        assert_eq!(budget.remaining(), 0);
        second.join().unwrap();
    }

    #[tokio::test]
    async fn failed_video_control_path_fetches_thumbnail_at_same_index_without_extra_slot() {
        let server = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_status(
                    500,
                    Vec::new(),
                ),
            ),
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_content_length(
                    3,
                    b"abc".to_vec(),
                ),
            ),
        ]);
        let direct_url = server.url("/direct");
        let fallback_url = server.url("/fallback");
        let request = twitter_video_request(
            3,
            "https://video.twimg.com/video/a.mp4",
            Some("https://pbs.twimg.com/media/a-thumb.jpg"),
        );
        let direct = get_http_client_no_redirect()
            .get(direct_url)
            .send()
            .await
            .unwrap();
        let budget = ExternalMediaBudget::new(100);
        let (index, file) =
            process_response_with_fallback(request, Some(direct), &budget, move |_| async move {
                get_http_client_no_redirect()
                    .get(fallback_url)
                    .send()
                    .await
                    .ok()
            })
            .await
            .expect("thumbnail fallback should produce a file");
        assert_eq!(index, 3);
        assert_eq!(file.kind, MediaKind::Image);
        assert_eq!(budget.remaining(), 97);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn thumbnail_fallback_does_not_make_prompt_claim_video_was_attached() {
        let server = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_status(
                    500,
                    Vec::new(),
                ),
            ),
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_content_length(
                    3,
                    b"img".to_vec(),
                ),
            ),
        ]);
        let content =
            twitter_content_with_video(Some("https://pbs.twimg.com/media/root-thumb.jpg"));
        let request = twitter_video_request(
            0,
            &content.video_urls[0],
            Some("https://pbs.twimg.com/media/root-thumb.jpg"),
        );
        let direct = get_http_client_no_redirect()
            .get(server.url("/direct"))
            .send()
            .await
            .unwrap();
        let fallback_url = server.url("/fallback");
        let budget = ExternalMediaBudget::new(100);
        let (_, file) =
            process_response_with_fallback(request, Some(direct), &budget, move |_| async move {
                get_http_client_no_redirect()
                    .get(fallback_url)
                    .send()
                    .await
                    .ok()
            })
            .await
            .expect("thumbnail fallback should produce an image");
        assert_eq!(file.kind, MediaKind::Image);
        assert!(content.formatted_content.contains("Videos available:"));
        assert!(!content.formatted_content.contains("Videos attached:"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn successful_direct_video_download_keeps_pre_download_prompt_truthful() {
        let server = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                3,
                b"vid".to_vec(),
            ),
        );
        let content = twitter_content_with_video(None);
        let request = twitter_video_request(0, &content.video_urls[0], None);
        let direct = get_http_client_no_redirect()
            .get(server.url("/direct"))
            .send()
            .await
            .unwrap();
        let budget = ExternalMediaBudget::new(100);
        let (_, file) =
            process_response_with_fallback(request, Some(direct), &budget, |_| async { None })
                .await
                .expect("direct video should produce a file");
        assert_eq!(file.kind, MediaKind::Video);
        assert!(content.formatted_content.contains("Videos available:"));
        assert!(!content.formatted_content.contains("Videos attached:"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn failed_video_without_fallback_keeps_prompt_truthful() {
        let server = TestServer::single_status("GET", "/direct", 500);
        let content = twitter_content_with_video(None);
        let request = twitter_video_request(0, &content.video_urls[0], None);
        let direct = get_http_client_no_redirect()
            .get(server.url("/direct"))
            .send()
            .await
            .unwrap();
        let budget = ExternalMediaBudget::new(100);
        let result =
            process_response_with_fallback(request, Some(direct), &budget, |_| async { None })
                .await;
        assert!(result.is_none());
        assert!(content.formatted_content.contains("Videos available:"));
        assert!(!content.formatted_content.contains("Videos attached:"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn shared_budget_carries_committed_telegraph_bytes_into_twitter_stage() {
        let telegraph = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                60,
                vec![b't'; 60],
            ),
        );
        let twitter = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                41,
                vec![b'x'; 41],
            ),
        );
        let telegraph_url = telegraph.url("/media");
        let twitter_url = twitter.url("/media");
        let budget = ExternalMediaBudget::new(100);
        let telegraph_files = collect_external_media(
            vec![ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/stage.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            }],
            &budget,
            4,
            move |_| {
                let url = telegraph_url.clone();
                async move { Ok(get_http_client_no_redirect().get(url).send().await?) }
            },
        )
        .await;
        assert_eq!(telegraph_files.len(), 1);
        assert_eq!(budget.remaining(), 40);
        telegraph.join().unwrap();

        let twitter_files = collect_external_media(
            vec![ExternalMediaRequest {
                index: 0,
                url: "https://pbs.twimg.com/media/stage.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Twitter,
                thumbnail_url: None,
            }],
            &budget,
            4,
            move |_| {
                let url = twitter_url.clone();
                async move { Ok(get_http_client_no_redirect().get(url).send().await?) }
            },
        )
        .await;
        assert!(twitter_files.is_empty());
        assert_eq!(budget.remaining(), 40);
        twitter.join().unwrap();
    }

    #[tokio::test]
    async fn collector_preserves_input_order() {
        let slow = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_content_length(
                    1,
                    b"0".to_vec(),
                ),
            )
            .delayed(std::time::Duration::from_millis(40)),
        ]);
        let fast = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                1,
                b"1".to_vec(),
            ),
        );
        let requests = vec![
            ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/0.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            },
            ExternalMediaRequest {
                index: 1,
                url: "https://telegra.ph/file/1.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            },
        ];
        let slow_url = slow.url("/media");
        let fast_url = fast.url("/media");
        let budget = ExternalMediaBudget::new(100);
        // Index 0 answers slower than index 1, so a naive first-finished
        // order would come back reversed; the collector must still restore
        // input order.
        let files = collect_external_media(requests, &budget, 4, move |request| {
            let slow_url = slow_url.clone();
            let fast_url = fast_url.clone();
            async move {
                let url = if request.index == 0 {
                    slow_url
                } else {
                    fast_url
                };
                Ok(get_http_client_no_redirect().get(url).send().await?)
            }
        })
        .await;
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].bytes(), b"0");
        assert_eq!(files[1].bytes(), b"1");
        slow.join().unwrap();
        fast.join().unwrap();
    }

    #[tokio::test]
    async fn collector_never_exceeds_fanout() {
        const FANOUT: usize = 2;
        const REQUEST_COUNT: usize = 5;
        let responses = (0..REQUEST_COUNT)
            .map(|i| {
                crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                    crate::tools::twitter_extractor::test_support::response_with_content_length(
                        1,
                        vec![b'a' + i as u8],
                    ),
                )
                .delayed(std::time::Duration::from_millis(30))
            })
            .collect();
        let server = TestServer::new(responses);
        let url = server.url("/media");
        let requests = (0..REQUEST_COUNT)
            .map(|index| ExternalMediaRequest {
                index,
                url: format!("https://telegra.ph/file/{index}.jpg"),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            })
            .collect::<Vec<_>>();
        let concurrent = Arc::new(AtomicUsize::new(0));
        let high_water = Arc::new(AtomicUsize::new(0));
        let budget = ExternalMediaBudget::new(1_000);
        let concurrent_for_start = concurrent.clone();
        let high_water_for_start = high_water.clone();
        let files = collect_external_media(requests, &budget, FANOUT, move |_request| {
            let url = url.clone();
            let concurrent = concurrent_for_start.clone();
            let high_water = high_water_for_start.clone();
            async move {
                let current = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                high_water.fetch_max(current, Ordering::SeqCst);
                let result = get_http_client_no_redirect().get(url).send().await;
                concurrent.fetch_sub(1, Ordering::SeqCst);
                Ok(result?)
            }
        })
        .await;
        assert_eq!(files.len(), REQUEST_COUNT);
        assert!(
            high_water.load(Ordering::SeqCst) <= FANOUT,
            "high water mark {} exceeded fanout {FANOUT}",
            high_water.load(Ordering::SeqCst)
        );
        assert_eq!(
            high_water.load(Ordering::SeqCst),
            FANOUT,
            "more requests than permits should saturate the fanout"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn collector_holds_permit_until_body_is_consumed() {
        // With fanout 1, the second request must not be able to start
        // while the first is still streaming its body: the fanout permit
        // is only released after process_response_with_fallback (which
        // reads the whole body) returns, not as soon as the response
        // headers arrive.
        let (slow_url, mut body_ready, release_body, slow_worker) =
            controlled_thirty_chunk_server();
        let fast = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                1,
                b"1".to_vec(),
            ),
        );
        let fast_url = fast.url("/media");
        let started = Arc::new(AtomicUsize::new(0));
        let started_for_start = started.clone();
        let requests = vec![
            ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/0.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            },
            ExternalMediaRequest {
                index: 1,
                url: "https://telegra.ph/file/1.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            },
        ];
        let budget = ExternalMediaBudget::new(200);
        let collection = collect_external_media(requests, &budget, 1, move |request| {
            started_for_start.fetch_add(1, Ordering::SeqCst);
            let slow_url = slow_url.clone();
            let fast_url = fast_url.clone();
            async move {
                let url = if request.index == 0 {
                    slow_url
                } else {
                    fast_url
                };
                Ok(get_http_client_no_redirect().get(url).send().await?)
            }
        });
        tokio::pin!(collection);

        // The slow leg's first chunk has landed, but its body is not yet
        // fully read (the second chunk is still gated on `release_body`).
        // Only one worker may have started at this point.
        tokio::time::timeout(SIGNAL_TIMEOUT, async {
            tokio::select! {
                result = &mut body_ready => result.expect("controlled first chunk should become ready"),
                _ = &mut collection => panic!("collection completed before the controlled body was released"),
            }
        })
        .await
        .expect("controlled body should become ready");
        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "the second request must not start while the first body is still streaming"
        );

        release_body.send(()).unwrap();
        let files = tokio::time::timeout(std::time::Duration::from_secs(2), collection)
            .await
            .expect("collection should finish once the body is released");
        assert_eq!(files.len(), 2);
        assert_eq!(started.load(Ordering::SeqCst), 2);
        slow_worker.join().unwrap();
        fast.join().unwrap();
    }

    #[tokio::test]
    async fn collector_times_out_and_returns_partial_results() {
        // Real listeners are created before pausing time: the fast one
        // answers over genuine (instant, real-time) localhost I/O, and the
        // hang one never answers, standing in for a provider that never
        // completes.
        let fast = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                1,
                b"x".to_vec(),
            ),
        );
        let hang_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let hang_addr = hang_listener.local_addr().unwrap();
        let hang_worker = std::thread::spawn(move || {
            let Ok((mut stream, _)) = hang_listener.accept() else {
                return;
            };
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer);
            // Never write a response; block until the collector aborts this
            // connection (or the read errors out), simulating a provider
            // that never completes.
            loop {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
        });

        let fast_url = fast.url("/media");
        let hang_url = url::Url::parse(&format!("http://{hang_addr}/media")).unwrap();
        let requests = vec![
            ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/0.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            },
            ExternalMediaRequest {
                index: 1,
                url: "https://telegra.ph/file/1.jpg".to_string(),
                kind: ExternalMediaKind::Image,
                source: MediaSource::Telegraph,
                thumbnail_url: None,
            },
        ];
        let budget = ExternalMediaBudget::new(100);
        // Signals when the fast leg's genuine (real-time) localhost round
        // trip has settled (success or failure), so the test can wait on
        // that instead of guessing how many scheduler ticks it needs under
        // load.
        let fast_done = Arc::new(tokio::sync::Notify::new());
        let fast_done_for_start = fast_done.clone();
        let handle = tokio::spawn(async move {
            collect_external_media(requests, &budget, 4, move |request| {
                let fast_url = fast_url.clone();
                let hang_url = hang_url.clone();
                let fast_done = fast_done_for_start.clone();
                async move {
                    if request.index == 0 {
                        let result = reqwest::Client::new().get(fast_url).send().await;
                        fast_done.notify_one();
                        return Ok(result?);
                    }
                    // A client without a request timeout, so the hang branch
                    // is blocked only on real I/O (no tokio timer for
                    // `advance` to fast-forward past) until the collector's
                    // own deadline aborts it.
                    Ok(reqwest::Client::new().get(hang_url).send().await?)
                }
            })
            .await
        });

        // Wait (in real, unpaused time, so a genuine stall fails fast
        // instead of hanging the suite) for the fast leg to settle, then a
        // handful of ticks for its (already-buffered, one-byte) body read.
        // Only once that is done do we freeze the clock and jump past the
        // deadline: `time::advance` itself only yields once, and jumping
        // past the deadline before the fast response is fully collected
        // would abort it too, alongside the hang leg it is meant to cut off.
        tokio::time::timeout(std::time::Duration::from_secs(10), fast_done.notified())
            .await
            .expect("fast leg should settle quickly over real localhost I/O");
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }

        tokio::time::pause();
        tokio::time::advance(
            EXTERNAL_MEDIA_COLLECTION_TIMEOUT + std::time::Duration::from_millis(1),
        )
        .await;
        let files = handle.await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].bytes(), b"x");
        fast.join().unwrap();
        // The hang worker unblocks once the collector aborts the connection
        // above; it is not joined here to keep the test from depending on
        // that abort tearing down the socket promptly.
        drop(hang_worker);
    }

    #[test]
    fn collector_logs_and_skips_failed_images() {
        // `tracing`'s per-callsite interest cache is process-global: under
        // `cargo test`'s default parallelism another thread's callsite can
        // race the rebuild `capture_json_events` triggers when it installs
        // its subscriber, so a run can (rarely) capture nothing even though
        // the `warn!` genuinely fired. Retry a few times rather than flake.
        let mut events = Vec::new();
        for _ in 0..5 {
            events = crate::utils::log_capture::capture_json_events(|| {
                // `warn!` is captured only on the thread the subscriber was
                // installed on, so this must poll on the current thread
                // rather than a multi-threaded runtime's worker threads.
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                runtime.block_on(async {
                    let server = TestServer::single_status("GET", "/media", 500);
                    let url = server.url("/media");
                    let requests = vec![ExternalMediaRequest {
                        index: 0,
                        url: "https://telegra.ph/file/0.jpg".to_string(),
                        kind: ExternalMediaKind::Image,
                        source: MediaSource::Telegraph,
                        thumbnail_url: None,
                    }];
                    let budget = ExternalMediaBudget::new(100);
                    // A client built fresh on this dedicated runtime: the
                    // shared `get_http_client_no_redirect()` client is
                    // lazily bound to whichever runtime first touches it,
                    // which may be a different test's, and reusing it here
                    // hangs on checkout.
                    let client = reqwest::Client::new();
                    let files = collect_external_media(requests, &budget, 4, move |_request| {
                        let url = url.clone();
                        let client = client.clone();
                        async move { Ok(client.get(url).send().await?) }
                    })
                    .await;
                    assert!(files.is_empty());
                    server.join().unwrap();
                });
            });
            if events
                .iter()
                .any(|event| event["fields"]["message"] == "Skipping image that failed to download")
            {
                break;
            }
        }

        // The http client's own connection-pool bookkeeping also emits trace
        // events on this subscriber; find our own warning among them rather
        // than assuming it is the only event captured.
        let warning = events
            .iter()
            .find(|event| event["fields"]["message"] == "Skipping image that failed to download")
            .unwrap_or_else(|| panic!("expected a download-failure warning; got {events:?}"));
        assert_eq!(warning["level"], "WARN");
        assert_eq!(warning["fields"]["source"], "telegraph");
        assert_eq!(
            warning["fields"]["media_url"],
            "https://telegra.ph/file/0.jpg"
        );
        assert!(warning["fields"]["error"].as_str().unwrap().contains("500"));
    }

    #[tokio::test]
    async fn collector_refunds_budget_on_overflow() {
        let server = TestServer::single(
            crate::tools::twitter_extractor::test_support::chunked_response(vec![
                vec![b'a'; 60],
                vec![b'b'; 60],
            ]),
        );
        let url = server.url("/media");
        let requests = vec![ExternalMediaRequest {
            index: 0,
            url: "https://telegra.ph/file/big.jpg".to_string(),
            kind: ExternalMediaKind::Image,
            source: MediaSource::Telegraph,
            thumbnail_url: None,
        }];
        let budget = ExternalMediaBudget::new(100);
        let files = collect_external_media(requests, &budget, 4, move |_request| {
            let url = url.clone();
            async move { Ok(get_http_client_no_redirect().get(url).send().await?) }
        })
        .await;
        assert!(files.is_empty());
        assert_eq!(
            budget.remaining(),
            100,
            "a mid-stream overflow must refund its partial reservation"
        );
        server.join().unwrap();
    }

    #[test]
    fn video_mime_preserves_url_then_bytes_then_default_precedence() {
        assert_eq!(
            resolve_video_mime_type(
                "https://video.twimg.com/v.mp4",
                Some("application/octet-stream"),
                b"bad"
            ),
            "video/mp4"
        );
        assert_eq!(
            resolve_video_mime_type(
                "https://video.twimg.com/v",
                Some("application/octet-stream"),
                &[0x1a, 0x45, 0xdf, 0xa3, 0x93, 0x42, 0x86, 0x81],
            ),
            "video/webm"
        );
        assert_eq!(
            resolve_video_mime_type("https://video.twimg.com/v", None, b"bad"),
            "video/mp4"
        );
    }
}

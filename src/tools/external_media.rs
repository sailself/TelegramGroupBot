use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use anyhow::{anyhow, bail, Result};
use reqwest::header::CONTENT_TYPE;
use reqwest::Response;
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tracing::{debug, warn};
use url::Url;

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, MediaFile, MediaKind};
use crate::tools::telegraph_extractor::TelegraphContent;
use crate::tools::twitter_extractor::{parse_allowed_media_url, TwitterContent};
use crate::utils::http::get_http_client_no_redirect;

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
    Image(&'static str),
    Video,
}

#[derive(Debug, Clone)]
pub(crate) struct ExternalMediaRequest {
    pub(crate) index: usize,
    pub(crate) url: String,
    pub(crate) kind: ExternalMediaKind,
    source: &'static str,
    thumbnail_url: Option<String>,
}

impl ExternalMediaRequest {
    pub(crate) fn thumbnail_fallback(&self) -> Option<Self> {
        self.thumbnail_url.as_ref().map(|url| Self {
            index: self.index,
            url: url.clone(),
            kind: ExternalMediaKind::Image(self.source),
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
        source: "twitter",
        thumbnail_url: thumbnail_url.map(str::to_string),
    }
}

fn telegraph_media_url(raw_url: &str) -> Result<Url> {
    let parsed = Url::parse(raw_url.trim())?;
    if parsed.scheme() != "https" {
        bail!("Telegraph media must use HTTPS")
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        bail!("Telegraph media must not contain credentials")
    }
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    if host != "telegra.ph" && host != "graph.org" {
        bail!("Telegraph media host is not allowlisted")
    }
    if parsed.port().is_some_and(|port| port != 443) {
        bail!("Telegraph media has a non-default port")
    }
    Ok(parsed)
}

fn validate_media_url(url: &str, source: &str) -> Result<Url> {
    if source == "twitter" {
        return parse_allowed_media_url(url);
    }
    telegraph_media_url(url)
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

async fn start_request(request: &ExternalMediaRequest) -> Option<Response> {
    let parsed_url = match validate_media_url(&request.url, request.source) {
        Ok(url) => url,
        Err(err) => {
            warn!(media_url = %request.url, error = %err, "Skipping disallowed external media URL");
            return None;
        }
    };
    if matches!(request.kind, ExternalMediaKind::Image(_))
        && request.url.to_ascii_lowercase().contains(".svg")
    {
        return None;
    }

    let response = match get_http_client_no_redirect().get(parsed_url).send().await {
        Ok(response) => response,
        Err(err) => {
            warn!(media_url = %request.url, error = %err, "Failed to fetch external media");
            return None;
        }
    };
    Some(response)
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
    if let ExternalMediaKind::Image(source) = request.kind {
        let content_type =
            content_type.or_else(|| image_mime_from_url(&request.url).map(ToString::to_string));
        let Some(content_type) = content_type else {
            warn!(source, media_url = %request.url, "Skipping image without Content-Type or URL MIME hint");
            return None;
        };
        if !content_type.starts_with("image/") || content_type == "image/svg+xml" {
            warn!(source, media_url = %request.url, content_type = %content_type, "Skipping non-image external media");
            return None;
        }
        let bytes = read_external_media_response(response, CONFIG.external_media_max_bytes, budget)
            .await
            .ok()?;
        return Some(MediaFile::new(
            bytes,
            content_type,
            MediaKind::Image,
            display_name_from_url(&request.url),
        ));
    }

    let bytes =
        match read_external_media_response(response, CONFIG.external_media_max_bytes, budget).await
        {
            Ok(bytes) => bytes,
            Err(err) => {
                debug!(media_url = %request.url, error = %err, "External video download failed");
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

async fn collect_external_media_with_starts_limit<F, Fut>(
    requests: Vec<ExternalMediaRequest>,
    max_files: usize,
    budget: &ExternalMediaBudget,
    start: F,
    fanout: usize,
) -> Vec<MediaFile>
where
    F: Fn(ExternalMediaRequest) -> Fut + Clone + Send + Sync + 'static,
    Fut: std::future::Future<Output = Option<Response>> + Send + 'static,
{
    if requests.is_empty() || max_files == 0 {
        return Vec::new();
    }
    let semaphore = Arc::new(Semaphore::new(fanout.max(1)));
    let mut receivers = Vec::with_capacity(requests.len());
    let mut senders = Vec::with_capacity(requests.len());
    for _ in 0..requests.len() {
        let (sender, receiver) =
            oneshot::channel::<(ExternalMediaRequest, Option<Response>, OwnedSemaphorePermit)>();
        senders.push(sender);
        receivers.push(receiver);
    }
    let launcher_semaphore = semaphore.clone();
    let launcher = tokio::spawn(async move {
        let mut workers = JoinSet::new();
        for (request, sender) in requests.into_iter().zip(senders) {
            let permit = match launcher_semaphore.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => return,
            };
            let start = start.clone();
            workers.spawn(async move {
                let response = start(request.clone()).await;
                let _ = sender.send((request, response, permit));
            });
        }
        while workers.join_next().await.is_some() {}
    });

    let mut collected = Vec::new();
    for receiver in receivers {
        let Ok((request, response, permit)) = receiver.await else {
            continue;
        };
        if let Some((index, file)) =
            process_response_with_fallback(request, response, budget, |fallback| async move {
                start_request(&fallback).await
            })
            .await
        {
            collected.push((index, file));
        }
        drop(permit);
    }
    let _ = launcher.await;
    collected.sort_by_key(|(index, _)| *index);
    collected
        .into_iter()
        .take(max_files)
        .map(|(_, file)| file)
        .collect()
}

async fn collect_external_media_with_starts<F, Fut>(
    requests: Vec<ExternalMediaRequest>,
    max_files: usize,
    budget: &ExternalMediaBudget,
    start: F,
) -> Vec<MediaFile>
where
    F: Fn(ExternalMediaRequest) -> Fut + Clone + Send + Sync + 'static,
    Fut: std::future::Future<Output = Option<Response>> + Send + 'static,
{
    collect_external_media_with_starts_limit(
        requests,
        max_files,
        budget,
        start,
        CONFIG.external_enrich_fanout,
    )
    .await
}

async fn collect_external_media(
    requests: Vec<ExternalMediaRequest>,
    max_files: usize,
    budget: &ExternalMediaBudget,
) -> Vec<MediaFile> {
    collect_external_media_with_starts(requests, max_files, budget, |request| async move {
        start_request(&request).await
    })
    .await
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
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
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
                source: "telegraph",
                thumbnail_url: None,
            });
            index += 1;
        }
    }
    collect_external_media(requests, max_files, budget).await
}

pub async fn download_twitter_media(
    contents: &[TwitterContent],
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
                kind: ExternalMediaKind::Image("twitter"),
                source: "twitter",
                thumbnail_url: None,
            });
            index += 1;
        }
        for url in &content.video_urls {
            if requests.len() >= max_files {
                break;
            }
            let thumbnail_url = content
                .video_thumbnail_fallbacks
                .iter()
                .find(|fallback| fallback.video_url == *url)
                .map(|fallback| fallback.thumbnail_url.as_str());
            requests.push(twitter_video_request(index, url, thumbnail_url));
            index += 1;
        }
    }
    collect_external_media(requests, max_files, budget).await
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread::JoinHandle;

    use super::*;
    use crate::tools::twitter_extractor::test_support::{redirect_response, TestServer};
    use crate::utils::http::get_http_client_no_redirect;

    fn controlled_body_server() -> (
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
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n1\r\nx\r\n",
                )
                .unwrap();
            stream.flush().unwrap();
            let _ = ready_sender.send(());
            let _ = release_receiver.recv_timeout(std::time::Duration::from_secs(2));
            stream.write_all(b"1\r\ny\r\n0\r\n\r\n").unwrap();
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
        assert!(matches!(fallback.kind, ExternalMediaKind::Image("twitter")));
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
        let first = TestServer::single(
            crate::tools::twitter_extractor::test_support::chunked_response(vec![
                vec![b'a'; 30],
                vec![b'b'; 30],
            ]),
        );
        let budget = ExternalMediaBudget::new(50);
        let response = get_http_client_no_redirect()
            .get(first.url("/media"))
            .send()
            .await
            .unwrap();
        assert!(read_external_media_response(response, 100, &budget)
            .await
            .is_err());
        assert_eq!(budget.remaining(), 50);
        first.join().unwrap();

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
    async fn inverted_response_delays_keep_source_index_priority_for_budget_survivors() {
        let slow = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_content_length(
                    2,
                    b"0a".to_vec(),
                ),
            )
            .delayed(std::time::Duration::from_millis(50)),
        ]);
        let fast = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::any(
                crate::tools::twitter_extractor::test_support::response_with_content_length(
                    2,
                    b"1b".to_vec(),
                ),
            )
            .delayed(std::time::Duration::from_millis(1)),
        ]);
        let requests = vec![
            ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/0.jpg".to_string(),
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            },
            ExternalMediaRequest {
                index: 1,
                url: "https://telegra.ph/file/1.jpg".to_string(),
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            },
        ];
        let slow_url = slow.url("/media");
        let fast_url = fast.url("/media");
        let budget = ExternalMediaBudget::new(2);
        let files = collect_external_media_with_starts(requests, 2, &budget, move |request| {
            let slow_url = slow_url.clone();
            let fast_url = fast_url.clone();
            async move {
                let url = if request.index == 0 {
                    slow_url
                } else {
                    fast_url
                };
                get_http_client_no_redirect().get(url).send().await.ok()
            }
        })
        .await;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].bytes(), b"0a");
        slow.join().unwrap();
        fast.join().unwrap();
    }

    #[tokio::test]
    async fn failed_lower_index_start_advances_without_deadlock_or_budget_leak() {
        let server = TestServer::single(
            crate::tools::twitter_extractor::test_support::response_with_content_length(
                2,
                b"1b".to_vec(),
            ),
        );
        let url = server.url("/media");
        let requests = vec![
            ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/0.jpg".to_string(),
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            },
            ExternalMediaRequest {
                index: 1,
                url: "https://telegra.ph/file/1.jpg".to_string(),
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            },
        ];
        let budget = ExternalMediaBudget::new(2);
        let files = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            collect_external_media_with_starts(requests, 2, &budget, move |request| {
                let url = url.clone();
                async move {
                    if request.index == 0 {
                        panic!("simulated lower-index start panic");
                    }
                    get_http_client_no_redirect().get(url).send().await.ok()
                }
            }),
        )
        .await
        .expect("a failed lower-index start must not hang");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].bytes(), b"1b");
        assert_eq!(budget.remaining(), 0);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn fanout_permit_stays_held_through_body_consumption() {
        let (slow_url, mut body_ready, release_body, slow_worker) = controlled_body_server();
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
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            },
            ExternalMediaRequest {
                index: 1,
                url: "https://telegra.ph/file/1.jpg".to_string(),
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            },
        ];
        let budget = ExternalMediaBudget::new(10);
        let collection = collect_external_media_with_starts_limit(
            requests,
            2,
            &budget,
            move |request| {
                started_for_start.fetch_add(1, Ordering::SeqCst);
                let url = if request.index == 0 {
                    slow_url.clone()
                } else {
                    fast_url.clone()
                };
                async move { get_http_client_no_redirect().get(url).send().await.ok() }
            },
            1,
        );
        tokio::pin!(collection);
        tokio::select! {
            _ = &mut body_ready => {}
            _ = &mut collection => panic!("collection completed before the controlled body was released"),
        }
        assert_eq!(started.load(Ordering::SeqCst), 1);
        release_body.send(()).unwrap();
        let files = tokio::time::timeout(std::time::Duration::from_secs(2), collection)
            .await
            .unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(started.load(Ordering::SeqCst), 2);
        slow_worker.join().unwrap();
        fast.join().unwrap();
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
        let telegraph_files = collect_external_media_with_starts(
            vec![ExternalMediaRequest {
                index: 0,
                url: "https://telegra.ph/file/stage.jpg".to_string(),
                kind: ExternalMediaKind::Image("telegraph"),
                source: "telegraph",
                thumbnail_url: None,
            }],
            1,
            &budget,
            move |_| {
                let url = telegraph_url.clone();
                async move { get_http_client_no_redirect().get(url).send().await.ok() }
            },
        )
        .await;
        assert_eq!(telegraph_files.len(), 1);
        assert_eq!(budget.remaining(), 40);
        telegraph.join().unwrap();

        let twitter_files = collect_external_media_with_starts(
            vec![ExternalMediaRequest {
                index: 0,
                url: "https://pbs.twimg.com/media/stage.jpg".to_string(),
                kind: ExternalMediaKind::Image("twitter"),
                source: "twitter",
                thumbnail_url: None,
            }],
            1,
            &budget,
            move |_| {
                let url = twitter_url.clone();
                async move { get_http_client_no_redirect().get(url).send().await.ok() }
            },
        )
        .await;
        assert!(twitter_files.is_empty());
        assert_eq!(budget.remaining(), 40);
        twitter.join().unwrap();
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

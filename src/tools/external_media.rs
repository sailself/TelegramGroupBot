use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use anyhow::{anyhow, bail, Result};
use reqwest::header::CONTENT_TYPE;
use reqwest::Response;
use tokio::sync::Semaphore;
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

async fn fetch_request(
    request: ExternalMediaRequest,
    budget: &ExternalMediaBudget,
) -> Option<MediaFile> {
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
    let mime_type = video_mime_from_url(&request.url)
        .map(ToString::to_string)
        .or(content_type)
        .or_else(|| detect_mime_type(&bytes))
        .unwrap_or_else(|| "video/mp4".to_string());
    Some(MediaFile::new(
        bytes,
        mime_type,
        MediaKind::Video,
        display_name_from_url(&request.url),
    ))
}

async fn fetch_request_with_fallback(
    request: ExternalMediaRequest,
    budget: &ExternalMediaBudget,
) -> Option<MediaFile> {
    let fallback = request.thumbnail_fallback();
    let direct = fetch_request(request, budget).await;
    if direct.is_some() {
        direct
    } else if let Some(fallback) = fallback {
        fetch_request(fallback, budget).await
    } else {
        None
    }
}

async fn collect_external_media(
    requests: Vec<ExternalMediaRequest>,
    max_files: usize,
    budget: &ExternalMediaBudget,
) -> Vec<MediaFile> {
    if requests.is_empty() || max_files == 0 {
        return Vec::new();
    }
    let semaphore = Arc::new(Semaphore::new(CONFIG.external_enrich_fanout));
    let mut join_set = JoinSet::new();
    for request in requests {
        let semaphore = semaphore.clone();
        let budget = budget.clone();
        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .expect("external enrich semaphore should remain open");
            (
                request.index,
                fetch_request_with_fallback(request, &budget).await,
            )
        });
    }
    let mut collected = Vec::new();
    while let Some(result) = join_set.join_next().await {
        if let Ok((index, Some(file))) = result {
            collected.push((index, file));
        }
    }
    collected.sort_by_key(|(index, _)| *index);
    collected
        .into_iter()
        .take(max_files)
        .map(|(_, file)| file)
        .collect()
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
    use super::*;
    use crate::tools::twitter_extractor::test_support::{redirect_response, TestServer};
    use crate::utils::http::get_http_client_no_redirect;

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
}

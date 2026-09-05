use std::{collections::HashSet, time::Duration};

use regex::Regex;
use std::sync::LazyLock;

use super::{
    read_limited_body, run_blocking_parser, ProviderError, ProviderErrorKind, TwitterFetchConfig,
    TwitterProvider,
};
use crate::tools::twitter_extractor::model::{
    is_usable_direct_video_url, is_usable_pbs_image_url, parse_allowed_media_url,
    validate_complete_post, XAuthor, XMedia, XPost,
};
use crate::tools::twitter_extractor::url::XStatusIdentity;
use crate::utils::http::NoRedirectClient;

static TIMESTAMP_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d{1,2}:\d{2}\s?[AP]M").unwrap());
static MEDIA_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[[^\]]*?\]\((https?://[^\)]+)\)").unwrap());
static LINK_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\]]*?)\]\((https?://[^\)]+)\)").unwrap());
static EMPTY_LINK_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\s*\]\((https?://[^\)]+)\)").unwrap());
static WHITESPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
type MediaBuckets = (Vec<String>, Vec<String>, Vec<String>);

fn error(
    kind: ProviderErrorKind,
    detail: &str,
    status: Option<reqwest::StatusCode>,
) -> ProviderError {
    ProviderError {
        provider: TwitterProvider::Jina,
        kind,
        status,
        detail: detail.to_owned(),
    }
}

pub(crate) fn parse(body: &[u8], identity: &XStatusIdentity) -> Result<XPost, ProviderError> {
    let text = std::str::from_utf8(body).map_err(|_| {
        error(
            ProviderErrorKind::Decode,
            "provider response was not valid UTF-8",
            None,
        )
    })?;
    let marker = "Markdown Content:\n";
    let marker_idx = text.find(marker).ok_or_else(|| {
        error(
            ProviderErrorKind::Decode,
            "provider response omitted markdown content",
            None,
        )
    })?;
    let lines = collect_relevant_lines(&text[marker_idx + marker.len()..]);
    let (cleaned, images, videos) = clean_lines_and_media(&lines)?;
    if cleaned.is_empty() && images.is_empty() && videos.is_empty() {
        return Err(error(
            ProviderErrorKind::Incomplete,
            "provider response had no usable content",
            None,
        ));
    }
    let (display_name, handle, handle_idx) = extract_metadata(&cleaned);
    let timestamp_idx = cleaned.iter().position(|line| looks_like_timestamp(line));
    let timestamp = timestamp_idx.map(|idx| cleaned[idx].clone());
    let display_idx = display_name
        .as_ref()
        .and_then(|name| cleaned.iter().position(|line| line == name));
    let body_text = strip_indices(&cleaned, &[handle_idx, display_idx, timestamp_idx])
        .join("\n")
        .trim()
        .to_owned();
    let advertised_media = !images.is_empty() || !videos.is_empty();
    let media = images
        .into_iter()
        .filter_map(|raw| {
            parse_allowed_media_url(&raw)
                .ok()
                .filter(is_usable_pbs_image_url)
                .map(|url| XMedia::Image {
                    url,
                    alt_text: None,
                })
        })
        .chain(videos.into_iter().filter_map(|raw| {
            parse_allowed_media_url(&raw)
                .ok()
                .filter(is_usable_direct_video_url)
                .map(|url| XMedia::Video {
                    url: Some(url),
                    thumbnail_url: None,
                    alt_text: None,
                    bitrate: None,
                })
        }))
        .collect::<Vec<_>>();
    if advertised_media && media.is_empty() {
        return Err(error(
            ProviderErrorKind::Incomplete,
            "provider response announced unusable media",
            None,
        ));
    }
    let author = match (display_name, handle) {
        (None, None) => None,
        (display_name, handle) => Some(XAuthor {
            display_name,
            handle,
        }),
    };
    let post = XPost {
        id: identity.id.clone(),
        author,
        text: body_text,
        created_at: timestamp,
        media,
        quote: None,
    };
    validate_complete_post(&post).map_err(|_| {
        error(
            ProviderErrorKind::Incomplete,
            "provider response was incomplete",
            None,
        )
    })?;
    Ok(post)
}

pub(crate) async fn fetch(
    client: &NoRedirectClient,
    config: &TwitterFetchConfig,
    identity: &XStatusIdentity,
    timeout: Duration,
) -> Result<XPost, ProviderError> {
    fetch_with_parser(client, config, identity, timeout, |body, identity| {
        parse(&body, &identity)
    })
    .await
}

async fn fetch_with_parser<F>(
    client: &NoRedirectClient,
    config: &TwitterFetchConfig,
    identity: &XStatusIdentity,
    timeout: Duration,
    parser: F,
) -> Result<XPost, ProviderError>
where
    F: FnOnce(Vec<u8>, XStatusIdentity) -> Result<XPost, ProviderError> + Send + 'static,
{
    let identity = identity.clone();
    let future = async {
        if identity.id.is_empty() || !identity.id.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(error(
                ProviderErrorKind::Incomplete,
                "requested status ID was invalid",
                None,
            ));
        }
        let mut endpoint = config.jina_reader_endpoint.clone();
        let mut path = endpoint.path().trim_end_matches('/').to_owned();
        path.push('/');
        path.push_str(identity.canonical_url.as_str());
        endpoint.set_path(&path);
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        let mut request = client.get(endpoint);
        if let Some(key) = config
            .jina_api_key
            .as_deref()
            .filter(|key| !key.trim().is_empty())
        {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|_| {
            error(
                ProviderErrorKind::Transport,
                "provider request failed",
                None,
            )
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(error(
                ProviderErrorKind::HttpStatus,
                "provider returned a non-success status",
                Some(status),
            ));
        }
        let body =
            read_limited_body(TwitterProvider::Jina, response, config.response_max_bytes).await?;
        run_blocking_parser(move || parser(body, identity))
            .await
            .map_err(|_| {
                error(
                    ProviderErrorKind::Incomplete,
                    "provider parser failed",
                    None,
                )
            })?
    };
    tokio::time::timeout(timeout, future).await.map_err(|_| {
        error(
            ProviderErrorKind::Timeout,
            "provider request timed out",
            None,
        )
    })?
}

fn looks_like_timestamp(text: &str) -> bool {
    if TIMESTAMP_REGEX.is_match(text) {
        return true;
    }
    let lowered = text.to_lowercase();
    if lowered.contains("am") || lowered.contains("pm") {
        [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
            "dec",
        ]
        .iter()
        .any(|month| lowered.contains(month))
    } else {
        false
    }
}

fn collect_relevant_lines(block: &str) -> Vec<String> {
    let stops = [
        "new to x?",
        "join x today",
        "sign up now to get your own personalized timeline!",
        "sign up",
        "log in",
        "tweet your reply",
        "trending now",
        "what's happening",
        "terms of service",
        "privacy policy",
        "cookie policy",
        "accessibility",
        "ads info",
    ];
    let prefixes = [
        "watch on",
        "show more",
        "related",
        "more replies",
        "explore",
        "tweet your reply",
    ];
    let mut out = Vec::new();
    let mut collecting = false;
    let mut seen = false;
    for line in block.lines() {
        let stripped = line.trim();
        let lowered = stripped.to_lowercase();
        if !collecting {
            if lowered == "conversation" {
                collecting = true;
            }
            continue;
        }
        if !seen {
            if stripped.is_empty() || stripped.chars().all(|c| c == '-') {
                continue;
            }
            seen = true;
        }
        if stripped.is_empty() {
            if out.last().is_some_and(|line: &String| !line.is_empty()) {
                out.push(String::new());
            }
            continue;
        }
        if stops.contains(&lowered.as_str())
            || prefixes.iter().any(|prefix| lowered.starts_with(prefix))
        {
            break;
        }
        out.push(line.to_owned());
    }
    if !out.is_empty() {
        return out;
    }
    let mut fallback = Vec::new();
    for line in block.lines() {
        let stripped = line.trim();
        let lowered = stripped.to_lowercase();
        if stripped.is_empty() {
            continue;
        }
        if stops.contains(&lowered.as_str())
            || prefixes.iter().any(|prefix| lowered.starts_with(prefix))
        {
            break;
        }
        fallback.push(line.to_owned());
    }
    fallback
}

fn normalize_media_url(raw: &str) -> Option<String> {
    let mut url = parse_allowed_media_url(raw).ok()?;
    if url
        .domain()
        .is_some_and(|domain| domain.ends_with("twimg.com"))
    {
        let mut pairs = url.query_pairs().into_owned().collect::<Vec<_>>();
        for pair in &mut pairs {
            if pair.0 == "name" {
                pair.1 = "orig".into();
            }
        }
        if !pairs.is_empty() {
            url.set_query(serde_urlencoded::to_string(pairs).ok().as_deref());
        }
    }
    Some(url.to_string())
}

fn clean_lines_and_media(lines: &[String]) -> Result<MediaBuckets, ProviderError> {
    let profile = [
        "profile_images",
        "profile_banners",
        "semantic_core_img",
        "/emoji/",
        "responsive-web/client-web",
    ];
    let video_ext = [".mp4", ".m3u8", ".mpd"];
    let punct = ['.', ',', ';', ':', ')', ']', '}', '!', '?'];
    let mut cleaned: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    let mut videos: Vec<String> = Vec::new();
    for line in lines {
        let mut working = line.clone();
        for caps in MEDIA_REGEX.captures_iter(&working) {
            let raw = caps[1].trim();
            let raw_lower = raw.to_ascii_lowercase();
            if raw_lower.contains(".svg") || profile.iter().any(|token| raw.contains(token)) {
                continue;
            }
            let Some(url) = normalize_media_url(raw) else {
                return Err(error(
                    ProviderErrorKind::Incomplete,
                    "provider response announced unusable media",
                    None,
                ));
            };
            if video_ext.iter().any(|ext| url.ends_with(ext)) || url.contains("video.twimg.com") {
                if !videos.contains(&url) {
                    videos.push(url);
                }
            } else if !images.contains(&url) {
                images.push(url);
            }
        }
        working = MEDIA_REGEX.replace_all(&working, "").to_string();
        working = EMPTY_LINK_REGEX.replace_all(&working, "").to_string();
        working = LINK_REGEX
            .replace_all(&working, |caps: &regex::Captures| caps[1].trim().to_owned())
            .to_string();
        working = WHITESPACE_REGEX
            .replace_all(&working, " ")
            .trim()
            .to_owned();
        if working.is_empty() {
            continue;
        }
        if let Some(last) = cleaned.last_mut() {
            if !last.ends_with(['.', '!', '?', ':'])
                && (working.starts_with('@') || working.starts_with('#'))
            {
                *last = format!("{last} {working}");
                continue;
            }
            if punct.iter().any(|prefix| working.starts_with(*prefix)) {
                last.push_str(&working);
                continue;
            }
        }
        cleaned.push(working);
    }
    Ok((cleaned, images, videos))
}

fn extract_metadata(lines: &[String]) -> (Option<String>, Option<String>, Option<usize>) {
    for (idx, line) in lines.iter().take(6).enumerate() {
        if line.starts_with('@') && !line.contains(' ') {
            let display = lines[..idx]
                .iter()
                .rev()
                .find(|line| !line.is_empty())
                .cloned();
            return (display, Some(line.clone()), Some(idx));
        }
    }
    (None, None, None)
}

fn strip_indices(lines: &[String], indexes: &[Option<usize>]) -> Vec<String> {
    let skip = indexes.iter().flatten().copied().collect::<HashSet<_>>();
    lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| (!skip.contains(&idx)).then_some(line.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::model::XMedia;
    use crate::tools::twitter_extractor::test_support::{
        response_with_content_length, ExpectedRequest, TestServer,
    };

    fn identity(id: &str) -> XStatusIdentity {
        XStatusIdentity {
            id: id.to_owned(),
            canonical_url: url::Url::parse(&format!("https://x.com/i/status/{id}")).unwrap(),
        }
    }

    #[test]
    fn parses_jina_photo_fixture_into_normalized_post() {
        let post = parse(
            include_bytes!("../fixtures/jina_photo.txt"),
            &identity("123"),
        )
        .unwrap();
        assert!(post.text.contains("fixture body"));
        assert_eq!(post.media.len(), 1);
    }

    #[test]
    fn parses_jina_video_thumbnail_as_image_fallback() {
        let post = parse(
            include_bytes!("../fixtures/jina_video_thumbnail.txt"),
            &identity("123"),
        )
        .unwrap();
        assert!(matches!(post.media[0], XMedia::Image { .. }));
    }

    #[test]
    fn parses_jina_markdown_without_conversation_heading() {
        let post = parse(
            include_bytes!("../fixtures/jina_no_conversation.txt"),
            &identity("123"),
        )
        .unwrap();
        assert!(post
            .text
            .contains("legacy body without conversation heading"));
    }

    #[test]
    fn rejects_jina_disallowed_announced_media() {
        let error = parse(
            include_bytes!("../fixtures/jina_disallowed_media.txt"),
            &identity("123"),
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Incomplete);
        assert!(!error.detail.contains("evil.example"));
    }

    #[test]
    fn rejects_jina_hls_only_media_even_when_root_text_exists() {
        let body = b"Title: fixture\nMarkdown Content:\nConversation\n\nAlice\n@alice\nroot text\n![video](https://video.twimg.com/video/list.m3u8)\n";
        let error = parse(body, &identity("123")).unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Incomplete);
    }

    fn config(endpoint: url::Url, key: Option<&str>) -> TwitterFetchConfig {
        TwitterFetchConfig {
            providers: vec![TwitterProvider::Jina],
            fxtwitter_api_base: endpoint.clone(),
            vxtwitter_api_base: endpoint.clone(),
            jina_reader_endpoint: endpoint,
            jina_api_key: key.map(str::to_owned),
            total_timeout: Duration::from_secs(5),
            provider_timeout: Duration::from_secs(2),
            response_max_bytes: 1024 * 1024,
        }
    }

    #[tokio::test]
    async fn jina_fetch_uses_configured_endpoint_and_optional_bearer() {
        let body = include_bytes!("../fixtures/jina_photo.txt");
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/https://x.com/i/status/123",
            response_with_content_length(body.len(), body.to_vec()),
        )
        .with_header("authorization", "Bearer test-jina-key")]);
        let post = fetch(
            crate::tools::twitter_extractor::providers::get_http_client_no_redirect(),
            &config(server.base_url(), Some("test-jina-key")),
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert!(post.text.contains("fixture body"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn jina_fetch_omits_blank_bearer_header() {
        let body = include_bytes!("../fixtures/jina_photo.txt");
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/https://x.com/i/status/123",
            response_with_content_length(body.len(), body.to_vec()),
        )
        .without_header("authorization")]);
        fetch(
            crate::tools::twitter_extractor::providers::get_http_client_no_redirect(),
            &config(server.base_url(), None),
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn jina_fetch_omits_whitespace_bearer_header() {
        let body = include_bytes!("../fixtures/jina_photo.txt");
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/https://x.com/i/status/123",
            response_with_content_length(body.len(), body.to_vec()),
        )
        .without_header("authorization")]);
        fetch(
            crate::tools::twitter_extractor::providers::get_http_client_no_redirect(),
            &config(server.base_url(), Some("   ")),
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn jina_fetch_times_out_while_blocking_parser_is_running() {
        let body = include_bytes!("../fixtures/jina_photo.txt");
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/https://x.com/i/status/123",
            response_with_content_length(body.len(), body.to_vec()),
        )]);
        // The parser blocks until the test releases it, so only the timeout
        // path can make fetch_with_parser return early.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let started = std::time::Instant::now();
        let error = fetch_with_parser(
            crate::tools::twitter_extractor::providers::get_http_client_no_redirect(),
            &config(server.base_url(), None),
            &identity("123"),
            Duration::from_millis(20),
            move |_body, _identity| {
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
                Err(error(ProviderErrorKind::Incomplete, "slow parser", None))
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_secs(2));
        let _ = release_tx.send(());
        server.join().unwrap();
    }
}

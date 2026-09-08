use std::time::Duration;

use serde::Deserialize;
#[cfg(test)]
use url::Url;

use super::{
    fetch_and_parse, ProviderError, ProviderErrorKind, ProviderFetch, TwitterFetchConfig,
    TwitterProvider, TWITTER_USER_AGENT,
};
use crate::tools::twitter_extractor::model::{
    is_usable_direct_video_url, is_usable_pbs_image_url, parse_allowed_media_url,
    validate_complete_post, XAuthor, XMedia, XPost,
};
use crate::tools::twitter_extractor::url::XStatusIdentity;
use crate::utils::http::NoRedirectClient;

#[derive(Debug, Deserialize)]
struct Envelope {
    #[serde(default, alias = "tweet")]
    status: Option<FxStatus>,
}

#[derive(Debug, Deserialize)]
struct FxStatus {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    author: Option<FxAuthor>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    media: Option<FxMedia>,
    #[serde(default)]
    quote: Option<Box<serde_json::value::RawValue>>,
}

#[derive(Debug, Deserialize)]
struct FxQuoteStatus {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    author: Option<FxAuthor>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    media: Option<FxMedia>,
}
#[derive(Debug, Deserialize)]
struct FxAuthor {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    screen_name: Option<String>,
}
#[derive(Debug, Deserialize)]
struct FxMedia {
    #[serde(default)]
    photos: Vec<FxPhoto>,
    #[serde(default)]
    videos: Vec<FxVideo>,
}
#[derive(Debug, Deserialize)]
struct FxPhoto {
    #[serde(default)]
    url: Option<String>,
    #[serde(default, alias = "altText")]
    alt_text: Option<String>,
}
#[derive(Debug, Deserialize)]
struct FxVideo {
    #[serde(default, alias = "thumbnail", alias = "thumbnailUrl")]
    thumbnail_url: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    variants: Vec<FxVariant>,
    #[serde(default, alias = "altText")]
    alt_text: Option<String>,
}
#[derive(Debug, Deserialize)]
struct FxVariant {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    bitrate: Option<u64>,
    #[serde(default)]
    content_type: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

fn error(
    kind: ProviderErrorKind,
    detail: &str,
    status: Option<reqwest::StatusCode>,
) -> ProviderError {
    ProviderError {
        provider: TwitterProvider::FxTwitter,
        kind,
        status,
        detail: detail.to_owned(),
    }
}

pub(crate) fn parse(body: &[u8], identity: &XStatusIdentity) -> Result<XPost, ProviderError> {
    let envelope: Envelope = serde_json::from_slice(body).map_err(|_| {
        error(
            ProviderErrorKind::Decode,
            "provider response was not valid JSON",
            None,
        )
    })?;
    let status = envelope.status.ok_or_else(|| {
        error(
            ProviderErrorKind::Incomplete,
            "provider response omitted status",
            None,
        )
    })?;
    let post = map_status(status, identity)?;
    validate_complete_post(&post).map_err(|_| {
        error(
            ProviderErrorKind::Incomplete,
            "provider response was incomplete",
            None,
        )
    })?;
    Ok(post)
}

fn map_status(status: FxStatus, identity: &XStatusIdentity) -> Result<XPost, ProviderError> {
    if let Some(id) = status.id.as_deref().filter(|id| !id.trim().is_empty()) {
        if id != identity.id {
            return Err(error(
                ProviderErrorKind::Incomplete,
                "provider returned a conflicting status ID",
                None,
            ));
        }
    }
    let (media, advertised) = map_media(status.media)?;
    let quote = status
        .quote
        .as_deref()
        .and_then(|quote| map_quote(quote, identity));
    let author = status.author.map(|author| XAuthor {
        display_name: author.name,
        handle: author.screen_name,
    });
    let post = XPost {
        id: status
            .id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| identity.id.clone()),
        author,
        text: status.text,
        created_at: status.created_at,
        media,
        quote,
    };
    if advertised && post.media.is_empty() {
        return Err(error(
            ProviderErrorKind::Incomplete,
            "provider media was unusable",
            None,
        ));
    }
    Ok(post)
}

fn map_quote(
    raw_quote: &serde_json::value::RawValue,
    identity: &XStatusIdentity,
) -> Option<Box<XPost>> {
    let quote: FxQuoteStatus = match serde_json::from_str(raw_quote.get()) {
        Ok(quote) => quote,
        Err(_) => {
            trace_quote_omitted(identity, ProviderErrorKind::Decode);
            return None;
        }
    };
    let (media, advertised) = match map_media(quote.media) {
        Ok(media) => media,
        Err(error) => {
            trace_quote_omitted(identity, error.kind);
            return None;
        }
    };
    if advertised && media.is_empty() {
        trace_quote_omitted(identity, ProviderErrorKind::Incomplete);
        return None;
    }
    let post = XPost {
        id: quote
            .id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_default(),
        author: quote.author.map(|author| XAuthor {
            display_name: author.name,
            handle: author.screen_name,
        }),
        text: quote.text,
        created_at: quote.created_at,
        media,
        quote: None,
    };
    if validate_complete_post(&post).is_err() {
        trace_quote_omitted(identity, ProviderErrorKind::Incomplete);
        return None;
    }
    Some(Box::new(post))
}

fn trace_quote_omitted(identity: &XStatusIdentity, category: ProviderErrorKind) {
    tracing::debug!(
        target: "tools.twitter",
        provider = TwitterProvider::FxTwitter.as_str(),
        status_id = %identity.id,
        category = category.as_str(),
        "Twitter quote omitted"
    );
}

fn map_media(media: Option<FxMedia>) -> Result<(Vec<XMedia>, bool), ProviderError> {
    let Some(media) = media else {
        return Ok((Vec::new(), false));
    };
    let advertised = !media.photos.is_empty() || !media.videos.is_empty();
    let mut mapped = Vec::new();
    for photo in media.photos {
        if let Some(raw) = photo.url.as_deref() {
            if let Ok(url) = parse_allowed_media_url(raw) {
                if !is_usable_pbs_image_url(&url) {
                    continue;
                }
                mapped.push(XMedia::Image {
                    url,
                    alt_text: photo.alt_text,
                });
            }
        }
    }
    for video in media.videos {
        let thumb = video
            .thumbnail_url
            .as_deref()
            .and_then(|raw| parse_allowed_media_url(raw).ok())
            .filter(is_usable_pbs_image_url);
        let direct = video
            .url
            .as_deref()
            .and_then(|raw| parse_allowed_media_url(raw).ok())
            .filter(is_usable_direct_video_url);
        let selected = video
            .variants
            .into_iter()
            .filter_map(|variant| {
                let raw = variant.url?;
                let kind = variant.content_type.or(variant.kind);
                if kind
                    .as_deref()
                    .is_some_and(|kind| !kind.eq_ignore_ascii_case("video/mp4"))
                {
                    return None;
                }
                let url = parse_allowed_media_url(&raw).ok()?;
                if !is_usable_direct_video_url(&url) {
                    return None;
                }
                Some((url, variant.bitrate))
            })
            .max_by_key(|(_, bitrate)| bitrate.unwrap_or(0));
        let (url, bitrate) = selected.map_or((direct, None), |(url, bitrate)| (Some(url), bitrate));
        if url.is_some() || thumb.is_some() {
            mapped.push(XMedia::Video {
                url,
                thumbnail_url: thumb,
                alt_text: video.alt_text,
                bitrate,
            });
        }
    }
    Ok((mapped, advertised))
}

pub(crate) async fn fetch(
    client: &NoRedirectClient,
    config: &TwitterFetchConfig,
    identity: &XStatusIdentity,
    timeout: Duration,
) -> Result<XPost, ProviderError> {
    if identity.id.is_empty() || !identity.id.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(error(
            ProviderErrorKind::Incomplete,
            "requested status ID was invalid",
            None,
        ));
    }
    let mut endpoint = config.fxtwitter_api_base.clone();
    let mut path = endpoint.path().trim_end_matches('/').to_owned();
    path.push_str("/i/status/");
    path.push_str(&identity.id);
    endpoint.set_path(&path);
    endpoint.set_query(None);
    endpoint.set_fragment(None);
    let identity = identity.clone();
    fetch_and_parse(
        ProviderFetch {
            url: endpoint,
            bearer: None,
            user_agent: Some(TWITTER_USER_AGENT),
            client,
            provider: TwitterProvider::FxTwitter,
        },
        config.response_max_bytes,
        timeout,
        move |body| parse(body, &identity),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::model::XMedia;
    use crate::tools::twitter_extractor::test_support::{
        redirect_response, response_with_content_length, ExpectedRequest, TestServer,
    };
    use crate::tools::twitter_extractor::url::XStatusIdentity;

    fn identity(id: &str) -> XStatusIdentity {
        XStatusIdentity {
            id: id.to_owned(),
            canonical_url: url::Url::parse(&format!("https://x.com/i/status/{id}")).unwrap(),
        }
    }

    #[test]
    fn parses_fxtwitter_photo_quote_fixture() {
        let post = parse(
            include_bytes!("../fixtures/fxtwitter_photo_quote.json"),
            &identity("123"),
        )
        .unwrap();
        assert_eq!(post.id, "123");
        assert_eq!(post.text, "root text");
        assert_eq!(post.media.len(), 1);
        assert_eq!(post.quote.as_ref().unwrap().text, "quoted text");
    }

    #[test]
    fn parses_fxtwitter_highest_bitrate_video() {
        let post = parse(
            include_bytes!("../fixtures/fxtwitter_video.json"),
            &identity("123"),
        )
        .unwrap();
        let XMedia::Video { url, bitrate, .. } = &post.media[0] else {
            panic!("expected video")
        };
        assert_eq!(
            url.as_ref().unwrap().as_str(),
            "https://video.twimg.com/video/high.mp4"
        );
        assert_eq!(bitrate, &Some(2_176_000));
    }

    #[test]
    fn parses_fxtwitter_tweet_envelope_alias() {
        let post = parse(
            br#"{"tweet":{"id":"123","text":"alias root","media":{"photos":[],"videos":[]}}}"#,
            &identity("123"),
        )
        .unwrap();
        assert_eq!(post.text, "alias root");
    }

    #[test]
    fn fxtwitter_accepts_absent_root_id_and_rejects_conflicting_root_id() {
        let without_id = parse(
            br#"{"status":{"text":"root","media":{"photos":[],"videos":[]}}}"#,
            &identity("123"),
        )
        .unwrap();
        assert_eq!(without_id.id, "123");

        let error = parse(
            br#"{"status":{"id":"999","text":"root","media":{"photos":[],"videos":[]}}}"#,
            &identity("123"),
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Incomplete);
    }

    #[test]
    fn rejects_fxtwitter_announced_media_with_only_invalid_urls() {
        let error = parse(
            br#"{"status":{"id":"123","text":"root","media":{"photos":[{"url":"https://evil.example/a.jpg"}],"videos":[]}}}"#,
            &identity("123"),
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Incomplete);
        assert!(!error.detail.contains("evil.example"));
    }

    #[test]
    fn rejects_fxtwitter_hls_only_video_without_thumbnail() {
        let error = parse(
            br#"{"status":{"id":"123","text":"root","media":{"photos":[],"videos":[{"url":"https://video.twimg.com/video/list.m3u8"}]}}}"#,
            &identity("123"),
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Incomplete);
    }

    #[test]
    fn malformed_fxtwitter_quote_does_not_erase_valid_root() {
        let post = parse(
            br#"{"status":{"id":"123","text":"root survives","quote":{"id":"456","text":{"unexpected":"object"}}}}"#,
            &identity("123"),
        )
        .unwrap();
        assert_eq!(post.text, "root survives");
        assert!(post.quote.is_none());
    }

    #[test]
    fn invalid_fxtwitter_quote_media_does_not_erase_valid_root() {
        let post = parse(
            br#"{"status":{"id":"123","text":"root survives","quote":{"id":"456","media":{"photos":[{"url":"https://evil.example/secret.jpg"}],"videos":[]}}}}"#,
            &identity("123"),
        )
        .unwrap();
        assert_eq!(post.text, "root survives");
        assert!(post.quote.is_none());
    }

    #[test]
    fn fxtwitter_keeps_one_quote_level_without_decoding_deeper_quote() {
        let post = parse(
            br#"{"status":{"id":"123","text":"root","quote":{"id":"456","text":"first quote","quote":{"id":"789","text":{"unexpected":"object"}}}}}"#,
            &identity("123"),
        )
        .unwrap();
        let quote = post.quote.expect("first quote should be retained");
        assert_eq!(quote.text, "first quote");
        assert!(quote.quote.is_none());
    }

    #[tokio::test]
    async fn fxtwitter_fetch_uses_status_endpoint_and_body_limit() {
        let server = TestServer::single_json(
            "GET",
            "/i/status/123",
            include_bytes!("../fixtures/fxtwitter_photo_quote.json"),
        );
        let mut config = test_config_with_fx_base(server.base_url());
        config.response_max_bytes = 1024 * 1024;
        let post = fetch(
            crate::utils::http::get_http_client_no_redirect(),
            &config,
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert_eq!(post.text, "root text");
        server.join().unwrap();
    }

    #[tokio::test]
    async fn fxtwitter_fetch_identifies_client_with_user_agent() {
        let body = include_bytes!("../fixtures/fxtwitter_photo_quote.json");
        let expected_user_agent =
            format!("telegram_group_helper_bot/{}", env!("CARGO_PKG_VERSION"));
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/i/status/123",
            response_with_content_length(body.len(), body.to_vec()),
        )
        .with_header("user-agent", &expected_user_agent)]);
        let config = test_config_with_fx_base(server.base_url());

        let result = fetch(
            crate::utils::http::get_http_client_no_redirect(),
            &config,
            &identity("123"),
            Duration::from_secs(1),
        )
        .await;
        server.join().unwrap();
        assert_eq!(result.unwrap().text, "root text");
    }

    #[tokio::test]
    async fn fxtwitter_fetch_rejects_adapter_local_oversized_response() {
        let server = TestServer::single_json("GET", "/i/status/123", &vec![b'x'; 2_048]);
        let mut config = test_config_with_fx_base(server.base_url());
        config.response_max_bytes = 1_024;
        let error = fetch(
            crate::utils::http::get_http_client_no_redirect(),
            &config,
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::BodyTooLarge);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn fxtwitter_fetch_rejects_redirect_without_following_location() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/i/status/123",
            redirect_response("/target"),
        )]);
        let config = test_config_with_fx_base(server.base_url());
        let error = fetch(
            crate::utils::http::get_http_client_no_redirect(),
            &config,
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::HttpStatus);
        assert_eq!(error.status, Some(reqwest::StatusCode::FOUND));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn fxtwitter_fetch_preserves_success_status_for_decode_errors() {
        let server = TestServer::single_json("GET", "/i/status/123", b"not json");
        let config = test_config_with_fx_base(server.base_url());
        let error = fetch(
            crate::utils::http::get_http_client_no_redirect(),
            &config,
            &identity("123"),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Decode);
        assert_eq!(error.status, Some(reqwest::StatusCode::OK));
        server.join().unwrap();
    }

    fn test_config_with_fx_base(base: Url) -> TwitterFetchConfig {
        TwitterFetchConfig {
            providers: vec![TwitterProvider::FxTwitter],
            fxtwitter_api_base: base.clone(),
            vxtwitter_api_base: base.clone(),
            jina_reader_endpoint: base,
            jina_api_key: None,
            total_timeout: Duration::from_secs(5),
            provider_timeout: Duration::from_secs(2),
            response_max_bytes: 1024 * 1024,
        }
    }
}

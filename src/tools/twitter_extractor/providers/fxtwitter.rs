#![allow(dead_code)]

use std::time::Duration;

use serde::Deserialize;
#[cfg(test)]
use url::Url;

use super::{
    read_limited_body, ProviderError, ProviderErrorKind, TwitterFetchConfig, TwitterProvider,
};
use crate::tools::twitter_extractor::model::{
    parse_allowed_media_url, validate_complete_post, XAuthor, XMedia, XPost,
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
    quote: Option<Box<FxStatus>>,
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
    let post = map_status(status, identity, true)?;
    validate_complete_post(&post).map_err(|_| {
        error(
            ProviderErrorKind::Incomplete,
            "provider response was incomplete",
            None,
        )
    })?;
    Ok(post)
}

fn map_status(
    status: FxStatus,
    identity: &XStatusIdentity,
    root: bool,
) -> Result<XPost, ProviderError> {
    if root {
        if let Some(id) = status.id.as_deref().filter(|id| !id.trim().is_empty()) {
            if id != identity.id {
                return Err(error(
                    ProviderErrorKind::Incomplete,
                    "provider returned a conflicting status ID",
                    None,
                ));
            }
        }
    }
    let (media, advertised) = map_media(status.media)?;
    let quote = if root {
        status
            .quote
            .map(|quote| map_status(*quote, identity, false))
            .transpose()?
            .map(Box::new)
    } else {
        None
    };
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

fn map_media(media: Option<FxMedia>) -> Result<(Vec<XMedia>, bool), ProviderError> {
    let Some(media) = media else {
        return Ok((Vec::new(), false));
    };
    let advertised = !media.photos.is_empty() || !media.videos.is_empty();
    let mut mapped = Vec::new();
    for photo in media.photos {
        if let Some(raw) = photo.url.as_deref() {
            if let Ok(url) = parse_allowed_media_url(raw) {
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
            .and_then(|raw| parse_allowed_media_url(raw).ok());
        let direct = video
            .url
            .as_deref()
            .and_then(|raw| parse_allowed_media_url(raw).ok())
            .filter(|url| {
                url.host_str() == Some("video.twimg.com")
                    && url.path().to_ascii_lowercase().ends_with(".mp4")
            });
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
                if url.host_str() != Some("video.twimg.com")
                    || !url.path().to_ascii_lowercase().ends_with(".mp4")
                {
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
        let mut endpoint = config.fxtwitter_api_base.clone();
        let mut path = endpoint.path().trim_end_matches('/').to_owned();
        path.push_str("/i/status/");
        path.push_str(&identity.id);
        endpoint.set_path(&path);
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        let response = client.get(endpoint).send().await.map_err(|_| {
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
        let body = read_limited_body(
            TwitterProvider::FxTwitter,
            response,
            config.response_max_bytes,
        )
        .await?;
        tokio::task::spawn_blocking(move || match parser(body, identity) {
            Ok(post) => Ok(post),
            Err(mut error) => {
                if error.status.is_none() {
                    error.status = Some(status);
                }
                Err(error)
            }
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::model::XMedia;
    use crate::tools::twitter_extractor::test_support::{
        redirect_response, ExpectedRequest, TestServer,
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
            super::super::get_http_client_no_redirect(),
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
    async fn fxtwitter_fetch_rejects_redirect_without_following_location() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/i/status/123",
            redirect_response("/target"),
        )]);
        let config = test_config_with_fx_base(server.base_url());
        let error = fetch(
            super::super::get_http_client_no_redirect(),
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
    async fn fxtwitter_fetch_times_out_while_blocking_parser_is_running() {
        let server = TestServer::single_json(
            "GET",
            "/i/status/123",
            include_bytes!("../fixtures/fxtwitter_photo_quote.json"),
        );
        let config = test_config_with_fx_base(server.base_url());
        let started = std::time::Instant::now();
        let error = fetch_with_parser(
            super::super::get_http_client_no_redirect(),
            &config,
            &identity("123"),
            Duration::from_millis(20),
            |_body, _identity| {
                std::thread::sleep(Duration::from_millis(250));
                Err(error(ProviderErrorKind::Incomplete, "slow parser", None))
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_millis(200));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn fxtwitter_fetch_preserves_success_status_for_decode_errors() {
        let server = TestServer::single_json("GET", "/i/status/123", b"not json");
        let config = test_config_with_fx_base(server.base_url());
        let error = fetch(
            super::super::get_http_client_no_redirect(),
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

#![allow(dead_code)]

use std::time::Duration;

use serde::Deserialize;

use super::{
    read_limited_body, ProviderError, ProviderErrorKind, TwitterFetchConfig, TwitterProvider,
};
use crate::tools::twitter_extractor::model::{
    parse_allowed_media_url, validate_complete_post, XAuthor, XMedia, XPost,
};
use crate::tools::twitter_extractor::url::XStatusIdentity;
use crate::utils::http::NoRedirectClient;

#[derive(Debug, Deserialize)]
struct VxStatus {
    #[serde(default, rename = "tweetID", alias = "tweet_id", alias = "id")]
    id: Option<String>,
    #[serde(default)]
    text: String,
    #[serde(default, alias = "userName")]
    user_name: Option<String>,
    #[serde(default, alias = "userScreenName")]
    user_screen_name: Option<String>,
    #[serde(default, alias = "created_at", alias = "createdAt")]
    date: Option<String>,
    #[serde(
        default,
        rename = "mediaURLs",
        alias = "media_urls",
        alias = "mediaUrls"
    )]
    media_urls: Option<Vec<String>>,
    #[serde(default, alias = "mediaExtended", alias = "media_extended")]
    media_extended: Option<Vec<VxMedia>>,
    #[serde(default, rename = "qrt", alias = "quote")]
    quote: Option<Box<VxStatus>>,
}

#[derive(Debug, Deserialize)]
struct VxMedia {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default, alias = "thumbnail", alias = "thumbnailUrl")]
    thumbnail_url: Option<String>,
    #[serde(default, alias = "altText")]
    alt_text: Option<String>,
}

fn error(
    kind: ProviderErrorKind,
    detail: &str,
    status: Option<reqwest::StatusCode>,
) -> ProviderError {
    ProviderError {
        provider: TwitterProvider::VxTwitter,
        kind,
        status,
        detail: detail.to_owned(),
    }
}

pub(crate) fn parse(body: &[u8], identity: &XStatusIdentity) -> Result<XPost, ProviderError> {
    let status: VxStatus = serde_json::from_slice(body).map_err(|_| {
        error(
            ProviderErrorKind::Decode,
            "provider response was not valid JSON",
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
    status: VxStatus,
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
    let (media, advertised) = map_media(status.media_extended, status.media_urls);
    if advertised && media.is_empty() {
        return Err(error(
            ProviderErrorKind::Incomplete,
            "provider media was unusable",
            None,
        ));
    }
    let quote = if root {
        status
            .quote
            .map(|quote| map_status(*quote, identity, false))
            .transpose()?
            .map(Box::new)
    } else {
        None
    };
    Ok(XPost {
        id: status
            .id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| identity.id.clone()),
        author: (status.user_name.is_some() || status.user_screen_name.is_some()).then_some({
            XAuthor {
                display_name: status.user_name,
                handle: status.user_screen_name,
            }
        }),
        text: status.text,
        created_at: status.date,
        media,
        quote,
    })
}

fn map_media(extended: Option<Vec<VxMedia>>, urls: Option<Vec<String>>) -> (Vec<XMedia>, bool) {
    let advertised = extended.as_ref().is_some_and(|items| !items.is_empty())
        || urls.as_ref().is_some_and(|items| !items.is_empty());
    let mut mapped = Vec::new();
    if let Some(items) = extended {
        for item in items {
            let kind = item.kind.as_deref().unwrap_or("image");
            if kind.eq_ignore_ascii_case("video") || kind.eq_ignore_ascii_case("gif") {
                let direct = item
                    .url
                    .as_deref()
                    .and_then(|raw| parse_allowed_media_url(raw).ok())
                    .filter(|url| {
                        url.host_str() == Some("video.twimg.com")
                            && url.path().to_ascii_lowercase().ends_with(".mp4")
                    });
                let thumb = item
                    .thumbnail_url
                    .as_deref()
                    .and_then(|raw| parse_allowed_media_url(raw).ok());
                if direct.is_some() || thumb.is_some() {
                    mapped.push(XMedia::Video {
                        url: direct,
                        thumbnail_url: thumb,
                        alt_text: item.alt_text,
                        bitrate: None,
                    });
                }
            } else if let Some(url) = item
                .url
                .as_deref()
                .and_then(|raw| parse_allowed_media_url(raw).ok())
            {
                mapped.push(XMedia::Image {
                    url,
                    alt_text: item.alt_text,
                });
            }
        }
    }
    if let Some(urls) = urls {
        for raw in urls {
            let Some(url) = parse_allowed_media_url(&raw).ok() else {
                continue;
            };
            let represented = mapped.iter().any(|media| match media {
                XMedia::Image { url: existing, .. } => existing == &url,
                XMedia::Video {
                    url: existing,
                    thumbnail_url,
                    ..
                } => existing.as_ref() == Some(&url) || thumbnail_url.as_ref() == Some(&url),
            });
            if represented {
                continue;
            }
            if url.host_str() == Some("video.twimg.com")
                && url.path().to_ascii_lowercase().ends_with(".mp4")
            {
                mapped.push(XMedia::Video {
                    url: Some(url),
                    thumbnail_url: None,
                    alt_text: None,
                    bitrate: None,
                });
            } else {
                mapped.push(XMedia::Image {
                    url,
                    alt_text: None,
                });
            }
        }
    }
    (mapped, advertised)
}

pub(crate) async fn fetch(
    client: &NoRedirectClient,
    config: &TwitterFetchConfig,
    identity: &XStatusIdentity,
    timeout: Duration,
) -> Result<XPost, ProviderError> {
    let future = async {
        if identity.id.is_empty() || !identity.id.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(error(
                ProviderErrorKind::Incomplete,
                "requested status ID was invalid",
                None,
            ));
        }
        let mut endpoint = config.vxtwitter_api_base.clone();
        let mut path = endpoint.path().trim_end_matches('/').to_owned();
        path.push_str("/Twitter/status/");
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
            TwitterProvider::VxTwitter,
            response,
            config.response_max_bytes,
        )
        .await?;
        parse(&body, identity)
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
    use crate::tools::twitter_extractor::test_support::TestServer;
    use crate::tools::twitter_extractor::url::XStatusIdentity;
    use std::time::Duration;
    use url::Url;

    fn identity(id: &str) -> XStatusIdentity {
        XStatusIdentity {
            id: id.to_owned(),
            canonical_url: Url::parse(&format!("https://x.com/i/status/{id}")).unwrap(),
        }
    }

    fn test_config_with_vx_base(base: Url) -> TwitterFetchConfig {
        TwitterFetchConfig {
            providers: vec![TwitterProvider::VxTwitter],
            fxtwitter_api_base: base.clone(),
            vxtwitter_api_base: base.clone(),
            jina_reader_endpoint: base,
            jina_api_key: None,
            total_timeout: Duration::from_secs(5),
            provider_timeout: Duration::from_secs(2),
            response_max_bytes: 1024 * 1024,
        }
    }

    #[test]
    fn parses_vxtwitter_extended_media_without_duplicates() {
        let post = parse(
            include_bytes!("../fixtures/vxtwitter_photo_quote.json"),
            &identity("123"),
        )
        .unwrap();
        assert_eq!(post.media.len(), 1);
        assert_eq!(post.quote.as_ref().unwrap().text, "quoted text");
    }

    #[test]
    fn rejects_vxtwitter_announced_media_with_only_disallowed_urls() {
        let body = br#"{"tweetID":"123","text":"root","media_extended":[{"type":"image","url":"https://example.com/a.jpg"}]}"#;
        assert_eq!(
            parse(body, &identity("123")).unwrap_err().kind,
            ProviderErrorKind::Incomplete
        );
    }

    #[test]
    fn parses_vxtwitter_direct_video_and_thumbnail() {
        let post = parse(
            include_bytes!("../fixtures/vxtwitter_video.json"),
            &identity("123"),
        )
        .unwrap();
        let XMedia::Video {
            url, thumbnail_url, ..
        } = &post.media[0]
        else {
            panic!("expected video")
        };
        assert_eq!(
            url.as_ref().unwrap().as_str(),
            "https://video.twimg.com/video/vx.mp4"
        );
        assert_eq!(
            thumbnail_url.as_ref().unwrap().as_str(),
            "https://pbs.twimg.com/media/vx-thumb.jpg"
        );
    }

    #[tokio::test]
    async fn vxtwitter_fetch_uses_twitter_status_endpoint() {
        let server = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("../fixtures/vxtwitter_photo_quote.json"),
        );
        let config = test_config_with_vx_base(server.base_url());
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
}

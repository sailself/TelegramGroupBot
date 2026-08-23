use std::collections::HashSet;

use ::url::Url;
use anyhow::{anyhow, Result};

use super::url::XStatusIdentity;

#[derive(Debug, Clone)]
pub struct TwitterContent {
    #[allow(dead_code)]
    pub url: String,
    pub text_content: String,
    pub image_urls: Vec<String>,
    pub video_urls: Vec<String>,
    pub formatted_content: String,
    #[allow(dead_code)]
    pub(crate) attachment_plan: Vec<TwitterAttachment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TwitterAttachment {
    Image {
        url: String,
    },
    Video {
        url: String,
        thumbnail_url: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XPost {
    pub id: String,
    pub author: Option<XAuthor>,
    pub text: String,
    pub created_at: Option<String>,
    pub media: Vec<XMedia>,
    pub quote: Option<Box<XPost>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XAuthor {
    pub display_name: Option<String>,
    pub handle: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XMedia {
    Image {
        url: Url,
        alt_text: Option<String>,
    },
    Video {
        url: Option<Url>,
        thumbnail_url: Option<Url>,
        alt_text: Option<String>,
        bitrate: Option<u64>,
    },
}

pub(crate) fn parse_allowed_media_url(raw_url: &str) -> Result<Url> {
    let parsed =
        Url::parse(raw_url.trim()).map_err(|error| anyhow!("invalid media URL: {error}"))?;
    if parsed.scheme() != "https" {
        return Err(anyhow!("media URL must use HTTPS"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(anyhow!("media URL must not contain credentials"));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("media URL has no host"))?
        .to_ascii_lowercase();
    if host != "pbs.twimg.com" && host != "video.twimg.com" {
        return Err(anyhow!("media URL host is not allowlisted"));
    }
    if parsed.port().is_some_and(|port| port != 443) {
        return Err(anyhow!("media URL has a non-default port"));
    }
    Ok(parsed)
}

pub(crate) fn validate_complete_post(post: &XPost) -> Result<()> {
    let root_has_text = !post.text.trim().is_empty();
    let root_has_media = post.media.iter().any(media_is_usable);
    let quote_has_text = post
        .quote
        .as_deref()
        .map(|quote| !quote.text.trim().is_empty())
        .unwrap_or(false);
    let quote_has_media = post
        .quote
        .as_deref()
        .map(|quote| quote.media.iter().any(media_is_usable))
        .unwrap_or(false);

    if root_has_text || root_has_media || quote_has_text || quote_has_media {
        Ok(())
    } else {
        Err(anyhow!(
            "Twitter/X post has no usable text, media, or quote"
        ))
    }
}

fn media_is_usable(media: &XMedia) -> bool {
    match media {
        XMedia::Image { url, .. } => is_usable_pbs_image_url(url),
        XMedia::Video {
            url, thumbnail_url, ..
        } => {
            url.as_ref().is_some_and(is_usable_direct_video_url)
                || thumbnail_url.as_ref().is_some_and(is_usable_pbs_image_url)
        }
    }
}

pub(crate) fn is_usable_pbs_image_url(url: &Url) -> bool {
    parse_allowed_media_url(url.as_str()).is_ok() && url.host_str() == Some("pbs.twimg.com")
}

pub(crate) fn is_usable_direct_video_url(url: &Url) -> bool {
    parse_allowed_media_url(url.as_str()).is_ok()
        && url.host_str() == Some("video.twimg.com")
        && url.path().to_ascii_lowercase().ends_with(".mp4")
}

pub(crate) fn build_twitter_content(
    identity: &XStatusIdentity,
    post: XPost,
) -> Result<TwitterContent> {
    validate_complete_post(&post)?;

    let mut attachment_plan = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut image_descriptions = Vec::new();
    let mut thumbnail_only_video = false;

    let mut text_sections = Vec::new();
    append_post_text(&post, &mut text_sections, false);
    append_media(
        &post.media,
        &mut attachment_plan,
        &mut image_descriptions,
        &mut thumbnail_only_video,
        &mut seen_urls,
    );

    if let Some(quote) = post.quote.as_deref() {
        let mut quote_sections = Vec::new();
        append_post_text(quote, &mut quote_sections, true);
        append_media(
            &quote.media,
            &mut attachment_plan,
            &mut image_descriptions,
            &mut thumbnail_only_video,
            &mut seen_urls,
        );
        if !quote_sections.is_empty() {
            text_sections.extend(quote_sections);
        }
    }

    text_sections.push(format!("Original link: {}", identity.canonical_url));
    if !image_descriptions.is_empty() {
        text_sections.extend(image_descriptions);
    }
    if thumbnail_only_video {
        text_sections.push("Video present; thumbnail attached".to_owned());
    }
    let image_urls = attachment_plan
        .iter()
        .filter_map(|attachment| match attachment {
            TwitterAttachment::Image { url } => Some(url.clone()),
            TwitterAttachment::Video { .. } => None,
        })
        .collect::<Vec<_>>();
    let video_urls = attachment_plan
        .iter()
        .filter_map(|attachment| match attachment {
            TwitterAttachment::Image { .. } => None,
            TwitterAttachment::Video { url, .. } => Some(url.clone()),
        })
        .collect::<Vec<_>>();
    let video_thumbnail_fallback_count = attachment_plan
        .iter()
        .filter(|attachment| {
            matches!(
                attachment,
                TwitterAttachment::Video {
                    thumbnail_url: Some(_),
                    ..
                }
            )
        })
        .count();
    if !video_urls.is_empty() {
        text_sections.push(format!(
            "Videos detected: {} video(s); thumbnail fallback available for {}",
            video_urls.len(),
            video_thumbnail_fallback_count
        ));
    }

    let text_content = text_sections.join("\n\n");
    let mut formatted_content = format!("\n\n--- Twitter Content ---\n{text_content}");
    if !image_urls.is_empty() {
        formatted_content.push_str(&format!(
            "\n\nImages attached: {} image(s)",
            image_urls.len()
        ));
    }
    if !video_urls.is_empty() {
        formatted_content.push_str(&format!(
            "\nVideos available: {} video(s)",
            video_urls.len()
        ));
    }
    formatted_content.push_str("\n--- End Twitter Content ---\n\n");

    Ok(TwitterContent {
        url: identity.canonical_url.to_string(),
        text_content,
        image_urls,
        video_urls,
        formatted_content,
        attachment_plan,
    })
}

fn append_post_text(post: &XPost, sections: &mut Vec<String>, quote: bool) {
    let mut header_parts = Vec::new();
    if let Some(author) = post.author.as_ref() {
        if let Some(name) = non_empty(author.display_name.as_deref()) {
            header_parts.push(name.to_owned());
        }
        if let Some(handle) = non_empty(author.handle.as_deref()) {
            if !header_parts.iter().any(|part| part == handle) {
                header_parts.push(handle.to_owned());
            }
        }
    }

    if quote {
        sections.push("Quoted post:".to_owned());
    }
    if !header_parts.is_empty() {
        let header = format!("Tweet by {}", header_parts.join(" "));
        if !quote {
            if let Some(created_at) = non_empty(post.created_at.as_deref()) {
                sections.push(format!("{header} at {created_at}"));
            } else {
                sections.push(header);
            }
        } else {
            sections.push(header);
        }
    } else if let Some(created_at) = non_empty(post.created_at.as_deref()) {
        if !quote {
            sections.push(format!("Tweet at {created_at}"));
        } else {
            sections.push(format!("Posted at {created_at}"));
        }
    }
    if let Some(text) = non_empty(Some(post.text.as_str())) {
        sections.push(text.to_owned());
    }
}

fn append_media(
    media: &[XMedia],
    attachment_plan: &mut Vec<TwitterAttachment>,
    image_descriptions: &mut Vec<String>,
    thumbnail_only_video: &mut bool,
    seen_urls: &mut HashSet<String>,
) {
    let mut selected_videos: Vec<(usize, &XMedia)> = Vec::new();
    let mut selected_by_thumbnail = std::collections::HashMap::<String, usize>::new();
    let valid_direct_thumbnail_keys = media
        .iter()
        .filter_map(|item| {
            let XMedia::Video {
                url: Some(url),
                thumbnail_url: Some(thumbnail_url),
                ..
            } = item
            else {
                return None;
            };
            if is_usable_direct_video_url(url) {
                usable_pbs_url(thumbnail_url).map(|url| url.to_string())
            } else {
                None
            }
        })
        .collect::<HashSet<_>>();
    for (media_index, item) in media.iter().enumerate() {
        let XMedia::Video {
            url,
            thumbnail_url,
            bitrate,
            ..
        } = item
        else {
            continue;
        };
        let key = thumbnail_url
            .as_ref()
            .and_then(usable_pbs_url)
            .map(|url| url.to_string());
        if key.as_ref().is_some_and(|key| {
            valid_direct_thumbnail_keys.contains(key)
                && !url.as_ref().is_some_and(is_usable_direct_video_url)
        }) {
            continue;
        }
        if let Some(key) = key {
            if let Some(index) = selected_by_thumbnail.get(&key).copied() {
                let current_bitrate = match selected_videos[index].1 {
                    XMedia::Video { bitrate, .. } => bitrate.unwrap_or(0),
                    XMedia::Image { .. } => 0,
                };
                if bitrate.unwrap_or(0) > current_bitrate {
                    selected_videos[index].1 = item;
                }
            } else {
                selected_by_thumbnail.insert(key, selected_videos.len());
                selected_videos.push((media_index, item));
            }
        } else {
            selected_videos.push((media_index, item));
        }
    }

    let selected_videos = selected_videos
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    for (media_index, item) in media.iter().enumerate() {
        match item {
            XMedia::Image { url, alt_text } => {
                let Some(url) = usable_pbs_url(url) else {
                    continue;
                };
                let url = url.to_string();
                if seen_urls.insert(url.clone()) {
                    attachment_plan.push(TwitterAttachment::Image { url });
                    append_alt_text(image_descriptions, alt_text.as_deref());
                }
            }
            XMedia::Video { .. } => {
                let Some(XMedia::Video {
                    url,
                    thumbnail_url,
                    alt_text,
                    ..
                }) = selected_videos.get(&media_index).copied()
                else {
                    continue;
                };
                let direct_url = url
                    .as_ref()
                    .filter(|url| is_usable_direct_video_url(url))
                    .map(ToString::to_string);
                let thumbnail = thumbnail_url
                    .as_ref()
                    .and_then(usable_pbs_url)
                    .map(ToString::to_string);
                append_alt_text(image_descriptions, alt_text.as_deref());
                match (direct_url, thumbnail) {
                    (Some(url), thumbnail_url) => {
                        if seen_urls.insert(url.clone()) {
                            attachment_plan.push(TwitterAttachment::Video { url, thumbnail_url });
                        }
                    }
                    (None, Some(url)) => {
                        if seen_urls.insert(url.clone()) {
                            attachment_plan.push(TwitterAttachment::Image { url });
                        }
                        *thumbnail_only_video = true;
                    }
                    (None, None) => {}
                }
            }
        }
    }
}

fn usable_pbs_url(url: &Url) -> Option<&Url> {
    if is_usable_pbs_image_url(url) {
        Some(url)
    } else {
        None
    }
}

fn append_alt_text(descriptions: &mut Vec<String>, alt_text: Option<&str>) {
    if let Some(alt_text) = non_empty(alt_text) {
        descriptions.push(format!("Image description: {alt_text}"));
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::url::XStatusIdentity;

    fn identity(id: &str) -> XStatusIdentity {
        XStatusIdentity {
            id: id.to_owned(),
            canonical_url: ::url::Url::parse(&format!("https://x.com/i/status/{id}")).unwrap(),
        }
    }

    fn author(name: &str, handle: &str) -> XAuthor {
        XAuthor {
            display_name: Some(name.to_owned()),
            handle: Some(handle.to_owned()),
        }
    }

    fn post_with_text(text: &str) -> XPost {
        XPost {
            id: "post".to_owned(),
            author: None,
            text: text.to_owned(),
            created_at: None,
            media: Vec::new(),
            quote: None,
        }
    }

    fn post_with_image(url: &str) -> XPost {
        let mut post = post_with_text("");
        post.media.push(XMedia::Image {
            url: ::url::Url::parse(url).unwrap(),
            alt_text: None,
        });
        post
    }

    fn post_with_quote(quote: XPost) -> XPost {
        let mut post = post_with_text("");
        post.quote = Some(Box::new(quote));
        post
    }

    fn empty_post() -> XPost {
        post_with_text("")
    }

    fn post_with_video_and_thumbnail() -> XPost {
        let mut post = post_with_text("video");
        post.media.push(XMedia::Video {
            url: Some(::url::Url::parse("https://video.twimg.com/video/low.mp4").unwrap()),
            thumbnail_url: Some(
                ::url::Url::parse("https://pbs.twimg.com/media/thumb.jpg").unwrap(),
            ),
            alt_text: None,
            bitrate: Some(100),
        });
        post.media.push(XMedia::Video {
            url: Some(::url::Url::parse("https://video.twimg.com/video/high.mp4").unwrap()),
            thumbnail_url: Some(
                ::url::Url::parse("https://pbs.twimg.com/media/thumb.jpg").unwrap(),
            ),
            alt_text: None,
            bitrate: Some(500),
        });
        post
    }

    fn post_with_thumbnail_only_video() -> XPost {
        let mut post = post_with_text("video");
        post.media.push(XMedia::Video {
            url: None,
            thumbnail_url: Some(
                ::url::Url::parse("https://pbs.twimg.com/media/thumb.jpg").unwrap(),
            ),
            alt_text: None,
            bitrate: None,
        });
        post
    }

    #[test]
    fn media_url_allowlist_is_exact_and_https_only() {
        assert!(parse_allowed_media_url("https://pbs.twimg.com/media/a.jpg").is_ok());
        assert!(parse_allowed_media_url("https://video.twimg.com/ext_tw_video/a.mp4").is_ok());
        assert!(parse_allowed_media_url("http://pbs.twimg.com/media/a.jpg").is_err());
        assert!(parse_allowed_media_url("https://evilpbs.twimg.com/media/a.jpg").is_err());
        assert!(parse_allowed_media_url("https://127.0.0.1/a.jpg").is_err());
        assert!(parse_allowed_media_url("https://user@pbs.twimg.com/a.jpg").is_err());
    }

    #[test]
    fn completeness_accepts_text_media_or_quote_and_rejects_empty_posts() {
        assert!(validate_complete_post(&post_with_text("hello")).is_ok());
        assert!(
            validate_complete_post(&post_with_image("https://pbs.twimg.com/media/a.jpg")).is_ok()
        );
        assert!(validate_complete_post(&post_with_quote(post_with_text("quoted"))).is_ok());
        assert!(validate_complete_post(&empty_post()).is_err());
    }

    #[test]
    fn completeness_rejects_security_allowed_but_semantically_unusable_media() {
        let video_host_image = post_with_image("https://video.twimg.com/video/not-an-image.jpg");
        assert!(validate_complete_post(&video_host_image).is_err());

        let mut video_host_thumbnail = empty_post();
        video_host_thumbnail.media.push(XMedia::Video {
            url: None,
            thumbnail_url: Some(
                ::url::Url::parse("https://video.twimg.com/video/not-a-thumbnail.jpg").unwrap(),
            ),
            alt_text: None,
            bitrate: None,
        });
        assert!(validate_complete_post(&video_host_thumbnail).is_err());
    }

    #[test]
    fn formatter_preserves_markers_and_flattens_one_quote_level() {
        let mut root = post_with_text("root text");
        root.author = Some(author("Alice", "alice"));
        root.quote = Some(Box::new(post_with_text("quoted text")));
        root.quote.as_mut().unwrap().quote = Some(Box::new(post_with_text("too deep")));

        let content = build_twitter_content(&identity("123"), root).unwrap();
        assert!(content
            .formatted_content
            .starts_with("\n\n--- Twitter Content ---\n"));
        assert!(content.text_content.contains("root text"));
        assert!(content.text_content.contains("Quoted post:"));
        assert!(content.text_content.contains("quoted text"));
        assert!(!content.text_content.contains("too deep"));
        assert!(content
            .text_content
            .contains("Original link: https://x.com/i/status/123"));
    }

    #[test]
    fn formatter_uses_direct_video_without_duplicate_thumbnail() {
        let content =
            build_twitter_content(&identity("123"), post_with_video_and_thumbnail()).unwrap();
        assert_eq!(
            content.video_urls,
            vec!["https://video.twimg.com/video/high.mp4"]
        );
        assert!(content.image_urls.is_empty());
        assert_eq!(
            content.attachment_plan,
            vec![TwitterAttachment::Video {
                url: "https://video.twimg.com/video/high.mp4".into(),
                thumbnail_url: Some("https://pbs.twimg.com/media/thumb.jpg".into()),
            }]
        );
        assert!(content
            .formatted_content
            .contains("Videos available: 1 video(s)"));
        assert!(!content.formatted_content.contains("Videos attached:"));
    }

    #[test]
    fn formatter_ignores_hls_variant_when_selecting_direct_mp4() {
        let thumbnail = ::url::Url::parse("https://pbs.twimg.com/media/thumb.jpg").unwrap();
        let mut post = post_with_text("video");
        post.media.push(XMedia::Video {
            url: Some(::url::Url::parse("https://video.twimg.com/video/low.mp4?tag=12").unwrap()),
            thumbnail_url: Some(thumbnail.clone()),
            alt_text: None,
            bitrate: Some(100),
        });
        post.media.push(XMedia::Video {
            url: Some(::url::Url::parse("https://video.twimg.com/video/high.m3u8").unwrap()),
            thumbnail_url: Some(thumbnail),
            alt_text: None,
            bitrate: Some(500),
        });

        let content = build_twitter_content(&identity("123"), post).unwrap();

        assert_eq!(
            content.video_urls,
            vec!["https://video.twimg.com/video/low.mp4?tag=12"]
        );
    }

    #[test]
    fn formatter_uses_thumbnail_when_video_has_no_direct_url() {
        let content =
            build_twitter_content(&identity("123"), post_with_thumbnail_only_video()).unwrap();
        assert!(content.video_urls.is_empty());
        assert_eq!(
            content.image_urls,
            vec!["https://pbs.twimg.com/media/thumb.jpg"]
        );
        assert!(content
            .text_content
            .contains("Video present; thumbnail attached"));
    }
}

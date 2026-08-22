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
    pub(crate) video_thumbnail_fallbacks: Vec<VideoThumbnailFallback>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VideoThumbnailFallback {
    pub video_url: String,
    pub thumbnail_url: String,
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
        XMedia::Image { url, .. } => parse_allowed_media_url(url.as_str()).is_ok(),
        XMedia::Video {
            url, thumbnail_url, ..
        } => {
            url.as_ref()
                .is_some_and(|url| parse_allowed_media_url(url.as_str()).is_ok())
                || thumbnail_url
                    .as_ref()
                    .is_some_and(|url| parse_allowed_media_url(url.as_str()).is_ok())
        }
    }
}

pub(crate) fn build_twitter_content(
    identity: &XStatusIdentity,
    post: XPost,
) -> Result<TwitterContent> {
    validate_complete_post(&post)?;

    let mut image_urls = Vec::new();
    let mut video_urls = Vec::new();
    let mut video_thumbnail_fallbacks = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut image_descriptions = Vec::new();
    let mut thumbnail_only_video = false;

    let mut text_sections = Vec::new();
    append_post_text(&post, &mut text_sections, false);
    append_media(
        &post.media,
        &mut image_urls,
        &mut video_urls,
        &mut video_thumbnail_fallbacks,
        &mut image_descriptions,
        &mut thumbnail_only_video,
        &mut seen_urls,
    );

    if let Some(quote) = post.quote.as_deref() {
        let mut quote_sections = Vec::new();
        append_post_text(quote, &mut quote_sections, true);
        append_media(
            &quote.media,
            &mut image_urls,
            &mut video_urls,
            &mut video_thumbnail_fallbacks,
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
    if !video_urls.is_empty() {
        text_sections.push(format!(
            "Videos detected: {} video(s); thumbnail fallback available for {}",
            video_urls.len(),
            video_thumbnail_fallbacks.len()
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
        formatted_content.push_str(&format!("\nVideos attached: {} video(s)", video_urls.len()));
    }
    formatted_content.push_str("\n--- End Twitter Content ---\n\n");

    Ok(TwitterContent {
        url: identity.canonical_url.to_string(),
        text_content,
        image_urls,
        video_urls,
        formatted_content,
        video_thumbnail_fallbacks,
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
    image_urls: &mut Vec<String>,
    video_urls: &mut Vec<String>,
    video_thumbnail_fallbacks: &mut Vec<VideoThumbnailFallback>,
    image_descriptions: &mut Vec<String>,
    thumbnail_only_video: &mut bool,
    seen_urls: &mut HashSet<String>,
) {
    let mut videos = Vec::new();
    for item in media {
        match item {
            XMedia::Image { url, alt_text } => {
                let Some(url) = allowed_url(url) else {
                    continue;
                };
                let url = url.to_string();
                if seen_urls.insert(url.clone()) {
                    image_urls.push(url);
                    append_alt_text(image_descriptions, alt_text.as_deref());
                }
            }
            XMedia::Video { .. } => videos.push(item),
        }
    }

    let mut selected_videos: Vec<&XMedia> = Vec::new();
    let mut selected_by_thumbnail = std::collections::HashMap::<String, usize>::new();
    for item in videos {
        let XMedia::Video {
            thumbnail_url,
            bitrate,
            ..
        } = item
        else {
            continue;
        };
        let key = thumbnail_url
            .as_ref()
            .and_then(allowed_url)
            .map(|url| url.to_string());
        if let Some(key) = key {
            if let Some(index) = selected_by_thumbnail.get(&key).copied() {
                let current_bitrate = match selected_videos[index] {
                    XMedia::Video { bitrate, .. } => bitrate.unwrap_or(0),
                    XMedia::Image { .. } => 0,
                };
                if bitrate.unwrap_or(0) > current_bitrate {
                    selected_videos[index] = item;
                }
            } else {
                selected_by_thumbnail.insert(key, selected_videos.len());
                selected_videos.push(item);
            }
        } else {
            selected_videos.push(item);
        }
    }

    for item in selected_videos {
        let XMedia::Video {
            url,
            thumbnail_url,
            alt_text,
            ..
        } = item
        else {
            continue;
        };
        let direct_url = url
            .as_ref()
            .and_then(allowed_url)
            .map(|url| url.to_string());
        let thumbnail = thumbnail_url
            .as_ref()
            .and_then(allowed_url)
            .map(|url| url.to_string());
        append_alt_text(image_descriptions, alt_text.as_deref());
        match (direct_url, thumbnail) {
            (Some(video_url), thumbnail) => {
                if seen_urls.insert(video_url.clone()) {
                    video_urls.push(video_url.clone());
                    if let Some(thumbnail_url) = thumbnail {
                        video_thumbnail_fallbacks.push(VideoThumbnailFallback {
                            video_url,
                            thumbnail_url,
                        });
                    }
                }
            }
            (None, Some(thumbnail_url)) => {
                if seen_urls.insert(thumbnail_url.clone()) {
                    image_urls.push(thumbnail_url);
                }
                *thumbnail_only_video = true;
            }
            (None, None) => {}
        }
    }
}

fn allowed_url(url: &Url) -> Option<&Url> {
    if parse_allowed_media_url(url.as_str()).is_ok() {
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
            content.video_thumbnail_fallbacks,
            vec![VideoThumbnailFallback {
                video_url: "https://video.twimg.com/video/high.mp4".into(),
                thumbnail_url: "https://pbs.twimg.com/media/thumb.jpg".into(),
            }]
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

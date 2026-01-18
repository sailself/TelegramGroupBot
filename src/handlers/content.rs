use std::time::Duration;

use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use teloxide::types::{MessageEntityKind, MessageEntityRef};
use tracing::warn;

use crate::config::CONFIG;
use crate::llm::media::download_media;
use crate::tools::telegraph_extractor::{extract_telegraph_content, TelegraphContent};
use crate::tools::twitter_extractor::{extract_twitter_content, TwitterContent};
use crate::utils::http::get_http_client;

fn markdown_to_telegraph_nodes(content: &str) -> Vec<serde_json::Value> {
    let mut nodes = Vec::new();
    let image_regex = Regex::new(r"^!\[([^\]]*)\]\(([^)]+)\)$").unwrap();
    for paragraph in content.split("\n\n") {
        let text = paragraph.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(captures) = image_regex.captures(text) {
            let alt = captures.get(1).map(|m| m.as_str()).unwrap_or("");
            let src = captures.get(2).map(|m| m.as_str()).unwrap_or("");
            if !src.is_empty() {
                nodes.push(json!({
                    "tag": "img",
                    "attrs": { "src": src }
                }));
                if !alt.is_empty() {
                    nodes.push(json!({
                        "tag": "figcaption",
                        "children": [alt]
                    }));
                }
                continue;
            }
        }
        nodes.push(json!({
            "tag": "p",
            "children": [text],
        }));
    }
    nodes
}

#[derive(Debug, Deserialize)]
struct TelegraphCreateResponse {
    ok: bool,
    result: Option<TelegraphCreateResult>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegraphCreateResult {
    url: String,
}

pub async fn create_telegraph_page(title: &str, content: &str) -> Option<String> {
    if CONFIG.telegraph_access_token.trim().is_empty() {
        warn!("Telegraph access token missing; skipping page creation");
        return None;
    }

    let nodes = markdown_to_telegraph_nodes(content);
    let content_json = serde_json::to_string(&nodes).unwrap_or_else(|_| "[]".to_string());
    let mut form = Vec::new();
    form.push(("access_token".to_string(), CONFIG.telegraph_access_token.clone()));
    form.push(("author_name".to_string(), CONFIG.telegraph_author_name.clone()));
    form.push(("author_url".to_string(), CONFIG.telegraph_author_url.clone()));
    form.push(("title".to_string(), title.to_string()));
    form.push(("content".to_string(), content_json));
    form.push(("return_content".to_string(), "false".to_string()));

    let client = get_http_client();
    let response = client
        .post("https://api.telegra.ph/createPage")
        .timeout(Duration::from_secs(10))
        .form(&form)
        .send()
        .await
        .ok()?;

    if !response.status().is_success() {
        warn!("Telegraph API call failed with status {}", response.status());
        return None;
    }

    let payload = response.json::<TelegraphCreateResponse>().await.ok()?;
    if payload.ok {
        return payload.result.map(|result| result.url);
    }

    warn!("Telegraph API error: {}", payload.error.unwrap_or_default());
    None
}

pub fn extract_youtube_urls(text: &str, max_urls: usize) -> (String, Vec<String>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

    let pattern = Regex::new(
        r"((?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})(?:[\?&][^\s]*)?)",
    )
    .unwrap();
    let mut matches = pattern.captures_iter(text).collect::<Vec<_>>();
    let mut urls = Vec::new();
    let mut new_text = text.to_string();
    let mut count = 0;

    matches.reverse();
    for caps in matches {
        if count >= max_urls {
            break;
        }
        let vid_id = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        if vid_id.is_empty() {
            continue;
        }
        let url = format!("https://www.youtube.com/watch?v={}", vid_id);
        urls.insert(0, url.clone());
        if let Some(m) = caps.get(0) {
            let start = m.start();
            let end = m.end();
            new_text.replace_range(start..end, &format!("YouTube_{}", vid_id));
        }
        count += 1;
    }

    (new_text, urls)
}

fn clean_url_candidate(url: &str) -> &str {
    url.trim_end_matches(|ch: char| matches!(ch, ')' | ']' | '}' | '>' | '"' | '\'' | ',' | '.' | ';' | ':'))
}

fn is_telegraph_url(url: &str) -> bool {
    let lowered = url.to_lowercase();
    lowered.contains("telegra.ph") || lowered.contains("t.me/")
}

pub async fn extract_telegraph_urls_and_content(
    text: &str,
    message_entities: Option<&[MessageEntityRef<'_>]>,
    max_urls: usize,
) -> (String, Vec<TelegraphContent>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

    let url_pattern = Regex::new(r#"https?://(?:telegra\.ph|t\.me)/[^\s\)>"]+"#).unwrap();
    let markdown_link_pattern = Regex::new(r#"\[[^\]]*\]\((https?://[^)]+)\)"#).unwrap();
    let html_link_pattern = Regex::new(r#"href=["'](https?://[^"']+)["']"#).unwrap();
    let mut urls = Vec::new();

    if let Some(entities) = message_entities {
        for entity in entities {
            if urls.len() >= max_urls {
                break;
            }
            let candidate = match entity.kind() {
                MessageEntityKind::Url => entity.text(),
                MessageEntityKind::TextLink { url } => url.as_str(),
                _ => continue,
            };
            let candidate = clean_url_candidate(candidate);
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }

    for m in url_pattern.find_iter(text) {
        urls.push(m.as_str().to_string());
    }
    for caps in markdown_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for caps in html_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }

    urls.sort();
    urls.dedup();

    let mut new_text = text.to_string();
    let mut extracted = Vec::new();
    for url in urls.into_iter().take(max_urls) {
        match extract_telegraph_content(&url).await {
            Ok(content) => {
                let formatted = format!("\n[Telegraph content extracted from {}]\n{}\n", url, content.text_content);
                new_text.push_str(&formatted);
                extracted.push(content);
            }
            Err(err) => {
                warn!("Telegraph extraction failed for {}: {}", url, err);
                new_text.push_str(&format!("\n[Telegraph content extraction failed for {}]\n", url));
            }
        }
    }

    (new_text, extracted)
}

pub async fn extract_twitter_urls_and_content(
    text: &str,
    message_entities: Option<&[MessageEntityRef<'_>]>,
    max_urls: usize,
) -> (String, Vec<TwitterContent>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

    let url_pattern = Regex::new(
        r#"(https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com|fxtwitter\.com|vxtwitter\.com|fixupx\.com|fixvx\.com|twittpr\.com|pxtwitter\.com|tweetpik\.com)/[^\s\)>"]+)"#,
    )
    .unwrap();
    let markdown_link_pattern = Regex::new(r#"\[[^\]]*\]\((https?://[^)]+)\)"#).unwrap();
    let html_link_pattern = Regex::new(r#"href=["'](https?://[^"']+)["']"#).unwrap();

    let mut urls = Vec::new();
    if let Some(entities) = message_entities {
        for entity in entities {
            if urls.len() >= max_urls {
                break;
            }
            let candidate = match entity.kind() {
                MessageEntityKind::Url => entity.text(),
                MessageEntityKind::TextLink { url } => url.as_str(),
                _ => continue,
            };
            let candidate = clean_url_candidate(candidate);
            if url_pattern.is_match(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for m in url_pattern.find_iter(text) {
        urls.push(m.as_str().to_string());
    }
    for caps in markdown_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if url_pattern.is_match(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for caps in html_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if url_pattern.is_match(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    urls.sort();
    urls.dedup();

    let mut new_text = text.to_string();
    let mut extracted = Vec::new();

    for url in urls.into_iter().take(max_urls) {
        match extract_twitter_content(&url).await {
            Ok(content) => {
                if !content.formatted_content.is_empty() {
                    new_text.push_str(&content.formatted_content);
                }
                extracted.push(content);
            }
            Err(err) => {
                warn!("Twitter extraction failed for {}: {}", url, err);
                new_text.push_str(&format!("\n[Twitter content extraction failed for {}]\n", url));
            }
        }
    }

    (new_text, extracted)
}

pub async fn download_telegraph_media(
    contents: &[TelegraphContent],
    max_images: usize,
    max_videos: usize,
) -> (Vec<Vec<u8>>, Option<Vec<u8>>, Option<String>) {
    let mut image_data_list = Vec::new();
    let mut video_data = None;
    let mut video_mime_type = None;

    for content in contents {
        for url in &content.image_urls {
            if image_data_list.len() >= max_images {
                break;
            }
            if let Some(bytes) = download_media(url).await {
                image_data_list.push(bytes);
            }
        }

        if video_data.is_some() || max_videos == 0 {
            continue;
        }

        for url in &content.video_urls {
            if let Some(bytes) = download_media(url).await {
                let lowered = url.to_lowercase();
                let mime = if lowered.ends_with(".m3u8") {
                    "application/x-mpegURL"
                } else if lowered.ends_with(".mpd") {
                    "application/dash+xml"
                } else if lowered.ends_with(".webm") {
                    "video/webm"
                } else {
                    "video/mp4"
                };
                video_data = Some(bytes);
                video_mime_type = Some(mime.to_string());
                break;
            }
        }
    }

    (image_data_list, video_data, video_mime_type)
}

pub async fn download_twitter_media(
    contents: &[TwitterContent],
    max_images: usize,
    max_videos: usize,
) -> (Vec<Vec<u8>>, Option<Vec<u8>>, Option<String>) {
    let mut image_data_list = Vec::new();
    let mut video_data = None;
    let mut video_mime_type = None;

    for content in contents {
        for url in &content.image_urls {
            if image_data_list.len() >= max_images {
                break;
            }
            if let Some(bytes) = download_media(url).await {
                image_data_list.push(bytes);
            }
        }

        if video_data.is_some() || max_videos == 0 {
            continue;
        }

        for url in &content.video_urls {
            if let Some(bytes) = download_media(url).await {
                let lowered = url.to_lowercase();
                let mime = if lowered.ends_with(".m3u8") {
                    "application/x-mpegURL"
                } else if lowered.ends_with(".mpd") {
                    "application/dash+xml"
                } else if lowered.ends_with(".webm") {
                    "video/webm"
                } else {
                    "video/mp4"
                };
                video_data = Some(bytes);
                video_mime_type = Some(mime.to_string());
                break;
            }
        }
    }

    (image_data_list, video_data, video_mime_type)
}

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};
use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use std::sync::LazyLock;
use teloxide::types::{MessageEntityKind, MessageEntityRef};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, warn};

use crate::config::CONFIG;
use crate::tools::telegraph_extractor::{extract_telegraph_content, TelegraphContent};
use crate::tools::twitter_extractor::{
    canonical_status_key, extract_twitter_content, is_supported_status_url, TwitterContent,
};
use crate::utils::http::get_http_client;
use crate::utils::text::truncate_with_ellipsis;
use crate::utils::ttl_cache::TtlCache;

const EXTRACTION_CACHE_TTL: Duration = Duration::from_secs(900);
const EXTRACTION_CACHE_MAX_ENTRIES: usize = 64;

static YOUTUBE_URL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"((?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})(?:[\?&][^\s]*)?)",
    )
    .expect("valid youtube regex")
});
static TELEGRAPH_URL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"https?://(?:telegra\.ph|t\.me)/[^\s\)>"]+"#).expect("valid telegraph url regex")
});
static HTTP_URL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)https?://[^\s<>\"']+"#).expect("valid HTTP URL regex"));
static MARKDOWN_LINK_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\[[^\]]*\]\((https?://[^)]+)\)"#).expect("valid markdown link regex")
});
static HTML_LINK_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"href=["'](https?://[^"']+)["']"#).expect("valid html link regex")
});

static TELEGRAPH_CACHE: LazyLock<Mutex<TtlCache<String, TelegraphContent>>> = LazyLock::new(|| {
    Mutex::new(TtlCache::new(
        EXTRACTION_CACHE_TTL,
        EXTRACTION_CACHE_MAX_ENTRIES,
    ))
});
static TWITTER_CACHE: LazyLock<Mutex<TtlCache<String, TwitterContent>>> = LazyLock::new(|| {
    Mutex::new(TtlCache::new(
        EXTRACTION_CACHE_TTL,
        EXTRACTION_CACHE_MAX_ENTRIES,
    ))
});

fn log_extracted_content(
    source: &str,
    url: &str,
    text: &str,
    images: usize,
    videos: usize,
    audios: usize,
) {
    debug!(
        target: "content.extract",
        source = source,
        url = url,
        images = images,
        videos = videos,
        audios = audios,
        text = %truncate_with_ellipsis(text, 200)
    );
}

fn markdown_to_telegraph_nodes(content: &str) -> Vec<serde_json::Value> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    #[derive(Debug)]
    struct NodeBuilder {
        tag: String,
        attrs: Option<serde_json::Map<String, serde_json::Value>>,
        children: Vec<serde_json::Value>,
    }

    #[derive(Debug, Default)]
    struct TableBuilder {
        rows: Vec<Vec<String>>,
        current_row: Option<Vec<String>>,
        current_cell: Option<String>,
    }

    enum StackEntry {
        Node(NodeBuilder),
        Image { src: String, alt: String },
        Table(TableBuilder),
    }

    fn push_text(children: &mut Vec<serde_json::Value>, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(serde_json::Value::String(existing)) = children.last_mut() {
            existing.push_str(text);
            return;
        }
        children.push(serde_json::Value::String(text.to_string()));
    }

    fn push_value(
        stack: &mut [StackEntry],
        root: &mut Vec<serde_json::Value>,
        value: serde_json::Value,
    ) {
        if let Some(StackEntry::Node(parent)) = stack.last_mut() {
            parent.children.push(value);
        } else {
            root.push(value);
        }
    }

    fn close_node(stack: &mut Vec<StackEntry>, root: &mut Vec<serde_json::Value>) {
        let Some(entry) = stack.pop() else {
            return;
        };
        match entry {
            StackEntry::Node(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert("tag".to_string(), serde_json::Value::String(node.tag));
                if let Some(attrs) = node.attrs {
                    obj.insert("attrs".to_string(), serde_json::Value::Object(attrs));
                }
                if !node.children.is_empty() {
                    obj.insert(
                        "children".to_string(),
                        serde_json::Value::Array(node.children),
                    );
                }
                push_value(stack, root, serde_json::Value::Object(obj));
            }
            StackEntry::Image { src, alt } => {
                if !src.is_empty() {
                    push_value(
                        stack,
                        root,
                        json!({
                            "tag": "img",
                            "attrs": { "src": src }
                        }),
                    );
                }
                if !alt.trim().is_empty() {
                    push_value(
                        stack,
                        root,
                        json!({
                            "tag": "figcaption",
                            "children": [alt.trim()]
                        }),
                    );
                }
            }
            StackEntry::Table(table) => {
                if let Some(value) = render_table_list(table) {
                    push_value(stack, root, value);
                }
            }
        }
    }

    fn active_table_mut(stack: &mut [StackEntry]) -> Option<&mut TableBuilder> {
        stack.iter_mut().rev().find_map(|entry| match entry {
            StackEntry::Table(table) => Some(table),
            StackEntry::Node(_) | StackEntry::Image { .. } => None,
        })
    }

    fn push_table_text(stack: &mut [StackEntry], text: &str) -> bool {
        let Some(table) = active_table_mut(stack) else {
            return false;
        };
        let Some(cell) = table.current_cell.as_mut() else {
            return true;
        };
        cell.push_str(text);
        true
    }

    fn finish_table_cell(stack: &mut [StackEntry]) -> bool {
        let Some(table) = active_table_mut(stack) else {
            return false;
        };
        if let Some(cell) = table.current_cell.take() {
            table
                .current_row
                .get_or_insert_with(Vec::new)
                .push(normalize_table_cell(&cell));
        }
        true
    }

    fn finish_table_row(stack: &mut [StackEntry]) -> bool {
        if active_table_mut(stack).is_none() {
            return false;
        }
        finish_table_cell(stack);
        let Some(table) = active_table_mut(stack) else {
            return false;
        };
        if let Some(row) = table.current_row.take() {
            if row.iter().any(|cell| !cell.trim().is_empty()) {
                table.rows.push(row);
            }
        }
        true
    }

    fn normalize_table_cell(value: &str) -> String {
        value.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn render_table_list(mut table: TableBuilder) -> Option<serde_json::Value> {
        if table.current_cell.is_some() {
            if let Some(cell) = table.current_cell.take() {
                table
                    .current_row
                    .get_or_insert_with(Vec::new)
                    .push(normalize_table_cell(&cell));
            }
        }
        if let Some(row) = table.current_row.take() {
            if row.iter().any(|cell| !cell.trim().is_empty()) {
                table.rows.push(row);
            }
        }

        let rows = table
            .rows
            .into_iter()
            .filter(|row| row.iter().any(|cell| !cell.trim().is_empty()))
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return None;
        }

        let headers = rows[0].clone();
        let data_rows = rows.iter().skip(1).collect::<Vec<_>>();
        if data_rows.is_empty() {
            return None;
        }

        let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut items = Vec::new();
        for row in data_rows {
            let mut children = Vec::new();
            for index in 0..columns {
                let cell = row.get(index).map(String::as_str).unwrap_or("").trim();
                if cell.is_empty() {
                    continue;
                }
                if !children.is_empty() {
                    children.push(json!({ "tag": "br" }));
                }
                let header = headers
                    .get(index)
                    .map(String::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or("Column");
                children.push(json!({
                    "tag": "strong",
                    "children": [header]
                }));
                children.push(serde_json::Value::String(format!(": {}", cell)));
            }
            if !children.is_empty() {
                items.push(json!({
                    "tag": "li",
                    "children": children
                }));
            }
        }

        if items.is_empty() {
            return None;
        }

        Some(json!({
            "tag": "ul",
            "children": items
        }))
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(content, options);
    let mut root: Vec<serde_json::Value> = Vec::new();
    let mut stack: Vec<StackEntry> = Vec::new();

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Table(_) => stack.push(StackEntry::Table(TableBuilder::default())),
                Tag::TableHead => {
                    if let Some(table) = active_table_mut(&mut stack) {
                        table.current_row = Some(Vec::new());
                    }
                }
                Tag::TableRow => {
                    if let Some(table) = active_table_mut(&mut stack) {
                        table.current_row = Some(Vec::new());
                    }
                }
                Tag::TableCell => {
                    if let Some(table) = active_table_mut(&mut stack) {
                        table.current_cell = Some(String::new());
                    }
                }
                _ if active_table_mut(&mut stack).is_some() => {}
                Tag::Paragraph => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "p".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Heading(level, _, _) => {
                    let tag_name = match level {
                        HeadingLevel::H1 | HeadingLevel::H2 | HeadingLevel::H3 => "h3",
                        _ => "h4",
                    };
                    stack.push(StackEntry::Node(NodeBuilder {
                        tag: tag_name.to_string(),
                        attrs: None,
                        children: Vec::new(),
                    }))
                }
                Tag::BlockQuote => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "blockquote".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::List(Some(_)) => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "ol".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::List(None) => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "ul".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Item => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "li".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Emphasis => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "em".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Strong => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "strong".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Strikethrough => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "s".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::CodeBlock(_kind) => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "pre".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Link(_, dest, _) => {
                    let mut attrs = serde_json::Map::new();
                    attrs.insert(
                        "href".to_string(),
                        serde_json::Value::String(dest.to_string()),
                    );
                    stack.push(StackEntry::Node(NodeBuilder {
                        tag: "a".to_string(),
                        attrs: Some(attrs),
                        children: Vec::new(),
                    }))
                }
                Tag::Image(_, dest, _) => stack.push(StackEntry::Image {
                    src: dest.to_string(),
                    alt: String::new(),
                }),
                _ => {}
            },
            Event::End(tag) => match tag {
                Tag::Table(_) => close_node(&mut stack, &mut root),
                Tag::TableHead => {
                    finish_table_row(&mut stack);
                }
                Tag::TableRow => {
                    finish_table_row(&mut stack);
                }
                Tag::TableCell => {
                    finish_table_cell(&mut stack);
                }
                _ if active_table_mut(&mut stack).is_some() => {}
                Tag::Image(_, _, _) => close_node(&mut stack, &mut root),
                Tag::Paragraph
                | Tag::Heading(..)
                | Tag::BlockQuote
                | Tag::List(_)
                | Tag::Item
                | Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Link(_, _, _)
                | Tag::CodeBlock(_) => close_node(&mut stack, &mut root),
                _ => {}
            },
            Event::Text(text) => {
                if push_table_text(&mut stack, &text) {
                    continue;
                } else if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(&text);
                } else if let Some(StackEntry::Node(parent)) = stack.last_mut() {
                    push_text(&mut parent.children, &text);
                } else {
                    push_text(&mut root, &text);
                }
            }
            Event::Code(text) => {
                if push_table_text(&mut stack, &text) {
                    continue;
                } else if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(&text);
                } else {
                    push_value(
                        &mut stack,
                        &mut root,
                        json!({
                            "tag": "code",
                            "children": [text.as_ref()]
                        }),
                    );
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if !push_table_text(&mut stack, " ") {
                    push_value(&mut stack, &mut root, json!({ "tag": "br" }));
                }
            }
            Event::Rule => {
                push_value(&mut stack, &mut root, json!({ "tag": "hr" }));
            }
            Event::Html(html) => {
                if push_table_text(&mut stack, &html) {
                    continue;
                } else if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(&html);
                } else if let Some(StackEntry::Node(parent)) = stack.last_mut() {
                    push_text(&mut parent.children, &html);
                } else {
                    push_text(&mut root, &html);
                }
            }
            Event::TaskListMarker(checked) => {
                let marker = if checked { "[x] " } else { "[ ] " };
                if push_table_text(&mut stack, marker) {
                    continue;
                } else if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(marker);
                } else if let Some(StackEntry::Node(parent)) = stack.last_mut() {
                    push_text(&mut parent.children, marker);
                } else {
                    push_text(&mut root, marker);
                }
            }
            Event::FootnoteReference(_) => {}
        }
    }

    while !stack.is_empty() {
        close_node(&mut stack, &mut root);
    }

    root
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
    let form = vec![
        (
            "access_token".to_string(),
            CONFIG.telegraph_access_token.clone(),
        ),
        (
            "author_name".to_string(),
            CONFIG.telegraph_author_name.clone(),
        ),
        (
            "author_url".to_string(),
            CONFIG.telegraph_author_url.clone(),
        ),
        ("title".to_string(), title.to_string()),
        ("content".to_string(), content_json),
        ("return_content".to_string(), "false".to_string()),
    ];

    let client = get_http_client();
    let response = client
        .post("https://api.telegra.ph/createPage")
        .timeout(Duration::from_secs(10))
        .form(&form)
        .send()
        .await
        .ok()?;

    if !response.status().is_success() {
        warn!(
            "Telegraph API call failed with status {}",
            response.status()
        );
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

    let mut matches = YOUTUBE_URL_REGEX.captures_iter(text).collect::<Vec<_>>();
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
    url.trim_end_matches(|ch: char| {
        matches!(
            ch,
            ')' | ']' | '}' | '>' | '"' | '\'' | ',' | '.' | ';' | ':'
        )
    })
}

pub(crate) fn twitter_cache_key(url: &str) -> anyhow::Result<String> {
    canonical_status_key(url)
}

pub(crate) fn discover_supported_status_urls(text: &str) -> Vec<String> {
    HTTP_URL_REGEX
        .find_iter(text)
        .map(|matched| clean_url_candidate(matched.as_str()))
        .filter(|candidate| is_supported_status_url(candidate))
        .map(ToString::to_string)
        .collect()
}

fn is_telegraph_url(url: &str) -> bool {
    let lowered = url.to_lowercase();
    lowered.contains("telegra.ph") || lowered.contains("t.me/")
}

/// Telegraph/`t.me` URLs mentioned in `text`, in scan order: bare URLs first,
/// then Markdown and HTML link targets. Callers deduplicate.
pub(crate) fn discover_telegraph_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    for matched in TELEGRAPH_URL_REGEX.find_iter(text) {
        urls.push(matched.as_str().to_string());
    }
    for caps in MARKDOWN_LINK_REGEX.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for caps in HTML_LINK_REGEX.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    urls
}

pub(crate) async fn extract_cached_telegraph_content(
    url: &str,
) -> anyhow::Result<TelegraphContent> {
    if let Some(content) = TELEGRAPH_CACHE.lock().get(url) {
        return Ok(content);
    }

    let content = extract_telegraph_content(url).await?;
    TELEGRAPH_CACHE
        .lock()
        .insert(url.to_string(), content.clone());
    Ok(content)
}

pub(crate) async fn extract_cached_twitter_content(url: &str) -> anyhow::Result<TwitterContent> {
    let cache_key = twitter_cache_key(url).ok();
    if let Some(cache_key) = cache_key.as_ref() {
        if let Some(content) = TWITTER_CACHE.lock().get(cache_key) {
            return Ok(content);
        }
    }

    let content = extract_twitter_content(url).await?;
    if let Some(cache_key) = cache_key {
        TWITTER_CACHE.lock().insert(cache_key, content.clone());
    }
    Ok(content)
}

pub async fn extract_telegraph_urls_and_content(
    text: &str,
    message_entities: Option<&[MessageEntityRef<'_>]>,
    max_urls: usize,
) -> (String, Vec<TelegraphContent>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

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

    urls.extend(discover_telegraph_urls(text));

    urls.sort();
    urls.dedup();

    let ordered_urls = urls.into_iter().take(max_urls).collect::<Vec<_>>();
    let semaphore = Arc::new(Semaphore::new(CONFIG.external_enrich_fanout));
    let mut join_set = JoinSet::new();
    for url in ordered_urls.iter().cloned() {
        let semaphore = semaphore.clone();
        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .expect("external enrich semaphore should remain open");
            let result = extract_cached_telegraph_content(&url).await;
            (url, result)
        });
    }
    let mut fetched = HashMap::new();
    while let Some(result) = join_set.join_next().await {
        if let Ok((url, content)) = result {
            fetched.insert(url, content);
        }
    }

    let mut new_text = text.to_string();
    let mut extracted = Vec::new();
    for url in ordered_urls {
        match fetched
            .remove(&url)
            .unwrap_or_else(|| Err(anyhow::anyhow!("Telegraph extraction task failed")))
        {
            Ok(content) => {
                log_extracted_content(
                    "telegraph",
                    &url,
                    &content.text_content,
                    content.image_urls.len(),
                    content.video_urls.len(),
                    0,
                );
                let formatted = format!(
                    "\n[Telegraph content extracted from {}]\n{}\n",
                    url, content.text_content
                );
                new_text.push_str(&formatted);
                extracted.push(content);
            }
            Err(err) => {
                warn!("Telegraph extraction failed for {}: {}", url, err);
                new_text.push_str(&format!(
                    "\n[Telegraph content extraction failed for {}]\n",
                    url
                ));
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
            if is_supported_status_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    urls.extend(discover_supported_status_urls(text));
    urls.sort();
    urls.dedup();

    let ordered_urls = urls.into_iter().take(max_urls).collect::<Vec<_>>();
    let semaphore = Arc::new(Semaphore::new(CONFIG.external_enrich_fanout));
    let mut join_set = JoinSet::new();
    for url in ordered_urls.iter().cloned() {
        let semaphore = semaphore.clone();
        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .expect("external enrich semaphore should remain open");
            let result = extract_cached_twitter_content(&url).await;
            (url, result)
        });
    }
    let mut fetched = HashMap::new();
    while let Some(result) = join_set.join_next().await {
        if let Ok((url, content)) = result {
            fetched.insert(url, content);
        }
    }

    let mut new_text = text.to_string();
    let mut extracted = Vec::new();
    for url in ordered_urls {
        match fetched
            .remove(&url)
            .unwrap_or_else(|| Err(anyhow::anyhow!("Twitter extraction task failed")))
        {
            Ok(content) => {
                log_extracted_content(
                    "twitter",
                    &url,
                    &content.text_content,
                    content.image_urls.len(),
                    content.video_urls.len(),
                    0,
                );
                if !content.formatted_content.is_empty() {
                    new_text.push_str(&content.formatted_content);
                }
                extracted.push(content);
            }
            Err(err) => {
                warn!("Twitter extraction failed for {}: {}", url, err);
                new_text.push_str(&format!(
                    "\n[Twitter content extraction failed for {}]\n",
                    url
                ));
            }
        }
    }

    (new_text, extracted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_tables_render_as_readable_telegraph_lists() {
        let nodes = markdown_to_telegraph_nodes(
            "Intro\n\n| Rank | Brand / division | Revenue |\n|---|---|---:|\n| 1 | **Gucci** | $5.99B |\n| 2 | Saint Laurent | $2.88B |\n\nAfter",
        );

        let list = nodes
            .iter()
            .find(|node| node.get("tag").and_then(|tag| tag.as_str()) == Some("ul"))
            .expect("table should render as a list");
        let list_text = serde_json::to_string(list).expect("list should serialize");

        assert!(list_text.contains("Brand / division"));
        assert!(list_text.contains("Gucci"));
        assert!(list_text.contains("$5.99B"));
        assert!(!list_text.contains("|---"));
    }

    #[test]
    fn markdown_table_output_preserves_surrounding_paragraphs() {
        let nodes =
            markdown_to_telegraph_nodes("Before\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nAfter");
        let tags = nodes
            .iter()
            .filter_map(|node| node.get("tag").and_then(|tag| tag.as_str()))
            .collect::<Vec<_>>();

        assert_eq!(tags, vec!["p", "ul", "p"]);
    }

    #[test]
    fn twitter_cache_key_deduplicates_hosts_and_tracking_parameters() {
        assert_eq!(
            twitter_cache_key("https://x.com/a/status/123?s=20").unwrap(),
            twitter_cache_key("https://mobile.twitter.com/a/status/123/photo/1").unwrap()
        );
    }

    #[test]
    fn twitter_url_discovery_rejects_embedded_or_suffix_confusion_urls() {
        assert!(!is_supported_status_url("https://evilx.com/a/status/123"));
        assert!(discover_supported_status_urls(
            "https://example.com/?next=https://x.com/a/status/123"
        )
        .is_empty());
        assert_eq!(
            discover_supported_status_urls("read https://x.com/a/status/123?s=20 now"),
            vec!["https://x.com/a/status/123?s=20"]
        );
    }

    #[test]
    fn twitter_url_discovery_accepts_uppercase_http_tokens() {
        assert_eq!(
            discover_supported_status_urls("HTTPS://X.COM/a/status/123"),
            vec!["HTTPS://X.COM/a/status/123"]
        );
        assert_eq!(
            discover_supported_status_urls("hTtPs://mobile.twitter.com/a/status/456/photo/1"),
            vec!["hTtPs://mobile.twitter.com/a/status/456/photo/1"]
        );
        assert!(is_supported_status_url("HTTPS://X.COM/a/status/123"));
    }

    #[test]
    fn twitter_cache_insert_never_leaves_more_than_64_entries() {
        let mut cache = TWITTER_CACHE.lock();
        for id in 0..=EXTRACTION_CACHE_MAX_ENTRIES {
            cache.insert(
                format!("capacity-test-{id}"),
                TwitterContent {
                    text_content: String::new(),
                    image_urls: Vec::new(),
                    video_urls: Vec::new(),
                    formatted_content: String::new(),
                    attachment_plan: Vec::new(),
                },
            );
        }
        // The oldest insertion is evicted; the newest 64 survive.
        assert!(cache.get("capacity-test-0").is_none());
        assert!(cache.get("capacity-test-1").is_some());
        assert!(cache
            .get(&format!("capacity-test-{EXTRACTION_CACHE_MAX_ENTRIES}"))
            .is_some());
    }
}

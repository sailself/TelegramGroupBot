use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

const BRAVE_IMAGE_SEARCH_ENDPOINT: &str = "https://api.search.brave.com/res/v1/images/search";
const DEFAULT_TIMEOUT_SECONDS: u64 = 30;

#[derive(Debug, Deserialize)]
struct BraveImageSearchResponse {
    results: Option<Vec<BraveImageSearchItem>>,
}

#[derive(Debug, Deserialize)]
struct BraveImageSearchItem {
    title: Option<String>,
    url: Option<String>,
    source: Option<String>,
    thumbnail: Option<BraveImageThumbnail>,
    properties: Option<BraveImageProperties>,
}

#[derive(Debug, Deserialize)]
struct BraveImageThumbnail {
    src: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BraveImageProperties {
    url: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraveImageSearchResult {
    pub title: String,
    pub page_url: String,
    pub image_url: String,
    pub thumbnail_url: Option<String>,
    pub source: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

fn normalize_image_url(raw: Option<String>) -> Option<String> {
    let url = raw.unwrap_or_default();
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn to_u32(value: Option<i64>) -> Option<u32> {
    value.and_then(|number| u32::try_from(number).ok())
}

fn normalize_result(item: BraveImageSearchItem) -> Option<BraveImageSearchResult> {
    let page_url = normalize_image_url(item.url)?;
    let image_url = normalize_image_url(
        item.properties
            .as_ref()
            .and_then(|properties| properties.url.clone())
            .or_else(|| {
                item.thumbnail
                    .as_ref()
                    .and_then(|thumbnail| thumbnail.src.clone())
            }),
    )?;
    let title = item
        .title
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| page_url.clone());
    let thumbnail_url = normalize_image_url(item.thumbnail.and_then(|thumbnail| thumbnail.src));
    let source = item
        .source
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let width = to_u32(
        item.properties
            .as_ref()
            .and_then(|properties| properties.width),
    );
    let height = to_u32(item.properties.and_then(|properties| properties.height));
    Some(BraveImageSearchResult {
        title,
        page_url,
        image_url,
        thumbnail_url,
        source,
        width,
        height,
    })
}

fn normalize_safesearch(value: &str) -> &'static str {
    if value.trim().eq_ignore_ascii_case("off") {
        "off"
    } else {
        "strict"
    }
}

pub async fn brave_image_search(
    query: &str,
    count: usize,
    safesearch: &str,
) -> Result<Vec<BraveImageSearchResult>> {
    if !CONFIG.enable_brave_search || CONFIG.brave_search_api_key.trim().is_empty() {
        return Err(anyhow!("BRAVE_SEARCH_API_KEY is not configured."));
    }
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let count = count.clamp(1, 200);
    let safesearch = normalize_safesearch(safesearch);

    info!(
        "Calling Brave image search endpoint {} with query: {}",
        BRAVE_IMAGE_SEARCH_ENDPOINT, query
    );

    let client = get_http_client();
    let response = client
        .get(BRAVE_IMAGE_SEARCH_ENDPOINT)
        .header("Accept", "application/json")
        .header("X-Subscription-Token", CONFIG.brave_search_api_key.clone())
        .query(&[
            ("q", query.trim()),
            ("count", &count.to_string()),
            ("safesearch", safesearch),
        ])
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECONDS))
        .send()
        .await
        .map_err(|err| anyhow!("Brave image search request failed: {err}"))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Brave image search request failed with status {}",
            response.status()
        ));
    }

    let data: BraveImageSearchResponse = response
        .json()
        .await
        .map_err(|err| anyhow!("Invalid Brave image response: {err}"))?;

    Ok(data
        .results
        .unwrap_or_default()
        .into_iter()
        .filter_map(normalize_result)
        .collect())
}

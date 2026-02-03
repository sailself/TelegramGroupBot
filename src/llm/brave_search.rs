use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use tracing::info;

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

const DEFAULT_TIMEOUT_SECONDS: u64 = 30;

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Option<Vec<BraveWebResult>>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: Option<String>,
    url: Option<String>,
    description: Option<String>,
    extra_snippets: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct BraveSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn extract_results(payload: BraveSearchResponse) -> Vec<BraveSearchResult> {
    let mut results = Vec::new();
    for item in payload
        .web
        .and_then(|web| web.results)
        .unwrap_or_default()
    {
        let url = item.url.unwrap_or_default();
        if url.trim().is_empty() {
            continue;
        }
        let title = item.title.unwrap_or_else(|| url.clone());
        let snippet = item
            .description
            .or_else(|| item.extra_snippets.and_then(|snippets| snippets.into_iter().next()))
            .unwrap_or_default();
        results.push(BraveSearchResult {
            title,
            url,
            snippet,
        });
    }
    results
}

pub async fn brave_search(query: &str, max_results: usize) -> Result<Vec<BraveSearchResult>> {
    if !CONFIG.enable_brave_search || CONFIG.brave_search_api_key.trim().is_empty() {
        return Err(anyhow!("BRAVE_SEARCH_API_KEY is not configured."));
    }
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let count = max_results.clamp(1, 20);
    info!(
        "Calling Brave search endpoint {} with query: {}",
        CONFIG.brave_search_endpoint, query
    );

    let client = get_http_client();
    let response = client
        .get(&CONFIG.brave_search_endpoint)
        .header("X-Subscription-Token", CONFIG.brave_search_api_key.clone())
        .query(&[("q", query), ("count", &count.to_string())])
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECONDS))
        .send()
        .await
        .map_err(|err| anyhow!("Brave search request failed: {err}"))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Brave search request failed with status {}",
            response.status()
        ));
    }

    let data: BraveSearchResponse = response
        .json()
        .await
        .map_err(|err| anyhow!("Invalid Brave response: {err}"))?;

    Ok(extract_results(data))
}

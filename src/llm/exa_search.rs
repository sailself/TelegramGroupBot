use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

const DEFAULT_TIMEOUT_SECONDS: u64 = 30;
const MAX_DEFAULT_RESULTS: usize = 5;

#[derive(Debug, Error)]
#[error("Exa search error: {0}")]
pub struct ExaSearchError(pub String);

#[derive(Debug, Deserialize)]
struct ExaResponse {
    results: Option<Vec<ExaResult>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ExaResult {
    url: Option<String>,
    title: Option<String>,
    highlight: Option<String>,
    snippet: Option<String>,
    text: Option<String>,
    summary: Option<String>,
}

fn normalise_snippet(value: Option<&str>) -> String {
    let snippet = value.unwrap_or("").replace('\n', " ");
    let snippet = snippet.trim();
    if snippet.chars().count() > 240 {
        let truncated: String = snippet.chars().take(240).collect();
        format!("{truncated}...")
    } else {
        snippet.to_string()
    }
}

fn extract_results(payload: ExaResponse) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    for item in payload.results.unwrap_or_default() {
        let url = item.url.unwrap_or_default();
        if url.trim().is_empty() {
            continue;
        }
        let title = item.title.unwrap_or_else(|| url.clone());
        let snippet = item
            .highlight
            .as_deref()
            .or(item.snippet.as_deref())
            .or(item.text.as_deref())
            .or(item.summary.as_deref());
        results.push((title, url, normalise_snippet(snippet)));
    }
    results
}

pub async fn exa_search(
    query: &str,
    max_results: Option<usize>,
) -> Result<Vec<(String, String, String)>, ExaSearchError> {
    if CONFIG.exa_api_key.trim().is_empty() {
        return Err(ExaSearchError("EXA_API_KEY is not configured.".to_string()));
    }
    if query.trim().is_empty() {
        return Err(ExaSearchError("query must not be empty".to_string()));
    }

    let payload = serde_json::json!({
        "query": query,
        "numResults": max_results.unwrap_or(MAX_DEFAULT_RESULTS).clamp(1, 10),
        "type": "auto"
    });

    info!(
        "Calling Exa search endpoint {} with query: {}",
        CONFIG.exa_search_endpoint, query
    );
    let client = get_http_client();
    let response = client
        .post(&CONFIG.exa_search_endpoint)
        .header("x-api-key", CONFIG.exa_api_key.clone())
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECONDS))
        .json(&payload)
        .send()
        .await
        .map_err(|err| ExaSearchError(format!("Exa search request failed: {err}")))?;

    if !response.status().is_success() {
        return Err(ExaSearchError(format!(
            "Exa search request failed with status {}",
            response.status()
        )));
    }

    let data: ExaResponse = response
        .json()
        .await
        .map_err(|err| ExaSearchError(format!("Invalid Exa response: {err}")))?;

    Ok(extract_results(data))
}

#[allow(dead_code)]
pub fn format_results_markdown(query: &str, results: &[(String, String, String)]) -> String {
    if results.is_empty() {
        return format!("No web results found for query: {}", query);
    }

    let mut lines = vec![format!("Search results for **{}**:", query)];
    for (idx, (title, url, snippet)) in results.iter().enumerate() {
        lines.push(format!("{}. [{}]({})", idx + 1, title, url));
        if !snippet.is_empty() {
            lines.push(format!("   {}", snippet));
        }
    }
    lines.join("\n")
}

#[allow(dead_code)]
pub async fn exa_search_tool(
    query: &str,
    max_results: Option<usize>,
) -> Result<String, ExaSearchError> {
    let results = exa_search(query, max_results).await?;
    Ok(format_results_markdown(query, &results))
}

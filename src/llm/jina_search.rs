use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use tracing::info;

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

const DEFAULT_READ_TIMEOUT: u64 = 30;
static TITLE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+)\]\s+Title:\s*(.+)").expect("valid jina title regex"));
static URL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+)\]\s+URL Source:\s*(.+)").expect("valid jina url regex"));
static SNIPPET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(\d+)\]\s+(Description|Snippet):\s*(.+)").expect("valid jina snippet regex")
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaSearchResponse {
    pub query: String,
    pub results: Vec<JinaSearchResult>,
}

fn parse_search_text(payload: &str, max_results: usize) -> Vec<JinaSearchResult> {
    let mut results: HashMap<i32, JinaSearchResult> = HashMap::new();

    for raw_line in payload.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("![") {
            continue;
        }

        if let Some(caps) = TITLE_REGEX.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| JinaSearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.title = caps[2].trim().to_string();
            continue;
        }

        if let Some(caps) = URL_REGEX.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| JinaSearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.url = caps[2].trim().to_string();
            continue;
        }

        if let Some(caps) = SNIPPET_REGEX.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| JinaSearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.snippet = caps[3].trim().to_string();
        }
    }

    let mut ordered: Vec<_> = results
        .into_iter()
        .filter(|(_, entry)| !entry.url.is_empty())
        .collect();
    ordered.sort_by_key(|(idx, _)| *idx);
    ordered
        .into_iter()
        .map(|(_, entry)| entry)
        .take(max_results)
        .collect()
}

pub async fn search_jina_web(query: &str, max_results: usize) -> Result<JinaSearchResponse> {
    search_jina_web_at(
        &CONFIG.jina_search_endpoint,
        &CONFIG.jina_ai_api_key,
        query,
        max_results,
    )
    .await
}

async fn search_jina_web_at(
    endpoint: &str,
    api_key: &str,
    query: &str,
    max_results: usize,
) -> Result<JinaSearchResponse> {
    if query.trim().is_empty() {
        anyhow::bail!("query must not be empty");
    }

    let payload = serde_json::json!({ "q": query });
    info!("Calling Jina search endpoint {endpoint} with query: {query}");

    let client = get_http_client();
    let mut request = client
        .post(endpoint)
        .timeout(Duration::from_secs(DEFAULT_READ_TIMEOUT))
        .json(&payload);

    if !api_key.trim().is_empty() {
        request = request.bearer_auth(api_key);
    }

    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Jina search request failed with status {status}");
    }
    let text = response.text().await?;
    let parsed = parse_search_text(&text, max_results);
    Ok(JinaSearchResponse {
        query: query.to_string(),
        results: parsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::test_support::{response_with_status, TestServer};

    #[tokio::test]
    async fn rejects_non_success_status_instead_of_returning_empty_results() {
        let server = TestServer::single_status("POST", "/search", 401);
        let endpoint = server.url("/search").to_string();

        let result = search_jina_web_at(&endpoint, "", "rust language", 5).await;

        assert!(
            result.is_err(),
            "a 401 must surface as an error, not as zero results"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn parses_successful_text_response() {
        let body = b"[1] Title: Rust\n[1] URL Source: https://www.rust-lang.org\n[1] Description: A language\n".to_vec();
        let server = TestServer::single(response_with_status(200, body));
        let endpoint = server.url("/search").to_string();

        let response = search_jina_web_at(&endpoint, "", "rust language", 5)
            .await
            .unwrap();

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].url, "https://www.rust-lang.org");
        server.join().unwrap();
    }
}

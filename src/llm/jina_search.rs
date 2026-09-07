use std::collections::HashMap;

use anyhow::{anyhow, Result};
use regex::Regex;
use std::sync::LazyLock;
use tracing::info;

use crate::config::CONFIG;
use crate::llm::audit::LlmUsageRecord;
use crate::llm::transport::{call_with_retry, read_body_limited, LlmCall};
use crate::llm::web_search::{
    BoxFuture, SearchProvider, SearchResult, WEB_SEARCH_REQUEST_TIMEOUT, WEB_SEARCH_RETRY_POLICY,
};
use crate::utils::http::get_http_client;

/// Jina returns plain text; a few hundred KiB covers any sane result page.
const JINA_MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
static TITLE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+)\]\s+Title:\s*(.+)").expect("valid jina title regex"));
static URL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+)\]\s+URL Source:\s*(.+)").expect("valid jina url regex"));
static SNIPPET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(\d+)\]\s+(Description|Snippet):\s*(.+)").expect("valid jina snippet regex")
});

/// Jina reader/search backend.
pub struct JinaSearch;

impl SearchProvider for JinaSearch {
    fn name(&self) -> &'static str {
        "jina"
    }

    fn is_enabled(&self) -> bool {
        CONFIG.enable_jina_mcp
    }

    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
        Box::pin(async move {
            if !self.is_enabled() {
                return Err(anyhow!("Jina search is disabled."));
            }
            search_jina_web_at(
                &CONFIG.jina_search_endpoint,
                &CONFIG.jina_ai_api_key,
                query,
                max_results,
            )
            .await
        })
    }
}

fn parse_search_text(payload: &str, max_results: usize) -> Vec<SearchResult> {
    let mut results: HashMap<i32, SearchResult> = HashMap::new();

    for raw_line in payload.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("![") {
            continue;
        }

        if let Some(caps) = TITLE_REGEX.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| SearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.title = caps[2].trim().to_string();
            continue;
        }

        if let Some(caps) = URL_REGEX.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| SearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.url = caps[2].trim().to_string();
            continue;
        }

        if let Some(caps) = SNIPPET_REGEX.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| SearchResult {
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

async fn search_jina_web_at(
    endpoint: &str,
    api_key: &str,
    query: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let payload = serde_json::json!({ "q": query });
    let payload = &payload;
    info!("Calling Jina search endpoint {endpoint} with query: {query}");

    let call = LlmCall::untracked("web-search", "jina");
    let text = call_with_retry(
        &call,
        &WEB_SEARCH_RETRY_POLICY,
        |_| async move {
            let mut request = get_http_client()
                .post(endpoint)
                .timeout(WEB_SEARCH_REQUEST_TIMEOUT)
                .json(payload);
            if !api_key.trim().is_empty() {
                request = request.bearer_auth(api_key);
            }
            Ok(request)
        },
        |_| {},
        |response| async move {
            let bytes = read_body_limited(response, "jina", JINA_MAX_BODY_BYTES).await?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        },
        |_| LlmUsageRecord::default(),
    )
    .await?;

    Ok(parse_search_text(&text, max_results))
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

        let results = search_jina_web_at(&endpoint, "", "rust language", 5)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://www.rust-lang.org");
        assert_eq!(results[0].title, "Rust");
        assert_eq!(results[0].snippet, "A language");
        server.join().unwrap();
    }
}

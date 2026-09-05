use anyhow::{anyhow, Result};
use serde::Deserialize;
use tracing::info;

use crate::config::CONFIG;
use crate::llm::audit::LlmUsageRecord;
use crate::llm::transport::{call_with_retry, read_json, LlmCall};
use crate::llm::web_search::{
    BoxFuture, SearchProvider, SearchResult, WEB_SEARCH_REQUEST_TIMEOUT, WEB_SEARCH_RETRY_POLICY,
};
use crate::utils::http::get_http_client;

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

/// Brave Search API backend.
pub struct BraveSearch;

impl SearchProvider for BraveSearch {
    fn name(&self) -> &'static str {
        "brave"
    }

    fn is_enabled(&self) -> bool {
        CONFIG.enable_brave_search && !CONFIG.brave_search_api_key.trim().is_empty()
    }

    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
        Box::pin(async move {
            if !self.is_enabled() {
                return Err(anyhow!("BRAVE_SEARCH_API_KEY is not configured."));
            }
            brave_search_at(
                &CONFIG.brave_search_endpoint,
                &CONFIG.brave_search_api_key,
                query,
                max_results,
            )
            .await
        })
    }
}

fn extract_results(payload: BraveSearchResponse) -> Vec<SearchResult> {
    let mut results = Vec::new();
    for item in payload.web.and_then(|web| web.results).unwrap_or_default() {
        let url = item.url.unwrap_or_default();
        if url.trim().is_empty() {
            continue;
        }
        let title = item.title.unwrap_or_else(|| url.clone());
        let snippet = item
            .description
            .or_else(|| {
                item.extra_snippets
                    .and_then(|snippets| snippets.into_iter().next())
            })
            .unwrap_or_default();
        results.push(SearchResult {
            title,
            url,
            snippet,
        });
    }
    results
}

async fn brave_search_at(
    endpoint: &str,
    api_key: &str,
    query: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let count = max_results.clamp(1, 20).to_string();
    let count = count.as_str();
    info!("Calling Brave search endpoint {endpoint} with query: {query}");

    let call = LlmCall::untracked("web-search", "brave");
    let data: BraveSearchResponse = call_with_retry(
        &call,
        &WEB_SEARCH_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .get(endpoint)
                .header("X-Subscription-Token", api_key)
                .query(&[("q", query), ("count", count)])
                .timeout(WEB_SEARCH_REQUEST_TIMEOUT))
        },
        |_| {},
        |response| read_json::<BraveSearchResponse>(response, "brave"),
        |_| LlmUsageRecord::default(),
    )
    .await?;

    Ok(extract_results(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::test_support::{
        response_with_headers, ExpectedRequest, TestServer,
    };

    #[tokio::test]
    async fn brave_results_are_parsed_and_transient_failures_retried() {
        let body = br#"{"web":{"results":[{"title":"Rust","url":"https://www.rust-lang.org","description":"A language"},{"url":"https://no-title.example","extra_snippets":["fallback snippet"]}]}}"#;
        // The query string is part of the path the mock sees, so match any
        // path and pin the behaviour through the header check instead.
        let server = TestServer::new(vec![
            ExpectedRequest::any(response_with_headers(503, &[], b"busy".to_vec())),
            ExpectedRequest::any(response_with_headers(200, &[], body.to_vec()))
                .with_header("x-subscription-token", "brave-key"),
        ]);
        let endpoint = server.url("/search").to_string();

        let results = brave_search_at(&endpoint, "brave-key", "rust language", 5)
            .await
            .expect("second attempt succeeds");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust");
        assert_eq!(results[0].url, "https://www.rust-lang.org");
        assert_eq!(results[0].snippet, "A language");
        assert_eq!(results[1].title, "https://no-title.example");
        assert_eq!(results[1].snippet, "fallback snippet");
        server.join().expect("both requests were served");
    }
}

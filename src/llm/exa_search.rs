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
struct ExaResponse {
    results: Option<Vec<ExaResult>>,
}

#[derive(Debug, Deserialize)]
struct ExaResult {
    url: Option<String>,
    title: Option<String>,
    highlight: Option<String>,
    snippet: Option<String>,
    text: Option<String>,
    summary: Option<String>,
}

/// Exa search backend.
pub struct ExaSearch;

impl SearchProvider for ExaSearch {
    fn name(&self) -> &'static str {
        "exa"
    }

    fn is_enabled(&self) -> bool {
        CONFIG.enable_exa_search && !CONFIG.exa_api_key.trim().is_empty()
    }

    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
        Box::pin(async move {
            if !self.is_enabled() {
                return Err(anyhow!("EXA_API_KEY is not configured."));
            }
            exa_search_at(
                &CONFIG.exa_search_endpoint,
                &CONFIG.exa_api_key,
                query,
                max_results,
            )
            .await
        })
    }
}

fn extract_results(payload: ExaResponse) -> Vec<SearchResult> {
    let mut results = Vec::new();
    for item in payload.results.unwrap_or_default() {
        let url = item.url.unwrap_or_default();
        if url.trim().is_empty() {
            continue;
        }
        let title = item.title.unwrap_or_else(|| url.clone());
        let snippet = item
            .highlight
            .or(item.snippet)
            .or(item.text)
            .or(item.summary)
            .unwrap_or_default();
        results.push(SearchResult {
            title,
            url,
            snippet,
        });
    }
    results
}

async fn exa_search_at(
    endpoint: &str,
    api_key: &str,
    query: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let payload = serde_json::json!({
        "query": query,
        "numResults": max_results.clamp(1, 10),
        "type": "auto"
    });
    let payload = &payload;
    info!("Calling Exa search endpoint {endpoint} with query: {query}");

    let call = LlmCall::untracked("web-search", "exa");
    let data: ExaResponse = call_with_retry(
        &call,
        &WEB_SEARCH_RETRY_POLICY,
        |_| async move {
            Ok(get_http_client()
                .post(endpoint)
                .header("x-api-key", api_key)
                .timeout(WEB_SEARCH_REQUEST_TIMEOUT)
                .json(payload))
        },
        |_| {},
        |response| read_json::<ExaResponse>(response, "exa"),
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
    async fn exa_results_are_parsed_with_the_first_available_snippet_field() {
        let body = br#"{"results":[{"url":"https://a.example","title":"A","highlight":"hl","text":"long text"},{"url":"https://b.example","summary":"only summary"}]}"#;
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/search",
            response_with_headers(200, &[], body.to_vec()),
        )
        .with_header("x-api-key", "exa-key")]);
        let endpoint = server.url("/search").to_string();

        let results = exa_search_at(&endpoint, "exa-key", "rust language", 5)
            .await
            .expect("results parse");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].snippet, "hl");
        assert_eq!(results[1].title, "https://b.example");
        assert_eq!(results[1].snippet, "only summary");
        server.join().expect("one request was served");
    }
}

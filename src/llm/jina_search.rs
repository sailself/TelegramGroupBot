use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

#[allow(dead_code)]
const DEFAULT_READ_TIMEOUT: u64 = 30;

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaSearchResponse {
    pub query: String,
    pub results: Vec<JinaSearchResult>,
}

#[allow(dead_code)]
fn parse_search_text(payload: &str, max_results: usize) -> Vec<JinaSearchResult> {
    let title_pattern = Regex::new(r"\[(\d+)\]\s+Title:\s*(.+)").unwrap();
    let url_pattern = Regex::new(r"\[(\d+)\]\s+URL Source:\s*(.+)").unwrap();
    let snippet_pattern = Regex::new(r"\[(\d+)\]\s+(Description|Snippet):\s*(.+)").unwrap();

    let mut results: HashMap<i32, JinaSearchResult> = HashMap::new();

    for raw_line in payload.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("![") {
            continue;
        }

        if let Some(caps) = title_pattern.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| JinaSearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.title = caps[2].trim().to_string();
            continue;
        }

        if let Some(caps) = url_pattern.captures(line) {
            let idx = caps[1].parse::<i32>().unwrap_or_default();
            let entry = results.entry(idx).or_insert_with(|| JinaSearchResult {
                title: String::new(),
                url: String::new(),
                snippet: String::new(),
            });
            entry.url = caps[2].trim().to_string();
            continue;
        }

        if let Some(caps) = snippet_pattern.captures(line) {
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

#[allow(dead_code)]
pub async fn search_jina_web(query: &str, max_results: usize) -> Result<JinaSearchResponse> {
    if query.trim().is_empty() {
        anyhow::bail!("query must not be empty");
    }

    let payload = serde_json::json!({ "q": query });
    info!("Calling Jina search endpoint {} with query: {}", CONFIG.jina_search_endpoint, query);

    let client = get_http_client();
    let mut request = client
        .post(&CONFIG.jina_search_endpoint)
        .timeout(Duration::from_secs(DEFAULT_READ_TIMEOUT))
        .json(&payload);

    if !CONFIG.jina_ai_api_key.trim().is_empty() {
        request = request.bearer_auth(&CONFIG.jina_ai_api_key);
    }

    let response = request.send().await?;
    let text = response.text().await?;
    let parsed = parse_search_text(&text, max_results);
    Ok(JinaSearchResponse {
        query: query.to_string(),
        results: parsed,
    })
}

#[allow(dead_code)]
pub async fn fetch_jina_reader(url: &str) -> Result<String> {
    if url.trim().is_empty() {
        anyhow::bail!("url must not be empty");
    }

    let target = format!("{}/{}", CONFIG.jina_reader_endpoint.trim_end_matches('/'), url.trim_start_matches('/'));
    info!("Calling Jina reader endpoint {}", target);

    let client = get_http_client();
    let mut request = client
        .get(target)
        .timeout(Duration::from_secs(60));

    if !CONFIG.jina_ai_api_key.trim().is_empty() {
        request = request.bearer_auth(&CONFIG.jina_ai_api_key);
    }

    let response = request.send().await?;
    let text = response.text().await?;
    Ok(text)
}

#[allow(dead_code)]
pub fn format_search_results_markdown(results: &JinaSearchResponse) -> String {
    if results.results.is_empty() {
        return format!("No results found for query: {}", results.query);
    }

    let mut lines = vec![format!("Search results for **{}**:", results.query)];
    for (idx, item) in results.results.iter().enumerate() {
        let mut snippet = item.snippet.replace('\n', " ");
        if snippet.len() > 200 {
            snippet = format!("{}...", &snippet[..200]);
        }
        let title = if item.title.is_empty() {
            &item.url
        } else {
            &item.title
        };
        lines.push(format!("{}. [{}]({})\n   {}", idx + 1, title, item.url, snippet));
    }

    lines.join("\n")
}

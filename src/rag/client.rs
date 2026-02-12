use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

#[derive(Debug, Clone, Serialize)]
pub struct RagMessageItem {
    pub message_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub date: String,
    pub reply_to_message_id: Option<i64>,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RagIngestResponse {
    #[serde(default)]
    pub accepted: usize,
    #[serde(default)]
    pub skipped: usize,
    #[serde(default)]
    pub failed: usize,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RagHit {
    pub message_id: i64,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub user_id: Option<i64>,
    #[serde(default)]
    pub reply_to_message_id: Option<i64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RagRetrieveResponse {
    #[serde(default)]
    pub hits: Vec<RagHit>,
}

#[derive(Debug, Serialize)]
struct RagIngestRequest {
    chat_id: i64,
    items: Vec<RagMessageItem>,
}

#[derive(Debug, Serialize)]
struct RagRetrieveRequest {
    chat_id: i64,
    query: String,
    top_k: usize,
}

fn rag_base_url() -> Option<String> {
    let base = CONFIG.rag_service_url.trim().trim_end_matches('/');
    if base.is_empty() {
        None
    } else {
        Some(base.to_string())
    }
}

pub fn is_rag_enabled() -> bool {
    CONFIG.enable_rag && rag_base_url().is_some()
}

pub fn build_ingest_item(
    message_id: i64,
    user_id: Option<i64>,
    username: Option<String>,
    date: DateTime<Utc>,
    reply_to_message_id: Option<i64>,
    text: &str,
) -> Option<RagMessageItem> {
    let normalized = normalize_ingest_text(text)?;
    Some(RagMessageItem {
        message_id,
        user_id,
        username,
        date: date.to_rfc3339(),
        reply_to_message_id,
        text: normalized,
    })
}

fn normalize_ingest_text(text: &str) -> Option<String> {
    let normalized = text.trim();
    if normalized.is_empty() || normalized.starts_with('/') {
        return None;
    }
    Some(normalized.to_string())
}

fn with_auth_headers(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    let api_key = CONFIG.rag_service_api_key.trim();
    if api_key.is_empty() {
        return request;
    }
    request.bearer_auth(api_key).header("X-API-Key", api_key)
}

pub async fn ingest_messages(
    chat_id: i64,
    items: Vec<RagMessageItem>,
) -> Result<RagIngestResponse> {
    let item_count = items.len();
    if item_count == 0 {
        return Ok(RagIngestResponse::default());
    }
    if !is_rag_enabled() {
        return Ok(RagIngestResponse {
            accepted: 0,
            skipped: item_count,
            failed: 0,
            errors: Vec::new(),
        });
    }

    let base_url = rag_base_url().ok_or_else(|| anyhow!("RAG service URL is not configured"))?;
    let payload = RagIngestRequest { chat_id, items };
    let endpoint = format!("{base_url}/v1/ingest/messages");

    let request = get_http_client()
        .post(&endpoint)
        .timeout(Duration::from_millis(CONFIG.rag_http_timeout_ms))
        .json(&payload);
    let response = with_auth_headers(request)
        .send()
        .await
        .map_err(|err| anyhow!("RAG ingest request failed: {err}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "RAG ingest request failed with status {}: {}",
            status,
            body
        ));
    }

    response
        .json::<RagIngestResponse>()
        .await
        .map_err(|err| anyhow!("Failed to parse RAG ingest response: {err}"))
}

pub async fn retrieve(chat_id: i64, query: &str, top_k: Option<usize>) -> Result<Vec<RagHit>> {
    if !is_rag_enabled() {
        return Err(anyhow!("RAG is disabled"));
    }

    let base_url = rag_base_url().ok_or_else(|| anyhow!("RAG service URL is not configured"))?;
    let top_k = top_k.unwrap_or(CONFIG.rag_query_top_k).max(1);
    let payload = RagRetrieveRequest {
        chat_id,
        query: query.to_string(),
        top_k,
    };
    let endpoint = format!("{base_url}/v1/retrieve");

    let request = get_http_client()
        .post(&endpoint)
        .timeout(Duration::from_millis(CONFIG.rag_http_timeout_ms))
        .json(&payload);
    let response = with_auth_headers(request)
        .send()
        .await
        .map_err(|err| anyhow!("RAG retrieve request failed: {err}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "RAG retrieve request failed with status {}: {}",
            status,
            body
        ));
    }

    let parsed = response
        .json::<RagRetrieveResponse>()
        .await
        .map_err(|err| anyhow!("Failed to parse RAG retrieve response: {err}"))?;
    Ok(parsed.hits)
}

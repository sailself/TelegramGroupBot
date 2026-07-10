use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use reqwest::StatusCode;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::audit::{
    log_llm_request_started, record_llm_request_success, LlmAuditContext, LlmUsageRecord,
};
use crate::llm::openai_codex;
use crate::llm::runtime_models::{
    selected_codex_model_record, CodexSelectedModelRecord, OPENAI_CODEX_SELECTED_MODEL_ID,
};
use crate::llm::tool_prompts::{tool_limit_guidance, TOOL_LIMIT_SYSTEM_PROMPT};
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::http::{get_http_client, get_http_client_no_compression};

const MAX_TOOL_CALL_ITERATIONS: usize = 3;
const RESPONSES_MAX_ATTEMPTS: usize = 3;
const RESPONSES_RETRY_BASE_DELAY_MS: u64 = 900;
const RESPONSES_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
const MODELS_ETAG_HEADER: &str = "x-models-etag";
const REQUEST_ID_HEADER: &str = "x-request-id";
const CODEX_RESPONSE_STYLE_ADDENDUM: &str = r#"# Style
Be direct, highly informative, and concise. Match depth to complexity.

# Framing
Prefer direct positive claims over negation-contrastive phrasing ("not X, but Y") when a plain statement is clearer. This is a preference, not an absolute: keep contrastive phrasing when evaluating or correcting a claim genuinely needs it.
- Weaker: 真正的创新者不是有创意的人，而是特质拉满的人。
- Better: 真正的创新者是特质拉满的人。

# Execution
- Answers first: lead with the core answer. For yes/no questions, answer first + one sentence of reasoning. For comparisons, recommend one + brief reasoning.
- Concept limits: keep conceptual explanations to 3-5 sentences; cover the essence. Do not restate points in "plain language".
- Code: provide code + non-trivial usage examples.
- Lists: at most 3-4 points per side; use bullets only for genuinely structural data, not decoration.
- Endings: stop after the final claim or concrete recommendation.

# Avoid (in whatever language you answer)
- Filler preambles (e.g. "Great question", "Certainly", "首先我们需要").
- Restating the user's prompt.
- Summary-label closers (e.g. "In conclusion", "Hope this helps", "总结一下").
- Conditional follow-up offers (e.g. "If you want, I can...", "如果你愿意，我可以...")."#;
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
struct ResponsesRequestDetails {
    provider: ThirdPartyProvider,
    display_name: &'static str,
    url: String,
    headers: Vec<(String, String)>,
    session_id: String,
    codex_account_id: Option<String>,
    payload: Value,
    streaming_sse: bool,
    request_timeout_secs: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ResponsesResponseMetadata {
    pub(crate) request_id: Option<String>,
    pub(crate) models_etag: Option<String>,
    pub(crate) codex_account_id: Option<String>,
    pub(crate) rate_limit_headers: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(crate) struct ResponsesApiResult {
    pub(crate) response: Value,
    pub(crate) metadata: ResponsesResponseMetadata,
}

#[derive(Debug, Default)]
struct CodexTurnState {
    value: Option<String>,
}

fn observe_response_metadata(provider: ThirdPartyProvider, metadata: &ResponsesResponseMetadata) {
    if provider != ThirdPartyProvider::OpenAICodex {
        return;
    }

    let Some(models_etag) = metadata
        .models_etag
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
    else {
        return;
    };
    let Some(account_id) = metadata
        .codex_account_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
    else {
        return;
    };

    tokio::spawn(async move {
        match crate::llm::runtime_models::refresh_selected_codex_model_for_etag(
            &models_etag,
            &account_id,
        )
        .await
        {
            Ok(true) => info!(
                models_etag = %models_etag,
                "Refreshed the selected Codex model metadata"
            ),
            Ok(false) => {}
            Err(err) => warn!(
                models_etag = %models_etag,
                error = %err,
                "Failed to refresh Codex model metadata"
            ),
        }
    });
}

impl CodexTurnState {
    fn capture(&mut self, headers: &reqwest::header::HeaderMap) {
        if self.value.is_some() {
            return;
        }
        self.value = headers
            .get(CODEX_TURN_STATE_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
    }

    fn apply(&self, headers: &mut Vec<(String, String)>) {
        if let Some(value) = self.value.as_ref() {
            headers.push((CODEX_TURN_STATE_HEADER.to_string(), value.clone()));
        }
    }
}

#[derive(Debug)]
enum ResponseBodyReadError {
    Transport(reqwest::Error),
    TooLarge { limit: usize },
}

impl ResponseBodyReadError {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::Transport(_))
    }
}

impl fmt::Display for ResponseBodyReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(err) => write!(formatter, "response body decode failed: {err}"),
            Self::TooLarge { limit } => {
                write!(formatter, "response body exceeded the {limit}-byte limit")
            }
        }
    }
}

impl std::error::Error for ResponseBodyReadError {}

#[derive(Debug)]
enum SseParseError {
    InvalidPayload {
        bytes: usize,
        source: serde_json::Error,
    },
    MissingCompletion,
    Incomplete(String),
    Failed(String),
}

impl SseParseError {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::InvalidPayload { .. } | Self::MissingCompletion)
    }
}

impl fmt::Display for SseParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayload { bytes, source } => {
                write!(
                    formatter,
                    "invalid {bytes}-byte SSE event payload: {source}"
                )
            }
            Self::MissingCompletion => {
                formatter.write_str("SSE stream ended before response.completed")
            }
            Self::Incomplete(detail) => write!(formatter, "SSE response incomplete: {detail}"),
            Self::Failed(detail) => write!(formatter, "SSE response failed: {detail}"),
        }
    }
}

impl std::error::Error for SseParseError {}

#[derive(Debug, Clone)]
struct ResponsesToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

fn generate_session_id() -> String {
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = chrono::Utc::now().timestamp_millis();
    format!("tg-codex-{now}-{counter}")
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn classify_incomplete_reason(value: Option<&str>) -> &'static str {
    match value {
        Some("max_output_tokens") => "max_output_tokens",
        Some("content_filter") => "content_filter",
        _ => "other",
    }
}

fn summarize_error_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "empty_response_body".to_string();
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let code_present = value
            .pointer("/error/code")
            .or_else(|| value.get("code"))
            .is_some();
        let type_present = value
            .pointer("/error/type")
            .or_else(|| value.get("type"))
            .is_some();
        let summary = format!(
            "json_error(code_present={}, type_present={}, bytes={})",
            code_present,
            type_present,
            body.len()
        );
        return summary;
    }

    format!("non_json_error(bytes={})", body.len())
}

fn summarize_responses_payload(payload: &Value) -> String {
    let model = payload
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let input_items = payload
        .get("input")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let input_images = payload
        .get("input")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("content").and_then(|value| value.as_array()))
                .flatten()
                .filter(|item| {
                    item.get("type").and_then(|value| value.as_str()) == Some("input_image")
                })
                .count()
        })
        .unwrap_or(0);
    let tool_names = payload
        .get("tools")
        .and_then(|value| value.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    tool.get("name")
                        .and_then(|value| value.as_str())
                        .or_else(|| tool.get("type").and_then(|value| value.as_str()))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let reasoning = payload
        .pointer("/reasoning/effort")
        .and_then(|value| value.as_str())
        .unwrap_or("default");
    let session_id = payload
        .get("prompt_cache_key")
        .and_then(|value| value.as_str())
        .unwrap_or("none");
    let stream = payload
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    format!(
        "model={model}, session_id={session_id}, input_items={input_items}, input_images={input_images}, tools={}, tool_names=[{}], stream={stream}, reasoning={reasoning}",
        tool_names.len(),
        tool_names.join(",")
    )
}

fn summarize_output_items(output_items: &[Value]) -> String {
    let mut counts = BTreeMap::new();
    for item in output_items {
        let item_type = item
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown");
        *counts.entry(item_type.to_string()).or_insert(0usize) += 1;
    }

    counts
        .into_iter()
        .map(|(item_type, count)| format!("{item_type}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn debug_model_label(model_config: &ThirdPartyModelConfig) -> &str {
    if model_config.provider == ThirdPartyProvider::OpenAICodex {
        model_config.name.as_str()
    } else {
        model_config.id.as_str()
    }
}

fn summarize_response_headers(headers: &reqwest::header::HeaderMap) -> String {
    let selected_headers = [
        reqwest::header::CONTENT_TYPE,
        reqwest::header::CONTENT_ENCODING,
        reqwest::header::TRANSFER_ENCODING,
        reqwest::header::CONTENT_LENGTH,
        reqwest::header::SERVER,
        reqwest::header::CACHE_CONTROL,
    ];

    selected_headers
        .iter()
        .filter_map(|name| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(|value| format!("{}={}", name.as_str(), truncate_for_log(value, 200)))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_codex_rate_limit_header(name: &str) -> bool {
    name.starts_with("x-codex-")
        && (name.contains("-primary-")
            || name.contains("-secondary-")
            || name.contains("-credits-")
            || name.ends_with("-limit-name")
            || name.ends_with("-rate-limit-reached-type"))
}

fn capture_response_metadata(
    headers: &reqwest::header::HeaderMap,
    codex_account_id: Option<&str>,
) -> ResponsesResponseMetadata {
    let header_string = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    let rate_limit_headers = headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str().to_ascii_lowercase();
            if !is_codex_rate_limit_header(&name) {
                return None;
            }
            value
                .to_str()
                .ok()
                .map(|value| (name, truncate_for_log(value, 200)))
        })
        .collect();

    ResponsesResponseMetadata {
        request_id: header_string(REQUEST_ID_HEADER),
        models_etag: header_string(MODELS_ETAG_HEADER),
        codex_account_id: codex_account_id.map(str::to_string),
        rate_limit_headers,
    }
}

async fn read_response_body_bytes(
    response: reqwest::Response,
    display_name: &str,
    model: &str,
    attempt: usize,
    streaming_sse: bool,
) -> std::result::Result<(Vec<u8>, String), ResponseBodyReadError> {
    read_response_body_bytes_with_limit(
        response,
        display_name,
        model,
        attempt,
        streaming_sse,
        RESPONSES_MAX_BODY_BYTES,
    )
    .await
}

async fn read_response_body_bytes_with_limit(
    mut response: reqwest::Response,
    display_name: &str,
    model: &str,
    attempt: usize,
    streaming_sse: bool,
    max_body_bytes: usize,
) -> std::result::Result<(Vec<u8>, String), ResponseBodyReadError> {
    let header_summary = summarize_response_headers(response.headers());
    let mut body = Vec::new();

    if response
        .content_length()
        .is_some_and(|length| length > max_body_bytes as u64)
    {
        error!(
            "{} response body rejected: model={}, attempt={}/{}, streaming_sse={}, headers=[{}], limit_bytes={}",
            display_name,
            model,
            attempt,
            RESPONSES_MAX_ATTEMPTS,
            streaming_sse,
            header_summary,
            max_body_bytes
        );
        return Err(ResponseBodyReadError::TooLarge {
            limit: max_body_bytes,
        });
    }

    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if body.len().saturating_add(chunk.len()) > max_body_bytes {
                    error!(
                        "{} response body rejected: model={}, attempt={}/{}, streaming_sse={}, headers=[{}], received_bytes={}, next_chunk_bytes={}, limit_bytes={}",
                        display_name,
                        model,
                        attempt,
                        RESPONSES_MAX_ATTEMPTS,
                        streaming_sse,
                        header_summary,
                        body.len(),
                        chunk.len(),
                        max_body_bytes
                    );
                    return Err(ResponseBodyReadError::TooLarge {
                        limit: max_body_bytes,
                    });
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => {
                return Ok((body, header_summary));
            }
            Err(err) => {
                if streaming_sse
                    && std::str::from_utf8(&body)
                        .ok()
                        .is_some_and(|partial| parse_sse_responses_body(partial).is_ok())
                {
                    warn!(
                        "{} response framing ended after a valid response.completed event: model={}, attempt={}/{}, headers=[{}], bytes={}, error={}",
                        display_name,
                        model,
                        attempt,
                        RESPONSES_MAX_ATTEMPTS,
                        header_summary,
                        body.len(),
                        err
                    );
                    return Ok((body, header_summary));
                }
                // A failure while streaming the body (e.g. an intermediary closing an
                // idle chunked connection mid-reasoning) is transient and safe to retry,
                // so log it as a warning while attempts remain and reserve `error!` for
                // the final, give-up attempt.
                let will_retry = attempt < RESPONSES_MAX_ATTEMPTS;
                let log_message = format!(
                    "{} response body decode failed: model={}, attempt={}/{}, streaming_sse={}, timeout={}, connect={}, headers=[{}], partial_bytes={}, retrying={}, error={}",
                    display_name,
                    model,
                    attempt,
                    RESPONSES_MAX_ATTEMPTS,
                    streaming_sse,
                    err.is_timeout(),
                    err.is_connect(),
                    header_summary,
                    body.len(),
                    will_retry,
                    err
                );
                if will_retry {
                    warn!("{log_message}");
                } else {
                    error!("{log_message}");
                }
                return Err(ResponseBodyReadError::Transport(err));
            }
        }
    }
}

fn responses_should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

fn responses_should_retry_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

fn responses_retry_delay(attempt: usize) -> Duration {
    let attempt = attempt.max(1) as u64;
    Duration::from_millis(RESPONSES_RETRY_BASE_DELAY_MS.saturating_mul(attempt))
}

fn responses_request_timeout_secs(provider: ThirdPartyProvider) -> u64 {
    match provider {
        ThirdPartyProvider::OpenAI => CONFIG.openai_request_timeout_secs,
        ThirdPartyProvider::OpenAICodex => CONFIG.openai_codex_request_timeout_secs,
        ThirdPartyProvider::OpenRouter
        | ThirdPartyProvider::Nvidia
        | ThirdPartyProvider::Ollama => 60,
    }
}

fn responses_tool_limit_guidance() -> String {
    tool_limit_guidance(MAX_TOOL_CALL_ITERATIONS)
}

fn build_responses_system_prompt(
    system_prompt: &str,
    model_config: &ThirdPartyModelConfig,
    extra_guidance: Option<&str>,
) -> String {
    let mut sections = vec![system_prompt.to_string()];

    if model_config.provider == ThirdPartyProvider::OpenAICodex {
        sections.push(CODEX_RESPONSE_STYLE_ADDENDUM.to_string());
    }

    if let Some(guidance) = extra_guidance
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        sections.push(guidance.to_string());
    }

    sections.join("\n\n")
}

fn build_responses_user_input(user_content: &str, image_data_list: &[Vec<u8>]) -> Vec<Value> {
    let mut content = vec![json!({
        "type": "input_text",
        "text": user_content.to_string(),
    })];

    for image_data in image_data_list {
        let mime_type = crate::llm::media::detect_mime_type(image_data)
            .unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        content.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": data_url,
        }));
    }

    vec![json!({
        "type": "message",
        "role": "user",
        "content": content,
    })]
}

fn build_responses_function_tools() -> Vec<Value> {
    if !web_search::is_search_enabled() {
        return Vec::new();
    }

    vec![json!({
        "type": "function",
        "name": "web_search",
        "description": "Search the web using the configured providers (Brave, Exa, Jina) and return a concise Markdown summary of the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        },
        "strict": false,
    })]
}

fn build_native_codex_web_search_tool(model_config: &ThirdPartyModelConfig) -> Option<Value> {
    if model_config.provider != ThirdPartyProvider::OpenAICodex {
        return None;
    }

    let record = selected_codex_model_record()?;
    if record.slug != model_config.model {
        return None;
    }

    openai_codex::build_native_web_search_tool_from_record(
        record.supports_search_tool,
        record.web_search_tool_type,
        openai_codex::native_web_search_mode(),
        &CONFIG.openai_codex_web_search_allowed_domains,
        Some(&CONFIG.openai_codex_web_search_context_size),
    )
}

fn convert_openai_function_tools_to_responses(tools: Vec<Value>) -> Vec<Value> {
    tools
        .into_iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            Some(json!({
                "type": "function",
                "name": function.get("name")?.as_str()?,
                "description": function
                    .get("description")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
                "parameters": function.get("parameters").cloned().unwrap_or_else(|| json!({})),
                "strict": false,
            }))
        })
        .collect()
}

fn responses_base_url(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/responses") {
        normalized.to_string()
    } else {
        format!("{normalized}/responses")
    }
}

/// Decide the `reasoning.effort` value for a request, if any.
///
/// `reasoning_override` is a per-call request (e.g. cheap agent steps asking
/// for "low"); it is validated against the selected Codex record's supported
/// levels when that record matches the requested model, and passed through
/// unvalidated for foreign slugs (the backend rejects unknown levels itself).
/// With no override this reproduces the original behavior: the globally
/// selected reasoning level of the matching Codex record.
fn reasoning_effort_for_request(
    provider: ThirdPartyProvider,
    model: &str,
    reasoning_override: Option<&str>,
    selected_record: Option<&CodexSelectedModelRecord>,
) -> Option<String> {
    if provider != ThirdPartyProvider::OpenAICodex {
        return None;
    }

    let matching_record = selected_record.filter(|record| record.slug == model);
    let recorded_level = matching_record.and_then(|record| record.selected_reasoning_level.clone());

    let Some(requested) = reasoning_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return recorded_level;
    };

    if let Some(record) = matching_record {
        let supported = record.supported_reasoning_levels.is_empty()
            || record
                .supported_reasoning_levels
                .iter()
                .any(|option| option.effort.eq_ignore_ascii_case(requested));
        if !supported {
            warn!(
                "Reasoning override '{}' is not supported by Codex model '{}'; keeping the selected level",
                requested, record.slug
            );
            return recorded_level;
        }
    }

    Some(requested.to_ascii_lowercase())
}

fn build_request_details(
    model_config: &ThirdPartyModelConfig,
    instructions: &str,
    input_items: Vec<Value>,
    tools: Option<Vec<Value>>,
    session_id: &str,
    reasoning_override: Option<&str>,
) -> Result<ResponsesRequestDetails> {
    let selected_record = if model_config.provider == ThirdPartyProvider::OpenAICodex {
        selected_codex_model_record()
    } else {
        None
    };
    let codex_account_id = if model_config.provider == ThirdPartyProvider::OpenAICodex {
        let account_id = crate::llm::runtime_models::current_codex_account_id()
            .ok_or_else(|| anyhow!("Codex auth token does not include a ChatGPT account id"))?;
        if model_config.id == OPENAI_CODEX_SELECTED_MODEL_ID {
            let record = selected_record
                .as_ref()
                .filter(|record| record.slug == model_config.model)
                .ok_or_else(|| anyhow!("The selected Codex model or account changed"))?;
            if record.account_id.as_deref().map(str::trim) != Some(account_id.as_str()) {
                return Err(anyhow!("The selected Codex model or account changed"));
            }
        }
        Some(account_id)
    } else {
        None
    };

    let (display_name, url, mut headers, streaming_sse) = match model_config.provider {
        ThirdPartyProvider::OpenAI => (
            "OpenAI",
            responses_base_url(&CONFIG.openai_base_url),
            vec![
                (
                    "Authorization".to_string(),
                    format!("Bearer {}", CONFIG.openai_api_key),
                ),
                (
                    "User-Agent".to_string(),
                    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                ),
            ],
            false,
        ),
        ThirdPartyProvider::OpenAICodex => (
            "OpenAI Codex",
            openai_codex::codex_response_url(),
            Vec::new(),
            true,
        ),
        _ => {
            return Err(anyhow!(
                "Unsupported responses provider {:?}",
                model_config.provider
            ))
        }
    };

    if streaming_sse {
        headers.push(("Accept".to_string(), "text/event-stream".to_string()));
    }

    let mut payload = json!({
        "model": model_config.model,
        "instructions": instructions,
        "input": input_items,
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "store": false,
        "stream": streaming_sse,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": session_id,
        "text": {
            "verbosity": "medium"
        },
    });

    if let Some(effort) = reasoning_effort_for_request(
        model_config.provider,
        &model_config.model,
        reasoning_override,
        selected_record.as_ref(),
    ) {
        payload["reasoning"] = json!({
            "effort": effort,
        });
    }

    if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
        payload["tools"] = Value::Array(tools);
    }

    Ok(ResponsesRequestDetails {
        provider: model_config.provider,
        display_name,
        url,
        headers,
        session_id: session_id.to_string(),
        codex_account_id,
        payload,
        streaming_sse,
        request_timeout_secs: responses_request_timeout_secs(model_config.provider),
    })
}

async fn resolve_request_headers_for_attempt(
    details: &ResponsesRequestDetails,
) -> Result<Vec<(String, String)>> {
    if details.provider != ThirdPartyProvider::OpenAICodex {
        return Ok(details.headers.clone());
    }

    let auth = openai_codex::get_valid_auth_context().await?;
    if details.codex_account_id.as_deref() != Some(auth.account_id.trim()) {
        return Err(anyhow!(
            "The active ChatGPT account changed while the Codex request was in progress"
        ));
    }
    let mut headers = openai_codex::codex_headers(&auth, Some(&details.session_id));
    headers.extend(details.headers.clone());
    Ok(headers)
}

fn parse_sse_responses_body(body: &str) -> std::result::Result<Value, SseParseError> {
    let mut output_items: Vec<Value> = Vec::new();
    let mut current_data_lines: Vec<String> = Vec::new();
    let mut response_id: Option<String> = None;
    let mut usage: Option<Value> = None;
    let mut completed = false;

    let flush_event = |lines: &mut Vec<String>,
                       output_items: &mut Vec<Value>,
                       response_id: &mut Option<String>,
                       usage: &mut Option<Value>,
                       completed: &mut bool|
     -> std::result::Result<(), SseParseError> {
        if lines.is_empty() {
            return Ok(());
        }
        let payload = lines.join("\n");
        lines.clear();
        if payload.trim().is_empty() || payload.trim() == "[DONE]" {
            return Ok(());
        }

        let value: Value =
            serde_json::from_str(&payload).map_err(|source| SseParseError::InvalidPayload {
                bytes: payload.len(),
                source,
            })?;
        if response_id.is_none() {
            *response_id = value
                .pointer("/response/id")
                .and_then(|value| value.as_str())
                .map(str::to_string);
        }
        match value.get("type").and_then(|value| value.as_str()) {
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item").cloned() {
                    output_items.push(item);
                }
            }
            Some("response.completed") => {
                *completed = true;
                if usage.is_none() {
                    *usage = value.pointer("/response/usage").cloned();
                }
                if output_items.is_empty() {
                    if let Some(items) = value
                        .get("response")
                        .and_then(|response| response.get("output"))
                        .and_then(|items| items.as_array())
                    {
                        output_items.extend(items.iter().cloned());
                    }
                }
            }
            Some("response.incomplete") => {
                let reason = value
                    .pointer("/response/incomplete_details/reason")
                    .and_then(|value| value.as_str());
                return Err(SseParseError::Incomplete(
                    classify_incomplete_reason(reason).to_string(),
                ));
            }
            Some("response.failed") => {
                let message_present = value
                    .pointer("/response/error/message")
                    .and_then(|value| value.as_str())
                    .or_else(|| {
                        value
                            .pointer("/error/message")
                            .and_then(|value| value.as_str())
                    })
                    .is_some();
                return Err(SseParseError::Failed(format!(
                    "response_failed(message_present={message_present})"
                )));
            }
            Some("error") => {
                let message_present = value
                    .pointer("/error/message")
                    .and_then(|value| value.as_str())
                    .or_else(|| value.get("message").and_then(|value| value.as_str()))
                    .is_some();
                return Err(SseParseError::Failed(format!(
                    "error_event(message_present={message_present})"
                )));
            }
            _ => {}
        }

        Ok(())
    };

    for line in body.lines() {
        if line.trim().is_empty() {
            flush_event(
                &mut current_data_lines,
                &mut output_items,
                &mut response_id,
                &mut usage,
                &mut completed,
            )?;
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            current_data_lines.push(data.trim_start().to_string());
        }
    }
    flush_event(
        &mut current_data_lines,
        &mut output_items,
        &mut response_id,
        &mut usage,
        &mut completed,
    )?;

    if !completed {
        return Err(SseParseError::MissingCompletion);
    }

    Ok(json!({
        "id": response_id,
        "output": output_items,
        "usage": usage,
    }))
}

async fn call_provider_api(
    details: &ResponsesRequestDetails,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
    turn_state: &mut CodexTurnState,
) -> Result<ResponsesApiResult> {
    let refresh_codex_auth = details.provider == ThirdPartyProvider::OpenAICodex;
    let result = call_provider_api_with_auth_hooks(
        details,
        audit_context,
        operation,
        turn_state,
        || resolve_request_headers_for_attempt(details),
        move || async move {
            if refresh_codex_auth {
                openai_codex::force_refresh_auth_tokens().await?;
            }
            Ok(())
        },
    )
    .await?;
    debug!(
        "{} completed response metadata: request_id={:?}, models_etag={:?}, rate_limit_header_count={}",
        details.display_name,
        result.metadata.request_id,
        result.metadata.models_etag,
        result.metadata.rate_limit_headers.len()
    );
    Ok(result)
}

async fn call_provider_api_with_auth_hooks<
    ResolveHeaders,
    ResolveHeadersFuture,
    RefreshAuth,
    RefreshAuthFuture,
>(
    details: &ResponsesRequestDetails,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
    turn_state: &mut CodexTurnState,
    mut resolve_headers: ResolveHeaders,
    mut refresh_auth: RefreshAuth,
) -> Result<ResponsesApiResult>
where
    ResolveHeaders: FnMut() -> ResolveHeadersFuture,
    ResolveHeadersFuture: Future<Output = Result<Vec<(String, String)>>>,
    RefreshAuth: FnMut() -> RefreshAuthFuture,
    RefreshAuthFuture: Future<Output = Result<()>>,
{
    let model = details
        .payload
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let started_at = chrono::Utc::now();
    let audit_metadata = json!({
        "request_summary": summarize_responses_payload(&details.payload),
        "streaming_sse": details.streaming_sse,
        "timeout_secs": details.request_timeout_secs
    });
    log_llm_request_started(
        details.provider.as_str(),
        model,
        operation,
        started_at,
        Some(&audit_metadata),
    );
    info!(
        "{} responses request starting: {}",
        details.display_name,
        summarize_responses_payload(&details.payload)
    );
    let client = if details.streaming_sse {
        get_http_client_no_compression()
    } else {
        get_http_client()
    };
    for attempt in 1..=RESPONSES_MAX_ATTEMPTS {
        let mut attempt_headers = resolve_headers().await?;
        if details.provider == ThirdPartyProvider::OpenAICodex {
            turn_state.apply(&mut attempt_headers);
        }
        let mut request = client
            .post(&details.url)
            .timeout(Duration::from_secs(details.request_timeout_secs));
        for (name, value) in &attempt_headers {
            request = request.header(name, value);
        }
        if details.streaming_sse {
            request = request.header(reqwest::header::ACCEPT_ENCODING, "identity");
        }
        debug!(
            "{} request timeout configured: model={}, timeout_secs={}, streaming_sse={}, attempt={}/{}",
            details.display_name,
            model,
            details.request_timeout_secs,
            details.streaming_sse,
            attempt,
            RESPONSES_MAX_ATTEMPTS
        );

        let response = match request.json(&details.payload).send().await {
            Ok(response) => response,
            Err(err) => {
                let should_retry =
                    responses_should_retry_error(&err) && attempt < RESPONSES_MAX_ATTEMPTS;
                let log_message = format!(
                    "{} responses request failed to send: model={}, error={}, timeout={}, connect={}, status={:?}, attempt={}/{}, retrying={}",
                    details.display_name,
                    model,
                    err,
                    err.is_timeout(),
                    err.is_connect(),
                    err.status(),
                    attempt,
                    RESPONSES_MAX_ATTEMPTS,
                    should_retry
                );
                if should_retry {
                    warn!("{log_message}");
                } else {
                    error!("{log_message}");
                }
                if should_retry {
                    tokio::time::sleep(responses_retry_delay(attempt)).await;
                    continue;
                }
                return Err(anyhow!("{} request failed: {}", details.display_name, err));
            }
        };

        let status = response.status();
        if details.provider == ThirdPartyProvider::OpenAICodex {
            turn_state.capture(response.headers());
        }
        let response_metadata =
            capture_response_metadata(response.headers(), details.codex_account_id.as_deref());
        observe_response_metadata(details.provider, &response_metadata);

        if !status.is_success() {
            if status == StatusCode::UNAUTHORIZED
                && details.provider == ThirdPartyProvider::OpenAICodex
                && attempt < RESPONSES_MAX_ATTEMPTS
            {
                warn!(
                    "{} responses request unauthorized for model={}; refreshing Codex auth and retrying (attempt={}/{})",
                    details.display_name,
                    model,
                    attempt,
                    RESPONSES_MAX_ATTEMPTS
                );
                refresh_auth().await?;
                continue;
            }
            let body = match read_response_body_bytes(
                response,
                details.display_name,
                model,
                attempt,
                details.streaming_sse,
            )
            .await
            {
                Ok((bytes, _)) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(err) => {
                    let should_retry = responses_should_retry_status(status)
                        && err.is_retryable()
                        && attempt < RESPONSES_MAX_ATTEMPTS;
                    if should_retry {
                        tokio::time::sleep(responses_retry_delay(attempt)).await;
                        continue;
                    }
                    return Err(anyhow!(
                        "{} request failed with status {} and its error body could not be read: {}",
                        details.display_name,
                        status,
                        err
                    ));
                }
            };
            let body_summary = summarize_error_body(&body);
            let should_retry =
                responses_should_retry_status(status) && attempt < RESPONSES_MAX_ATTEMPTS;
            let log_message = format!(
                "{} responses API error: model={}, status={}, error_summary={}, attempt={}/{}, retrying={}",
                details.display_name,
                model,
                status,
                body_summary,
                attempt,
                RESPONSES_MAX_ATTEMPTS,
                should_retry
            );
            if should_retry {
                warn!("{log_message}");
            } else {
                error!("{log_message}");
            }
            if should_retry {
                tokio::time::sleep(responses_retry_delay(attempt)).await;
                continue;
            }
            return Err(anyhow!(
                "{} request failed with status {}: {}",
                details.display_name,
                status,
                body_summary
            ));
        }

        debug!(
            "{} response metadata: model={}, request_id={:?}, models_etag={:?}, rate_limit_headers={:?}",
            details.display_name,
            model,
            response_metadata.request_id,
            response_metadata.models_etag,
            response_metadata.rate_limit_headers
        );

        let (body_bytes, header_summary) = match read_response_body_bytes(
            response,
            details.display_name,
            model,
            attempt,
            details.streaming_sse,
        )
        .await
        {
            Ok(result) => result,
            // The body stream was interrupted (e.g. a Cloudflare-fronted chunked SSE
            // connection dropped while the model was reasoning and emitting no events).
            // Re-issuing is safe: the payload is unchanged across attempts, and
            // `store=false` means the failed attempt left no server-side state behind.
            // So retry with the same backoff used for send-phase failures instead of
            // surfacing the failure to the user. `read_response_body_bytes` already
            // logged the diagnostics, including whether a retry would follow.
            Err(err) => {
                if err.is_retryable() && attempt < RESPONSES_MAX_ATTEMPTS {
                    tokio::time::sleep(responses_retry_delay(attempt)).await;
                    continue;
                }
                return Err(err.into());
            }
        };
        debug!(
            "{} response headers for model={}: [{}]",
            details.display_name, model, header_summary
        );
        let body = match String::from_utf8(body_bytes) {
            Ok(body) => body,
            Err(err) => {
                let bytes = err.into_bytes();
                let should_retry = details.streaming_sse && attempt < RESPONSES_MAX_ATTEMPTS;
                let log_message = format!(
                    "{} response body was not valid UTF-8: model={}, attempt={}/{}, headers=[{}], bytes={}, retrying={}",
                    details.display_name,
                    model,
                    attempt,
                    RESPONSES_MAX_ATTEMPTS,
                    header_summary,
                    bytes.len(),
                    should_retry
                );
                if should_retry {
                    warn!("{log_message}");
                    tokio::time::sleep(responses_retry_delay(attempt)).await;
                    continue;
                }
                error!("{log_message}");
                return Err(anyhow!(
                    "{} response body was not valid UTF-8",
                    details.display_name
                ));
            }
        };

        let value = if details.streaming_sse {
            match parse_sse_responses_body(&body) {
                Ok(value) => value,
                Err(err) => {
                    let should_retry = err.is_retryable() && attempt < RESPONSES_MAX_ATTEMPTS;
                    let log_message = format!(
                        "{} SSE response rejected: model={}, attempt={}/{}, headers=[{}], retrying={}, error={}",
                        details.display_name,
                        model,
                        attempt,
                        RESPONSES_MAX_ATTEMPTS,
                        header_summary,
                        should_retry,
                        err
                    );
                    if should_retry {
                        warn!("{log_message}");
                        tokio::time::sleep(responses_retry_delay(attempt)).await;
                        continue;
                    }
                    error!("{log_message}");
                    return Err(err.into());
                }
            }
        } else {
            match serde_json::from_str::<Value>(&body) {
                Ok(value) => value,
                Err(err) => {
                    error!(
                        "{} JSON parse failed: model={}, attempt={}/{}, headers=[{}], bytes={}, error={}",
                        details.display_name,
                        model,
                        attempt,
                        RESPONSES_MAX_ATTEMPTS,
                        header_summary,
                        body.len(),
                        err
                    );
                    return Err(anyhow!(
                        "{} response JSON parse failed: {}",
                        details.display_name,
                        err
                    ));
                }
            }
        };
        let output_items = extract_response_output_items(&value);
        info!(
            "{} responses request completed: model={}, output_items={}, output_summary=[{}]",
            details.display_name,
            model,
            output_items.len(),
            summarize_output_items(&output_items)
        );
        let usage = extract_responses_usage(&value);
        record_llm_request_success(
            audit_context,
            details.provider.as_str(),
            model,
            operation,
            started_at,
            chrono::Utc::now(),
            usage,
        )
        .await;
        return Ok(ResponsesApiResult {
            response: value,
            metadata: response_metadata,
        });
    }

    unreachable!("responses provider retry loop exhausted")
}

fn extract_response_output_items(response: &Value) -> Vec<Value> {
    response
        .get("output")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default()
}

fn extract_responses_usage(response: &Value) -> LlmUsageRecord {
    let usage_value = response.get("usage").cloned();
    let input_tokens = usage_value
        .as_ref()
        .and_then(|usage| usage.get("input_tokens"))
        .and_then(|value| value.as_i64());
    let output_tokens = usage_value
        .as_ref()
        .and_then(|usage| usage.get("output_tokens"))
        .and_then(|value| value.as_i64());
    let total_tokens = usage_value
        .as_ref()
        .and_then(|usage| usage.get("total_tokens"))
        .and_then(|value| value.as_i64())
        .or_else(|| match (input_tokens, output_tokens) {
            (Some(input_tokens), Some(output_tokens)) => Some(input_tokens + output_tokens),
            _ => None,
        });
    let reasoning_tokens = usage_value
        .as_ref()
        .and_then(|usage| usage.pointer("/output_tokens_details/reasoning_tokens"))
        .and_then(|value| value.as_i64());
    let cached_input_tokens = usage_value
        .as_ref()
        .and_then(|usage| usage.pointer("/input_tokens_details/cached_tokens"))
        .and_then(|value| value.as_i64());

    LlmUsageRecord {
        response_id: response
            .get("id")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        input_tokens,
        output_tokens,
        total_tokens,
        reasoning_tokens,
        cached_input_tokens,
        raw_usage_json: usage_value.map(|usage| usage.to_string()),
    }
}

fn extract_response_text(output_items: &[Value]) -> String {
    let mut text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    for item in output_items {
        match item.get("type").and_then(|value| value.as_str()) {
            Some("message") => {
                if let Some(content_items) = item.get("content").and_then(|value| value.as_array())
                {
                    for content_item in content_items {
                        let item_type = content_item.get("type").and_then(|value| value.as_str());
                        if matches!(item_type, Some("output_text") | Some("text")) {
                            if let Some(text) =
                                content_item.get("text").and_then(|value| value.as_str())
                            {
                                let trimmed = text.trim();
                                if !trimmed.is_empty() {
                                    text_parts.push(trimmed.to_string());
                                }
                            }
                        }
                    }
                }
            }
            Some("reasoning") => {
                if let Some(summary_items) = item.get("summary").and_then(|value| value.as_array())
                {
                    for summary_item in summary_items {
                        if let Some(text) =
                            summary_item.get("text").and_then(|value| value.as_str())
                        {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                reasoning_parts.push(trimmed.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        return text_parts.join("\n");
    }
    reasoning_parts.join("\n")
}

fn extract_response_tool_calls(output_items: &[Value]) -> Vec<ResponsesToolCall> {
    output_items
        .iter()
        .filter(|item| item.get("type").and_then(|value| value.as_str()) == Some("function_call"))
        .filter_map(|item| {
            Some(ResponsesToolCall {
                call_id: item.get("call_id")?.as_str()?.to_string(),
                name: item.get("name")?.as_str()?.to_string(),
                arguments: item
                    .get("arguments")
                    .and_then(|value| value.as_str())
                    .unwrap_or("{}")
                    .to_string(),
            })
        })
        .collect()
}

async fn execute_function_tool(name: &str, arguments: &Value) -> Result<String> {
    let argument_keys = arguments
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    debug!(
        "Responses tool call requested: name={}, argument_keys={:?}, argument_bytes={}",
        name,
        argument_keys,
        arguments.to_string().len()
    );
    match name {
        "web_search" => {
            let query = arguments
                .get("query")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let max_results = arguments
                .get("max_results")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize);
            match web_search_tool(query, max_results).await {
                Ok(result) => {
                    debug!(
                        "Responses tool call web_search completed: chars={}",
                        result.chars().count()
                    );
                    Ok(result)
                }
                Err(err) => Err(err),
            }
        }
        _ => Ok(String::from("Unsupported tool call")),
    }
}

async fn responses_completion_with_tools(
    instructions: &str,
    mut input_items: Vec<Value>,
    model_config: &ThirdPartyModelConfig,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
    reasoning_override: Option<&str>,
) -> Result<String> {
    let tools = build_responses_function_tools();
    let session_id = generate_session_id();
    let mut turn_state = CodexTurnState::default();
    let model_label = debug_model_label(model_config);
    debug!(
        "Responses tool loop starting: model={}, session_id={}, tools_enabled={}",
        model_label,
        session_id,
        !tools.is_empty()
    );

    for iteration in 0..MAX_TOOL_CALL_ITERATIONS {
        debug!(
            "Responses tool iteration {}/{} for model={} session_id={}",
            iteration + 1,
            MAX_TOOL_CALL_ITERATIONS,
            model_label,
            session_id
        );
        let details = build_request_details(
            model_config,
            instructions,
            input_items.clone(),
            Some(tools.clone()),
            &session_id,
            reasoning_override,
        )?;
        let ResponsesApiResult {
            response,
            metadata: _,
        } = call_provider_api(&details, audit_context, operation, &mut turn_state).await?;
        let output_items = extract_response_output_items(&response);
        let tool_calls = extract_response_tool_calls(&output_items);
        let content = extract_response_text(&output_items);

        if tool_calls.is_empty() {
            return Ok(content);
        }

        debug!(
            "Responses tool iteration {}/{} returned {} tool call(s) for model={} session_id={}",
            iteration + 1,
            MAX_TOOL_CALL_ITERATIONS,
            tool_calls.len(),
            model_label,
            session_id
        );

        input_items.extend(output_items.clone());

        for tool_call in tool_calls {
            let args_value: Value =
                serde_json::from_str(&tool_call.arguments).unwrap_or(Value::Null);
            let result = execute_function_tool(&tool_call.name, &args_value)
                .await
                .unwrap_or_else(|err| err.to_string());
            input_items.push(json!({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": result,
            }));
        }

        if iteration + 1 == MAX_TOOL_CALL_ITERATIONS {
            let final_instructions = format!("{instructions}\n\n{TOOL_LIMIT_SYSTEM_PROMPT}");
            let details = build_request_details(
                model_config,
                &final_instructions,
                input_items,
                None,
                &session_id,
                reasoning_override,
            )?;
            let ResponsesApiResult {
                response,
                metadata: _,
            } = call_provider_api(&details, audit_context, operation, &mut turn_state).await?;
            return Ok(extract_response_text(&extract_response_output_items(
                &response,
            )));
        }
    }

    unreachable!("responses tool loop exhausted without returning")
}

#[allow(clippy::too_many_arguments)]
async fn responses_completion_with_tool_runtime(
    instructions: &str,
    mut input_items: Vec<Value>,
    model_config: &ThirdPartyModelConfig,
    runtime: &mut ToolRuntime,
    native_codex_web_search_tool: Option<Value>,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
    reasoning_override: Option<&str>,
) -> Result<String> {
    let mut tools =
        convert_openai_function_tools_to_responses(runtime.build_openai_function_tools());
    let has_native_codex_web_search = native_codex_web_search_tool.is_some();
    let model_label = debug_model_label(model_config);
    if has_native_codex_web_search {
        tools
            .retain(|tool| tool.get("name").and_then(|value| value.as_str()) != Some("web_search"));
    }
    if let Some(native_tool) = native_codex_web_search_tool {
        tools.push(native_tool);
    }
    let mut tools_enabled = !tools.is_empty();
    let session_id = generate_session_id();
    let mut turn_state = CodexTurnState::default();
    let mut final_answer_requested = false;
    debug!(
        "Responses runtime tool loop starting: model={}, session_id={}, tools_enabled={}, native_codex_web_search={}",
        model_label,
        session_id,
        tools_enabled,
        has_native_codex_web_search
    );

    for iteration in 0..runtime.max_total_successful_calls().saturating_add(2) {
        debug!(
            "Responses runtime iteration {} for model={} session_id={} tools_enabled={}",
            iteration + 1,
            model_label,
            session_id,
            tools_enabled
        );
        let details = build_request_details(
            model_config,
            instructions,
            input_items.clone(),
            tools_enabled.then_some(tools.clone()),
            &session_id,
            reasoning_override,
        )?;
        let ResponsesApiResult {
            response,
            metadata: _,
        } = call_provider_api(&details, audit_context, operation, &mut turn_state).await?;
        let output_items = extract_response_output_items(&response);
        let tool_calls = if tools_enabled {
            extract_response_tool_calls(&output_items)
        } else {
            Vec::new()
        };

        if tool_calls.is_empty() {
            return Ok(extract_response_text(&output_items));
        }

        debug!(
            "Responses runtime iteration {} returned {} tool call(s) for model={} session_id={}",
            iteration + 1,
            tool_calls.len(),
            model_label,
            session_id
        );

        input_items.extend(output_items.clone());

        for tool_call in tool_calls {
            let args_value: Value =
                serde_json::from_str(&tool_call.arguments).unwrap_or_else(|_| json!({}));
            let result = runtime.execute_tool(&tool_call.name, &args_value).await;
            input_items.push(json!({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": result,
            }));
        }

        if runtime.force_final_answer() && !final_answer_requested {
            final_answer_requested = true;
            tools_enabled = false;
        }
    }

    let final_instructions = format!("{instructions}\n\n{TOOL_LIMIT_SYSTEM_PROMPT}");
    let details = build_request_details(
        model_config,
        &final_instructions,
        input_items,
        None,
        &session_id,
        reasoning_override,
    )?;
    let ResponsesApiResult {
        response,
        metadata: _,
    } = call_provider_api(&details, audit_context, operation, &mut turn_state).await?;
    Ok(extract_response_text(&extract_response_output_items(
        &response,
    )))
}

#[allow(clippy::too_many_arguments)]
pub async fn call_responses_provider(
    system_prompt: &str,
    user_content: &str,
    model_config: &ThirdPartyModelConfig,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    supports_tools: bool,
    audit_context: Option<&LlmAuditContext>,
    reasoning_override: Option<&str>,
) -> Result<String> {
    let native_codex_web_search_tool = supports_tools
        .then(|| build_native_codex_web_search_tool(model_config))
        .flatten();
    let custom_tools_enabled =
        supports_tools && native_codex_web_search_tool.is_none() && web_search::is_search_enabled();
    let model_label = debug_model_label(model_config);
    debug!(
        "Responses provider selected: provider={}, model={}, response_title={}, supports_tools={}, custom_tools_enabled={}, native_codex_web_search={}, image_count={}",
        model_config.provider.as_str(),
        model_label,
        response_title,
        supports_tools,
        custom_tools_enabled,
        native_codex_web_search_tool.is_some(),
        image_data_list.len()
    );
    let tool_limit_guidance = custom_tools_enabled.then(responses_tool_limit_guidance);
    let instructions =
        build_responses_system_prompt(system_prompt, model_config, tool_limit_guidance.as_deref());
    let input_items = build_responses_user_input(user_content, image_data_list);
    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);
    if custom_tools_enabled {
        return responses_completion_with_tools(
            &instructions,
            input_items,
            model_config,
            audit_context,
            &operation,
            reasoning_override,
        )
        .await;
    }

    let session_id = generate_session_id();
    let mut turn_state = CodexTurnState::default();
    let details = build_request_details(
        model_config,
        &instructions,
        input_items,
        native_codex_web_search_tool.map(|tool| vec![tool]),
        &session_id,
        reasoning_override,
    )?;
    let ResponsesApiResult {
        response,
        metadata: _,
    } = call_provider_api(&details, audit_context, &operation, &mut turn_state).await?;
    Ok(extract_response_text(&extract_response_output_items(
        &response,
    )))
}

#[allow(clippy::too_many_arguments)]
pub async fn call_responses_provider_with_tool_runtime(
    system_prompt: &str,
    user_content: &str,
    model_config: &ThirdPartyModelConfig,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    runtime: &mut ToolRuntime,
    audit_context: Option<&LlmAuditContext>,
    reasoning_override: Option<&str>,
) -> Result<String> {
    let runtime_guidance = runtime.tool_limit_guidance();
    let instructions =
        build_responses_system_prompt(system_prompt, model_config, Some(&runtime_guidance));
    let input_items = build_responses_user_input(user_content, image_data_list);
    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);
    let native_codex_web_search_tool = runtime
        .allows_web_search()
        .then(|| build_native_codex_web_search_tool(model_config))
        .flatten();
    let model_label = debug_model_label(model_config);
    debug!(
        "Responses provider runtime selected: provider={}, model={}, response_title={}, native_codex_web_search={}, image_count={}",
        model_config.provider.as_str(),
        model_label,
        response_title,
        native_codex_web_search_tool.is_some(),
        image_data_list.len()
    );

    responses_completion_with_tool_runtime(
        &instructions,
        input_items,
        model_config,
        runtime,
        native_codex_web_search_tool.clone(),
        audit_context,
        &operation,
        reasoning_override,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ThirdPartyProvider;

    fn read_http_request_headers(stream: &mut std::net::TcpStream) -> String {
        use std::io::Read;

        let mut request = Vec::new();
        let mut chunk = [0u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let bytes_read = stream.read(&mut chunk).expect("read HTTP request");
            if bytes_read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..bytes_read]);
        }
        String::from_utf8_lossy(&request).into_owned()
    }

    fn model_config(provider: ThirdPartyProvider, model: &str) -> ThirdPartyModelConfig {
        ThirdPartyModelConfig {
            id: format!("{}:{}", provider.as_str(), model),
            provider,
            name: model.to_string(),
            model: model.to_string(),
            image: false,
            video: false,
            audio: false,
            tools: true,
        }
    }

    fn codex_record(
        slug: &str,
        supported: &[&str],
        selected: Option<&str>,
    ) -> CodexSelectedModelRecord {
        CodexSelectedModelRecord {
            account_id: None,
            slug: slug.to_string(),
            display_name: slug.to_string(),
            description: None,
            input_modalities: vec!["text".to_string()],
            priority: 0,
            etag: None,
            default_reasoning_level: None,
            supported_reasoning_levels: supported
                .iter()
                .map(
                    |effort| crate::llm::openai_codex::CodexReasoningEffortOption {
                        effort: effort.to_string(),
                        description: effort.to_string(),
                    },
                )
                .collect(),
            selected_reasoning_level: selected.map(str::to_string),
            web_search_tool_type: Default::default(),
            supports_search_tool: false,
            fetched_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn reasoning_override_ignored_for_non_codex_providers() {
        assert_eq!(
            reasoning_effort_for_request(ThirdPartyProvider::OpenAI, "gpt-5.4", Some("low"), None),
            None
        );
    }

    #[test]
    fn reasoning_without_override_uses_selected_record_level() {
        let record = codex_record("gpt-5.5", &["low", "xhigh"], Some("xhigh"));
        assert_eq!(
            reasoning_effort_for_request(
                ThirdPartyProvider::OpenAICodex,
                "gpt-5.5",
                None,
                Some(&record)
            ),
            Some("xhigh".to_string())
        );
        assert_eq!(
            reasoning_effort_for_request(ThirdPartyProvider::OpenAICodex, "gpt-5.5", None, None),
            None
        );
    }

    #[test]
    fn reasoning_override_applies_when_supported() {
        let record = codex_record("gpt-5.5", &["low", "medium", "xhigh"], Some("xhigh"));
        assert_eq!(
            reasoning_effort_for_request(
                ThirdPartyProvider::OpenAICodex,
                "gpt-5.5",
                Some("Low"),
                Some(&record)
            ),
            Some("low".to_string())
        );
    }

    #[test]
    fn unsupported_reasoning_override_falls_back_to_selected_level() {
        let record = codex_record("gpt-5.5", &["medium", "xhigh"], Some("xhigh"));
        assert_eq!(
            reasoning_effort_for_request(
                ThirdPartyProvider::OpenAICodex,
                "gpt-5.5",
                Some("low"),
                Some(&record)
            ),
            Some("xhigh".to_string())
        );
    }

    #[test]
    fn reasoning_override_passes_through_for_foreign_slugs() {
        let record = codex_record("gpt-5.5", &["medium"], Some("medium"));
        assert_eq!(
            reasoning_effort_for_request(
                ThirdPartyProvider::OpenAICodex,
                "gpt-5.4-mini",
                Some("low"),
                Some(&record)
            ),
            Some("low".to_string())
        );
        assert_eq!(
            reasoning_effort_for_request(
                ThirdPartyProvider::OpenAICodex,
                "gpt-5.4-mini",
                Some("  "),
                Some(&record)
            ),
            None
        );
    }

    #[test]
    fn extract_response_text_reads_output_text_blocks() {
        let output = vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "output_text", "text": "hello" },
                { "type": "output_text", "text": "world" }
            ]
        })];

        assert_eq!(extract_response_text(&output), "hello\nworld");
    }

    #[test]
    fn extract_response_tool_calls_reads_function_calls() {
        let output = vec![json!({
            "type": "function_call",
            "call_id": "call_123",
            "name": "web_search",
            "arguments": "{\"query\":\"rust\"}"
        })];

        let calls = extract_response_tool_calls(&output);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_123");
        assert_eq!(calls[0].name, "web_search");
    }

    #[test]
    fn responses_base_url_appends_suffix_once() {
        assert_eq!(
            responses_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            responses_base_url("https://chatgpt.com/backend-api/codex/responses"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn codex_system_prompt_appends_extra_style_guidance() {
        let instructions = build_responses_system_prompt(
            "Base prompt",
            &model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.4"),
            Some("Tool guidance"),
        );

        assert!(instructions.contains("Base prompt"));
        assert!(instructions.contains("Answers first"));
        assert!(instructions.contains("Tool guidance"));
        // The addendum must be left-aligned: no source indentation may bleed
        // into the prompt the model receives.
        assert!(instructions.contains("\n# Style"));
        assert!(!instructions.contains("    # Style"));
    }

    #[test]
    fn public_openai_system_prompt_skips_codex_style_guidance() {
        let instructions = build_responses_system_prompt(
            "Base prompt",
            &model_config(ThirdPartyProvider::OpenAI, "gpt-5.4"),
            None,
        );

        assert_eq!(instructions, "Base prompt");
    }

    #[test]
    fn responses_request_timeout_uses_provider_config() {
        assert_eq!(
            responses_request_timeout_secs(ThirdPartyProvider::OpenAI),
            CONFIG.openai_request_timeout_secs
        );
        assert_eq!(
            responses_request_timeout_secs(ThirdPartyProvider::OpenAICodex),
            CONFIG.openai_codex_request_timeout_secs
        );
    }

    #[test]
    fn extract_responses_usage_reads_token_counts_and_reasoning_details() {
        let response = json!({
            "id": "resp_123",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {
                    "cached_tokens": 3
                },
                "output_tokens_details": {
                    "reasoning_tokens": 7
                }
            }
        });

        let usage = extract_responses_usage(&response);

        assert_eq!(usage.response_id.as_deref(), Some("resp_123"));
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(30));
        assert_eq!(usage.reasoning_tokens, Some(7));
        assert_eq!(usage.cached_input_tokens, Some(3));
    }

    #[test]
    fn parse_sse_responses_body_collects_output_items() {
        let body = r#"event: response.created
data: {"type":"response.created","response":{"id":"resp1"}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"web_search","arguments":"{}"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp1","output":[],"usage":{"input_tokens":12,"output_tokens":8}}}
"#;

        let parsed = parse_sse_responses_body(body).expect("SSE body should parse");
        let output = parsed
            .get("output")
            .and_then(|value| value.as_array())
            .expect("output array");

        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(
            parsed.get("id").and_then(|value| value.as_str()),
            Some("resp1")
        );
        assert_eq!(
            parsed
                .pointer("/usage/input_tokens")
                .and_then(|value| value.as_i64()),
            Some(12)
        );
    }

    #[test]
    fn parse_sse_responses_body_rejects_clean_eof_without_completion() {
        let body = r#"event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}}

"#;

        let err = parse_sse_responses_body(body).expect_err("completion is mandatory");
        assert!(matches!(err, SseParseError::MissingCompletion));
    }

    #[test]
    fn parse_sse_responses_body_reports_incomplete_terminal_event() {
        let body = r#"event: response.incomplete
data: {"type":"response.incomplete","response":{"id":"resp1","incomplete_details":{"reason":"max_output_tokens"}}}

"#;

        let err = parse_sse_responses_body(body).expect_err("incomplete is not success");
        assert!(matches!(
            err,
            SseParseError::Incomplete(detail) if detail == "max_output_tokens"
        ));
    }

    #[test]
    fn incomplete_sse_does_not_retain_remote_error_message() {
        let body = r#"event: response.incomplete
data: {"type":"response.incomplete","response":{"id":"resp1","error":{"message":"secret APIKEY123"}}}

"#;

        let err = parse_sse_responses_body(body).expect_err("incomplete is not success");
        assert!(matches!(
            err,
            SseParseError::Incomplete(detail) if detail == "other"
        ));
    }

    #[test]
    fn parse_sse_responses_body_reports_failed_terminal_event() {
        let body = r#"event: response.failed
data: {"type":"response.failed","response":{"id":"resp1","error":{"message":"backend rejected request"}}}

"#;

        let err = parse_sse_responses_body(body).expect_err("failed is not success");
        assert!(matches!(
            err,
            SseParseError::Failed(detail) if detail == "response_failed(message_present=true)"
        ));
    }

    #[test]
    fn provider_error_summary_does_not_retain_remote_message_or_body() {
        let secret = "user prompt and bearer token";
        let json_body = format!(
            r#"{{"error":{{"code":"rate_limit","type":"request_error","message":"{secret}"}}}}"#
        );
        let json_summary = summarize_error_body(&json_body);
        let text_summary = summarize_error_body(secret);

        assert_eq!(
            json_summary,
            format!(
                "json_error(code_present=true, type_present=true, bytes={})",
                json_body.len()
            )
        );
        assert_eq!(
            text_summary,
            format!("non_json_error(bytes={})", secret.len())
        );
        assert!(!json_summary.contains(secret));
        assert!(!text_summary.contains(secret));
    }

    #[test]
    fn response_metadata_captures_observability_headers_without_turn_state() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, "req-123".parse().unwrap());
        headers.insert(MODELS_ETAG_HEADER, "models-v2".parse().unwrap());
        headers.insert(CODEX_TURN_STATE_HEADER, "sticky-secret".parse().unwrap());
        headers.insert("x-codex-primary-used-percent", "41.5".parse().unwrap());

        let metadata = capture_response_metadata(&headers, Some("acct-1"));

        assert_eq!(metadata.request_id.as_deref(), Some("req-123"));
        assert_eq!(metadata.models_etag.as_deref(), Some("models-v2"));
        assert_eq!(metadata.codex_account_id.as_deref(), Some("acct-1"));
        assert_eq!(
            metadata
                .rate_limit_headers
                .get("x-codex-primary-used-percent")
                .map(String::as_str),
            Some("41.5")
        );
        assert!(!metadata
            .rate_limit_headers
            .contains_key(CODEX_TURN_STATE_HEADER));
    }

    #[test]
    fn codex_turn_state_is_first_value_wins_and_fresh_state_is_empty() {
        let mut first_headers = reqwest::header::HeaderMap::new();
        first_headers.insert(CODEX_TURN_STATE_HEADER, "sticky-1".parse().unwrap());
        let mut later_headers = reqwest::header::HeaderMap::new();
        later_headers.insert(CODEX_TURN_STATE_HEADER, "sticky-2".parse().unwrap());

        let mut turn_state = CodexTurnState::default();
        turn_state.capture(&first_headers);
        turn_state.capture(&later_headers);
        let mut replay_headers = Vec::new();
        turn_state.apply(&mut replay_headers);
        assert_eq!(
            replay_headers,
            vec![(CODEX_TURN_STATE_HEADER.to_string(), "sticky-1".to_string())]
        );

        let mut next_turn_headers = Vec::new();
        CodexTurnState::default().apply(&mut next_turn_headers);
        assert!(next_turn_headers.is_empty());
    }

    #[tokio::test]
    async fn read_response_body_bytes_errors_on_truncated_chunked_stream() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            // A valid first chunk, then the socket is dropped without the terminating
            // `0\r\n\r\n` chunk — the exact "truncated mid-stream" shape from production.
            let partial = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n";
            let _ = stream.write_all(partial.as_bytes());
            let _ = stream.flush();
        });

        let client = reqwest::Client::builder().build().unwrap();
        let response = client
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("request headers should arrive");
        let result = read_response_body_bytes(response, "Test", "test-model", 1, true).await;
        assert!(
            result.is_err(),
            "a truncated chunked body must surface as an error so the loop can retry"
        );
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn read_response_body_bytes_reads_complete_chunked_stream() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            let full = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
            let _ = stream.write_all(full.as_bytes());
            let _ = stream.flush();
        });

        let client = reqwest::Client::builder().build().unwrap();
        let response = client
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("request headers should arrive");
        let (body, _headers) = read_response_body_bytes(response, "Test", "test-model", 1, true)
            .await
            .expect("a complete chunked body should read cleanly");
        assert_eq!(String::from_utf8(body).unwrap(), "hello");
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn read_response_body_accepts_completed_sse_before_truncated_chunk_terminator() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            let body = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-complete\",\"output\":[]}}\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{}\r\n",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let response = reqwest::Client::new()
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("request headers should arrive");
        let (body, _) = read_response_body_bytes(response, "Test", "test-model", 1, true)
            .await
            .expect("a semantically complete SSE body must not be retried");
        let parsed = parse_sse_responses_body(std::str::from_utf8(&body).unwrap())
            .expect("the buffered SSE body should be complete");
        assert_eq!(parsed["id"], "resp-complete");
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn read_response_body_bytes_rejects_body_above_limit() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            let response =
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nhello";
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });

        let response = reqwest::Client::new()
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("request headers should arrive");
        let err = read_response_body_bytes_with_limit(response, "Test", "test-model", 1, true, 4)
            .await
            .expect_err("the configured body limit must be enforced");

        assert!(matches!(err, ResponseBodyReadError::TooLarge { limit: 4 }));
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn call_provider_api_refreshes_headers_after_unauthorized() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;
        use std::sync::{Arc, Mutex};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request_headers(&mut stream).to_ascii_lowercase();
                assert!(request.contains("authorization: bearer token-a"));
                assert!(!request.contains(CODEX_TURN_STATE_HEADER));
                let response = "HTTP/1.1 401 Unauthorized\r\nx-codex-turn-state: sticky-1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
            {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request_headers(&mut stream).to_ascii_lowercase();
                assert!(request.contains("authorization: bearer token-b"));
                assert!(request.contains("x-codex-turn-state: sticky-1"));
                let body = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-auth\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"authenticated\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nx-request-id: req-auth\r\nx-models-etag: models-v2\r\nx-codex-primary-used-percent: 12.5\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let details = ResponsesRequestDetails {
            provider: ThirdPartyProvider::OpenAICodex,
            display_name: "OpenAI Codex",
            url: format!("http://{addr}/responses"),
            headers: Vec::new(),
            session_id: "test-session".to_string(),
            codex_account_id: None,
            payload: json!({ "model": "gpt-5.5", "stream": true }),
            streaming_sse: true,
            request_timeout_secs: 30,
        };
        let tokens = Arc::new(Mutex::new(std::collections::VecDeque::from([
            "token-a", "token-b",
        ])));
        let resolve_tokens = Arc::clone(&tokens);
        let refresh_calls = Arc::new(AtomicUsize::new(0));
        let refresh_counter = Arc::clone(&refresh_calls);
        let mut turn_state = CodexTurnState::default();

        let result = call_provider_api_with_auth_hooks(
            &details,
            None,
            "test:auth-refresh",
            &mut turn_state,
            move || {
                let token = resolve_tokens
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("one token per attempt");
                std::future::ready(Ok(vec![(
                    "Authorization".to_string(),
                    format!("Bearer {token}"),
                )]))
            },
            move || {
                refresh_counter.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Ok(()))
            },
        )
        .await
        .expect("the second attempt should use refreshed headers");

        assert_eq!(refresh_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            extract_response_text(&extract_response_output_items(&result.response)),
            "authenticated"
        );
        assert_eq!(result.metadata.request_id.as_deref(), Some("req-auth"));
        assert_eq!(result.metadata.models_etag.as_deref(), Some("models-v2"));
        assert_eq!(
            result
                .metadata
                .rate_limit_headers
                .get("x-codex-primary-used-percent")
                .map(String::as_str),
            Some("12.5")
        );
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn call_provider_api_retries_clean_eof_and_replays_turn_state() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request_headers(&mut stream).to_ascii_lowercase();
                assert!(!request.contains(CODEX_TURN_STATE_HEADER));
                let body = "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"partial\"}]}}\n\n";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nx-codex-turn-state: sticky-eof\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
            {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request_headers(&mut stream).to_ascii_lowercase();
                assert!(request.contains("x-codex-turn-state: sticky-eof"));
                let body = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-retry\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"complete\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let details = ResponsesRequestDetails {
            provider: ThirdPartyProvider::OpenAICodex,
            display_name: "OpenAI Codex",
            url: format!("http://{addr}/responses"),
            headers: Vec::new(),
            session_id: "test-session".to_string(),
            codex_account_id: None,
            payload: json!({ "model": "gpt-5.5", "stream": true }),
            streaming_sse: true,
            request_timeout_secs: 30,
        };
        let mut turn_state = CodexTurnState::default();
        let result = call_provider_api_with_auth_hooks(
            &details,
            None,
            "test:semantic-retry",
            &mut turn_state,
            || {
                std::future::ready(Ok(vec![(
                    "Authorization".to_string(),
                    "Bearer test-token".to_string(),
                )]))
            },
            || std::future::ready(Ok(())),
        )
        .await
        .expect("a clean EOF without response.completed should retry");

        assert_eq!(
            extract_response_text(&extract_response_output_items(&result.response)),
            "complete"
        );
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn call_provider_api_retries_after_truncated_stream() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            // Attempt 1: a truncated chunked SSE stream (connection drops mid-stream).
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let partial = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n";
                let _ = stream.write_all(partial.as_bytes());
                let _ = stream.flush();
            }
            // Attempt 2: a complete SSE response the parser can consume.
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let body = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp1\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"recovered\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        let details = ResponsesRequestDetails {
            provider: ThirdPartyProvider::OpenAI,
            display_name: "OpenAI",
            url: format!("http://{addr}/responses"),
            headers: Vec::new(),
            session_id: "test-session".to_string(),
            codex_account_id: None,
            payload: json!({ "model": "gpt-5.5", "stream": true }),
            streaming_sse: true,
            request_timeout_secs: 30,
        };

        let mut turn_state = CodexTurnState::default();
        let result = call_provider_api(&details, None, "test:qa", &mut turn_state)
            .await
            .expect("the loop should recover from a single truncated stream by retrying");
        let output = extract_response_output_items(&result.response);
        assert_eq!(extract_response_text(&output), "recovered");
        handle.join().unwrap();
    }
}

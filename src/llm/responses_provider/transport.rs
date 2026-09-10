//! HTTP transport for the Responses API: request assembly, retries, and
//! response/metadata capture.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::audit::LlmAuditContext;
use crate::llm::openai_codex;
use crate::llm::transport::{
    call_with_retry, read_body_limited_or_partial, usage, BodyRead, LlmCall, ProviderError,
    RetryPolicy,
};
use crate::utils::http::{get_http_client, get_http_client_no_compression};
use crate::utils::text::truncate_for_log;

use super::codex_identity::CodexRequestIdentity;
use super::payload::{
    build_responses_payload, summarize_output_items, summarize_responses_payload,
};
use super::sse::{
    extract_response_output_items, interrupted_body_is_complete_sse, parse_sse_responses_body,
};

const RESPONSES_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(3, Duration::from_millis(900));
pub(super) const RESPONSES_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
pub(super) const CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
pub(super) const CODEX_RESPONSES_LITE_HEADER: &str = "x-openai-internal-codex-responses-lite";
pub(super) const MODELS_ETAG_HEADER: &str = "x-models-etag";
pub(super) const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone)]
pub(super) struct ResponsesRequestDetails {
    pub(super) provider: ThirdPartyProvider,
    pub(super) display_name: &'static str,
    pub(super) url: String,
    pub(super) headers: Vec<(String, String)>,
    pub(super) session_id: String,
    pub(super) codex_account_id: Option<String>,
    pub(super) payload: Value,
    pub(super) streaming_sse: bool,
    pub(super) request_timeout_secs: u64,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ResponsesResponseMetadata {
    pub(super) request_id: Option<String>,
    pub(super) models_etag: Option<String>,
    pub(super) codex_account_id: Option<String>,
    pub(super) rate_limit_headers: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(super) struct ResponsesApiResult {
    pub(super) response: Value,
    pub(super) metadata: ResponsesResponseMetadata,
}

#[derive(Debug, Default)]
pub(super) struct CodexTurnState {
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
    pub(super) fn capture(&mut self, headers: &reqwest::header::HeaderMap) {
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

    pub(super) fn apply(&self, headers: &mut Vec<(String, String)>) {
        if let Some(value) = self.value.as_ref() {
            headers.push((CODEX_TURN_STATE_HEADER.to_string(), value.clone()));
        }
    }
}

pub(super) fn summarize_error_body(body: &str) -> String {
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

pub(super) fn capture_response_metadata(
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

pub(super) fn responses_request_timeout_secs(provider: ThirdPartyProvider) -> u64 {
    match provider {
        ThirdPartyProvider::OpenAI => CONFIG.openai.request_timeout_secs,
        ThirdPartyProvider::OpenAICodex => CONFIG.codex.request_timeout_secs,
        ThirdPartyProvider::OpenRouter
        | ThirdPartyProvider::Nvidia
        | ThirdPartyProvider::Ollama => 60,
    }
}

pub(super) fn responses_base_url(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/responses") {
        normalized.to_string()
    } else {
        format!("{normalized}/responses")
    }
}

pub(super) fn add_codex_responses_lite_header(
    headers: &mut Vec<(String, String)>,
    use_responses_lite: bool,
) {
    if use_responses_lite {
        headers.push((CODEX_RESPONSES_LITE_HEADER.to_string(), "true".to_string()));
    }
}

pub(super) fn build_request_details(
    model_config: &ThirdPartyModelConfig,
    instructions: &str,
    input_items: Vec<Value>,
    tools: Option<Vec<Value>>,
    session_id: &str,
    identity: Option<&CodexRequestIdentity>,
) -> Result<ResponsesRequestDetails> {
    let (display_name, url, mut headers, streaming_sse) = match model_config.provider {
        ThirdPartyProvider::OpenAI => (
            "OpenAI",
            responses_base_url(&CONFIG.openai.base_url),
            vec![
                (
                    "Authorization".to_string(),
                    format!("Bearer {}", CONFIG.openai.api_key),
                ),
                (
                    "User-Agent".to_string(),
                    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                ),
            ],
            false,
        ),
        ThirdPartyProvider::OpenAICodex => {
            if identity.is_none() {
                return Err(anyhow!("Codex requests need a resolved request identity"));
            }
            (
                "OpenAI Codex",
                openai_codex::codex_response_url(),
                Vec::new(),
                true,
            )
        }
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

    let (payload, use_responses_lite) = build_responses_payload(
        model_config,
        instructions,
        input_items,
        tools,
        session_id,
        identity,
        streaming_sse,
    );
    add_codex_responses_lite_header(&mut headers, use_responses_lite);

    Ok(ResponsesRequestDetails {
        provider: model_config.provider,
        display_name,
        url,
        headers,
        session_id: session_id.to_string(),
        codex_account_id: identity.map(|identity| identity.account_id.clone()),
        payload,
        streaming_sse,
        request_timeout_secs: responses_request_timeout_secs(model_config.provider),
    })
}

pub(super) fn merge_request_headers_for_attempt(
    mut attempt_headers: Vec<(String, String)>,
    request_headers: &[(String, String)],
) -> Vec<(String, String)> {
    attempt_headers.extend(request_headers.iter().cloned());
    attempt_headers
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
    Ok(merge_request_headers_for_attempt(
        openai_codex::codex_headers(&auth, Some(&details.session_id)),
        &details.headers,
    ))
}

pub(super) async fn call_provider_api(
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

pub(super) async fn call_provider_api_with_auth_hooks<
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
    let audit_metadata = json!({
        "request_summary": summarize_responses_payload(&details.payload),
        "streaming_sse": details.streaming_sse,
        "timeout_secs": details.request_timeout_secs
    });
    let call = LlmCall::begin(
        details.provider.as_str(),
        model,
        operation,
        audit_context,
        Some(&audit_metadata),
    )
    .with_label(details.display_name);
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
    let timeout = Duration::from_secs(details.request_timeout_secs);
    let is_codex = details.provider == ThirdPartyProvider::OpenAICodex;
    // Only Codex credentials can be refreshed; for OpenAI a 401 is final.
    let policy = RetryPolicy {
        refresh_auth_on_unauthorized: is_codex,
        ..RESPONSES_RETRY_POLICY
    };
    let display_name = details.display_name;
    let streaming_sse = details.streaming_sse;
    let turn_state = parking_lot::Mutex::new(turn_state);
    let last_metadata = parking_lot::Mutex::new(None::<ResponsesResponseMetadata>);

    let result = call_with_retry(
        &call,
        &policy,
        |attempt| {
            // Refresh first (when the previous attempt was rejected with 401),
            // then resolve the per-attempt headers so they see the new tokens.
            let refresh = if attempt.previous_unauthorized {
                Some(refresh_auth())
            } else {
                None
            };
            let headers = resolve_headers();
            let turn_state = &turn_state;
            async move {
                if let Some(refresh) = refresh {
                    refresh
                        .await
                        .map_err(|err| ProviderError::rejected(err.to_string()))?;
                }
                let mut attempt_headers = headers
                    .await
                    .map_err(|err| ProviderError::rejected(err.to_string()))?;
                if is_codex {
                    turn_state.lock().apply(&mut attempt_headers);
                }
                let mut request = client.post(&details.url).timeout(timeout);
                for (name, value) in &attempt_headers {
                    request = request.header(name, value);
                }
                if streaming_sse {
                    request = request.header(reqwest::header::ACCEPT_ENCODING, "identity");
                }
                debug!(
                    "{} request timeout configured: model={}, timeout_secs={}, streaming_sse={}, attempt={}/{}",
                    display_name,
                    model,
                    details.request_timeout_secs,
                    streaming_sse,
                    attempt.number,
                    attempt.max_attempts
                );
                Ok(request.json(&details.payload))
            }
        },
        |response| {
            // Every response (including a 401) may carry the sticky turn state
            // and rate-limit metadata.
            if is_codex {
                turn_state.lock().capture(response.headers());
            }
            *last_metadata.lock() = Some(capture_response_metadata(
                response.headers(),
                details.codex_account_id.as_deref(),
            ));
        },
        |response| async move {
            let header_summary = summarize_response_headers(response.headers());
            debug!(
                "{} response headers for model={}: [{}]",
                display_name, model, header_summary
            );
            let body_bytes =
                match read_body_limited_or_partial(response, display_name, RESPONSES_MAX_BODY_BYTES)
                    .await?
                {
                    BodyRead::Complete(bytes) => bytes,
                    BodyRead::Interrupted { partial, error } => {
                        if streaming_sse && interrupted_body_is_complete_sse(&partial) {
                            warn!(
                                "{} response framing ended after a valid response.completed event: model={}, headers=[{}], bytes={}, error={}",
                                display_name,
                                model,
                                header_summary,
                                partial.len(),
                                error
                            );
                            partial
                        } else {
                            // An intermediary closing an idle chunked connection
                            // mid-reasoning is transient. Re-issuing is safe: the
                            // payload is unchanged and `store=false` left no
                            // server-side state behind.
                            return Err(error);
                        }
                    }
                };
            let body = String::from_utf8(body_bytes).map_err(|err| {
                ProviderError::decode(
                    display_name,
                    format!(
                        "response body was not valid UTF-8 ({} bytes, headers=[{header_summary}])",
                        err.as_bytes().len()
                    ),
                    streaming_sse,
                )
            })?;
            if streaming_sse {
                parse_sse_responses_body(&body).map_err(|err| {
                    ProviderError::decode(
                        display_name,
                        format!("SSE response rejected (headers=[{header_summary}]): {err}"),
                        err.is_retryable(),
                    )
                })
            } else {
                serde_json::from_str::<Value>(&body).map_err(|err| {
                    ProviderError::decode(
                        display_name,
                        format!(
                            "response JSON parse failed ({} bytes, headers=[{header_summary}]): {err}",
                            body.len()
                        ),
                        false,
                    )
                })
            }
        },
        usage::from_responses,
    )
    .await;

    let value = result.map_err(|err| responses_provider_error(display_name, err))?;
    // Only a successful response carries model metadata worth acting on.
    let metadata = last_metadata.into_inner().unwrap_or_default();
    observe_response_metadata(details.provider, &metadata);
    let output_items = extract_response_output_items(&value);
    info!(
        "{} responses request completed: model={}, output_items={}, output_summary=[{}]",
        display_name,
        model,
        output_items.len(),
        summarize_output_items(&output_items)
    );
    Ok(ResponsesApiResult {
        response: value,
        metadata,
    })
}

/// Keep remote error text out of user-facing messages: HTTP failures are
/// summarised structurally and decode failures carry only our own wording.
fn responses_provider_error(display_name: &str, err: ProviderError) -> anyhow::Error {
    match err {
        ProviderError::Http {
            status,
            body_snippet,
            ..
        } => anyhow!(
            "{} request failed with status {}: {}",
            display_name,
            status,
            summarize_error_body(&body_snippet)
        ),
        ProviderError::Decode { detail, .. } => anyhow!("{detail}"),
        other => anyhow!("{other}"),
    }
}

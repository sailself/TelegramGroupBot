use std::collections::BTreeMap;
use std::future::Future;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::audit::LlmAuditContext;
use crate::llm::openai_codex;
use crate::llm::runtime_models::ResolvedExplicitCodexModel;
use crate::llm::tool_loop::{
    clamp_request_timeout_secs, run_tool_loop, BoxFuture, ModelTurn, ToolCall, ToolProtocol,
    TurnDeadline,
};
use crate::llm::tool_prompts::TOOL_LIMIT_SYSTEM_PROMPT;
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::transport::{
    call_with_retry, read_body_limited_or_partial, usage, BodyRead, LlmCall, ProviderError,
    RetryPolicy,
};
use crate::utils::http::{get_http_client, get_http_client_no_compression};
use crate::utils::text::truncate_for_log;

mod codex_identity;
mod payload;
mod sse;

pub(crate) use codex_identity::{effective_reasoning_effort, CodexRequestIdentity};
use payload::{
    build_native_codex_web_search_tool_from_record, build_responses_payload,
    build_responses_system_prompt, build_responses_user_input, debug_model_label,
    generate_session_id, summarize_output_items, summarize_responses_payload,
};
use sse::{
    extract_response_output_items, extract_response_text, interrupted_body_is_complete_sse,
    parse_sse_responses_body,
};

const RESPONSES_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(3, Duration::from_millis(900));
const RESPONSES_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
const CODEX_RESPONSES_LITE_HEADER: &str = "x-openai-internal-codex-responses-lite";
const MODELS_ETAG_HEADER: &str = "x-models-etag";
const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone)]
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

#[derive(Debug, Clone)]
struct ResponsesToolCall {
    call_id: String,
    name: String,
    arguments: String,
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

fn responses_request_timeout_secs(provider: ThirdPartyProvider) -> u64 {
    match provider {
        ThirdPartyProvider::OpenAI => CONFIG.openai_request_timeout_secs,
        ThirdPartyProvider::OpenAICodex => CONFIG.openai_codex_request_timeout_secs,
        ThirdPartyProvider::OpenRouter
        | ThirdPartyProvider::Nvidia
        | ThirdPartyProvider::Ollama => 60,
    }
}

fn responses_base_url(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/responses") {
        normalized.to_string()
    } else {
        format!("{normalized}/responses")
    }
}

fn add_codex_responses_lite_header(headers: &mut Vec<(String, String)>, use_responses_lite: bool) {
    if use_responses_lite {
        headers.push((CODEX_RESPONSES_LITE_HEADER.to_string(), "true".to_string()));
    }
}

fn build_request_details(
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

fn merge_request_headers_for_attempt(
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

/// OpenAI Responses half of the shared tool loop.
struct ResponsesProtocol<'a> {
    model_config: &'a ThirdPartyModelConfig,
    instructions: String,
    session_id: String,
    turn_state: CodexTurnState,
    native_codex_web_search_tool: Option<Value>,
    audit_context: Option<&'a LlmAuditContext>,
    operation: &'a str,
    identity: Option<CodexRequestIdentity>,
}

impl ToolProtocol for ResponsesProtocol<'_> {
    type Item = Value;

    fn tool_declarations(&self, runtime: &ToolRuntime) -> Vec<Value> {
        // The runtime already withholds the function-call web_search when the
        // model searches natively (see ToolRuntime::use_native_web_search).
        let mut tools = runtime.build_responses_tools();
        if let Some(native_tool) = &self.native_codex_web_search_tool {
            tools.push(native_tool.clone());
        }
        tools
    }

    fn complete<'a>(
        &'a mut self,
        transcript: &'a [Value],
        tools: Option<&'a [Value]>,
        request_timeout: Duration,
    ) -> BoxFuture<'a, Result<ModelTurn<Value>>> {
        Box::pin(async move {
            let mut details = build_request_details(
                self.model_config,
                &self.instructions,
                transcript.to_vec(),
                tools.map(<[Value]>::to_vec),
                &self.session_id,
                self.identity.as_ref(),
            )?;
            details.request_timeout_secs =
                clamp_request_timeout_secs(details.request_timeout_secs, request_timeout);
            let ResponsesApiResult {
                response,
                metadata: _,
            } = call_provider_api(
                &details,
                self.audit_context,
                self.operation,
                &mut self.turn_state,
            )
            .await?;
            let output_items = extract_response_output_items(&response);
            let tool_calls = extract_response_tool_calls(&output_items)
                .into_iter()
                .map(|call| ToolCall::from_argument_text(call.call_id, call.name, &call.arguments))
                .collect();
            Ok(ModelTurn {
                text: extract_response_text(&output_items),
                tool_calls,
                transcript: output_items,
            })
        })
    }

    fn tool_results(&self, results: Vec<(ToolCall, String)>) -> Vec<Value> {
        results
            .into_iter()
            .map(|(call, output)| {
                json!({
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": output,
                })
            })
            .collect()
    }

    fn begin_final_pass(&mut self) {
        self.instructions = format!("{}\n\n{TOOL_LIMIT_SYSTEM_PROMPT}", self.instructions);
    }
}

#[allow(clippy::too_many_arguments)]
async fn responses_completion_with_tool_runtime(
    instructions: &str,
    input_items: Vec<Value>,
    model_config: &ThirdPartyModelConfig,
    runtime: &mut ToolRuntime,
    native_codex_web_search_tool: Option<Value>,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
    identity: Option<CodexRequestIdentity>,
) -> Result<String> {
    let per_request = Duration::from_secs(responses_request_timeout_secs(model_config.provider));
    let deadline = TurnDeadline::for_runtime(per_request, runtime);
    let mut protocol = ResponsesProtocol {
        model_config,
        instructions: instructions.to_string(),
        session_id: generate_session_id(),
        turn_state: CodexTurnState::default(),
        native_codex_web_search_tool,
        audit_context,
        operation,
        identity,
    };
    debug!(
        "Responses runtime tool loop starting: model={}, session_id={}, native_codex_web_search={}",
        debug_model_label(model_config),
        protocol.session_id,
        protocol.native_codex_web_search_tool.is_some()
    );
    run_tool_loop(&mut protocol, runtime, input_items, &deadline).await
}

/// Answer with an OpenAI Responses provider. `tools` runs the shared tool
/// loop over that runtime's budget, letting Codex use its native web search
/// where the profile allows; `None` is a single request without tools.
#[allow(clippy::too_many_arguments)]
pub async fn call_responses_provider(
    system_prompt: &str,
    user_content: &str,
    model_config: &ThirdPartyModelConfig,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    tools: Option<&mut ToolRuntime>,
    audit_context: Option<&LlmAuditContext>,
    reasoning_override: Option<&str>,
    explicit_codex: Option<&ResolvedExplicitCodexModel>,
    codex_prompt_style: crate::llm::CodexPromptStyle,
) -> Result<String> {
    crate::llm::runtime_models::ensure_selected_codex_model_metadata_current(model_config).await?;
    let identity = CodexRequestIdentity::resolve(
        model_config,
        explicit_codex.map(|explicit| &explicit.record),
        reasoning_override,
    )?;
    let model_label = debug_model_label(model_config);
    let input_items = build_responses_user_input(user_content, image_data_list);
    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);

    let Some(runtime) = tools else {
        debug!(
            "Responses provider selected: provider={}, model={}, response_title={}, tools=false, image_count={}",
            model_config.provider.as_str(),
            model_label,
            response_title,
            image_data_list.len()
        );
        let instructions =
            build_responses_system_prompt(system_prompt, model_config, codex_prompt_style, None);
        let session_id = generate_session_id();
        let mut turn_state = CodexTurnState::default();
        let details = build_request_details(
            model_config,
            &instructions,
            input_items,
            None,
            &session_id,
            identity.as_ref(),
        )?;
        let ResponsesApiResult {
            response,
            metadata: _,
        } = call_provider_api(&details, audit_context, &operation, &mut turn_state).await?;
        return Ok(extract_response_text(&extract_response_output_items(
            &response,
        )));
    };

    let native_codex_web_search_tool = if runtime.allows_native_web_search() {
        identity
            .as_ref()
            .and_then(|identity| identity.record.as_ref())
            .and_then(|record| build_native_codex_web_search_tool_from_record(model_config, record))
    } else {
        None
    };
    if native_codex_web_search_tool.is_some() {
        // Decided before the guidance is rendered, so the prompt describes
        // web_search while the function-call variant stays undeclared.
        runtime.use_native_web_search();
    }
    let runtime_guidance = runtime.tool_limit_guidance();
    let instructions = build_responses_system_prompt(
        system_prompt,
        model_config,
        codex_prompt_style,
        Some(&runtime_guidance),
    );
    debug!(
        "Responses provider selected: provider={}, model={}, response_title={}, tools=true, native_codex_web_search={}, image_count={}",
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
        native_codex_web_search_tool,
        audit_context,
        &operation,
        identity,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ThirdPartyProvider;
    use crate::llm::runtime_models::CodexSelectedModelRecord;
    use sse::SseParseError;
    use std::sync::atomic::Ordering;

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

    /// Identity for the fixture account, binding `record` (if any) to it.
    fn test_identity(
        config: &ThirdPartyModelConfig,
        record: Option<&CodexSelectedModelRecord>,
        reasoning_override: Option<&str>,
    ) -> CodexRequestIdentity {
        let record = record.cloned().map(|mut record| {
            record.account_id = Some("acct-1".to_string());
            record
        });
        CodexRequestIdentity::resolve_with(
            config,
            None,
            record.as_ref(),
            "acct-1",
            reasoning_override,
        )
        .expect("the test identity resolves")
    }

    fn codex_record(
        slug: &str,
        supported: &[&str],
        selected: Option<&str>,
        use_responses_lite: bool,
    ) -> CodexSelectedModelRecord {
        CodexSelectedModelRecord {
            metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
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
            use_responses_lite,
            fetched_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn responses_lite_payload_uses_developer_input_contract() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-luna");
        let record = codex_record("gpt-5.6-luna", &["medium", "max"], Some("max"), true);
        let tools = vec![json!({
            "type": "function",
            "name": "web_search",
            "parameters": {"type": "object"}
        })];

        let (payload, use_lite) = build_responses_payload(
            &config,
            "System instructions",
            vec![json!({"type": "message", "role": "user", "content": []})],
            Some(tools.clone()),
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        assert!(use_lite);
        assert!(payload.get("instructions").is_none());
        assert!(payload.get("tools").is_none());
        assert_eq!(payload["parallel_tool_calls"], false);
        assert_eq!(payload["reasoning"]["effort"], "max");
        assert_eq!(payload["reasoning"]["context"], "all_turns");
        assert_eq!(payload["input"][0]["type"], "additional_tools");
        assert_eq!(payload["input"][0]["tools"], Value::Array(tools));
        assert_eq!(payload["input"][1]["role"], "developer");
        assert_eq!(payload["input"][2]["role"], "user");
    }

    #[test]
    fn responses_payloads_apply_supported_internal_reasoning_override() {
        for use_lite in [false, true] {
            let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-luna");
            let record = codex_record(
                "gpt-5.6-luna",
                &["low", "medium", "max"],
                Some("max"),
                use_lite,
            );

            let (payload, actual_use_lite) = build_responses_payload(
                &config,
                "System instructions",
                vec![json!({"type": "message", "role": "user", "content": []})],
                None,
                "session-internal",
                Some(&test_identity(&config, Some(&record), Some("low"))),
                false,
            );

            assert_eq!(actual_use_lite, use_lite);
            assert_eq!(payload["reasoning"]["effort"], "low");
        }
    }

    #[test]
    fn explicit_codex_metadata_controls_responses_lite() {
        let selected_luna_config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-luna");
        let selected_luna_record = codex_record("gpt-5.6-luna", &[], None, false);
        let mut explicit_terra_config =
            model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-terra");
        explicit_terra_config.id = "openai-codex:gpt-5.6-terra".to_string();
        let explicit_terra_record = codex_record("gpt-5.6-terra", &[], None, true);

        let (_, selected_actual_use_lite) = build_responses_payload(
            &selected_luna_config,
            "",
            vec![],
            None,
            "session-selected",
            Some(&test_identity(
                &selected_luna_config,
                Some(&selected_luna_record),
                None,
            )),
            true,
        );
        let (_, explicit_actual_use_lite) = build_responses_payload(
            &explicit_terra_config,
            "",
            vec![],
            None,
            "session-explicit",
            Some(&test_identity(
                &explicit_terra_config,
                Some(&explicit_terra_record),
                None,
            )),
            true,
        );

        assert!(!selected_actual_use_lite);
        assert!(explicit_actual_use_lite);
    }

    #[test]
    fn request_identity_rejects_fresh_account_mismatch_from_validated_metadata() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-terra");
        let mut record = codex_record("gpt-5.6-terra", &[], None, true);
        record.account_id = Some("acct-1".to_string());

        assert!(
            CodexRequestIdentity::resolve_with(&config, None, Some(&record), "acct-2", None)
                .is_err()
        );
        assert!(
            CodexRequestIdentity::resolve_with(&config, None, Some(&record), "  ", None).is_err(),
            "an empty account id never resolves"
        );
        assert_eq!(
            CodexRequestIdentity::resolve(
                &model_config(ThirdPartyProvider::OpenAI, "gpt-5.4"),
                None,
                None
            )
            .expect("public OpenAI should not require Codex metadata"),
            None
        );
    }

    #[test]
    fn an_explicit_quick_identity_is_resolved_once_and_reused_by_every_iteration() {
        let mut config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-terra");
        config.id = "openai-codex:gpt-5.6-terra".to_string();
        let mut record = codex_record("gpt-5.6-terra", &["medium"], None, true);
        record.account_id = Some("acct-1".to_string());
        record.default_reasoning_level = Some("medium".to_string());

        let identity =
            CodexRequestIdentity::resolve_with(&config, Some(&record), None, "acct-1", Some("low"))
                .expect("the explicit record resolves for its account");
        assert_eq!(identity.account_id, "acct-1");
        assert_eq!(
            identity.reasoning_effort.as_deref(),
            Some("medium"),
            "an unsupported override falls back to the catalog level"
        );
        assert!(identity.use_responses_lite);

        for session_id in ["iteration-1", "iteration-2"] {
            let details = build_request_details(
                &config,
                "instructions",
                vec![json!({"type": "message", "role": "user", "content": []})],
                Some(vec![json!({"type": "function", "name": "web_search"})]),
                session_id,
                Some(&identity),
            )
            .expect("details build from the identity alone");

            assert_eq!(details.codex_account_id.as_deref(), Some("acct-1"));
            assert_eq!(details.payload["model"], "gpt-5.6-terra");
            assert_eq!(details.payload["reasoning"]["effort"], "medium");
            assert!(details
                .headers
                .iter()
                .any(|(name, value)| { name == CODEX_RESPONSES_LITE_HEADER && value == "true" }));
        }

        assert!(
            CodexRequestIdentity::resolve_with(&config, Some(&record), None, "acct-2", Some("low"))
                .is_err(),
            "another account cannot use the record"
        );
        let mut other_model = config.clone();
        other_model.model = "gpt-5.6-luna".to_string();
        assert!(
            CodexRequestIdentity::resolve_with(&other_model, Some(&record), None, "acct-1", None)
                .is_err(),
            "the record must name the requested model"
        );
        assert!(
            build_request_details(&config, "", vec![], None, "no-identity", None).is_err(),
            "Codex requests need an identity"
        );
    }

    #[test]
    fn an_empty_quick_override_uses_the_catalog_level() {
        let mut config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-terra");
        config.id = "openai-codex:gpt-5.6-terra".to_string();
        let mut record = codex_record("gpt-5.6-terra", &["medium"], None, true);
        record.account_id = Some("acct-1".to_string());
        record.default_reasoning_level = Some("medium".to_string());
        let identity =
            CodexRequestIdentity::resolve_with(&config, Some(&record), None, "acct-1", Some("  "))
                .expect("the catalog default resolves");

        let details =
            build_request_details(&config, "", vec![], None, "empty-override", Some(&identity))
                .expect("details build");

        assert_eq!(details.payload["reasoning"]["effort"], "medium");
    }

    #[test]
    fn synthesized_and_configured_foreign_codex_requests_use_current_account_without_catalog_metadata(
    ) {
        let mut synthesized_agent_step =
            model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.4-mini");
        synthesized_agent_step.tools = false;
        let configured_q_model =
            model_config(ThirdPartyProvider::OpenAICodex, "configured-q-model");

        for config in [&synthesized_agent_step, &configured_q_model] {
            let identity =
                CodexRequestIdentity::resolve_with(config, None, None, "acct-1", Some("low"))
                    .expect("foreign Codex requests use the active account");
            assert_eq!(identity.account_id, "acct-1");
            assert!(identity.record.is_none());
            assert!(!identity.use_responses_lite);

            let (payload, use_lite) = build_responses_payload(
                config,
                "",
                vec![],
                None,
                "foreign-session",
                Some(&identity),
                true,
            );
            assert_eq!(payload["model"], config.model);
            assert_eq!(payload["reasoning"]["effort"], "low");
            assert!(!use_lite);
        }
    }

    #[test]
    fn responses_lite_empty_tools_and_instructions_keep_one_empty_prefix() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "lite-model");
        let record = codex_record("lite-model", &[], None, true);

        let (payload, use_lite) = build_responses_payload(
            &config,
            "",
            vec![json!({"type": "message", "role": "user", "content": []})],
            Some(Vec::new()),
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        let input = payload["input"].as_array().expect("input array");
        assert!(use_lite);
        assert_eq!(
            input
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"))
                .count(),
            1
        );
        assert_eq!(input[0]["tools"], json!([]));
        assert!(!input.iter().any(|item| {
            item.get("type").and_then(Value::as_str) == Some("message")
                && item.get("role").and_then(Value::as_str) == Some("developer")
        }));
    }

    #[test]
    fn responses_lite_requires_codex_provider_and_matching_slug() {
        let mut record = codex_record("selected-model", &[], None, true);
        record.account_id = Some("acct-1".to_string());
        let mismatched = model_config(ThirdPartyProvider::OpenAICodex, "other-model");
        let public = model_config(ThirdPartyProvider::OpenAI, "selected-model");

        assert!(
            CodexRequestIdentity::resolve_with(&mismatched, None, Some(&record), "acct-1", None)
                .is_err(),
            "a record for another slug cannot shape this request"
        );
        assert_eq!(
            CodexRequestIdentity::resolve(&public, None, None).expect("non-Codex resolves"),
            None,
            "non-Codex providers carry no Codex identity"
        );
        let (_, use_lite) = build_responses_payload(&public, "", vec![], None, "s", None, false);
        assert!(!use_lite);
    }

    #[test]
    fn responses_lite_rebuild_from_iteration_history_does_not_accumulate_prefixes() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "lite-model");
        let record = codex_record("lite-model", &[], None, true);
        let mut history = vec![json!({"type": "message", "role": "user", "content": []})];

        let (first, _) = build_responses_payload(
            &config,
            "instructions",
            history.clone(),
            Some(Vec::new()),
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );
        history.push(json!({
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "result"
        }));
        let (second, _) = build_responses_payload(
            &config,
            "instructions",
            history,
            Some(Vec::new()),
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        for payload in [&first, &second] {
            assert_eq!(
                payload["input"]
                    .as_array()
                    .expect("input array")
                    .iter()
                    .filter(
                        |item| item.get("type").and_then(Value::as_str) == Some("additional_tools")
                    )
                    .count(),
                1
            );
        }
    }

    #[test]
    fn responses_lite_removes_detail_from_initial_input_images() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "lite-model");
        let record = codex_record("lite-model", &[], None, true);
        let input = build_responses_user_input("describe", &[vec![1, 2, 3]]);

        let (payload, use_lite) = build_responses_payload(
            &config,
            "instructions",
            input,
            None,
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        assert!(use_lite);
        assert_eq!(payload["input"][2]["content"][1]["type"], "input_image");
        assert!(payload["input"][2]["content"][1].get("detail").is_none());
    }

    #[test]
    fn responses_lite_removes_detail_from_nested_input_images() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "lite-model");
        let record = codex_record("lite-model", &[], None, true);
        let input = vec![json!({
            "type": "function_call_output",
            "call_id": "call-1",
            "output": {
                "structured": [{
                    "type": "input_image",
                    "detail": "high",
                    "image_url": "data:image/png;base64,AA=="
                }]
            }
        })];

        let (payload, _) = build_responses_payload(
            &config,
            "",
            input,
            None,
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        let image = &payload["input"][1]["output"]["structured"][0];
        assert_eq!(image["type"], "input_image");
        assert!(image.get("detail").is_none());
    }

    #[test]
    fn normal_responses_preserve_input_image_detail() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "normal-model");
        let record = codex_record("normal-model", &[], None, false);
        let input = build_responses_user_input("describe", &[vec![1, 2, 3]]);

        let (payload, use_lite) = build_responses_payload(
            &config,
            "instructions",
            input,
            None,
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        assert!(!use_lite);
        assert_eq!(payload["input"][0]["content"][1]["detail"], "auto");
    }

    #[test]
    fn normal_responses_payload_keeps_top_level_contract() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.5");
        let record = codex_record("gpt-5.5", &["medium"], Some("medium"), false);
        let tools = vec![json!({"type": "web_search"})];

        let (payload, use_lite) = build_responses_payload(
            &config,
            "System instructions",
            vec![json!({"type": "message", "role": "user", "content": []})],
            Some(tools.clone()),
            "session-1",
            Some(&test_identity(&config, Some(&record), None)),
            true,
        );

        assert!(!use_lite);
        assert_eq!(payload["instructions"], "System instructions");
        assert_eq!(payload["tools"], Value::Array(tools));
        assert_eq!(payload["parallel_tool_calls"], true);
        assert!(payload["reasoning"].get("context").is_none());
        assert_eq!(payload["input"][0]["role"], "user");
    }

    #[test]
    fn responses_lite_header_is_added_only_for_lite_requests() {
        let mut lite_headers = Vec::new();
        add_codex_responses_lite_header(&mut lite_headers, true);
        assert_eq!(
            lite_headers,
            vec![(CODEX_RESPONSES_LITE_HEADER.to_string(), "true".to_string())]
        );

        let mut normal_headers = Vec::new();
        add_codex_responses_lite_header(&mut normal_headers, false);
        assert!(normal_headers.is_empty());
    }

    #[test]
    fn responses_lite_header_is_carried_by_every_retry_header_assembly() {
        let mut request_headers = Vec::new();
        add_codex_responses_lite_header(&mut request_headers, true);

        for token in ["token-a", "token-b"] {
            let headers = merge_request_headers_for_attempt(
                vec![("Authorization".to_string(), format!("Bearer {token}"))],
                &request_headers,
            );
            assert!(headers
                .iter()
                .any(|(name, value)| { name == CODEX_RESPONSES_LITE_HEADER && value == "true" }));
        }
    }

    #[test]
    fn responses_lite_summary_reads_redacted_tool_names_from_leading_input_item() {
        let payload = json!({
            "model": "lite-model",
            "input": [{
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "function",
                        "name": "web_search",
                        "description": "secret-description",
                        "parameters": {"secret-schema": true}
                    },
                    {"type": "computer"}
                ]
            }],
            "stream": true
        });

        let summary = summarize_responses_payload(&payload);

        assert!(summary.contains("tools=2, tool_names=[web_search,computer]"));
        assert!(!summary.contains("secret-description"));
        assert!(!summary.contains("secret-schema"));
    }

    #[test]
    fn normal_responses_summary_keeps_top_level_tool_names() {
        let payload = json!({
            "model": "normal-model",
            "input": [],
            "tools": [{"type": "function", "name": "lookup"}],
            "stream": false
        });

        let summary = summarize_responses_payload(&payload);

        assert!(summary.contains("tools=1, tool_names=[lookup]"));
    }

    #[test]
    fn responses_lite_does_not_use_native_hosted_web_search() {
        let config = model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.6-luna");
        let mut record = codex_record("gpt-5.6-luna", &["medium"], Some("medium"), true);
        record.supports_search_tool = true;

        assert!(build_native_codex_web_search_tool_from_record(&config, &record).is_none());
    }

    #[tokio::test]
    async fn selected_metadata_guard_leaves_public_openai_requests_unchanged() {
        let config = model_config(ThirdPartyProvider::OpenAI, "public-model");

        crate::llm::runtime_models::ensure_selected_codex_model_metadata_current(&config)
            .await
            .expect("public OpenAI must not require Codex selected-model metadata");
    }

    #[test]
    fn reasoning_without_override_uses_selected_record_level() {
        let record = codex_record("gpt-5.5", &["low", "xhigh"], Some("xhigh"), false);
        assert_eq!(
            effective_reasoning_effort("gpt-5.5", Some(&record), None),
            Some("xhigh".to_string())
        );
        assert_eq!(effective_reasoning_effort("gpt-5.5", None, None), None);
    }

    #[test]
    fn reasoning_without_override_or_selection_uses_the_catalog_default() {
        let mut record = codex_record("gpt-5.5", &["medium"], None, false);
        record.default_reasoning_level = Some("Medium".to_string());
        assert_eq!(
            effective_reasoning_effort("gpt-5.5", Some(&record), None),
            Some("medium".to_string())
        );
    }

    #[test]
    fn reasoning_override_applies_when_supported() {
        let record = codex_record("gpt-5.5", &["low", "medium", "xhigh"], Some("xhigh"), false);
        assert_eq!(
            effective_reasoning_effort("gpt-5.5", Some(&record), Some("Low")),
            Some("low".to_string())
        );
    }

    #[test]
    fn unsupported_reasoning_override_falls_back_to_selected_level() {
        let record = codex_record("gpt-5.5", &["medium", "xhigh"], Some("xhigh"), false);
        assert_eq!(
            effective_reasoning_effort("gpt-5.5", Some(&record), Some("low")),
            Some("xhigh".to_string())
        );
    }

    #[test]
    fn unsupported_reasoning_override_falls_back_to_catalog_default_when_selection_is_empty() {
        let mut record = codex_record("gpt-5.5", &["medium"], None, false);
        record.default_reasoning_level = Some("medium".to_string());

        assert_eq!(
            effective_reasoning_effort("gpt-5.5", Some(&record), Some("low")),
            Some("medium".to_string())
        );
    }

    #[test]
    fn reasoning_override_passes_through_for_foreign_slugs() {
        let record = codex_record("gpt-5.5", &["medium"], Some("medium"), false);
        assert_eq!(
            effective_reasoning_effort("gpt-5.4-mini", Some(&record), Some("low")),
            Some("low".to_string())
        );
        assert_eq!(
            effective_reasoning_effort("gpt-5.4-mini", Some(&record), Some("  ")),
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
    fn codex_freeform_system_prompt_appends_compact_style_guidance_once() {
        let instructions = build_responses_system_prompt(
            "Base prompt",
            &model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.4"),
            crate::llm::CodexPromptStyle::FreeformAnswer,
            Some("Tool guidance"),
        );

        assert_eq!(
            instructions.matches("Keep the answer substantive").count(),
            1
        );
        assert_eq!(instructions.matches("Tool guidance").count(), 1);
    }

    #[test]
    fn codex_task_specific_system_prompt_skips_style_guidance() {
        let instructions = build_responses_system_prompt(
            "Base prompt",
            &model_config(ThirdPartyProvider::OpenAICodex, "gpt-5.4"),
            crate::llm::CodexPromptStyle::TaskSpecific,
            Some("Tool guidance"),
        );

        assert_eq!(instructions, "Base prompt\n\nTool guidance");
    }

    #[test]
    fn public_openai_system_prompt_skips_codex_style_guidance() {
        let instructions = build_responses_system_prompt(
            "Base prompt",
            &model_config(ThirdPartyProvider::OpenAI, "gpt-5.4"),
            crate::llm::CodexPromptStyle::FreeformAnswer,
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
        let read = read_body_limited_or_partial(response, "Test", RESPONSES_MAX_BODY_BYTES)
            .await
            .expect("only the size limit is a hard error");
        let BodyRead::Interrupted { partial, error } = read else {
            panic!("a truncated chunked body must surface as interrupted so the loop can retry");
        };
        assert_eq!(partial, b"hello");
        assert!(error.is_retryable());
        assert!(!interrupted_body_is_complete_sse(&partial));
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
        let read = read_body_limited_or_partial(response, "Test", RESPONSES_MAX_BODY_BYTES)
            .await
            .expect("a complete chunked body should read cleanly");
        let BodyRead::Complete(body) = read else {
            panic!("a complete chunked body must not be reported as interrupted");
        };
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
        let read = read_body_limited_or_partial(response, "Test", RESPONSES_MAX_BODY_BYTES)
            .await
            .expect("only the size limit is a hard error");
        let BodyRead::Interrupted { partial, .. } = read else {
            panic!("the missing chunk terminator must be reported as an interruption");
        };
        assert!(
            interrupted_body_is_complete_sse(&partial),
            "a semantically complete SSE body must be accepted instead of retried"
        );
        let parsed = parse_sse_responses_body(std::str::from_utf8(&partial).unwrap())
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
        let err = read_body_limited_or_partial(response, "Test", 4)
            .await
            .expect_err("the configured body limit must be enforced");

        assert!(matches!(err, ProviderError::BodyTooLarge { limit: 4, .. }));
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

use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::LazyLock;
use tracing::{debug, warn};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::audit::LlmAuditContext;
use crate::llm::media::{MediaFile, MediaKind};
use crate::llm::responses_provider::call_responses_provider;
use crate::llm::runtime_models::{
    is_runtime_provider_ready, runtime_model_config, ResolvedExplicitCodexModel,
};
use crate::llm::tool_loop::{
    clamp_request_timeout_secs, run_tool_loop, BoxFuture, ModelTurn, ToolCall, ToolProtocol,
    TurnDeadline,
};
use crate::llm::tool_prompts::TOOL_LIMIT_SYSTEM_PROMPT;
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::transport::{call_with_retry, read_json, usage, LlmCall, RetryPolicy};
use crate::llm::CodexPromptStyle;
use crate::utils::http::get_http_client;
use crate::utils::text::truncate_for_log;

const THIRD_PARTY_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(3, Duration::from_millis(900));
const OPENROUTER_REFERER: &str = "https://github.com/sailself/TelegramGroupHelperBot";
const OPENROUTER_TITLE: &str = "TelegramGroupHelperBot";

#[derive(Clone, Copy)]
pub struct ThirdPartyCallOptions<'a> {
    audit_context: Option<&'a LlmAuditContext>,
    reasoning_override: Option<&'a str>,
    /// An explicit Codex model (config plus catalog record) the caller already
    /// resolved, so the call does not depend on the runtime catalog for it.
    explicit_codex: Option<&'a ResolvedExplicitCodexModel>,
    codex_prompt_style: CodexPromptStyle,
}

impl<'a> ThirdPartyCallOptions<'a> {
    pub(crate) fn new(
        audit_context: Option<&'a LlmAuditContext>,
        codex_prompt_style: CodexPromptStyle,
    ) -> Self {
        Self {
            audit_context,
            reasoning_override: None,
            explicit_codex: None,
            codex_prompt_style,
        }
    }

    pub(crate) fn with_reasoning_override(mut self, reasoning_override: Option<&'a str>) -> Self {
        self.reasoning_override = reasoning_override;
        self
    }

    pub(crate) fn with_explicit_codex_model(
        mut self,
        explicit_codex: Option<&'a ResolvedExplicitCodexModel>,
    ) -> Self {
        self.explicit_codex = explicit_codex;
        self
    }
}

fn model_config_for_call(
    model_id: &str,
    explicit_codex: Option<&ResolvedExplicitCodexModel>,
) -> Result<ThirdPartyModelConfig> {
    if let Some(explicit) = explicit_codex {
        if explicit.config.id != model_id {
            return Err(anyhow!("The explicit Codex model changed"));
        }
        return Ok(explicit.config.clone());
    }

    CONFIG
        .get_third_party_model_config(model_id)
        .cloned()
        .or_else(|| runtime_model_config(model_id))
        .ok_or_else(|| anyhow!("Unknown third-party model '{}'", model_id))
}

#[derive(Debug, Clone)]
struct ProviderRuntimeConfig {
    provider: ThirdPartyProvider,
    display_name: &'static str,
    base_url: String,
    api_key: String,
    temperature: f32,
    top_p: f32,
    top_k: Option<i32>,
    request_timeout_secs: u64,
}

#[derive(Debug, Clone)]
struct ProviderRequestDetails {
    display_name: &'static str,
    url: String,
    headers: Vec<(String, String)>,
    payload: Value,
    request_timeout_secs: u64,
}

fn summarize_payload(payload: &Value) -> String {
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let message_count = payload
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|messages| messages.len())
        .unwrap_or(0);
    let tool_names = payload
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tool_choice = payload
        .get("tool_choice")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");

    format!(
        "model={}, messages={}, tools={}, tool_choice={}, tool_names=[{}]",
        model,
        message_count,
        tool_names.len(),
        tool_choice,
        tool_names.join(",")
    )
}

static HARMONY_TAG_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|.*?\|>").expect("valid harmony tag regex"));
// `(?s)` lets `.` span newlines: reasoning blocks are normally multi-line.
static THINK_BLOCK_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>(.*)").expect("valid think block regex"));

fn parse_gpt_content(content: &str) -> String {
    if let Some(last_pos) = content.rfind("<|message|>") {
        let analysis = &content[..last_pos];
        let final_text = &content[last_pos + "<|message|>".len()..];
        let final_clean = HARMONY_TAG_REGEX
            .replace_all(final_text, "")
            .trim()
            .to_string();
        if !final_clean.is_empty() {
            return final_clean;
        }
        let analysis_clean = HARMONY_TAG_REGEX
            .replace_all(analysis, "")
            .trim()
            .to_string();
        return analysis_clean;
    }
    content.to_string()
}

fn parse_qwen_content(content: &str) -> String {
    if let Some(caps) = THINK_BLOCK_REGEX.captures(content) {
        let final_text = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let final_text = final_text.trim();
        if !final_text.is_empty() {
            return final_text.to_string();
        }
        let analysis = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        return analysis.trim().to_string();
    }
    content.trim().to_string()
}

/// Remove inline reasoning markup that some models put in `content`: a
/// leading Qwen/DeepSeek/GLM-style `<think>...</think>` block, or GPT-OSS
/// harmony channel tokens. Detection is content-based rather than keyed on
/// provider or model name, because the same model families are served by
/// OpenRouter, NVIDIA and Ollama alike and the markers never appear unless a
/// model actually produced them.
fn strip_reasoning_markup(content: &str) -> String {
    if content.contains("<|message|>") {
        return parse_gpt_content(content);
    }
    // Only a block at the very start is reasoning; an answer that merely
    // mentions the tags must be left intact.
    if content.trim_start().starts_with("<think>") {
        return parse_qwen_content(content);
    }
    content.to_string()
}

fn parse_third_party_response(model_config: &ThirdPartyModelConfig, content: &str) -> String {
    let cleaned = strip_reasoning_markup(content);
    if cleaned != content {
        debug!(
            model = %model_config.id,
            provider = model_config.provider.as_str(),
            "stripped inline reasoning markup from response"
        );
    }
    cleaned
}

fn extract_reasoning_text(message: &Value) -> Option<String> {
    // `reasoning` (OpenRouter), `reasoning_content` (NVIDIA NIM / DeepSeek),
    // `thinking` (Ollama): whichever the provider used for the separate
    // reasoning stream.
    for key in ["reasoning", "reasoning_content", "thinking"] {
        if let Some(reasoning) = message.get(key).and_then(|v| v.as_str()) {
            let trimmed = reasoning.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    let details = message
        .get("reasoning_details")
        .and_then(|v| v.as_array())?;
    let mut parts = Vec::new();
    for detail in details {
        let text = detail.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let text = text.trim();
        if !text.is_empty() {
            parts.push(text.to_string());
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_message_content(message: &Value) -> String {
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if !content.is_empty() {
        return content;
    }

    extract_reasoning_text(message).unwrap_or_default()
}

fn image_data_list_from_media(media_files: &[MediaFile]) -> Vec<Vec<u8>> {
    media_files
        .iter()
        .filter(|file| file.kind == MediaKind::Image)
        .map(|file| file.bytes().to_vec())
        .collect()
}

fn build_message_content(user_content: &str, media_files: &[MediaFile]) -> Value {
    let supported_media = media_files
        .iter()
        .filter(|file| {
            matches!(
                file.kind,
                MediaKind::Image | MediaKind::Video | MediaKind::Audio
            )
        })
        .collect::<Vec<_>>();
    if supported_media.is_empty() {
        return Value::String(user_content.to_string());
    }

    let mut parts = Vec::new();
    parts.push(json!({
        "type": "text",
        "text": user_content
    }));

    for file in supported_media {
        let fallback_mime_type = match file.kind {
            MediaKind::Image => "image/png",
            MediaKind::Video => "video/mp4",
            MediaKind::Audio => "audio/mpeg",
            MediaKind::Document => continue,
        };
        let mime_type = crate::llm::media::detect_mime_type(file.bytes())
            .or_else(|| (!file.mime_type.trim().is_empty()).then(|| file.mime_type.clone()))
            .unwrap_or_else(|| fallback_mime_type.to_string());
        let encoded = general_purpose::STANDARD.encode(file.bytes());
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        match file.kind {
            MediaKind::Image => parts.push(json!({
                "type": "image_url",
                "image_url": { "url": data_url }
            })),
            MediaKind::Video => parts.push(json!({
                "type": "video_url",
                "video_url": { "url": data_url }
            })),
            MediaKind::Audio => parts.push(json!({
                "type": "audio_url",
                "audio_url": { "url": data_url }
            })),
            MediaKind::Document => {}
        }
    }

    Value::Array(parts)
}

fn provider_runtime_config(provider: ThirdPartyProvider) -> Result<ProviderRuntimeConfig> {
    let config = match provider {
        ThirdPartyProvider::OpenRouter => ProviderRuntimeConfig {
            provider,
            display_name: "OpenRouter",
            base_url: CONFIG.openrouter_base_url.clone(),
            api_key: CONFIG.openrouter_api_key.clone(),
            temperature: CONFIG.openrouter_temperature,
            top_p: CONFIG.openrouter_top_p,
            top_k: Some(CONFIG.openrouter_top_k),
            request_timeout_secs: CONFIG.openrouter_request_timeout_secs,
        },
        ThirdPartyProvider::Nvidia => ProviderRuntimeConfig {
            provider,
            display_name: "NVIDIA",
            base_url: CONFIG.nvidia_base_url.clone(),
            api_key: CONFIG.nvidia_api_key.clone(),
            temperature: CONFIG.nvidia_temperature,
            top_p: CONFIG.nvidia_top_p,
            top_k: None,
            request_timeout_secs: CONFIG.nvidia_request_timeout_secs,
        },
        ThirdPartyProvider::Ollama => ProviderRuntimeConfig {
            provider,
            display_name: "Ollama",
            base_url: CONFIG.ollama_base_url.clone(),
            api_key: CONFIG.ollama_api_key.clone(),
            temperature: CONFIG.ollama_temperature,
            top_p: CONFIG.ollama_top_p,
            top_k: None,
            request_timeout_secs: CONFIG.ollama_request_timeout_secs,
        },
        ThirdPartyProvider::OpenAI | ThirdPartyProvider::OpenAICodex => {
            return Err(anyhow!(
                "Responses providers are handled by the responses provider adapter"
            ));
        }
    };

    if !is_runtime_provider_ready(provider) {
        return Err(anyhow!(
            "{} is not enabled or its API key is missing",
            config.display_name
        ));
    }

    Ok(config)
}

/// Convert an f32 sampling parameter to JSON without f32->f64 widening noise
/// (0.4f32 would otherwise serialize as 0.4000000059604645, which at least one
/// OpenRouter upstream rejects with an opaque 400). Round-tripping through the
/// shortest decimal representation keeps the wire value equal to the
/// configured literal.
fn sampling_param(value: f32) -> Value {
    json!(value.to_string().parse::<f64>().unwrap_or(f64::from(value)))
}

fn build_request_details_for_runtime(
    model_config: &ThirdPartyModelConfig,
    runtime: &ProviderRuntimeConfig,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<&str>,
) -> ProviderRequestDetails {
    let mut headers = vec![(
        "Authorization".to_string(),
        format!("Bearer {}", runtime.api_key),
    )];
    if runtime.provider == ThirdPartyProvider::OpenRouter {
        headers.push(("HTTP-Referer".to_string(), OPENROUTER_REFERER.to_string()));
        headers.push(("X-Title".to_string(), OPENROUTER_TITLE.to_string()));
    }

    let mut payload = json!({
        "model": model_config.model,
        "messages": messages,
        "temperature": sampling_param(runtime.temperature),
        "top_p": sampling_param(runtime.top_p),
    });

    if let Some(top_k) = runtime.top_k {
        payload["top_k"] = json!(top_k);
    }

    if let Some(tools) = tools {
        payload["tools"] = Value::Array(tools);
        payload["tool_choice"] = Value::String(tool_choice.unwrap_or("auto").to_string());
    }

    ProviderRequestDetails {
        display_name: runtime.display_name,
        url: format!(
            "{}/chat/completions",
            runtime.base_url.trim_end_matches('/')
        ),
        headers,
        payload,
        request_timeout_secs: runtime.request_timeout_secs,
    }
}

fn build_request_details(
    model_config: &ThirdPartyModelConfig,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<&str>,
) -> Result<ProviderRequestDetails> {
    let runtime = provider_runtime_config(model_config.provider)?;
    Ok(build_request_details_for_runtime(
        model_config,
        &runtime,
        messages,
        tools,
        tool_choice,
    ))
}

async fn call_provider_api(
    details: &ProviderRequestDetails,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<Value> {
    debug!(
        "{} request: {}",
        details.display_name,
        summarize_payload(&details.payload)
    );
    let model = details
        .payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let metadata = json!({
        "request_summary": summarize_payload(&details.payload),
        "timeout_secs": details.request_timeout_secs
    });
    let call = LlmCall::begin(
        details.display_name,
        model,
        operation,
        audit_context,
        Some(&metadata),
    );
    let timeout = Duration::from_secs(details.request_timeout_secs);

    let value = call_with_retry(
        &call,
        &THIRD_PARTY_RETRY_POLICY,
        |attempt| async move {
            debug!(
                "{} request timeout configured: model={}, timeout_secs={}, attempt={}/{}",
                details.display_name,
                model,
                details.request_timeout_secs,
                attempt.number,
                attempt.max_attempts
            );
            let mut request = get_http_client().post(&details.url).timeout(timeout);
            for (name, value) in &details.headers {
                request = request.header(name, value);
            }
            Ok(request.json(&details.payload))
        },
        |_| {},
        |response| read_json::<Value>(response, details.display_name),
        usage::from_chat_completions,
    )
    .await?;

    debug!(
        "{} response received for model={}",
        details.display_name, model
    );
    Ok(value)
}

fn extract_response_message(response: &Value) -> Value {
    response
        .get("choices")
        .and_then(|v| v.get(0))
        .and_then(|v| v.get("message"))
        .cloned()
        .unwrap_or(Value::Null)
}

fn extract_tool_calls(message: &Value) -> Vec<Value> {
    message
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
}

/// Chat Completions half of the shared tool loop.
struct ChatCompletionsProtocol<'a> {
    model_config: &'a ThirdPartyModelConfig,
    audit_context: Option<&'a LlmAuditContext>,
    operation: &'a str,
}

impl ToolProtocol for ChatCompletionsProtocol<'_> {
    type Item = Value;

    fn tool_declarations(&self, runtime: &ToolRuntime) -> Vec<Value> {
        runtime.build_openai_function_tools()
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
                transcript.to_vec(),
                tools.map(<[Value]>::to_vec),
                tools.is_some().then_some("auto"),
            )?;
            details.request_timeout_secs =
                clamp_request_timeout_secs(details.request_timeout_secs, request_timeout);
            let response = call_provider_api(&details, self.audit_context, self.operation).await?;
            let message = extract_response_message(&response);
            let content = extract_message_content(&message);
            let tool_calls = extract_tool_calls(&message)
                .iter()
                .map(chat_tool_call)
                .collect::<Vec<_>>();
            if tool_calls.is_empty() && content.trim().is_empty() {
                warn!(
                    "{} response had empty content and no tool calls: {}",
                    details.display_name,
                    truncate_for_log(&response.to_string(), 2000)
                );
            }
            Ok(ModelTurn {
                text: parse_third_party_response(self.model_config, &content),
                tool_calls,
                transcript: vec![message],
            })
        })
    }

    fn tool_results(&self, results: Vec<(ToolCall, String)>) -> Vec<Value> {
        results
            .into_iter()
            .map(|(call, output)| {
                json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                })
            })
            .collect()
    }

    fn budget_exhausted_notice(&self) -> Option<Value> {
        Some(json!({
            "role": "system",
            "content": TOOL_LIMIT_SYSTEM_PROMPT,
        }))
    }
}

/// A Chat Completions `tool_calls[]` entry as a [`ToolCall`].
fn chat_tool_call(tool_call: &Value) -> ToolCall {
    let function = tool_call.get("function");
    ToolCall::from_argument_text(
        tool_call.get("id").and_then(Value::as_str).unwrap_or(""),
        function
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        function
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("{}"),
    )
}

async fn chat_completion_with_tool_runtime(
    messages: Vec<Value>,
    model_config: &ThirdPartyModelConfig,
    runtime: &mut ToolRuntime,
    audit_context: Option<&LlmAuditContext>,
    operation: &str,
) -> Result<String> {
    let per_request =
        Duration::from_secs(provider_runtime_config(model_config.provider)?.request_timeout_secs);
    let deadline = TurnDeadline::for_runtime(per_request, runtime);
    let mut protocol = ChatCompletionsProtocol {
        model_config,
        audit_context,
        operation,
    };
    run_tool_loop(&mut protocol, runtime, messages, &deadline).await
}

/// [`call_third_party`] with a tool runtime, for the `/qc`, `/s` and
/// quick-answer paths that manage their own runtime.
pub async fn call_third_party_with_tool_runtime(
    system_prompt: &str,
    user_content: &str,
    model_id: &str,
    response_title: &str,
    media_files: &[MediaFile],
    runtime: &mut ToolRuntime,
    options: ThirdPartyCallOptions<'_>,
) -> Result<String> {
    call_third_party(
        system_prompt,
        user_content,
        model_id,
        response_title,
        media_files,
        Some(runtime),
        options,
    )
    .await
}

/// Answer with a third-party model. `tools` runs the shared tool loop over
/// that runtime's budget; `None` is a single request without tools.
pub async fn call_third_party(
    system_prompt: &str,
    user_content: &str,
    model_id: &str,
    response_title: &str,
    media_files: &[MediaFile],
    tools: Option<&mut ToolRuntime>,
    options: ThirdPartyCallOptions<'_>,
) -> Result<String> {
    if model_id.trim().is_empty() {
        return Err(anyhow!("Model identifier is required"));
    }

    let model_config = model_config_for_call(model_id, options.explicit_codex)?;
    call_third_party_with_reasoning_config(
        system_prompt,
        user_content,
        &model_config,
        response_title,
        media_files,
        tools,
        options,
    )
    .await
}

/// Variant of [`call_third_party`] that takes an already resolved model config,
/// so callers can use synthesized configs that are not in the runtime catalog
/// (e.g. a foreign Codex slug used as agent step model).
pub async fn call_third_party_with_reasoning_config(
    system_prompt: &str,
    user_content: &str,
    model_config: &ThirdPartyModelConfig,
    response_title: &str,
    media_files: &[MediaFile],
    tools: Option<&mut ToolRuntime>,
    options: ThirdPartyCallOptions<'_>,
) -> Result<String> {
    let ThirdPartyCallOptions {
        audit_context,
        reasoning_override,
        explicit_codex,
        codex_prompt_style,
    } = options;
    if matches!(
        model_config.provider,
        ThirdPartyProvider::OpenAI | ThirdPartyProvider::OpenAICodex
    ) {
        let image_data_list = image_data_list_from_media(media_files);
        return call_responses_provider(
            system_prompt,
            user_content,
            model_config,
            response_title,
            &image_data_list,
            tools,
            audit_context,
            reasoning_override,
            explicit_codex,
            codex_prompt_style,
        )
        .await;
    }

    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);
    let message_content = build_message_content(user_content, media_files);
    let Some(runtime) = tools else {
        let messages = vec![
            json!({ "role": "system", "content": system_prompt }),
            json!({ "role": "user", "content": message_content }),
        ];
        let details = build_request_details(model_config, messages, None, None)?;
        let response = call_provider_api(&details, audit_context, &operation).await?;
        let content = response
            .get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .map(extract_message_content)
            .unwrap_or_default();
        return Ok(parse_third_party_response(model_config, &content));
    };

    let system_prompt = format!("{}\n\n{}", system_prompt, runtime.tool_limit_guidance());
    let messages = vec![
        json!({ "role": "system", "content": system_prompt }),
        json!({ "role": "user", "content": message_content }),
    ];
    chat_completion_with_tool_runtime(messages, model_config, runtime, audit_context, &operation)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: ThirdPartyProvider, name: &str, raw_model: &str) -> ThirdPartyModelConfig {
        ThirdPartyModelConfig {
            id: format!("{}:{}", provider.as_str(), raw_model),
            provider,
            name: name.to_string(),
            model: raw_model.to_string(),
            image: false,
            video: false,
            audio: false,
            tools: true,
        }
    }

    fn mock_runtime(base_url: &str) -> ProviderRuntimeConfig {
        ProviderRuntimeConfig {
            provider: ThirdPartyProvider::OpenRouter,
            display_name: "OpenRouter",
            base_url: base_url.to_string(),
            api_key: "test-openrouter".to_string(),
            temperature: 0.7,
            top_p: 0.95,
            top_k: Some(40),
            request_timeout_secs: 5,
        }
    }

    #[tokio::test]
    async fn provider_api_retries_transient_failures_and_sends_provider_headers() {
        use crate::tools::twitter_extractor::test_support::{
            response_with_headers, ExpectedRequest, TestServer,
        };
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/chat/completions",
                response_with_headers(503, &[], b"busy".to_vec()),
            ),
            ExpectedRequest::new(
                "POST",
                "/chat/completions",
                response_with_headers(
                    200,
                    &[("content-type", "application/json")],
                    br#"{"id":"chatcmpl_1","choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#.to_vec(),
                ),
            )
            .with_header("authorization", "Bearer test-openrouter")
            .with_header("http-referer", OPENROUTER_REFERER),
        ]);
        let runtime = mock_runtime(server.base_url().to_string().trim_end_matches('/'));
        let details = build_request_details_for_runtime(
            &model(ThirdPartyProvider::OpenRouter, "Test", "vendor/model"),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        let value = call_provider_api(&details, None, "test")
            .await
            .expect("second attempt succeeds");

        assert_eq!(value["choices"][0]["message"]["content"], "hi");
        server.join().expect("both requests were served");
    }

    #[tokio::test]
    async fn provider_api_decode_failures_name_the_content_type_and_body() {
        use crate::tools::twitter_extractor::test_support::{
            response_with_headers, ExpectedRequest, TestServer,
        };
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/chat/completions",
            response_with_headers(
                200,
                &[("content-type", "text/html")],
                b"<html>upstream oops</html>".to_vec(),
            ),
        )]);
        let runtime = mock_runtime(server.base_url().to_string().trim_end_matches('/'));
        let details = build_request_details_for_runtime(
            &model(ThirdPartyProvider::OpenRouter, "Test", "vendor/model"),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        let err = call_provider_api(&details, None, "test")
            .await
            .expect_err("html is not a chat completion");

        let text = err.to_string();
        assert!(text.contains("text/html"), "{text}");
        assert!(text.contains("upstream oops"), "{text}");
        server.join().expect("one request was served");
    }

    #[test]
    fn openrouter_request_details_keep_headers_and_top_k() {
        let runtime = ProviderRuntimeConfig {
            provider: ThirdPartyProvider::OpenRouter,
            display_name: "OpenRouter",
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: "test-openrouter".to_string(),
            temperature: 0.7,
            top_p: 0.95,
            top_k: Some(40),
            request_timeout_secs: 75,
        };
        let details = build_request_details_for_runtime(
            &model(
                ThirdPartyProvider::OpenRouter,
                "Qwen 3",
                "qwen/qwen3-next-80b-a3b-instruct:free",
            ),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        assert_eq!(details.url, "https://openrouter.ai/api/v1/chat/completions");
        assert!(details
            .headers
            .iter()
            .any(|(name, value)| name == "HTTP-Referer" && value == OPENROUTER_REFERER));
        assert!(details
            .headers
            .iter()
            .any(|(name, value)| name == "X-Title" && value == OPENROUTER_TITLE));
        assert_eq!(
            details.payload.get("top_k").and_then(|v| v.as_i64()),
            Some(40)
        );
        assert_eq!(details.request_timeout_secs, 75);
    }

    #[test]
    fn request_details_serialize_f32_sampling_params_without_float_noise() {
        // 0.4f32 widened to f64 becomes 0.4000000059604645; at least one
        // OpenRouter upstream (stealth/ox-alpha) rejects such values with an
        // opaque 400, so the wire value must match the configured literal.
        let runtime = ProviderRuntimeConfig {
            provider: ThirdPartyProvider::OpenRouter,
            display_name: "OpenRouter",
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: "test-openrouter".to_string(),
            temperature: 0.4,
            top_p: 0.95,
            top_k: Some(40),
            request_timeout_secs: 60,
        };
        let details = build_request_details_for_runtime(
            &model(
                ThirdPartyProvider::OpenRouter,
                "Ox Alpha",
                "stealth/ox-alpha",
            ),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        assert_eq!(details.payload["temperature"].to_string(), "0.4");
        assert_eq!(details.payload["top_p"].to_string(), "0.95");
    }

    #[test]
    fn nvidia_request_details_omit_top_k_and_openrouter_headers() {
        let runtime = ProviderRuntimeConfig {
            provider: ThirdPartyProvider::Nvidia,
            display_name: "NVIDIA",
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
            api_key: "test-nvidia".to_string(),
            temperature: 0.4,
            top_p: 0.8,
            top_k: None,
            request_timeout_secs: 120,
        };
        let details = build_request_details_for_runtime(
            &model(
                ThirdPartyProvider::Nvidia,
                "Gemma 3n",
                "google/gemma-3n-e4b-it",
            ),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        assert_eq!(
            details.url,
            "https://integrate.api.nvidia.com/v1/chat/completions"
        );
        assert!(!details
            .headers
            .iter()
            .any(|(name, _)| name == "HTTP-Referer" || name == "X-Title"));
        assert!(details.payload.get("top_k").is_none());
        assert_eq!(details.request_timeout_secs, 120);
    }

    #[test]
    fn message_content_includes_video_url_parts_for_video_media() {
        let media = vec![MediaFile::new(
            b"video-bytes".to_vec(),
            "video/mp4".to_string(),
            MediaKind::Video,
            None,
        )];

        let content = build_message_content("analyze this", &media);
        let parts = content.as_array().expect("content should be media parts");

        assert_eq!(parts[0], json!({"type": "text", "text": "analyze this"}));
        assert_eq!(parts[1]["type"], "video_url");
        assert_eq!(
            parts[1]["video_url"]["url"],
            format!(
                "data:video/mp4;base64,{}",
                general_purpose::STANDARD.encode(b"video-bytes")
            )
        );
    }

    #[test]
    fn message_content_includes_audio_url_parts_for_audio_media() {
        let media = vec![MediaFile::new(
            b"audio-bytes".to_vec(),
            "audio/mpeg".to_string(),
            MediaKind::Audio,
            None,
        )];

        let content = build_message_content("transcribe this", &media);
        let parts = content.as_array().expect("content should be media parts");

        assert_eq!(parts[0], json!({"type": "text", "text": "transcribe this"}));
        assert_eq!(parts[1]["type"], "audio_url");
        assert_eq!(
            parts[1]["audio_url"]["url"],
            format!(
                "data:audio/mpeg;base64,{}",
                general_purpose::STANDARD.encode(b"audio-bytes")
            )
        );
    }

    #[test]
    fn message_content_keeps_large_audio_as_base64_data_url() {
        let media = vec![MediaFile::new(
            vec![b'x'; 190 * 1024],
            "audio/mpeg".to_string(),
            MediaKind::Audio,
            None,
        )];

        let content = build_message_content("transcribe this", &media);
        let parts = content.as_array().expect("content should be media parts");
        let url = parts[1]["audio_url"]["url"]
            .as_str()
            .expect("audio url should be a string");

        assert!(url.starts_with("data:audio/mpeg;base64,"));
        assert!(!url.contains("asset_id"));
    }

    #[test]
    fn ollama_request_details_use_cloud_endpoint_and_bearer_auth() {
        let runtime = ProviderRuntimeConfig {
            provider: ThirdPartyProvider::Ollama,
            display_name: "Ollama",
            base_url: "https://ollama.com/v1".to_string(),
            api_key: "test-ollama".to_string(),
            temperature: 0.3,
            top_p: 0.7,
            top_k: None,
            request_timeout_secs: 90,
        };
        let details = build_request_details_for_runtime(
            &model(ThirdPartyProvider::Ollama, "Qwen 3 32B", "qwen3:32b"),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            Some(vec![json!({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": { "type": "object" }
                }
            })]),
            Some("auto"),
        );

        assert_eq!(details.url, "https://ollama.com/v1/chat/completions");
        assert!(details
            .headers
            .iter()
            .any(|(name, value)| { name == "Authorization" && value == "Bearer test-ollama" }));
        assert!(!details
            .headers
            .iter()
            .any(|(name, _)| name == "HTTP-Referer" || name == "X-Title"));
        assert!(details.payload.get("top_k").is_none());
        assert_eq!(
            details
                .payload
                .get("tool_choice")
                .and_then(|value| value.as_str()),
            Some("auto")
        );
        assert_eq!(details.request_timeout_secs, 90);
    }

    #[test]
    fn parse_qwen_content_strips_multiline_think_block() {
        let content = "<think>\nLet me reason.\nStep two.\n</think>\n\nThe answer is 42.";
        assert_eq!(parse_qwen_content(content), "The answer is 42.");
    }

    #[test]
    fn parse_qwen_content_strips_single_line_think_block() {
        assert_eq!(
            parse_qwen_content("<think>hmm</think> Final answer"),
            "Final answer"
        );
    }

    #[test]
    fn parse_qwen_content_falls_back_to_reasoning_when_answer_is_empty() {
        assert_eq!(
            parse_qwen_content("<think>\nonly reasoning\n</think>"),
            "only reasoning"
        );
    }

    #[test]
    fn parse_qwen_content_returns_plain_text_unchanged() {
        assert_eq!(parse_qwen_content("  plain answer  "), "plain answer");
    }

    #[test]
    fn inline_reasoning_markup_is_stripped_for_every_provider() {
        let think = "<think>\nplanning\n</think>\nThe answer.";
        for provider in [
            ThirdPartyProvider::Nvidia,
            ThirdPartyProvider::Ollama,
            ThirdPartyProvider::OpenRouter,
        ] {
            let config = model(provider, "DeepSeek", "deepseek-ai/deepseek-v4-flash");
            assert_eq!(
                parse_third_party_response(&config, think),
                "The answer.",
                "provider {provider:?}"
            );
        }

        let harmony = "<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>Answer";
        let config = model(ThirdPartyProvider::Nvidia, "GPT OSS", "openai/gpt-oss-120b");
        assert_eq!(parse_third_party_response(&config, harmony), "Answer");
    }

    #[test]
    fn think_tags_mentioned_inside_an_answer_are_left_alone() {
        let text = "Qwen wraps its reasoning in <think>...</think> tags.";
        let config = model(ThirdPartyProvider::OpenRouter, "Qwen", "qwen/qwen3-next");
        assert_eq!(parse_third_party_response(&config, text), text);
    }

    #[test]
    fn empty_content_falls_back_to_provider_specific_reasoning_fields() {
        let nvidia = json!({ "content": "", "reasoning_content": "  deep thought  " });
        assert_eq!(extract_message_content(&nvidia), "deep thought");

        let ollama = json!({ "content": "", "thinking": "hmm" });
        assert_eq!(extract_message_content(&ollama), "hmm");

        let openrouter = json!({ "content": "", "reasoning": "existing path" });
        assert_eq!(extract_message_content(&openrouter), "existing path");
    }
}

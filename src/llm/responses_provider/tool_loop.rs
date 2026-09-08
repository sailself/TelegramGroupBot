//! The Responses half of the shared tool-runtime loop.

use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Value};
use tracing::debug;

use crate::config::ThirdPartyModelConfig;
use crate::llm::audit::LlmAuditContext;
use crate::llm::tool_loop::{
    clamp_request_timeout_secs, run_tool_loop, BoxFuture, ModelTurn, ToolCall, ToolProtocol,
    TurnDeadline,
};
use crate::llm::tool_prompts::TOOL_LIMIT_SYSTEM_PROMPT;
use crate::llm::tool_runtime::ToolRuntime;

use super::codex_identity::CodexRequestIdentity;
use super::payload::{debug_model_label, generate_session_id};
use super::sse::{extract_response_output_items, extract_response_text};
use super::transport::{
    build_request_details, call_provider_api, responses_request_timeout_secs, CodexTurnState,
    ResponsesApiResult,
};

#[derive(Debug, Clone)]
pub(super) struct ResponsesToolCall {
    pub(super) call_id: String,
    pub(super) name: String,
    pub(super) arguments: String,
}

pub(super) fn extract_response_tool_calls(output_items: &[Value]) -> Vec<ResponsesToolCall> {
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
pub(super) async fn responses_completion_with_tool_runtime(
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

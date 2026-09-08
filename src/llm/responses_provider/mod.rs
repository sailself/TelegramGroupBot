use anyhow::Result;
use tracing::debug;

use crate::config::ThirdPartyModelConfig;
use crate::llm::audit::LlmAuditContext;
use crate::llm::runtime_models::ResolvedExplicitCodexModel;
use crate::llm::tool_runtime::ToolRuntime;

mod codex_identity;
mod payload;
mod sse;
mod tool_loop;
mod transport;

pub(crate) use codex_identity::{effective_reasoning_effort, CodexRequestIdentity};
use payload::{
    build_native_codex_web_search_tool_from_record, build_responses_system_prompt,
    build_responses_user_input, debug_model_label, generate_session_id,
};
use sse::{extract_response_output_items, extract_response_text};
use tool_loop::responses_completion_with_tool_runtime;
use transport::{build_request_details, call_provider_api, CodexTurnState, ResponsesApiResult};

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
mod tests;

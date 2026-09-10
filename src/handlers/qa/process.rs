//! Running a prepared `/q`-family request against the selected model and
//! rendering the answer back to the chat.

use anyhow::Result;
use serde_json::Value;
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ParseMode,
};
use tokio::sync::OwnedSemaphorePermit;
use tracing::{error, info};

use crate::config::{ThirdPartyProvider, CONFIG};
use crate::handlers::enrichment::{render_sources, EnrichmentBudget};
use crate::handlers::responses::send_response;
use crate::llm::audit::{audit_context_from_id, LlmAuditContext};
use crate::llm::media::{summarize_media_files, MediaFile, MediaSummary};
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::{
    call_gemini, call_gemini_with_tool_runtime, call_third_party,
    call_third_party_with_tool_runtime, CodexPromptStyle, GeminiCallRequest, ThirdPartyCallOptions,
};
use crate::state::{AppState, PendingQRequest, QaCommandMode};
use crate::utils::markdown::markdown_to_telegram_html;
use crate::utils::progress::ProgressReporter;
use crate::utils::telegram::start_chat_action_heartbeat;
use crate::utils::text::escape_html;

use super::chat_search::{
    chat_search_rebuilding_message, process_chat_search_request, warn_on_unverified_chat_links,
};
use super::model_resolution::{format_llm_error_message, ModelCatalogSnapshot, QaModel};
use super::prompt::{
    build_chat_context_system_prompt, build_quick_system_prompt, build_system_prompt,
};

pub(super) const QUICK_SEARCH_FOOTER: &str =
    "_Quick mode used its one web-search round. Use /q for deeper verification or research._";

pub(super) fn qa_mode_label(mode: QaCommandMode) -> &'static str {
    match mode {
        QaCommandMode::Standard => "standard",
        QaCommandMode::Quick => "quick",
        QaCommandMode::ChatContext => "chat_context",
        QaCommandMode::ChatSearch => "chat_search",
    }
}

fn qa_mode_command_name(mode: QaCommandMode) -> &'static str {
    match mode {
        QaCommandMode::Standard => "q",
        QaCommandMode::Quick => "qq",
        QaCommandMode::ChatContext => "qc",
        QaCommandMode::ChatSearch => "s",
    }
}

pub(super) fn reasoning_override_for_qa_mode(
    mode: QaCommandMode,
    provider: ThirdPartyProvider,
    configured_effort: &str,
) -> Option<&str> {
    (mode == QaCommandMode::Quick && provider == ThirdPartyProvider::OpenAICodex)
        .then(|| configured_effort.trim())
        .filter(|effort| !effort.is_empty())
}

pub(super) fn uses_quick_tool_runtime(mode: QaCommandMode, supports_tools: bool) -> bool {
    mode == QaCommandMode::Quick && supports_tools
}

pub(super) fn append_quick_search_footer(
    mut response: String,
    mode: QaCommandMode,
    web_search_attempted: bool,
) -> String {
    if mode == QaCommandMode::Quick
        && web_search_attempted
        && !response.contains(QUICK_SEARCH_FOOTER)
    {
        response.push_str("\n\n");
        response.push_str(QUICK_SEARCH_FOOTER);
    }
    response
}

/// One model call: the prompt, its inputs, and the knobs the providers differ
/// on. Built per mode so [`dispatch`] stays the only place that knows which
/// provider function answers a request.
pub(super) struct QaCall<'a> {
    pub(super) system_prompt: String,
    pub(super) user_content: String,
    /// Gemini logs this in place of the system prompt text; third-party
    /// providers use it as the response title in their audit rows.
    pub(super) label: &'static str,
    pub(super) media_files: Option<Vec<MediaFile>>,
    pub(super) youtube_urls: Option<Vec<String>>,
    /// Borrowed, not owned: the caller reads the runtime's web-search flag and
    /// accumulated message ids after the call returns.
    pub(super) tools: Option<&'a mut ToolRuntime>,
    pub(super) reasoning_override: Option<String>,
    pub(super) response_schema: Option<Value>,
    pub(super) prompt_style: CodexPromptStyle,
    pub(super) use_pro: bool,
    pub(super) search_grounding: bool,
    pub(super) audit_context: Option<&'a LlmAuditContext>,
}

/// The one place a `/q`-family request reaches a model. Four calls: Gemini and
/// third-party, each with and without a tool runtime.
pub(super) async fn dispatch(
    model: &QaModel,
    call: QaCall<'_>,
) -> Result<(String, Option<String>)> {
    let QaCall {
        system_prompt,
        user_content,
        label,
        media_files,
        youtube_urls,
        tools,
        reasoning_override,
        response_schema,
        prompt_style,
        use_pro,
        search_grounding,
        audit_context,
    } = call;
    let third_party_options = ThirdPartyCallOptions::new(audit_context, prompt_style)
        .with_reasoning_override(reasoning_override.as_deref())
        .with_explicit_codex_model(model.explicit_codex());

    match (model, tools) {
        (QaModel::Gemini, Some(runtime)) => call_gemini_with_tool_runtime(
            &system_prompt,
            &user_content,
            runtime,
            use_pro,
            media_files,
            youtube_urls,
            Some(label),
            response_schema,
            audit_context,
        )
        .await
        .map(|result| (result.text, Some(result.model_used))),
        (QaModel::Gemini, None) => call_gemini(GeminiCallRequest {
            system_prompt: &system_prompt,
            user_content: &user_content,
            use_search_grounding: search_grounding,
            use_pro_model: use_pro,
            media_files: media_files.unwrap_or_default(),
            youtube_urls: youtube_urls.unwrap_or_default(),
            system_prompt_label: Some(label),
            audit_context,
        })
        .await
        .map(|result| (result.text, Some(result.model_used))),
        (QaModel::ThirdParty { .. }, Some(runtime)) => call_third_party_with_tool_runtime(
            &system_prompt,
            &user_content,
            model.model_id(),
            label,
            &media_files.unwrap_or_default(),
            runtime,
            third_party_options,
        )
        .await
        .map(|response| (response, None)),
        (QaModel::ThirdParty { .. }, None) => call_third_party(
            &system_prompt,
            &user_content,
            model.model_id(),
            label,
            &media_files.unwrap_or_default(),
            None,
            third_party_options,
        )
        .await
        .map(|response| (response, None)),
    }
}

/// Gemini names the prompt it logs; third-party providers title the response.
/// Every mode names both, and the resolved model picks which one it uses.
pub(super) fn qa_call_label(model: &QaModel, mode: QaCommandMode) -> &'static str {
    match (mode, model) {
        (QaCommandMode::Standard, QaModel::Gemini) => "Q_SYSTEM_PROMPT",
        (QaCommandMode::Standard, QaModel::ThirdParty { .. }) => "Answer to Your Question",
        (QaCommandMode::Quick, QaModel::Gemini) => "QUICK_Q_SYSTEM_PROMPT",
        (QaCommandMode::Quick, QaModel::ThirdParty { .. }) => "Quick Answer",
        (QaCommandMode::ChatContext, QaModel::Gemini) => "QC_SYSTEM_PROMPT",
        (QaCommandMode::ChatContext, QaModel::ThirdParty { .. }) => "Answer about Chat",
        (QaCommandMode::ChatSearch, QaModel::Gemini) => "CHAT_SEARCH_SYSTEM_PROMPT",
        (QaCommandMode::ChatSearch, QaModel::ThirdParty { .. }) => "Chat Search",
    }
}

/// Gemini's pro model earns its cost when the request carries media or video.
fn use_pro_gemini(request: &PendingQRequest) -> bool {
    !request.enrichment.media_files.is_empty() || !request.enrichment.youtube_urls.is_empty()
}

/// `/q`: one answer, with web search when the model can search.
async fn run_standard_request(
    model: &QaModel,
    request: &PendingQRequest,
    system_prompt: String,
    query: String,
    audit_context: Option<&LlmAuditContext>,
) -> Result<(String, Option<String>)> {
    let mut web_tools = (!matches!(model, QaModel::Gemini) && model.supports_tools())
        .then(ToolRuntime::for_web_search);
    dispatch(
        model,
        QaCall {
            system_prompt,
            user_content: query,
            label: qa_call_label(model, QaCommandMode::Standard),
            media_files: Some(request.enrichment.media_files.clone()),
            youtube_urls: Some(request.enrichment.youtube_urls.clone()),
            tools: web_tools.as_mut(),
            reasoning_override: None,
            response_schema: None,
            prompt_style: CodexPromptStyle::FreeformAnswer,
            use_pro: use_pro_gemini(request),
            search_grounding: true,
            audit_context,
        },
    )
    .await
}

/// The quick call, with or without the tool runtime that gives it its one
/// web-search round.
fn quick_call<'a>(
    model: &QaModel,
    request: &PendingQRequest,
    system_prompt: String,
    query: String,
    reasoning_override: Option<String>,
    tools: Option<&'a mut ToolRuntime>,
    audit_context: Option<&'a LlmAuditContext>,
) -> QaCall<'a> {
    QaCall {
        system_prompt,
        user_content: query,
        label: qa_call_label(model, QaCommandMode::Quick),
        media_files: Some(request.enrichment.media_files.clone()),
        youtube_urls: Some(request.enrichment.youtube_urls.clone()),
        tools,
        reasoning_override,
        response_schema: None,
        prompt_style: CodexPromptStyle::FreeformAnswer,
        use_pro: use_pro_gemini(request),
        search_grounding: false,
        audit_context,
    }
}

/// `/qq`: one bounded answer, with at most one web-search round. Reports
/// whether that round happened, for the footer that tells the user so.
async fn run_quick_request(
    model: &QaModel,
    state: &AppState,
    request: &PendingQRequest,
    system_prompt: String,
    query: String,
    audit_context: Option<&LlmAuditContext>,
) -> (Result<(String, Option<String>)>, bool) {
    let reasoning_override = reasoning_override_for_qa_mode(
        QaCommandMode::Quick,
        model.third_party_provider(),
        &CONFIG.quick_reasoning_effort,
    )
    .map(str::to_string);

    if !uses_quick_tool_runtime(QaCommandMode::Quick, model.supports_tools()) {
        let call = quick_call(
            model,
            request,
            system_prompt,
            query,
            reasoning_override,
            None,
            audit_context,
        );
        return (dispatch(model, call).await, false);
    }

    let mut runtime = ToolRuntime::for_quick(state.db.clone(), request.chat_id);
    let call = quick_call(
        model,
        request,
        system_prompt,
        query,
        reasoning_override,
        Some(&mut runtime),
        audit_context,
    );
    let result = dispatch(model, call).await;
    let web_search_attempted = runtime.web_search_attempted();
    (result, web_search_attempted)
}

/// `/qc`: the agentic pipeline when it is enabled and takes the request,
/// otherwise the legacy tool loop. Reports the message ids the tools actually
/// returned, so a fabricated citation can be spotted.
async fn run_chat_context_request(
    bot: &Bot,
    state: &AppState,
    model: &QaModel,
    request: &PendingQRequest,
    system_prompt: String,
    query: String,
    audit_context: Option<&LlmAuditContext>,
) -> (Result<(String, Option<String>)>, Vec<i64>) {
    if CONFIG.enable_agentic_qc {
        let mut progress_reporter = ProgressReporter::new(
            bot.clone(),
            ChatId(request.chat_id),
            MessageId(request.selection_message_id as i32),
        );
        match crate::agents::qc::run_qc_pipeline(
            crate::agents::qc::QcRequest {
                db: &state.db,
                chat_id: request.chat_id,
                query: &query,
                model_name: model.model_id(),
                system_prompt: &system_prompt,
                media_files: &request.enrichment.media_files,
                youtube_urls: &request.enrichment.youtube_urls,
                audit_context,
            },
            &mut progress_reporter,
        )
        .await
        {
            Ok(crate::agents::qc::QcPipelineResult::Answer(outcome)) => {
                return (
                    Ok((outcome.answer, outcome.gemini_model_used)),
                    outcome.valid_message_ids,
                );
            }
            Ok(crate::agents::qc::QcPipelineResult::UseLegacy(reason)) => {
                info!("Agentic /qc fell back to the legacy tool loop: {reason}");
            }
            Err(err) => return (Err(err), Vec::new()),
        }
    }

    let mut runtime = ToolRuntime::for_qc(state.db.clone(), request.chat_id);
    // Gemini needs the tool budget spelled out in its system prompt; the
    // third-party call appends the same guidance itself.
    let system_prompt = match model {
        QaModel::Gemini => format!("{}\n\n{}", system_prompt, runtime.tool_limit_guidance()),
        QaModel::ThirdParty { .. } => system_prompt,
    };
    let result = dispatch(
        model,
        QaCall {
            system_prompt,
            user_content: query,
            label: qa_call_label(model, QaCommandMode::ChatContext),
            media_files: Some(request.enrichment.media_files.clone()),
            youtube_urls: Some(request.enrichment.youtube_urls.clone()),
            tools: Some(&mut runtime),
            reasoning_override: None,
            response_schema: None,
            prompt_style: CodexPromptStyle::FreeformAnswer,
            use_pro: use_pro_gemini(request),
            search_grounding: false,
            audit_context,
        },
    )
    .await;
    let valid_message_ids = runtime.accumulated_message_ids();
    (result, valid_message_ids)
}

/// What a QA call was: logged once when it starts and again, with the error,
/// if it fails.
struct QaRequestSummary {
    mode: &'static str,
    provider: &'static str,
    model: String,
    chat_id: i64,
    user_id: i64,
    message_id: i64,
    selection_message_id: i64,
    tools_enabled: bool,
    media: MediaSummary,
    youtube_urls: usize,
    query_len: usize,
}

impl QaRequestSummary {
    fn new(
        model: &QaModel,
        request: &PendingQRequest,
        model_label: &str,
        query_len: usize,
    ) -> Self {
        Self {
            mode: qa_mode_label(request.mode),
            provider: model.provider_label(),
            model: model_label.to_string(),
            chat_id: request.chat_id,
            user_id: request.user_id,
            message_id: request.message_id,
            selection_message_id: request.selection_message_id,
            tools_enabled: model.supports_tools(),
            media: summarize_media_files(&request.enrichment.media_files),
            youtube_urls: request.enrichment.youtube_urls.len(),
            query_len,
        }
    }
}

impl std::fmt::Display for QaRequestSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "mode={}, provider={}, model={}, chat_id={}, user_id={}, message_id={}, selection_message_id={}, tools_enabled={}, images={}, videos={}, audios={}, documents={}, youtube_urls={}, query_len={}",
            self.mode,
            self.provider,
            self.model,
            self.chat_id,
            self.user_id,
            self.message_id,
            self.selection_message_id,
            self.tools_enabled,
            self.media.images,
            self.media.videos,
            self.media.audios,
            self.media.documents,
            self.youtube_urls,
            self.query_len
        )
    }
}

fn build_mode_system_prompt(request: &PendingQRequest) -> String {
    let language = request.telegram_language_code.as_deref();
    match request.mode {
        QaCommandMode::Standard => build_system_prompt(language),
        QaCommandMode::Quick => build_quick_system_prompt(language),
        QaCommandMode::ChatContext => build_chat_context_system_prompt(language),
        // Chat search builds its own prompt around the result target.
        QaCommandMode::ChatSearch => String::new(),
    }
}

/// The question, with any fetched link content quoted after it — fenced and
/// budgeted, so remote text is data the model may cite and never instructions.
fn build_user_content(request: &PendingQRequest) -> String {
    let mut content = request.query.clone();
    let rendered_sources = render_sources(
        &request.enrichment.sources,
        &EnrichmentBudget::for_question(),
    );
    if !rendered_sources.is_empty() {
        content.push_str("\n\n");
        content.push_str(&rendered_sources);
    }
    content
}

/// Render the answer into the message the user is already looking at, naming
/// the model that produced it.
async fn send_qa_answer(
    bot: &Bot,
    request: &PendingQRequest,
    model: &QaModel,
    snapshot: &ModelCatalogSnapshot,
    response: String,
    gemini_model_used: Option<&str>,
    quick_search_attempted: bool,
) -> Result<()> {
    let response_text = append_quick_search_footer(response, request.mode, quick_search_attempted);
    let mut rendered_response = markdown_to_telegram_html(&response_text);
    if !model.model_id().is_empty() {
        let display_model = model.result_display_name(snapshot, request.mode, gemini_model_used);
        rendered_response.push_str(&format!("\n\nModel: {}", escape_html(&display_model)));
    }

    send_response(
        bot,
        ChatId(request.chat_id),
        MessageId(request.selection_message_id as i32),
        &rendered_response,
        if request.mode == QaCommandMode::ChatContext {
            "Answer about Chat"
        } else {
            "Answer to Your Question"
        },
        ParseMode::Html,
    )
    .await
}

/// Run a prepared request against `model`. `heavy_permit` is the permit a
/// caller already holds (the direct `/q` path throttles its own preparation);
/// passing it through avoids taking a second slot from the same semaphore,
/// which could exhaust the heavy-command lane and deadlock it.
pub(super) async fn process_request(
    bot: &Bot,
    state: &AppState,
    request: PendingQRequest,
    model: &QaModel,
    snapshot: &ModelCatalogSnapshot,
    heavy_permit: Option<OwnedSemaphorePermit>,
) -> Result<()> {
    if matches!(model, QaModel::Gemini) && !CONFIG.gemini_api_available() {
        bot.edit_message_text(
            ChatId(request.chat_id),
            MessageId(request.selection_message_id as i32),
            "Gemini is disabled or not configured. Please choose another model.",
        )
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await?;
        return Ok(());
    }

    let _heavy_permit = state.reuse_or_acquire_heavy_permit(heavy_permit).await;
    let audit_context = audit_context_from_id(&state.db, request.llm_invocation_id);
    if request.mode.requires_chat_search_index() && !state.db.is_search_ready() {
        bot.edit_message_text(
            ChatId(request.chat_id),
            MessageId(request.selection_message_id as i32),
            chat_search_rebuilding_message(qa_mode_command_name(request.mode)),
        )
        .await?;
        return Ok(());
    }

    let system_prompt = build_mode_system_prompt(&request);
    let query = build_user_content(&request);

    let logged_model_name = model.display_name(snapshot, request.mode);
    let summary = QaRequestSummary::new(model, &request, &logged_model_name, query.chars().count());
    info!("Processing QA request: {summary}");

    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), ChatId(request.chat_id), ChatAction::Typing);

    let mut qc_valid_message_ids: Vec<i64> = Vec::new();
    let mut quick_search_attempted = false;
    let response = match request.mode {
        QaCommandMode::ChatSearch => {
            return process_chat_search_request(
                bot,
                state,
                &request,
                &query,
                model,
                snapshot,
                audit_context.as_ref(),
            )
            .await;
        }
        QaCommandMode::Standard => {
            run_standard_request(
                model,
                &request,
                system_prompt,
                query,
                audit_context.as_ref(),
            )
            .await
        }
        QaCommandMode::Quick => {
            let (result, web_search_attempted) = run_quick_request(
                model,
                state,
                &request,
                system_prompt,
                query,
                audit_context.as_ref(),
            )
            .await;
            quick_search_attempted = web_search_attempted;
            result
        }
        QaCommandMode::ChatContext => {
            let (result, valid_message_ids) = run_chat_context_request(
                bot,
                state,
                model,
                &request,
                system_prompt,
                query,
                audit_context.as_ref(),
            )
            .await;
            qc_valid_message_ids = valid_message_ids;
            result
        }
    };
    let (response, gemini_model_used) = match response {
        Ok(response) => response,
        Err(err) => {
            error!("QA request failed: {summary}, error={err:#}");
            let message = format_llm_error_message(model, &logged_model_name, &err);
            bot.edit_message_text(
                ChatId(request.chat_id),
                MessageId(request.selection_message_id as i32),
                message,
            )
            .await?;
            return Err(err);
        }
    };

    if response.trim().is_empty() {
        bot.edit_message_text(ChatId(request.chat_id), MessageId(request.selection_message_id as i32), "I couldn't find an answer to your question. Please try rephrasing or asking something else.")
            .await?;
        return Ok(());
    }

    if request.mode == QaCommandMode::ChatContext {
        warn_on_unverified_chat_links(
            &response,
            request.chat_id,
            &qc_valid_message_ids,
            request.message_id,
        );
    }

    send_qa_answer(
        bot,
        &request,
        model,
        snapshot,
        response,
        gemini_model_used.as_deref(),
        quick_search_attempted,
    )
    .await
}

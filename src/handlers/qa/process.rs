//! Running a prepared `/q`-family request against the selected model and
//! rendering the answer back to the chat.

use anyhow::{anyhow, Result};
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ParseMode,
};
use tokio::sync::OwnedSemaphorePermit;
use tracing::{error, info};

use crate::config::{ThirdPartyProvider, CONFIG};
use crate::handlers::responses::send_response;
use crate::llm::audit::audit_context_from_id;
use crate::llm::media::summarize_media_files;
use crate::llm::runtime_models::{runtime_model_config, ResolvedExplicitCodexModel};
use crate::llm::text_model::MODEL_GEMINI;
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::{
    call_gemini, call_gemini_with_tool_runtime, call_third_party,
    call_third_party_with_tool_runtime, GeminiCallRequest,
};
use crate::state::{AppState, PendingQRequest, QaCommandMode};
use crate::utils::markdown::markdown_to_telegram_html;
use crate::utils::progress::ProgressReporter;
use crate::utils::telegram::start_chat_action_heartbeat;
use crate::utils::text::escape_html;

use super::chat_search::{
    chat_search_rebuilding_message, process_chat_search_request, warn_on_unverified_chat_links,
};
use super::model_resolution::{
    codex_quick_result_label, configured_model_display_name, format_llm_error_message,
    result_model_display_name, third_party_provider_label,
};
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

/// Run a prepared request against `model_name`. `heavy_permit` is the permit a
/// caller already holds (the direct `/q` path throttles its own preparation);
/// passing it through avoids taking a second slot from the same semaphore,
/// which could exhaust the heavy-command lane and deadlock it.
pub(super) async fn process_request(
    bot: &Bot,
    state: &AppState,
    request: PendingQRequest,
    model_name: &str,
    explicit_codex: Option<&ResolvedExplicitCodexModel>,
    heavy_permit: Option<OwnedSemaphorePermit>,
) -> Result<()> {
    if model_name == MODEL_GEMINI && !CONFIG.gemini_api_available() {
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

    let runtime_config = if explicit_codex.is_none() && model_name != MODEL_GEMINI {
        runtime_model_config(model_name)
    } else {
        None
    };
    let request_model_config = match explicit_codex {
        Some(explicit) => {
            if explicit.config.id != model_name {
                return Err(anyhow!("The explicit Codex model changed"));
            }
            Some(&explicit.config)
        }
        None => runtime_config.as_ref(),
    };

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

    let system_prompt = match request.mode {
        QaCommandMode::Standard => build_system_prompt(request.telegram_language_code.as_deref()),
        QaCommandMode::Quick => {
            build_quick_system_prompt(request.telegram_language_code.as_deref())
        }
        QaCommandMode::ChatContext => {
            build_chat_context_system_prompt(request.telegram_language_code.as_deref())
        }
        QaCommandMode::ChatSearch => String::new(),
    };

    let mut query = request.query.clone();
    for content in &request.telegraph_contents {
        query.push_str("\n\n");
        query.push_str(content);
    }
    for content in &request.twitter_contents {
        query.push_str("\n\n");
        query.push_str(content);
    }

    let supports_tools = if model_name == MODEL_GEMINI {
        true
    } else {
        request_model_config.is_some_and(|config| config.tools)
    };
    let media_summary = summarize_media_files(&request.media_files);
    let provider_label = if model_name == MODEL_GEMINI {
        "Gemini".to_string()
    } else {
        request_model_config
            .map(|config| third_party_provider_label(config.provider).to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    };
    let logged_model_name = explicit_codex
        .map(|explicit| {
            codex_quick_result_label(
                &explicit.config,
                Some(&explicit.record),
                Some(&CONFIG.quick_reasoning_effort),
            )
        })
        .unwrap_or_else(|| configured_model_display_name(model_name));

    info!(
        "Processing QA request: mode={}, provider={}, model={}, chat_id={}, user_id={}, message_id={}, selection_message_id={}, tools_enabled={}, images={}, videos={}, audios={}, documents={}, youtube_urls={}, query_len={}",
        qa_mode_label(request.mode),
        provider_label,
        logged_model_name,
        request.chat_id,
        request.user_id,
        request.message_id,
        request.selection_message_id,
        supports_tools,
        media_summary.images,
        media_summary.videos,
        media_summary.audios,
        media_summary.documents,
        request.youtube_urls.len(),
        query.chars().count()
    );

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
                model_name,
                audit_context.as_ref(),
            )
            .await;
        }
        QaCommandMode::Standard => {
            if model_name == MODEL_GEMINI {
                let use_pro = !request.media_files.is_empty() || !request.youtube_urls.is_empty();
                call_gemini(GeminiCallRequest {
                    system_prompt: &system_prompt,
                    user_content: &query,
                    use_search_grounding: true,
                    use_pro_model: use_pro,
                    media_files: request.media_files.clone(),
                    youtube_urls: request.youtube_urls.clone(),
                    system_prompt_label: Some("Q_SYSTEM_PROMPT"),
                    audit_context: audit_context.as_ref(),
                })
                .await
                .map(|result| (result.text, Some(result.model_used)))
            } else {
                let mut web_tools = supports_tools.then(ToolRuntime::for_web_search);
                call_third_party(
                    &system_prompt,
                    &query,
                    model_name,
                    "Answer to Your Question",
                    &request.media_files,
                    web_tools.as_mut(),
                    crate::llm::ThirdPartyCallOptions::new(
                        audit_context.as_ref(),
                        crate::llm::CodexPromptStyle::FreeformAnswer,
                    ),
                )
                .await
                .map(|result| (result, None))
            }
        }
        QaCommandMode::Quick => {
            let use_pro = !request.media_files.is_empty() || !request.youtube_urls.is_empty();
            if uses_quick_tool_runtime(request.mode, supports_tools) {
                let mut runtime = ToolRuntime::for_quick(state.db.clone(), request.chat_id);
                let result = if model_name == MODEL_GEMINI {
                    call_gemini_with_tool_runtime(
                        &system_prompt,
                        &query,
                        &mut runtime,
                        use_pro,
                        Some(request.media_files.clone()),
                        Some(request.youtube_urls.clone()),
                        Some("QUICK_Q_SYSTEM_PROMPT"),
                        None,
                        audit_context.as_ref(),
                    )
                    .await
                    .map(|result| (result.text, Some(result.model_used)))
                } else {
                    let provider = request_model_config
                        .map(|config| config.provider)
                        .unwrap_or(ThirdPartyProvider::OpenRouter);
                    let reasoning_override = reasoning_override_for_qa_mode(
                        request.mode,
                        provider,
                        &CONFIG.quick_reasoning_effort,
                    );
                    call_third_party_with_tool_runtime(
                        &system_prompt,
                        &query,
                        model_name,
                        "Quick Answer",
                        &request.media_files,
                        &mut runtime,
                        crate::llm::ThirdPartyCallOptions::new(
                            audit_context.as_ref(),
                            crate::llm::CodexPromptStyle::FreeformAnswer,
                        )
                        .with_reasoning_override(reasoning_override)
                        .with_explicit_codex_model(explicit_codex),
                    )
                    .await
                    .map(|result| (result, None))
                };
                quick_search_attempted = runtime.web_search_attempted();
                result
            } else {
                let provider = request_model_config
                    .map(|config| config.provider)
                    .unwrap_or(ThirdPartyProvider::OpenRouter);
                let reasoning_override = reasoning_override_for_qa_mode(
                    request.mode,
                    provider,
                    &CONFIG.quick_reasoning_effort,
                );
                call_third_party(
                    &system_prompt,
                    &query,
                    model_name,
                    "Quick Answer",
                    &request.media_files,
                    None,
                    crate::llm::ThirdPartyCallOptions::new(
                        audit_context.as_ref(),
                        crate::llm::CodexPromptStyle::FreeformAnswer,
                    )
                    .with_reasoning_override(reasoning_override)
                    .with_explicit_codex_model(explicit_codex),
                )
                .await
                .map(|result| (result, None))
            }
        }
        QaCommandMode::ChatContext => {
            let mut agentic_result: Option<Result<(String, Option<String>)>> = None;
            if CONFIG.enable_agentic_qc {
                let mut progress_reporter = ProgressReporter::new(
                    bot.clone(),
                    ChatId(request.chat_id),
                    MessageId(request.selection_message_id as i32),
                );
                match crate::agents::qc::run_qc_pipeline(
                    &state.db,
                    request.chat_id,
                    &query,
                    model_name,
                    &system_prompt,
                    &request.media_files,
                    &request.youtube_urls,
                    audit_context.as_ref(),
                    &mut progress_reporter,
                )
                .await
                {
                    Ok(crate::agents::qc::QcPipelineResult::Answer(outcome)) => {
                        qc_valid_message_ids = outcome.valid_message_ids;
                        agentic_result = Some(Ok((outcome.answer, outcome.gemini_model_used)));
                    }
                    Ok(crate::agents::qc::QcPipelineResult::UseLegacy(reason)) => {
                        info!("Agentic /qc fell back to the legacy tool loop: {reason}");
                    }
                    Err(err) => {
                        agentic_result = Some(Err(err));
                    }
                }
            }

            if let Some(result) = agentic_result {
                result
            } else {
                let mut runtime = ToolRuntime::for_qc(state.db.clone(), request.chat_id);
                let qc_result = if model_name == MODEL_GEMINI {
                    let use_pro =
                        !request.media_files.is_empty() || !request.youtube_urls.is_empty();
                    call_gemini_with_tool_runtime(
                        &format!("{}\n\n{}", system_prompt, runtime.tool_limit_guidance()),
                        &query,
                        &mut runtime,
                        use_pro,
                        Some(request.media_files.clone()),
                        Some(request.youtube_urls.clone()),
                        Some("QC_SYSTEM_PROMPT"),
                        None,
                        audit_context.as_ref(),
                    )
                    .await
                    .map(|result| (result.text, Some(result.model_used)))
                } else {
                    call_third_party_with_tool_runtime(
                        &system_prompt,
                        &query,
                        model_name,
                        "Answer about Chat",
                        &request.media_files,
                        &mut runtime,
                        crate::llm::ThirdPartyCallOptions::new(
                            audit_context.as_ref(),
                            crate::llm::CodexPromptStyle::FreeformAnswer,
                        ),
                    )
                    .await
                    .map(|result| (result, None))
                };
                qc_valid_message_ids = runtime.accumulated_message_ids();
                qc_result
            }
        }
    };
    let (response, gemini_model_used) = match response {
        Ok(response) => response,
        Err(err) => {
            error!(
                "QA request failed: mode={}, provider={}, model={}, chat_id={}, user_id={}, message_id={}, selection_message_id={}, tools_enabled={}, images={}, videos={}, audios={}, documents={}, youtube_urls={}, query_len={}, error={:#}",
                qa_mode_label(request.mode),
                provider_label,
                logged_model_name,
                request.chat_id,
                request.user_id,
                request.message_id,
                request.selection_message_id,
                supports_tools,
                media_summary.images,
                media_summary.videos,
                media_summary.audios,
                media_summary.documents,
                request.youtube_urls.len(),
                query.chars().count(),
                err
            );
            let message = format_llm_error_message(model_name, &err);
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

    let response_text = append_quick_search_footer(response, request.mode, quick_search_attempted);
    let mut rendered_response = markdown_to_telegram_html(&response_text);
    if !model_name.is_empty() {
        let display_model = result_model_display_name(
            model_name,
            gemini_model_used.as_deref(),
            request.mode,
            explicit_codex,
        );
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
    .await?;

    Ok(())
}

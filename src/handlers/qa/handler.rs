//! `/q`, `/qc`, `/qq`, and `/s` command entry points: request intake, media
//! and link extraction, direct-model vs. picker routing.

use std::time::Duration;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::ParseMode;
use tracing::warn;

use crate::config::CONFIG;
use crate::db::database::build_message_insert;
use crate::db::models::MessageInsert;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::content::{
    download_telegraph_media, download_twitter_media, extract_telegraph_urls_and_content,
    extract_twitter_urls_and_content,
};
use crate::handlers::media::{collect_message_media, MediaCollectionOptions};
use crate::llm::audit::{
    create_audit_context_from_message, LlmAuditContext, LLM_TRIGGER_KIND_AUTO_Q,
    LLM_TRIGGER_KIND_COMMAND,
};
use crate::llm::media::summarize_media_files;
use crate::llm::runtime_models::runtime_model_count;
use crate::llm::text_model::{
    has_available_third_party_models_for_request, resolve_default_text_model_for_request,
    ModelRequestCapabilities,
};
use crate::state::{AppState, PendingQRequest, QaCommandMode};
use crate::tools::external_media::ExternalMediaBudget;
use crate::utils::telegram::{
    message_entities_for_text, message_text_or_caption, reply_with_retry,
};
use crate::utils::timing::{complete_command_timer, now_unix_seconds, start_command_timer};

use super::chat_search::{build_chat_search_pending_request, chat_search_rebuilding_message};
use super::model_resolution::{
    configured_model_display_name, resolve_quick_text_model_for_request,
    selectable_model_ids_for_request, should_use_default_model_without_selection,
    video_request_has_capable_model,
};
use super::process::process_request;
use super::prompt::{
    build_media_only_qa_prompt, prepare_youtube_inputs_for_qa, NO_VIDEO_CAPABLE_MODEL_MESSAGE,
};
use super::selection_ui::{
    create_model_selection_keyboard, process_timed_out_q_request_with_default_model,
};

pub(super) const USER_ERROR_DETAIL_LIMIT: usize = 400;

async fn create_q_audit_context(
    state: &AppState,
    message: &Message,
    command_name: &str,
) -> Option<LlmAuditContext> {
    let trigger_kind = if message_text_or_caption(message)
        .map(|text| text.trim_start().starts_with('/'))
        .unwrap_or(false)
    {
        LLM_TRIGGER_KIND_COMMAND
    } else {
        LLM_TRIGGER_KIND_AUTO_Q
    };

    create_audit_context_from_message(&state.db, trigger_kind, command_name, message).await
}

/// Row recording a `/q`-family command in the chat history and search index.
/// Built once per request so every path (direct model, picker, auto-`/q`,
/// `/s`) records the question the same way.
pub(super) fn build_q_command_insert(
    message: &Message,
    user_id: i64,
    username: &str,
    original_query: &str,
    db_query_text: &str,
    command_name: &str,
) -> MessageInsert {
    build_message_insert(
        Some(user_id),
        Some(username.to_string()),
        message
            .text()
            .map(|value| value.to_string())
            .or_else(|| message.caption().map(|value| value.to_string()))
            .or_else(|| Some(original_query.to_string())),
        None,
        message.date,
        message.reply_to_message().map(|msg| msg.id.0 as i64),
        Some(message.chat.id.0),
        Some(message.id.0 as i64),
        Some(db_query_text.to_string()),
        true,
        Some(command_name.to_string()),
        true,
        true,
    )
}

async fn q_handler_internal(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
    command_name: &str,
    mode: QaCommandMode,
) -> Result<()> {
    if !check_access_control(&bot, &message, command_name).await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        reply_with_retry(
            &bot,
            message.chat.id,
            "You're sending commands too quickly. Please wait a moment before trying again.",
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }
    let heavy_permit = state.acquire_heavy_command_permit().await;

    let query_text_raw = query.unwrap_or_default();
    let query_entities = message_entities_for_text(&message);
    let reply_message = message.reply_to_message();
    let mut reply_text_raw = String::new();
    let mut reply_text = String::new();
    let mut telegraph_contents = Vec::new();
    let mut twitter_contents = Vec::new();

    if let Some(reply) = reply_message {
        reply_text_raw = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if !reply_text_raw.trim().is_empty() {
            let reply_entities = message_entities_for_text(reply);
            let (reply_text_processed, reply_telegraph) =
                extract_telegraph_urls_and_content(&reply_text_raw, reply_entities.as_deref(), 5)
                    .await;
            let (reply_text_processed, reply_twitter) = extract_twitter_urls_and_content(
                &reply_text_processed,
                reply_entities.as_deref(),
                5,
            )
            .await;
            telegraph_contents.extend(reply_telegraph);
            twitter_contents.extend(reply_twitter);
            reply_text = reply_text_processed;
        }
    }

    let media_options = MediaCollectionOptions::for_qa();
    let max_files = media_options.max_files;
    let media = collect_message_media(&bot, &state, &message, media_options).await;
    let mut media_files = media.files;
    let initial_media_summary = summarize_media_files(&media_files);

    let original_query = if query_text_raw.trim().is_empty() {
        if reply_text_raw.trim().is_empty() {
            build_media_only_qa_prompt(&initial_media_summary).unwrap_or_default()
        } else {
            reply_text_raw.clone()
        }
    } else {
        query_text_raw.clone()
    };

    if original_query.trim().is_empty() {
        reply_with_retry(
            &bot,
            message.chat.id,
            &format!(
                "Please provide a question or reply to a message with /{}.",
                command_name
            ),
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    if mode == QaCommandMode::ChatContext && !state.db.is_search_ready() {
        reply_with_retry(
            &bot,
            message.chat.id,
            &chat_search_rebuilding_message("qc"),
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let mut query_text = query_text_raw.clone();
    if !query_text.trim().is_empty() {
        let (query_text_processed, query_telegraph) =
            extract_telegraph_urls_and_content(&query_text, query_entities.as_deref(), 5).await;
        let (query_text_processed, query_twitter) =
            extract_twitter_urls_and_content(&query_text_processed, query_entities.as_deref(), 5)
                .await;
        telegraph_contents.extend(query_telegraph);
        twitter_contents.extend(query_twitter);
        query_text = query_text_processed;
    }

    let query_base = if query_text.trim().is_empty() {
        if reply_text.trim().is_empty() {
            original_query.clone()
        } else {
            reply_text.clone()
        }
    } else if reply_text.trim().is_empty() {
        query_text.clone()
    } else {
        format!(
            "Context from replied message: \"{}\"\n\nQuestion: {}",
            reply_text, query_text
        )
    };

    let (mut query_text, mut youtube_urls) =
        prepare_youtube_inputs_for_qa(&query_base, mode, None, CONFIG.gemini_api_available());

    let user_language_code = message
        .from
        .as_ref()
        .and_then(|user| user.language_code.as_deref());

    let username = message
        .from
        .as_ref()
        .map(|user| user.full_name())
        .unwrap_or_else(|| "Anonymous".to_string());

    let db_query_text = if let Some(reply) = message.reply_to_message() {
        let replied_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if replied_text.is_empty() {
            query_text.clone()
        } else {
            format!(
                "Context from replied message: \"{}\"\n\nQuestion: {}",
                replied_text, query_text
            )
        }
    } else {
        query_text.clone()
    };

    // Record the question now so it lands in the chat history and search
    // index no matter which path (direct model, picker, timeout) answers it.
    let db_insert = build_q_command_insert(
        &message,
        user_id,
        &username,
        &original_query,
        &db_query_text,
        command_name,
    );
    if let Err(err) = state.db.queue_message_insert(db_insert).await {
        warn!("Failed to queue /{command_name} message insert: {err}");
    }

    let mut remaining = max_files.saturating_sub(media_files.len());
    let external_media_budget = ExternalMediaBudget::new(CONFIG.external_media_total_max_bytes);
    if remaining > 0 {
        let telegraph_files =
            download_telegraph_media(&telegraph_contents, remaining, &external_media_budget).await;
        remaining = remaining.saturating_sub(telegraph_files.len());
        media_files.extend(telegraph_files);
    }

    if remaining > 0 {
        let twitter_files =
            download_twitter_media(&twitter_contents, remaining, &external_media_budget).await;
        media_files.extend(twitter_files);
    }
    let audit_context = create_q_audit_context(&state, &message, command_name).await;

    let media_summary = summarize_media_files(&media_files);
    let has_images = media_summary.images > 0;
    let has_video = media_summary.videos > 0;
    let has_audio = media_summary.audios > 0;
    let has_documents = media_summary.documents > 0;

    let require_tools = mode.requires_custom_tools();
    let request_capabilities = ModelRequestCapabilities {
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    };
    let third_party_models_available_for_request = has_available_third_party_models_for_request(
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    );
    if has_video
        && !video_request_has_capable_model(
            CONFIG.gemini_api_available(),
            third_party_models_available_for_request,
        )
    {
        reply_with_retry(
            &bot,
            message.chat.id,
            NO_VIDEO_CAPABLE_MODEL_MESSAGE,
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let query_message_is_from_bot = message
        .from
        .as_ref()
        .map(|user| user.is_bot)
        .unwrap_or(false);
    let must_use_default_model = should_use_default_model_without_selection(
        mode,
        request_capabilities,
        !youtube_urls.is_empty(),
        CONFIG.gemini_api_available(),
        third_party_models_available_for_request,
        runtime_model_count(),
        query_message_is_from_bot,
    );
    let direct_model = if must_use_default_model {
        let resolved = if mode == QaCommandMode::Quick {
            resolve_quick_text_model_for_request(has_images, has_video, has_audio, has_documents)
                .await
                .map(|model| {
                    (
                        model.model_id,
                        "default_quick_text_model",
                        model.explicit_codex,
                    )
                })
        } else {
            resolve_default_text_model_for_request(request_capabilities)
                .map(|model| (model, "default_text_model", None))
        };
        match resolved {
            Ok(model) => Some(model),
            Err(err) => {
                reply_with_retry(
                    &bot,
                    message.chat.id,
                    &err.to_string(),
                    Some(message.id),
                    None,
                    None,
                )
                .await?;
                return Ok(());
            }
        }
    } else {
        let selectable_model_ids = selectable_model_ids_for_request(
            has_images,
            has_video,
            has_audio,
            has_documents,
            require_tools,
        );
        if selectable_model_ids.len() == 1 {
            selectable_model_ids
                .into_iter()
                .next()
                .map(|model| (model, "single_selectable_model", None))
        } else {
            None
        }
    };

    if let Some((selected_model, timer_detail, explicit_codex)) = direct_model {
        if mode == QaCommandMode::Quick {
            (query_text, youtube_urls) = prepare_youtube_inputs_for_qa(
                &query_base,
                mode,
                Some(&selected_model),
                CONFIG.gemini_api_available(),
            );
        }
        let display_name = explicit_codex
            .as_ref()
            .map(|explicit| explicit.config.name.clone())
            .unwrap_or_else(|| configured_model_display_name(&selected_model));
        let processing_message_text = if has_video {
            format!(
                "Analyzing video and processing your question with {}...",
                display_name
            )
        } else if has_audio {
            format!(
                "Analyzing audio and processing your question with {}...",
                display_name
            )
        } else if has_images {
            format!(
                "Analyzing {} image(s) and processing your question with {}...",
                media_summary.images, display_name
            )
        } else if has_documents {
            format!(
                "Analyzing {} document(s) and processing your question with {}...",
                media_summary.documents, display_name
            )
        } else if !twitter_contents.is_empty() {
            format!(
                "Analyzing {} Twitter post(s) and processing your question with {}...",
                twitter_contents.len(),
                display_name
            )
        } else if !youtube_urls.is_empty() {
            format!(
                "Analyzing {} YouTube video(s) and processing your question with {}...",
                youtube_urls.len(),
                display_name
            )
        } else {
            format!("Processing your question with {}...", display_name)
        };
        let processing_message = reply_with_retry(
            &bot,
            message.chat.id,
            &processing_message_text,
            Some(message.id),
            None,
            None,
        )
        .await?;
        let mut timer = start_command_timer(command_name, &message);
        let pending_request = PendingQRequest {
            user_id,
            query: query_text.clone(),
            telegram_language_code: user_language_code.map(str::to_string),
            media_files,
            youtube_urls,
            telegraph_contents: telegraph_contents
                .iter()
                .map(|c| c.text_content.clone())
                .collect(),
            twitter_contents: twitter_contents
                .iter()
                .map(|c| c.text_content.clone())
                .collect(),
            chat_id: message.chat.id.0,
            message_id: message.id.0 as i64,
            selection_message_id: processing_message.id.0 as i64,
            original_user_id: user_id,
            llm_invocation_id: audit_context.as_ref().map(|context| context.invocation_id),
            timestamp: now_unix_seconds(),
            command_timer: None,
            mode,
        };

        let result = process_request(
            &bot,
            &state,
            pending_request,
            &selected_model,
            explicit_codex.as_ref(),
            Some(heavy_permit),
        )
        .await;
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(&mut timer, status, Some(timer_detail.to_string()));
        result?;
        return Ok(());
    }

    let has_media = has_images || has_video || has_audio || has_documents;
    let mut selection_text = "Please select which AI model to use for your question:".to_string();
    if has_media {
        selection_text.push_str("\n\n<i>Note: Only models that support media are shown.</i>");
    }

    let keyboard = create_model_selection_keyboard(
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    );
    let selection_message = reply_with_retry(
        &bot,
        message.chat.id,
        &selection_text,
        Some(message.id),
        Some(ParseMode::Html),
        Some(keyboard),
    )
    .await?;

    let request_key = format!("{}_{}", message.chat.id.0, selection_message.id.0);
    let timer = start_command_timer(command_name, &message);

    let pending_request = PendingQRequest {
        user_id,
        query: query_text.clone(),
        telegram_language_code: user_language_code.map(str::to_string),
        media_files,
        youtube_urls,
        telegraph_contents: telegraph_contents
            .iter()
            .map(|c| c.text_content.clone())
            .collect(),
        twitter_contents: twitter_contents
            .iter()
            .map(|c| c.text_content.clone())
            .collect(),
        chat_id: message.chat.id.0,
        message_id: message.id.0 as i64,
        selection_message_id: selection_message.id.0 as i64,
        original_user_id: user_id,
        llm_invocation_id: audit_context.as_ref().map(|context| context.invocation_id),
        timestamp: now_unix_seconds(),
        command_timer: Some(timer),
        mode,
    };

    let timeout_bot = bot.clone();
    let timeout_state = state.clone();
    state.pending_q_requests.insert_with_timeout(
        request_key,
        pending_request,
        Duration::from_secs(CONFIG.model_selection_timeout),
        move |request| async move {
            process_timed_out_q_request_with_default_model(&timeout_bot, &timeout_state, request)
                .await;
        },
    );

    Ok(())
}

pub async fn q_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
    command_name: &str,
) -> Result<()> {
    q_handler_internal(
        bot,
        state,
        message,
        query,
        command_name,
        QaCommandMode::Standard,
    )
    .await
}

pub async fn qc_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    q_handler_internal(bot, state, message, query, "qc", QaCommandMode::ChatContext).await
}

pub async fn s_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "s").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        reply_with_retry(
            &bot,
            message.chat.id,
            "You're sending commands too quickly. Please wait a moment before trying again.",
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let query_text = query
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            message.reply_to_message().and_then(|reply| {
                reply
                    .text()
                    .map(|value| value.to_string())
                    .or_else(|| reply.caption().map(|value| value.to_string()))
            })
        })
        .unwrap_or_default();
    if query_text.trim().is_empty() {
        reply_with_retry(
            &bot,
            message.chat.id,
            "Please provide a search query or reply to a message with /s.",
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let username = message
        .from
        .as_ref()
        .map(|user| user.full_name())
        .unwrap_or_else(|| "Anonymous".to_string());
    let db_insert =
        build_q_command_insert(&message, user_id, &username, &query_text, &query_text, "s");
    if let Err(err) = state.db.queue_message_insert(db_insert).await {
        warn!("Failed to queue /s message insert: {err}");
    }

    if !state.db.is_search_ready() {
        reply_with_retry(
            &bot,
            message.chat.id,
            &chat_search_rebuilding_message("s"),
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }
    let audit_context = create_q_audit_context(&state, &message, "s").await;

    let request_capabilities = ModelRequestCapabilities {
        require_tools: true,
        ..ModelRequestCapabilities::default()
    };
    let third_party_models_available_for_request =
        has_available_third_party_models_for_request(false, false, false, false, true);
    let must_use_default_model = should_use_default_model_without_selection(
        QaCommandMode::ChatSearch,
        request_capabilities,
        false,
        CONFIG.gemini_api_available(),
        third_party_models_available_for_request,
        runtime_model_count(),
        false,
    );
    let direct_model = if must_use_default_model {
        match resolve_default_text_model_for_request(request_capabilities) {
            Ok(model) => Some((model, "default_text_model")),
            Err(err) => {
                reply_with_retry(
                    &bot,
                    message.chat.id,
                    &err.to_string(),
                    Some(message.id),
                    None,
                    None,
                )
                .await?;
                return Ok(());
            }
        }
    } else {
        let selectable_model_ids =
            selectable_model_ids_for_request(false, false, false, false, true);
        if selectable_model_ids.is_empty() {
            reply_with_retry(
                &bot,
                message.chat.id,
                "No tool-capable AI model is available for /s. Enable Gemini or configure a ready third-party model with tools=true.",
                Some(message.id),
                None,
                None,
            )
            .await?;
            return Ok(());
        }
        if selectable_model_ids.len() == 1 {
            selectable_model_ids
                .into_iter()
                .next()
                .map(|model| (model, "single_selectable_model"))
        } else {
            None
        }
    };

    if let Some((selected_model, timer_detail)) = direct_model {
        let display_name = configured_model_display_name(&selected_model);
        let processing_message = reply_with_retry(
            &bot,
            message.chat.id,
            &format!("Searching this chat with {}...", display_name),
            Some(message.id),
            None,
            None,
        )
        .await?;
        let mut timer = start_command_timer("s", &message);
        let pending_request = build_chat_search_pending_request(
            &message,
            user_id,
            &query_text,
            processing_message.id.0 as i64,
            audit_context.as_ref(),
            None,
        );

        let result =
            process_request(&bot, &state, pending_request, &selected_model, None, None).await;
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(&mut timer, status, Some(timer_detail.to_string()));
        result?;
        return Ok(());
    }

    let keyboard = create_model_selection_keyboard(false, false, false, false, true);
    let selection_message = reply_with_retry(
        &bot,
        message.chat.id,
        "Please select which AI model to use for chat search:",
        Some(message.id),
        None,
        Some(keyboard),
    )
    .await?;
    let request_key = format!("{}_{}", message.chat.id.0, selection_message.id.0);
    let timer = start_command_timer("s", &message);
    let pending_request = build_chat_search_pending_request(
        &message,
        user_id,
        &query_text,
        selection_message.id.0 as i64,
        audit_context.as_ref(),
        Some(timer),
    );

    let timeout_bot = bot.clone();
    let timeout_state = state.clone();
    state.pending_q_requests.insert_with_timeout(
        request_key,
        pending_request,
        Duration::from_secs(CONFIG.model_selection_timeout),
        move |request| async move {
            process_timed_out_q_request_with_default_model(&timeout_bot, &timeout_state, request)
                .await;
        },
    );

    Ok(())
}

pub async fn qq_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    q_handler_internal(bot, state, message, query, "qq", QaCommandMode::Quick).await
}

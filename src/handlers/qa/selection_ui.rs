//! Inline-keyboard model picker: callback-data encoding, pending-request
//! resolution, and the `/q`-family callback-query handler.

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, MessageId};
use tracing::error;

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::media::summarize_media_files;
use crate::llm::runtime_models::runtime_models;
use crate::llm::text_model::{
    model_supports_media_for_request, normalize_model_identifier, ready_runtime_providers,
    resolve_default_text_model_for_request, resolve_exact_model_identifier_with_models,
    ModelRequestCapabilities, MODEL_GEMINI,
};
use crate::state::{AppState, PendingEntryGuard, PendingQRequest, QaCommandMode};
use crate::utils::timing::{complete_command_timer, now_unix_seconds};

use super::model_resolution::{
    configured_model_display_name, default_model_selection_key,
    selectable_model_ids_for_request_with_models,
};
use super::process::process_request;

pub const MODEL_CALLBACK_PREFIX: &str = "model_select:";
const MODEL_CALLBACK_COMPACT_PREFIX: &str = "m:";
pub(super) const TELEGRAM_CALLBACK_DATA_LIMIT: usize = 64;

fn compact_model_callback_hash(model_identifier: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in model_identifier.trim().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn compact_model_callback_token(model_identifier: &str) -> String {
    format!(
        "{}{:016x}",
        MODEL_CALLBACK_COMPACT_PREFIX,
        compact_model_callback_hash(model_identifier)
    )
}

fn model_selection_callback_data(model_identifier: &str) -> String {
    let model_identifier = model_identifier.trim();
    let full_callback = format!("{}{}", MODEL_CALLBACK_PREFIX, model_identifier);
    if full_callback.len() <= TELEGRAM_CALLBACK_DATA_LIMIT {
        full_callback
    } else {
        format!(
            "{}{}",
            MODEL_CALLBACK_PREFIX,
            compact_model_callback_token(model_identifier)
        )
    }
}

pub(super) fn resolve_model_callback_token_with_models(
    token: &str,
    models: &[ThirdPartyModelConfig],
) -> Option<String> {
    let token = token.trim();
    if token.eq_ignore_ascii_case(MODEL_GEMINI) {
        return Some(MODEL_GEMINI.to_string());
    }

    if token.starts_with(MODEL_CALLBACK_COMPACT_PREFIX) {
        return models
            .iter()
            .find(|config| compact_model_callback_token(&config.id) == token)
            .map(|config| config.id.clone());
    }

    resolve_exact_model_identifier_with_models(token, models)
}

pub(super) enum PendingQRequestCallbackAction {
    Missing,
    Ignored,
    InvalidSelection,
    UseDefault(PendingQRequest),
    UseSelected(PendingQRequest),
}

pub(super) fn take_pending_q_request_for_callback<F>(
    pending: &mut PendingEntryGuard<'_, PendingQRequest>,
    query_user_id: i64,
    now: i64,
    timeout_secs: u64,
    selected_model_is_allowed: F,
) -> PendingQRequestCallbackAction
where
    F: FnOnce(&PendingQRequest) -> bool,
{
    let Some(request) = pending.get() else {
        return PendingQRequestCallbackAction::Missing;
    };

    if request.original_user_id != query_user_id {
        return PendingQRequestCallbackAction::Ignored;
    }

    let timeout_secs = i64::try_from(timeout_secs).unwrap_or(i64::MAX);
    if now.saturating_sub(request.timestamp) > timeout_secs {
        return pending
            .take()
            .map(PendingQRequestCallbackAction::UseDefault)
            .unwrap_or(PendingQRequestCallbackAction::Missing);
    }

    if !selected_model_is_allowed(request) {
        return PendingQRequestCallbackAction::InvalidSelection;
    }

    pending
        .take()
        .map(PendingQRequestCallbackAction::UseSelected)
        .unwrap_or(PendingQRequestCallbackAction::Missing)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_model_selection_keyboard_with_models(
    models: &[ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    gemini_available: bool,
    default_model: &str,
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> InlineKeyboardMarkup {
    let mut keyboard: Vec<Vec<InlineKeyboardButton>> = Vec::new();
    let default_model_key = default_model_selection_key(default_model, models);
    let selectable_model_ids = selectable_model_ids_for_request_with_models(
        models,
        ready_providers,
        gemini_available,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    );
    let mut model_buttons = selectable_model_ids
        .iter()
        .map(|model_id| {
            InlineKeyboardButton::callback(
                configured_model_display_name(model_id),
                model_selection_callback_data(model_id),
            )
        })
        .collect::<Vec<_>>();

    let default_callback = model_selection_callback_data(&default_model_key);
    if let Some(default_index) = model_buttons.iter().position(|button| match &button.kind {
        teloxide::types::InlineKeyboardButtonKind::CallbackData(data) => data == &default_callback,
        _ => false,
    }) {
        let default_button = model_buttons.remove(default_index);
        model_buttons.insert(0, default_button);
    }

    for chunk in model_buttons.chunks(2) {
        keyboard.push(chunk.to_vec());
    }

    InlineKeyboardMarkup::new(keyboard)
}

pub(super) fn create_model_selection_keyboard(
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> InlineKeyboardMarkup {
    let models = runtime_models();
    let ready_providers = ready_runtime_providers(&models);
    create_model_selection_keyboard_with_models(
        &models,
        &ready_providers,
        CONFIG.gemini_api_available(),
        &CONFIG.default_text_model,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    )
}

pub(super) async fn process_timed_out_q_request_with_default_model(
    bot: &Bot,
    state: &AppState,
    mut request: PendingQRequest,
) {
    let summary = summarize_media_files(&request.media_files);
    let has_images = summary.images > 0;
    let has_video = summary.videos > 0;
    let has_audio = summary.audios > 0;
    let has_documents = summary.documents > 0;
    let default_model = match resolve_default_text_model_for_request(ModelRequestCapabilities {
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools: request.mode.requires_custom_tools(),
    }) {
        Ok(model) => model,
        Err(err) => {
            let _ = bot
                .edit_message_text(
                    ChatId(request.chat_id),
                    MessageId(request.selection_message_id as i32),
                    err.to_string(),
                )
                .reply_markup(InlineKeyboardMarkup::new(
                    Vec::<Vec<InlineKeyboardButton>>::new(),
                ))
                .await;
            if let Some(mut timer) = request.command_timer.take() {
                complete_command_timer(
                    &mut timer,
                    "error",
                    Some("default_text_model_unavailable".to_string()),
                );
            }
            return;
        }
    };

    let _ = bot
        .edit_message_text(
            ChatId(request.chat_id),
            MessageId(request.selection_message_id as i32),
            "No model selected in time. Using default model...",
        )
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await;

    let command_timer = request.command_timer.take();
    let result = process_request(bot, state, request, &default_model, None, None).await;
    if let Some(mut timer) = command_timer {
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(
            &mut timer,
            status,
            Some("timeout_default_model".to_string()),
        );
    }
    if let Err(err) = result {
        error!(
            "Timed-out QA request failed after default-model fallback: model={}, error={:#}",
            configured_model_display_name(&default_model),
            err
        );
    }
}

pub async fn model_selection_callback(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> Result<()> {
    bot.answer_callback_query(query.id.clone()).await?;

    let Some(data) = &query.data else {
        return Ok(());
    };
    if !data.starts_with(MODEL_CALLBACK_PREFIX) {
        return Ok(());
    }

    let selected_token = data.trim_start_matches(MODEL_CALLBACK_PREFIX);
    let models = runtime_models();
    let selected_model = resolve_model_callback_token_with_models(selected_token, &models)
        .unwrap_or_else(|| normalize_model_identifier(selected_token));

    let message = match query.message.clone() {
        Some(msg) => msg,
        None => return Ok(()),
    };

    let request_key = format!("{}_{}", message.chat().id.0, message.id().0);
    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();
    let action = {
        let mut pending = state.pending_q_requests.entry(&request_key);
        take_pending_q_request_for_callback(
            &mut pending,
            query_user_id,
            now_unix_seconds(),
            CONFIG.model_selection_timeout,
            |request| {
                let summary = summarize_media_files(&request.media_files);
                let has_images = summary.images > 0;
                let has_video = summary.videos > 0;
                let has_audio = summary.audios > 0;
                let has_documents = summary.documents > 0;
                model_supports_media_for_request(
                    &selected_model,
                    has_images,
                    has_video,
                    has_audio,
                    has_documents,
                    request.mode.requires_custom_tools(),
                )
            },
        )
    };

    let mut request = match action {
        PendingQRequestCallbackAction::UseSelected(request) => request,
        PendingQRequestCallbackAction::UseDefault(request) => {
            process_timed_out_q_request_with_default_model(&bot, &state, request).await;
            return Ok(());
        }
        PendingQRequestCallbackAction::Missing
        | PendingQRequestCallbackAction::Ignored
        | PendingQRequestCallbackAction::InvalidSelection => return Ok(()),
    };

    let summary = summarize_media_files(&request.media_files);

    let display_name = configured_model_display_name(&selected_model);

    let processing_text = if request.mode == QaCommandMode::ChatSearch {
        format!("Searching this chat with {}...", display_name)
    } else if summary.videos > 0 {
        format!(
            "Analyzing video and processing your question with {}...",
            display_name
        )
    } else if summary.audios > 0 {
        format!(
            "Analyzing audio and processing your question with {}...",
            display_name
        )
    } else if summary.images > 0 {
        format!(
            "Analyzing {} image(s) and processing your question with {}...",
            summary.images, display_name
        )
    } else if summary.documents > 0 {
        format!(
            "Analyzing {} document(s) and processing your question with {}...",
            summary.documents, display_name
        )
    } else {
        format!("Processing your question with {}...", display_name)
    };

    bot.edit_message_text(message.chat().id, message.id(), processing_text)
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await?;

    let command_timer = request.command_timer.take();
    let result = process_request(&bot, &state, request, &selected_model, None, None).await;
    if let Some(mut timer) = command_timer {
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(&mut timer, status, None);
    }

    result?;
    Ok(())
}

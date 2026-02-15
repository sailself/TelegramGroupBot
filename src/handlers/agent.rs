use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{
    InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ParseMode, ReplyParameters,
};
use tracing::{error, warn};

use crate::agent::runtime::{cancel_pending_action, continue_after_confirmation, start_agent_run};
use crate::agent::types::AgentRunOutcome;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::responses::send_response;
use crate::state::AppState;

pub const AGENT_CONFIRM_CALLBACK_PREFIX: &str = "agent_confirm:";
pub const AGENT_CANCEL_CALLBACK_PREFIX: &str = "agent_cancel:";

fn build_confirmation_keyboard(key: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback(
            "Confirm",
            format!("{}{}", AGENT_CONFIRM_CALLBACK_PREFIX, key),
        ),
        InlineKeyboardButton::callback(
            "Cancel",
            format!("{}{}", AGENT_CANCEL_CALLBACK_PREFIX, key),
        ),
    ]])
}

fn build_prompt_from_message(message: &Message, provided_prompt: Option<String>) -> Option<String> {
    let provided = provided_prompt.unwrap_or_default().trim().to_string();
    if !provided.is_empty() {
        return Some(provided);
    }

    let reply = message.reply_to_message()?;
    let text = reply
        .text()
        .or_else(|| reply.caption())
        .unwrap_or("")
        .trim()
        .to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

async fn edit_processing_message(bot: &Bot, chat_id: ChatId, message_id: i64, text: &str) {
    if let Err(err) = bot
        .edit_message_text(chat_id, MessageId(message_id as i32), text.to_string())
        .await
    {
        warn!("Failed to edit agent processing message: {}", err);
    }
}

async fn delete_confirmation_message(bot: &Bot, query: &CallbackQuery) {
    if let Some(message) = query.message.as_ref() {
        if let Err(err) = bot.delete_message(message.chat().id, message.id()).await {
            warn!("Failed to delete agent confirmation message: {}", err);
        }
    }
}

async fn handle_agent_outcome(
    bot: &Bot,
    state: &AppState,
    chat_id: ChatId,
    request_message_id: MessageId,
    processing_message_id: MessageId,
    outcome: AgentRunOutcome,
) -> Result<()> {
    match outcome {
        AgentRunOutcome::Completed {
            session_id,
            response_text,
            selected_skills,
        } => {
            let mut response = response_text;
            if !selected_skills.is_empty() {
                response.push_str("\n\nSkills: ");
                response.push_str(&selected_skills.join(", "));
            }
            response.push_str(&format!("\nSession: {}", session_id));
            send_response(
                bot,
                chat_id,
                processing_message_id,
                &response,
                "Agent Response",
                ParseMode::MarkdownV2,
            )
            .await?;
        }
        AgentRunOutcome::AwaitingConfirmation {
            confirmation_key,
            notice_text,
            ..
        } => {
            bot.send_message(chat_id, notice_text)
                .reply_parameters(ReplyParameters::new(request_message_id))
                .reply_markup(build_confirmation_keyboard(&confirmation_key))
                .await?;

            let pending_processing_id = {
                state
                    .pending_agent_actions
                    .lock()
                    .get(&confirmation_key)
                    .map(|pending| pending.processing_message_id)
            };
            if let Some(processing_message_id) = pending_processing_id {
                edit_processing_message(
                    bot,
                    chat_id,
                    processing_message_id,
                    "Awaiting confirmation for a side-effect tool call.",
                )
                .await;
            }
        }
    }

    Ok(())
}

pub async fn agent_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    prompt: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "agent").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let Some(prompt_text) = build_prompt_from_message(&message, prompt) else {
        bot.send_message(
            message.chat.id,
            "Usage: /agent <prompt>\nOr reply to a text message with /agent",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    };

    let processing_message = bot
        .send_message(message.chat.id, "Running agent...")
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;

    let outcome = start_agent_run(
        &state,
        user_id,
        message.chat.id.0,
        processing_message.id.0 as i64,
        &prompt_text,
    )
    .await;

    match outcome {
        Ok(result) => {
            handle_agent_outcome(
                &bot,
                &state,
                message.chat.id,
                message.id,
                processing_message.id,
                result,
            )
            .await?;
        }
        Err(err) => {
            let error_text = format!("Agent run failed: {}", err);
            error!("{}", error_text);
            edit_processing_message(
                &bot,
                message.chat.id,
                processing_message.id.0 as i64,
                &error_text,
            )
            .await;
        }
    }

    Ok(())
}

pub async fn agent_confirmation_callback(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> Result<()> {
    bot.answer_callback_query(query.id.clone()).await?;

    let Some(data) = query.data.clone() else {
        return Ok(());
    };

    let (is_confirm, key) = if let Some(key) = data.strip_prefix(AGENT_CONFIRM_CALLBACK_PREFIX) {
        (true, key.to_string())
    } else if let Some(key) = data.strip_prefix(AGENT_CANCEL_CALLBACK_PREFIX) {
        (false, key.to_string())
    } else {
        return Ok(());
    };

    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();

    let pending_preview = {
        let pending_map = state.pending_agent_actions.lock();
        pending_map.get(&key).cloned()
    };
    let Some(pending_preview) = pending_preview else {
        if let Some(message) = query.message {
            bot.edit_message_text(
                message.chat().id,
                message.id(),
                "This confirmation has expired.",
            )
            .await?;
        }
        return Ok(());
    };

    if pending_preview.user_id != query_user_id {
        return Ok(());
    }

    let pending = {
        let mut pending_map = state.pending_agent_actions.lock();
        pending_map.remove(&key)
    };
    let Some(pending) = pending else {
        return Ok(());
    };

    if !is_confirm {
        cancel_pending_action(&state, &pending).await?;
        delete_confirmation_message(&bot, &query).await;
        edit_processing_message(
            &bot,
            ChatId(pending.chat_id),
            pending.processing_message_id,
            "Cancelled side-effect tool execution.",
        )
        .await;
        return Ok(());
    }

    delete_confirmation_message(&bot, &query).await;

    let outcome = continue_after_confirmation(&state, pending.clone(), query_user_id).await;
    match outcome {
        Ok(result) => {
            handle_agent_outcome(
                &bot,
                &state,
                ChatId(pending.chat_id),
                MessageId(pending.processing_message_id as i32),
                MessageId(pending.processing_message_id as i32),
                result,
            )
            .await?;
        }
        Err(err) => {
            let text = format!("Failed to continue agent run: {}", err);
            warn!("{}", text);
            edit_processing_message(
                &bot,
                ChatId(pending.chat_id),
                pending.processing_message_id,
                &text,
            )
            .await;
        }
    }

    Ok(())
}

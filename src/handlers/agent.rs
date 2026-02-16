use anyhow::Result;
use serde_json::from_str;
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, MessageEntityRef, MessageId, ParseMode,
    ReplyParameters,
};
use tracing::{error, warn};

use crate::agent::runtime::{cancel_pending_action, continue_after_confirmation, start_agent_run};
use crate::agent::types::AgentRunOutcome;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::content::{
    extract_telegraph_urls_and_content, extract_twitter_urls_and_content,
};
use crate::handlers::media::{
    collect_message_media, summarize_media_files, MediaCollectionOptions,
};
use crate::handlers::responses::send_response;
use crate::state::AppState;
use crate::utils::telegram::start_chat_action_heartbeat;

pub const AGENT_CONFIRM_CALLBACK_PREFIX: &str = "agent_confirm:";
pub const AGENT_CANCEL_CALLBACK_PREFIX: &str = "agent_cancel:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentPromptSource {
    Provided,
    Reply,
}

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

fn message_entities_for_text(message: &Message) -> Option<Vec<MessageEntityRef<'_>>> {
    if message.text().is_some() {
        message.parse_entities()
    } else {
        message.parse_caption_entities()
    }
}

fn build_prompt_from_message(
    message: &Message,
    provided_prompt: Option<String>,
) -> Option<(String, AgentPromptSource)> {
    let provided = provided_prompt.unwrap_or_default().trim().to_string();
    if !provided.is_empty() {
        return Some((provided, AgentPromptSource::Provided));
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
        Some((text, AgentPromptSource::Reply))
    }
}

async fn preprocess_agent_prompt(
    message: &Message,
    prompt_text: &str,
    source: AgentPromptSource,
) -> String {
    let mut prompt_processed = prompt_text.to_string();

    match source {
        AgentPromptSource::Provided => {
            let query_entities = message_entities_for_text(message);
            let (query_processed, query_telegraph) =
                extract_telegraph_urls_and_content(&prompt_processed, query_entities.as_deref(), 5)
                    .await;
            let (query_processed, query_twitter) =
                extract_twitter_urls_and_content(&query_processed, query_entities.as_deref(), 5)
                    .await;
            prompt_processed = query_processed;
            let _ = query_telegraph;
            let _ = query_twitter;

            if let Some(reply) = message.reply_to_message() {
                let reply_raw = reply
                    .text()
                    .or_else(|| reply.caption())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !reply_raw.is_empty() {
                    let reply_entities = message_entities_for_text(reply);
                    let (reply_processed, reply_telegraph) = extract_telegraph_urls_and_content(
                        &reply_raw,
                        reply_entities.as_deref(),
                        5,
                    )
                    .await;
                    let (reply_processed, reply_twitter) = extract_twitter_urls_and_content(
                        &reply_processed,
                        reply_entities.as_deref(),
                        5,
                    )
                    .await;
                    let _ = reply_telegraph;
                    let _ = reply_twitter;
                    prompt_processed = format!(
                        "Context from replied message: \"{}\"\n\nTask: {}",
                        reply_processed, prompt_processed
                    );
                }
            }
        }
        AgentPromptSource::Reply => {
            if let Some(reply) = message.reply_to_message() {
                let reply_entities = message_entities_for_text(reply);
                let (reply_processed, reply_telegraph) = extract_telegraph_urls_and_content(
                    &prompt_processed,
                    reply_entities.as_deref(),
                    5,
                )
                .await;
                let (reply_processed, reply_twitter) = extract_twitter_urls_and_content(
                    &reply_processed,
                    reply_entities.as_deref(),
                    5,
                )
                .await;
                prompt_processed = reply_processed;
                let _ = reply_telegraph;
                let _ = reply_twitter;
            } else {
                let query_entities = message_entities_for_text(message);
                let (query_processed, query_telegraph) = extract_telegraph_urls_and_content(
                    &prompt_processed,
                    query_entities.as_deref(),
                    5,
                )
                .await;
                let (query_processed, query_twitter) = extract_twitter_urls_and_content(
                    &query_processed,
                    query_entities.as_deref(),
                    5,
                )
                .await;
                prompt_processed = query_processed;
                let _ = query_telegraph;
                let _ = query_twitter;
            }
        }
    }

    prompt_processed
}

fn parse_resume_argument(raw: &str) -> Option<(i64, Option<String>)> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (session_part, tail) = match trimmed.find(char::is_whitespace) {
        Some(index) => (&trimmed[..index], Some(trimmed[index..].trim().to_string())),
        None => (trimmed, None),
    };
    let session_id = session_part.parse::<i64>().ok()?;
    if session_id <= 0 {
        return None;
    }
    let instruction = tail.filter(|value| !value.is_empty());
    Some((session_id, instruction))
}

fn summarize_text(value: &str, max_chars: usize) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        return normalized;
    }
    let mut out = String::new();
    for ch in normalized.chars().take(max_chars) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn build_resume_prompt(
    session_id: i64,
    original_prompt: &str,
    steps: &[crate::db::models::AgentStepRow],
    instruction: Option<&str>,
) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Resume context from previous session #{}.",
        session_id
    ));
    lines.push(format!(
        "Original task: {}",
        summarize_text(original_prompt, 400)
    ));
    lines.push("Recent transcript:".to_string());

    if steps.is_empty() {
        lines.push("- (no recorded steps)".to_string());
    } else {
        for step in steps {
            let content = step
                .content
                .as_deref()
                .map(|value| summarize_text(value, 320))
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "(empty)".to_string());
            lines.push(format!("- [{}] {}", step.role, content));
        }
    }

    if let Some(instruction) = instruction {
        lines.push(format!(
            "New instruction: {}",
            summarize_text(instruction, 600)
        ));
    } else {
        lines.push(
            "New instruction: Continue from this context and produce the next best response."
                .to_string(),
        );
    }

    lines.join("\n")
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

    let prompt_context = build_prompt_from_message(&message, prompt);
    let media =
        collect_message_media(&bot, &state, &message, MediaCollectionOptions::for_qa()).await;
    let media_files = media.files;

    if prompt_context.is_none() && media_files.is_empty() {
        bot.send_message(
            message.chat.id,
            "Usage: /agent <prompt>\nOr reply to a text message with /agent",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let (prompt_text_raw, prompt_source) =
        prompt_context.unwrap_or_else(|| (String::new(), AgentPromptSource::Provided));
    let mut prompt_text = if prompt_text_raw.trim().is_empty() {
        String::new()
    } else {
        preprocess_agent_prompt(&message, &prompt_text_raw, prompt_source).await
    };
    if prompt_text.trim().is_empty() && !media_files.is_empty() {
        prompt_text = "Analyze the attached media and answer the user request. If no specific question was provided, give a concise summary of the media.".to_string();
    }

    if prompt_text.trim().is_empty() {
        bot.send_message(
            message.chat.id,
            "Please provide a task or reply to a text message with /agent.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let media_summary = summarize_media_files(&media_files);
    let processing_text = if media_summary.total > 0 {
        format!(
            "Running agent with {} attachment(s) (images: {}, videos: {}, audio: {}, documents: {})...",
            media_summary.total,
            media_summary.images,
            media_summary.videos,
            media_summary.audios,
            media_summary.documents
        )
    } else {
        "Running agent...".to_string()
    };
    let processing_message = bot
        .send_message(message.chat.id, processing_text)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;

    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
    let outcome = start_agent_run(
        &state,
        user_id,
        message.chat.id.0,
        processing_message.id.0 as i64,
        &prompt_text,
        media_files,
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

pub async fn agent_status_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "agent").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();

    let sessions = state
        .db
        .list_recent_agent_sessions_for_user(message.chat.id.0, user_id, 5)
        .await?;
    let pending_count = {
        state
            .pending_agent_actions
            .lock()
            .values()
            .filter(|pending| pending.chat_id == message.chat.id.0 && pending.user_id == user_id)
            .count()
    };

    let mut lines = Vec::new();
    lines.push("Agent session status".to_string());
    lines.push(format!("pending_confirmations: {}", pending_count));

    if sessions.is_empty() {
        lines.push("No previous sessions found for this chat/user.".to_string());
        lines.push("Start with: /agent <task>".to_string());
    } else {
        lines.push("Recent sessions:".to_string());
        for session in sessions {
            let skills = session
                .selected_skills_json
                .as_deref()
                .and_then(|raw| from_str::<Vec<String>>(raw).ok())
                .unwrap_or_default();
            let skill_text = if skills.is_empty() {
                "-".to_string()
            } else {
                summarize_text(&skills.join(", "), 80)
            };
            lines.push(format!(
                "- #{} [{}] model={} updated={} skills={}",
                session.id, session.status, session.model_name, session.updated_at, skill_text
            ));
        }
    }

    bot.send_message(message.chat.id, lines.join("\n"))
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    Ok(())
}

pub async fn agent_resume_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    arg: Option<String>,
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

    let Some(raw_arg) = arg else {
        bot.send_message(
            message.chat.id,
            "Usage: /agent_resume <session_id> [new instruction]",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    };

    let Some((session_id, instruction)) = parse_resume_argument(&raw_arg) else {
        bot.send_message(
            message.chat.id,
            "Usage: /agent_resume <session_id> [new instruction]",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    };

    let Some(session) = state
        .db
        .get_agent_session_for_user(session_id, message.chat.id.0, user_id)
        .await?
    else {
        bot.send_message(
            message.chat.id,
            format!(
                "Session #{} was not found in this chat for your account.",
                session_id
            ),
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    };

    let steps = state.db.list_agent_steps(session_id, 12).await?;
    let resumed_prompt =
        build_resume_prompt(session_id, &session.prompt, &steps, instruction.as_deref());

    let processing_message = bot
        .send_message(
            message.chat.id,
            format!("Resuming session #{}...", session_id),
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;

    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
    let outcome = start_agent_run(
        &state,
        user_id,
        message.chat.id.0,
        processing_message.id.0 as i64,
        &resumed_prompt,
        Vec::new(),
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
            let error_text = format!("Agent resume failed: {}", err);
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

pub async fn agent_new_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "agent").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();

    let pending_to_cancel = {
        let mut pending_map = state.pending_agent_actions.lock();
        let keys = pending_map
            .iter()
            .filter(|(_, pending)| {
                pending.chat_id == message.chat.id.0 && pending.user_id == user_id
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();

        let mut removed = Vec::new();
        for key in keys {
            if let Some(pending) = pending_map.remove(&key) {
                removed.push(pending);
            }
        }
        removed
    };

    for pending in &pending_to_cancel {
        if let Err(err) = cancel_pending_action(&state, pending).await {
            warn!(
                "Failed to cancel pending action for session {}: {}",
                pending.session_id, err
            );
        }
    }

    let superseded = state
        .db
        .supersede_active_agent_sessions(message.chat.id.0, user_id)
        .await?;

    let text = format!(
        "Started a fresh agent lane.\nCancelled pending confirmations: {}\nSuperseded active sessions: {}\nUse /agent <task> to begin.",
        pending_to_cancel.len(),
        superseded
    );
    bot.send_message(message.chat.id, text)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
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

    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), ChatId(pending.chat_id), ChatAction::Typing);
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

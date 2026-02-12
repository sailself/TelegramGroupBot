use std::time::Duration;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};
use tracing::{error, warn};

use crate::config::CONFIG;
use crate::db::database::build_message_insert;
use crate::handlers::content::create_telegraph_page;
use crate::rag::client::{build_ingest_item, ingest_messages, is_rag_enabled};
use crate::state::AppState;
use crate::utils::language::detect_language_or_fallback;

async fn edit_text_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    text: &str,
    parse_mode: Option<ParseMode>,
) -> Result<()> {
    let mut delay = Duration::from_secs_f32(1.5);
    for attempt in 0..3 {
        let request = bot.edit_message_text(chat_id, message_id, text.to_string());
        let request = if let Some(mode) = parse_mode {
            request.parse_mode(mode)
        } else {
            request
        };

        match request.await {
            Ok(_) => return Ok(()),
            Err(err) => {
                if attempt == 2 {
                    return Err(err.into());
                }
                warn!("edit_message_text failed: {err}");
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
    }

    Ok(())
}

#[allow(deprecated)]
pub async fn send_response(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    response: &str,
    title: &str,
    parse_mode: ParseMode,
) -> Result<()> {
    let line_count = response.lines().count();

    if line_count > 22 || response.len() > CONFIG.telegram_max_length {
        let telegraph_url = create_telegraph_page(title, response).await;
        if let Some(url) = telegraph_url {
            edit_text_with_retry(
                bot,
                chat_id,
                message_id,
                &format!("I have too much to say. [View it here]({})", url),
                Some(ParseMode::Markdown),
            )
            .await?;
            return Ok(());
        }

        let truncated = if response.len() > CONFIG.telegram_max_length {
            format!(
                "{}...\n\n(Response was truncated due to length)",
                &response[..CONFIG.telegram_max_length.saturating_sub(100)]
            )
        } else {
            response.to_string()
        };
        edit_text_with_retry(bot, chat_id, message_id, &truncated, None).await?;
        return Ok(());
    }

    if let Err(err) =
        edit_text_with_retry(bot, chat_id, message_id, response, Some(parse_mode)).await
    {
        warn!("Failed to send formatted response: {err}");
        edit_text_with_retry(bot, chat_id, message_id, response, None).await?;
    }

    Ok(())
}

pub async fn log_message(state: &AppState, message: &Message) {
    let text = message
        .text()
        .map(|value| value.to_string())
        .or_else(|| message.caption().map(|value| value.to_string()));

    let Some(text) = text else {
        return;
    };

    let user_language_code = message
        .from
        .as_ref()
        .and_then(|user| user.language_code.as_deref());
    let language = detect_language_or_fallback(&[&text], user_language_code, "Chinese");

    let username = if let Some(user) = message.from.as_ref() {
        if !user.full_name().is_empty() {
            user.full_name()
        } else if let Some(username) = &user.username {
            username.clone()
        } else {
            "Anonymous".to_string()
        }
    } else {
        "Anonymous".to_string()
    };

    let insert = build_message_insert(
        message
            .from
            .as_ref()
            .and_then(|user| i64::try_from(user.id.0).ok()),
        Some(username.clone()),
        Some(text.clone()),
        Some(language.clone()),
        message.date,
        message.reply_to_message().map(|msg| msg.id.0 as i64),
        Some(message.chat.id.0),
        Some(message.id.0 as i64),
    );

    if let Err(err) = state.db.queue_message_insert(insert).await {
        error!("Failed to queue message insert: {err}");
    }

    if !is_rag_enabled() {
        return;
    }

    if let Some(item) = build_ingest_item(
        message.id.0 as i64,
        message
            .from
            .as_ref()
            .and_then(|user| i64::try_from(user.id.0).ok()),
        Some(username),
        message.date,
        message.reply_to_message().map(|msg| msg.id.0 as i64),
        &text,
    ) {
        let chat_id = message.chat.id.0;
        tokio::spawn(async move {
            if let Err(err) = ingest_messages(chat_id, vec![item]).await {
                warn!("Failed to ingest live message into RAG service: {err}");
            }
        });
    }
}

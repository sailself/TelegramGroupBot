use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};
use tracing::{error, warn};

use crate::config::CONFIG;
use crate::db::database::build_message_insert;
use crate::db::search::derive_search_provenance;
use crate::handlers::content::create_telegraph_page;
use crate::state::AppState;
use crate::utils::telegram::retry_telegram;
use crate::utils::text::truncate_to_chars;

/// Edit a message, retrying only transient Telegram failures. Permanent
/// rejections (bad markup, unmodified text) surface immediately so callers
/// can fall back to plain text without a multi-second retry delay.
async fn edit_text_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    text: &str,
    parse_mode: Option<ParseMode>,
) -> Result<()> {
    retry_telegram("edit_message_text", || {
        let request = bot.edit_message_text(chat_id, message_id, text.to_string());
        match parse_mode {
            Some(mode) => request.parse_mode(mode),
            None => request,
        }
    })
    .await?;
    Ok(())
}

const OVERFLOW_LINE_LIMIT: usize = 22;
const OVERFLOW_TRUNCATION_MARGIN: usize = 100;

/// Whether a response is too long for a single Telegram message and should
/// go to Telegraph (or be truncated when Telegraph is unavailable).
fn needs_overflow_delivery(response: &str, max_chars: usize) -> bool {
    response.lines().count() > OVERFLOW_LINE_LIMIT || response.chars().count() > max_chars
}

/// Plain-text fallback used when Telegraph publishing fails.
fn overflow_fallback_text(response: &str, max_chars: usize) -> String {
    if response.chars().count() > max_chars {
        format!(
            "{}...\n\n(Response was truncated due to length)",
            truncate_to_chars(
                response,
                max_chars.saturating_sub(OVERFLOW_TRUNCATION_MARGIN)
            )
        )
    } else {
        response.to_string()
    }
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
    if needs_overflow_delivery(response, CONFIG.telegram_max_length) {
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

        let truncated = overflow_fallback_text(response, CONFIG.telegram_max_length);
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

    let provenance = derive_search_provenance(&text);
    let insert = build_message_insert(
        message
            .from
            .as_ref()
            .and_then(|user| i64::try_from(user.id.0).ok()),
        Some(username),
        Some(text.clone()),
        None,
        message.date,
        message.reply_to_message().map(|msg| msg.id.0 as i64),
        Some(message.chat.id.0),
        Some(message.id.0 as i64),
        None,
        provenance.asks_ai,
        provenance.ai_command,
        provenance.is_command,
        false,
    );

    if let Err(err) = state.db.queue_message_insert(insert).await {
        error!("Failed to queue message insert: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_detection_counts_characters_not_bytes() {
        // 2,000 CJK chars are 6,000 bytes but fit in a 4,000-char message.
        let response = "好".repeat(2_000);
        assert!(!needs_overflow_delivery(&response, 4_000));
        assert!(needs_overflow_delivery(&"好".repeat(4_001), 4_000));
        assert!(needs_overflow_delivery(&"x\n".repeat(30), 4_000));
    }

    #[test]
    fn overflow_fallback_truncates_on_char_boundary() {
        // 1 ASCII byte then 3-byte chars: the old byte slice at 3,900 landed
        // mid-character and panicked.
        let response = format!("a{}", "好".repeat(4_000));
        let truncated = overflow_fallback_text(&response, 4_000);
        assert!(truncated.ends_with("(Response was truncated due to length)"));
        let body = truncated.split("...").next().unwrap();
        assert_eq!(body.chars().count(), 3_900);
    }

    #[test]
    fn overflow_fallback_returns_short_response_unchanged() {
        assert_eq!(overflow_fallback_text("short", 4_000), "short");
    }
}

use std::future::IntoFuture;
use std::time::Duration;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, InlineKeyboardMarkup, MessageEntityRef, MessageId, ParseMode, ReplyParameters,
};
use teloxide::RequestError;
use tokio::task::JoinHandle;
use tracing::warn;

const CHAT_ACTION_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);

/// How many times a Telegram request is attempted before its error is
/// returned to the caller.
pub const TELEGRAM_RETRY_ATTEMPTS: usize = 3;
const TELEGRAM_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(1_500);

/// Whether a Telegram request failure is transient and worth retrying.
/// API rejections (bad markup, unmodified message, missing rights) are
/// permanent and must surface immediately.
pub fn telegram_error_is_retryable(err: &RequestError) -> bool {
    matches!(
        err,
        RequestError::Network(_) | RequestError::RetryAfter(_) | RequestError::Io(_)
    )
}

/// Run a Telegram request built by `make_request`, retrying transient
/// failures up to [`TELEGRAM_RETRY_ATTEMPTS`] times. Flood-control errors
/// wait the server-specified interval; other transient errors back off
/// exponentially from 1.5s. Permanent errors are returned at once.
pub async fn retry_telegram<T, R, F>(op_name: &str, mut make_request: F) -> Result<T, RequestError>
where
    F: FnMut() -> R,
    R: IntoFuture<Output = Result<T, RequestError>>,
{
    let mut delay = TELEGRAM_RETRY_INITIAL_BACKOFF;
    let mut attempt = 0;
    loop {
        attempt += 1;
        match make_request().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if !telegram_error_is_retryable(&err) || attempt >= TELEGRAM_RETRY_ATTEMPTS {
                    return Err(err);
                }
                warn!("{op_name} attempt {attempt} failed: {err}");
                if let RequestError::RetryAfter(wait) = err {
                    tokio::time::sleep(wait.duration()).await;
                } else {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }
    }
}

pub struct ChatActionHeartbeat {
    task_handle: Option<JoinHandle<()>>,
}

impl Drop for ChatActionHeartbeat {
    fn drop(&mut self) {
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
    }
}

pub fn start_chat_action_heartbeat(
    bot: Bot,
    chat_id: ChatId,
    action: ChatAction,
) -> ChatActionHeartbeat {
    let task_handle = tokio::spawn(async move {
        loop {
            if let Err(err) = bot.send_chat_action(chat_id, action).await {
                warn!("send_chat_action failed: {err}");
            }
            tokio::time::sleep(CHAT_ACTION_HEARTBEAT_INTERVAL).await;
        }
    });

    ChatActionHeartbeat {
        task_handle: Some(task_handle),
    }
}

pub fn normalize_supergroup_chat_id_for_link(chat_id: i64) -> Option<String> {
    let raw = chat_id.to_string();
    if let Some(normalized) = raw.strip_prefix("-100") {
        if normalized.is_empty() {
            None
        } else {
            Some(normalized.to_string())
        }
    } else {
        None
    }
}

pub fn build_message_link(chat_id: i64, message_id: i64) -> Option<String> {
    if let Some(normalized_chat_id) = normalize_supergroup_chat_id_for_link(chat_id) {
        return Some(format!(
            "https://t.me/c/{}/{}",
            normalized_chat_id, message_id
        ));
    }

    if chat_id > 0 {
        return Some(format!(
            "tg://openmessage?user_id={}&message_id={}",
            chat_id, message_id
        ));
    }

    None
}

pub(crate) fn strip_command_prefix(text: &str, command_prefix: &str) -> String {
    let Some(stripped) = text.strip_prefix(command_prefix) else {
        return text.to_string();
    };
    // Telegram appends `@botname` to the command when it is addressed to a
    // specific bot (`/img@MyBot ...`); that mention is not part of the prompt.
    let stripped = match stripped.strip_prefix('@') {
        Some(after_at) => {
            after_at.trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_')
        }
        None => stripped,
    };
    stripped.trim().to_string()
}

pub(crate) fn message_entities_for_text(message: &Message) -> Option<Vec<MessageEntityRef<'_>>> {
    if message.text().is_some() {
        message.parse_entities()
    } else {
        message.parse_caption_entities()
    }
}

pub(crate) fn message_text_or_caption(message: &Message) -> Option<&str> {
    message.text().or_else(|| message.caption())
}

pub(crate) async fn send_message_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
) -> Result<Message> {
    send_message_with_retry_parse_mode(bot, chat_id, text, reply_to, None).await
}

pub(crate) async fn send_message_with_retry_parse_mode(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
    parse_mode: Option<ParseMode>,
) -> Result<Message> {
    retry_telegram("send_message", || {
        let mut request = bot.send_message(chat_id, text.to_string());
        if let Some(reply_to) = reply_to {
            request = request.reply_parameters(ReplyParameters::new(reply_to));
        }
        if let Some(parse_mode) = parse_mode {
            request = request.parse_mode(parse_mode);
        }
        request
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn edit_message_text_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    text: &str,
) -> Result<()> {
    retry_telegram("edit_message_text", || {
        bot.edit_message_text(chat_id, message_id, text.to_string())
    })
    .await?;
    Ok(())
}

/// Like [`send_message_with_retry_parse_mode`] but also supports an inline
/// keyboard reply markup; used by the interactive `/q` flows that attach
/// model-selection buttons.
pub(crate) async fn reply_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
    parse_mode: Option<ParseMode>,
    reply_markup: Option<InlineKeyboardMarkup>,
) -> Result<Message> {
    retry_telegram("send_message", || {
        let mut request = bot.send_message(chat_id, text.to_string());
        if let Some(reply_to) = reply_to {
            request = request.reply_parameters(ReplyParameters::new(reply_to));
        }
        if let Some(mode) = parse_mode {
            request = request.parse_mode(mode);
        }
        if let Some(markup) = reply_markup.clone() {
            request = request.reply_markup(markup);
        }
        request
    })
    .await
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use teloxide::types::Seconds;
    use teloxide::{ApiError, RequestError};

    use super::*;

    fn io_error() -> RequestError {
        RequestError::Io(Arc::new(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "reset",
        )))
    }

    #[tokio::test(start_paused = true)]
    async fn retry_telegram_retries_transient_errors_and_returns_the_first_success() {
        let calls = AtomicUsize::new(0);
        let result = retry_telegram("send_message", || {
            let attempt = calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(if attempt < 2 { Err(io_error()) } else { Ok(42) })
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_telegram_returns_permanent_errors_without_retrying() {
        let calls = AtomicUsize::new(0);
        let result: Result<(), RequestError> = retry_telegram("edit_message_text", || {
            calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Err(RequestError::Api(ApiError::MessageNotModified)))
        })
        .await;

        assert!(matches!(
            result,
            Err(RequestError::Api(ApiError::MessageNotModified))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_telegram_gives_up_after_the_attempt_budget() {
        let calls = AtomicUsize::new(0);
        let result: Result<(), RequestError> = retry_telegram("send_video", || {
            calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Err(io_error()))
        })
        .await;

        assert!(matches!(result, Err(RequestError::Io(_))));
        assert_eq!(calls.load(Ordering::SeqCst), TELEGRAM_RETRY_ATTEMPTS);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_telegram_waits_out_flood_control_before_retrying() {
        let started = tokio::time::Instant::now();
        let calls = AtomicUsize::new(0);
        let result = retry_telegram("send_message", || {
            let attempt = calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(if attempt == 0 {
                Err(RequestError::RetryAfter(Seconds::from_seconds(7)))
            } else {
                Ok(())
            })
        })
        .await;

        assert!(result.is_ok());
        assert!(started.elapsed() >= Duration::from_secs(7));
    }

    #[test]
    fn normalizes_supergroup_chat_ids_for_links() {
        assert_eq!(
            normalize_supergroup_chat_id_for_link(-1001374348669),
            Some("1374348669".to_string())
        );
        assert_eq!(
            build_message_link(-1001374348669, 2229136),
            Some("https://t.me/c/1374348669/2229136".to_string())
        );
    }

    #[test]
    fn rejects_non_supergroup_chat_ids_for_links() {
        assert_eq!(normalize_supergroup_chat_id_for_link(-4679676827), None);
        assert_eq!(build_message_link(-4679676827, 42), None);
    }

    #[test]
    fn builds_private_chat_deep_links_for_user_chats() {
        assert_eq!(
            build_message_link(351987360, 42),
            Some("tg://openmessage?user_id=351987360&message_id=42".to_string())
        );
    }

    #[test]
    fn strip_command_prefix_removes_command_and_attached_bot_mention() {
        assert_eq!(strip_command_prefix("/img a cat", "/img"), "a cat");
        assert_eq!(strip_command_prefix("/img@MyBot a cat", "/img"), "a cat");
        assert_eq!(strip_command_prefix("/image@My_Bot2", "/image"), "");
    }

    #[test]
    fn strip_command_prefix_keeps_mentions_inside_the_prompt() {
        assert_eq!(
            strip_command_prefix("/img @alice as a knight", "/img"),
            "@alice as a knight"
        );
        assert_eq!(strip_command_prefix("draw a cat", "/img"), "draw a cat");
    }
}

use std::future::IntoFuture;
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::ChatAction;
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
}

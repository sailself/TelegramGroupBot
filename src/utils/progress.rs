//! Throttled progress updates for long-running agentic commands.
//!
//! Edits the command's processing message with brief phase descriptions
//! ("Researching claim 2/4…"). Edits are best-effort: failures are logged and
//! swallowed, repeated edits are throttled, and unchanged text is skipped so
//! the bot never trips Telegram flood control over cosmetic updates. Final
//! answers and error messages are NOT delivered through this type — they keep
//! going through `send_response` / the handlers' error edits.

use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use teloxide::prelude::*;
use teloxide::types::MessageId;
use teloxide::RequestError;
use tracing::{debug, warn};

const MIN_EDIT_INTERVAL: Duration = Duration::from_millis(2_500);
const UPDATE_NOW_MAX_ATTEMPTS: usize = 2;

/// Minimal seam so a caller that reports per-item progress (e.g.
/// `agents::common::map_bounded`) can be driven by a fake sink in tests
/// instead of a real `ProgressReporter` bound to a live Telegram message.
pub trait ProgressSink {
    fn update_text<'a>(&'a mut self, text: String)
        -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

pub struct ProgressReporter {
    bot: Bot,
    chat_id: ChatId,
    message_id: MessageId,
    min_interval: Duration,
    /// Earliest moment the next edit may be sent (advanced on success and on
    /// `RetryAfter`, so flood-control waits are respected across updates).
    next_edit_allowed_at: Option<Instant>,
    last_text: String,
    note: Option<String>,
}

pub(crate) fn status_with_note<'a>(text: &'a str, note: Option<&str>) -> Cow<'a, str> {
    match note {
        Some(note) => Cow::Owned(format!("{text}\n\n{note}")),
        None => Cow::Borrowed(text),
    }
}

/// Pure throttle decision: emit only when the text changed and the
/// next-allowed instant has passed.
fn should_emit(
    next_edit_allowed_at: Option<Instant>,
    now: Instant,
    last_text: &str,
    new_text: &str,
) -> bool {
    if new_text.trim().is_empty() || new_text == last_text {
        return false;
    }
    match next_edit_allowed_at {
        Some(allowed_at) => now >= allowed_at,
        None => true,
    }
}

impl ProgressReporter {
    pub fn new(bot: Bot, chat_id: ChatId, message_id: MessageId) -> Self {
        Self {
            bot,
            chat_id,
            message_id,
            min_interval: MIN_EDIT_INTERVAL,
            next_edit_allowed_at: None,
            last_text: String::new(),
            note: None,
        }
    }

    pub(crate) fn with_note(mut self, note: Option<&str>) -> Self {
        self.note = note.map(str::to_owned);
        self
    }

    /// Throttled, best-effort progress edit. Skips when called too soon after
    /// the previous edit or when the text is unchanged; never returns an error.
    pub async fn update(&mut self, text: &str) {
        if !should_emit(
            self.next_edit_allowed_at,
            Instant::now(),
            &self.last_text,
            text,
        ) {
            debug!("progress update skipped (throttled or unchanged): {text}");
            return;
        }
        self.try_edit(text).await;
    }

    /// Progress edit that bypasses the throttle, for phase-terminal states.
    /// Still best-effort, but retries once honoring `RetryAfter`.
    pub async fn update_now(&mut self, text: &str) {
        if text.trim().is_empty() || text == self.last_text {
            return;
        }
        for attempt in 1..=UPDATE_NOW_MAX_ATTEMPTS {
            if self.try_edit(text).await {
                return;
            }
            if attempt < UPDATE_NOW_MAX_ATTEMPTS {
                if let Some(allowed_at) = self.next_edit_allowed_at {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(allowed_at)).await;
                }
            }
        }
    }

    async fn try_edit(&mut self, text: &str) -> bool {
        match self
            .bot
            .edit_message_text(
                self.chat_id,
                self.message_id,
                status_with_note(text, self.note.as_deref()).into_owned(),
            )
            .await
        {
            Ok(_) => {
                self.last_text = text.to_string();
                self.next_edit_allowed_at = Some(Instant::now() + self.min_interval);
                true
            }
            Err(RequestError::RetryAfter(wait)) => {
                let wait = wait.duration().max(self.min_interval);
                warn!("progress edit hit flood control; backing off {wait:?}");
                self.next_edit_allowed_at = Some(Instant::now() + wait);
                false
            }
            Err(err) => {
                debug!("progress edit failed (ignored): {err}");
                // Leave the throttle window unchanged on unrelated errors so a
                // later phase update can still try.
                self.next_edit_allowed_at = Some(Instant::now() + self.min_interval);
                false
            }
        }
    }
}

impl ProgressSink for ProgressReporter {
    fn update_text<'a>(
        &'a mut self,
        text: String,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move { self.update(&text).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};

    #[test]
    fn emits_first_update_immediately() {
        assert!(should_emit(None, Instant::now(), "", "Extracting claims…"));
    }

    #[test]
    fn skips_unchanged_or_empty_text() {
        let now = Instant::now();
        assert!(!should_emit(None, now, "same", "same"));
        assert!(!should_emit(None, now, "old", "   "));
    }

    #[test]
    fn respects_throttle_window() {
        let now = Instant::now();
        let blocked_until = now + Duration::from_secs(2);
        assert!(!should_emit(Some(blocked_until), now, "old", "new"));
        assert!(should_emit(
            Some(now),
            now + Duration::from_millis(1),
            "old",
            "new"
        ));
    }

    #[tokio::test]
    async fn phase_updates_keep_the_status_note() {
        let server = TestServer::new(
            ["Extracting claims", "Composing the report"]
                .into_iter()
                .map(|phase| {
                    ExpectedRequest::new(
                        "POST",
                        "/bot123:test/EditMessageText",
                        response_with_status(
                            200,
                            br#"{"ok":true,"result":{"message_id":3,"date":1,"chat":{"id":2,"type":"private"},"text":"status"}}"#
                                .to_vec(),
                        ),
                    )
                    .with_body_fragment(phase.as_bytes())
                    .with_body_fragment(b"1 rich-message attachment could not be included")
                })
                .collect(),
        );
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let mut progress = ProgressReporter::new(bot, ChatId(2), MessageId(3))
            .with_note(Some("1 rich-message attachment could not be included"));

        progress.update("Extracting claims").await;
        progress.update_now("Composing the report").await;

        server.join().unwrap();
    }
}

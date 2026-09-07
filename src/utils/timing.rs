use std::time::{Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use teloxide::types::Message;
use tracing::info;

use crate::utils::text::truncate_to_chars;

pub fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug)]
pub struct CommandTimer {
    command: String,
    chat_id: Option<i64>,
    user_id: Option<i64>,
    username: Option<String>,
    message_id: Option<i64>,
    text: Option<String>,
    started_at: DateTime<Utc>,
    started_perf: Instant,
    status: String,
    detail: Option<String>,
    completed: bool,
}

const COMMAND_TEXT_EXCERPT_MAX_CHARS: usize = 300;

/// Single-line excerpt of the command text for the timing log.
fn command_text_excerpt(text: &str) -> String {
    let flattened = text.replace('\n', " ");
    truncate_to_chars(&flattened, COMMAND_TEXT_EXCERPT_MAX_CHARS).to_string()
}

impl CommandTimer {
    /// Start a timer for `command` from already-extracted message identity.
    pub fn new(
        command: &str,
        chat_id: Option<i64>,
        user_id: Option<i64>,
        username: Option<String>,
        message_id: Option<i64>,
        text: Option<String>,
    ) -> Self {
        CommandTimer {
            command: command.to_string(),
            chat_id,
            user_id,
            username,
            message_id,
            text: text.as_deref().map(command_text_excerpt),
            started_at: Utc::now(),
            started_perf: Instant::now(),
            status: "success".to_string(),
            detail: None,
            completed: false,
        }
    }

    pub fn from_message(command: &str, message: &Message) -> Self {
        let user = message.from.as_ref();
        Self::new(
            command,
            Some(message.chat.id.0),
            user.and_then(|u| i64::try_from(u.id.0).ok()),
            user.and_then(|u| u.username.clone()),
            Some(message.id.0 as i64),
            message
                .text()
                .or_else(|| message.caption())
                .map(str::to_string),
        )
    }

    /// Emitted as structured fields (not a key=value message string) so the
    /// JSON timing layer yields queryable columns.
    pub fn log_received(&self) {
        info!(
            target: "bot.timing",
            event = "command_received",
            command = %self.command,
            chat_id = self.chat_id,
            user_id = self.user_id,
            username = self.username.as_deref(),
            message_id = self.message_id,
            received_at = %self.started_at.to_rfc3339(),
            text = self.text.as_deref(),
        );
    }

    pub fn mark_status(&mut self, status: &str, detail: Option<String>) {
        self.status = status.to_string();
        self.detail = detail;
    }

    pub fn log_completed(&mut self) {
        if self.completed {
            return;
        }
        self.completed = true;
        let completed_at = Utc::now();
        let duration = self.started_perf.elapsed().as_secs_f64();
        info!(
            target: "bot.timing",
            event = "command_completed",
            command = %self.command,
            chat_id = self.chat_id,
            user_id = self.user_id,
            message_id = self.message_id,
            started_at = %self.started_at.to_rfc3339(),
            response_sent_at = %completed_at.to_rfc3339(),
            duration_s = duration,
            status = %self.status,
            detail = self.detail.as_deref(),
        );
    }
}

pub fn start_command_timer(command: &str, message: &Message) -> CommandTimer {
    let timer = CommandTimer::from_message(command, message);
    timer.log_received();
    timer
}

pub fn complete_command_timer(timer: &mut CommandTimer, status: &str, detail: Option<String>) {
    timer.mark_status(status, detail);
    timer.log_completed();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::log_capture::capture_json_events;

    #[test]
    fn command_text_excerpt_flattens_newlines_and_keeps_short_text() {
        assert_eq!(command_text_excerpt("/q hello\nworld"), "/q hello world");
    }

    #[test]
    fn command_text_excerpt_does_not_split_multibyte_chars() {
        // 4 ASCII bytes followed by 3-byte CJK chars: byte offset 300 falls
        // inside a character, which used to panic.
        let text = format!("/qq {}", "好".repeat(400));
        let excerpt = command_text_excerpt(&text);
        assert!(excerpt.starts_with("/qq "));
        assert_eq!(excerpt.chars().count(), COMMAND_TEXT_EXCERPT_MAX_CHARS);
    }

    #[test]
    fn command_timer_emits_structured_fields_for_the_json_layer() {
        let events = capture_json_events(|| {
            let mut timer = CommandTimer::new(
                "q",
                Some(-100),
                Some(7),
                Some("alice".to_string()),
                Some(99),
                Some("/q hi".to_string()),
            );
            timer.log_received();
            complete_command_timer(&mut timer, "success", Some("cached".to_string()));
        });

        assert_eq!(events.len(), 2, "{events:?}");
        let received = &events[0]["fields"];
        assert_eq!(events[0]["target"], "bot.timing");
        assert_eq!(received["event"], "command_received");
        assert_eq!(received["command"], "q");
        assert_eq!(received["chat_id"], -100);
        assert_eq!(received["user_id"], 7);
        assert_eq!(received["username"], "alice");
        assert_eq!(received["message_id"], 99);
        assert_eq!(received["text"], "/q hi");

        let completed = &events[1]["fields"];
        assert_eq!(completed["event"], "command_completed");
        assert_eq!(completed["command"], "q");
        assert_eq!(completed["status"], "success");
        assert_eq!(completed["detail"], "cached");
        assert!(completed["duration_s"].is_number(), "{completed:?}");
    }
}

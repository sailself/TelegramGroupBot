use std::time::Instant;

use chrono::{DateTime, Utc};
use teloxide::types::Message;
use tracing::info;

use crate::utils::text::truncate_to_chars;

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
    pub fn from_message(command: &str, message: &Message) -> Self {
        let text = message
            .text()
            .or_else(|| message.caption())
            .map(command_text_excerpt);

        let user = message.from.as_ref();
        CommandTimer {
            command: command.to_string(),
            chat_id: Some(message.chat.id.0),
            user_id: user.and_then(|u| i64::try_from(u.id.0).ok()),
            username: user.and_then(|u| u.username.clone()),
            message_id: Some(message.id.0 as i64),
            text,
            started_at: Utc::now(),
            started_perf: Instant::now(),
            status: "success".to_string(),
            detail: None,
            completed: false,
        }
    }

    pub fn log_received(&self) {
        info!(
            target: "bot.timing",
            "event=command_received command={} chat_id={:?} user_id={:?} username={:?} message_id={:?} received_at={} text={:?}",
            self.command,
            self.chat_id,
            self.user_id,
            self.username,
            self.message_id,
            self.started_at.to_rfc3339(),
            self.text
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
            "event=command_completed command={} chat_id={:?} user_id={:?} message_id={:?} started_at={} response_sent_at={} duration_s={:.3} status={} detail={}",
            self.command,
            self.chat_id,
            self.user_id,
            self.message_id,
            self.started_at.to_rfc3339(),
            completed_at.to_rfc3339(),
            duration,
            self.status,
            self.detail.clone().unwrap_or_default()
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
}

use std::time::Instant;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use teloxide::types::Message;
use tracing::info;

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

impl CommandTimer {
    pub fn from_message(command: &str, message: &Message) -> Self {
        let text = message
            .text()
            .map(|value| value.replace('\n', " "))
            .or_else(|| message.caption().map(|value| value.replace('\n', " ")))
            .map(|value| {
                if value.len() > 300 {
                    value[..300].to_string()
                } else {
                    value
                }
            });

        let user = message.from();
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

pub async fn log_llm_timing<T, F, Fut>(
    provider: &str,
    model: &str,
    operation: &str,
    metadata: Option<JsonValue>,
    call: F,
) -> Result<T, anyhow::Error>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, anyhow::Error>>,
{
    let started_at = Utc::now();
    let started_perf = Instant::now();
    let metadata_text = metadata
        .as_ref()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "{}".to_string());
    info!(
        target: "bot.timing",
        "event=llm_request provider={} model={} operation={} started_at={} metadata={}",
        provider,
        model,
        operation,
        started_at.to_rfc3339(),
        metadata_text
    );

    let mut status = "success";
    let result = call().await;
    if result.is_err() {
        status = "error";
    }

    let completed_at = Utc::now();
    let duration = started_perf.elapsed().as_secs_f64();
    info!(
        target: "bot.timing",
        "event=llm_response provider={} model={} operation={} completed_at={} duration_s={:.3} status={} metadata={}",
        provider,
        model,
        operation,
        completed_at.to_rfc3339(),
        duration,
        status,
        metadata_text
    );

    result
}

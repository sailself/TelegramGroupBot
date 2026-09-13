use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};
use tracing::{error, warn};

use crate::config::CONFIG;
use crate::db::database::build_message_insert;
use crate::db::search::derive_search_provenance;
use crate::handlers::content::create_telegraph_answer;
use crate::state::AppState;
use crate::utils::markdown::{markdown_to_plain_text, markdown_to_telegram_html};
use crate::utils::telegram::retry_telegram;
use crate::utils::text::{escape_html, truncate_to_chars};

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

/// Original content, formatted only at the destination.
pub struct ResponseContent {
    pub markdown: String,
    pub model_label: Option<String>,
}
impl ResponseContent {
    pub fn new(markdown: impl Into<String>) -> Self {
        Self {
            markdown: markdown.into(),
            model_label: None,
        }
    }
    pub fn with_model(mut self, label: impl Into<String>) -> Self {
        self.model_label = Some(label.into());
        self
    }
    fn plain_text(&self, include_link_destinations: bool) -> String {
        let mut body = markdown_to_plain_text(&self.markdown, include_link_destinations);
        if let Some(label) = &self.model_label {
            body.push_str(&format!("\n\nModel: {label}"));
        }
        body
    }
    fn html(&self) -> String {
        let mut body = markdown_to_telegram_html(&self.markdown);
        if let Some(label) = &self.model_label {
            body.push_str(&format!("\n\nModel: {}", escape_html(label)));
        }
        body
    }
}

const OVERFLOW_LINE_LIMIT: usize = 22;

/// Whether a response is too long for a single Telegram message and should
/// go to Telegraph (or be truncated when Telegraph is unavailable).
fn needs_overflow_delivery(response: &str, max_chars: usize) -> bool {
    response.lines().count() > OVERFLOW_LINE_LIMIT || response.chars().count() > max_chars
}

/// Plain-text fallback used when Telegraph publishing fails.
fn overflow_fallback_text(response: &str, max_chars: usize) -> String {
    const NOTICE: &str = "...\n\n(Response was truncated due to length)";
    if response.chars().count() > max_chars {
        let suffix = truncate_to_chars(NOTICE, max_chars);
        format!(
            "{}{}",
            truncate_to_chars(response, max_chars.saturating_sub(suffix.chars().count())),
            suffix
        )
    } else {
        response.to_string()
    }
}

pub async fn send_response(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    response: &ResponseContent,
    title: &str,
) -> Result<()> {
    deliver_response(response, CONFIG.telegram.max_length,
        |markdown, label: Option<String>| async move { create_telegraph_answer(title, &markdown, label.as_deref()).await },
        |text, mode| async move { edit_text_with_retry(bot, chat_id, message_id, &text, mode).await },
    ).await
}

/// Delivery policy is shared by real Telegram/Telegraph I/O and local tests.
async fn deliver_response<P, PF, E, EF>(
    response: &ResponseContent,
    max_chars: usize,
    publish: P,
    mut edit: E,
) -> Result<()>
where
    P: FnOnce(String, Option<String>) -> PF,
    PF: std::future::Future<Output = Option<String>>,
    E: FnMut(String, Option<ParseMode>) -> EF,
    EF: std::future::Future<Output = Result<()>>,
{
    let visible = response.plain_text(false);
    if needs_overflow_delivery(&visible, max_chars) {
        if let Some(url) = publish(response.markdown.clone(), response.model_label.clone()).await {
            edit(
                format!(
                    "I have too much to say. <a href=\"{}\">View it here</a>",
                    escape_html(&url)
                ),
                Some(ParseMode::Html),
            )
            .await?;
        } else {
            edit(
                overflow_fallback_text(&response.plain_text(true), max_chars),
                None,
            )
            .await?;
        }
        return Ok(());
    }
    if let Err(error) = edit(response.html(), Some(ParseMode::Html)).await {
        warn!("Failed to send formatted response: {error}");
        edit(
            overflow_fallback_text(&response.plain_text(true), max_chars),
            None,
        )
        .await?;
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
        assert!(body.chars().count() < 4_000);
        assert_eq!(truncated.chars().count(), 4_000);
    }

    #[test]
    fn overflow_fallback_returns_short_response_unchanged() {
        assert_eq!(overflow_fallback_text("short", 4_000), "short");
    }
}

#[cfg(test)]
mod content_tests {
    use super::*;
    #[test]
    fn destinations_keep_markdown_and_literal_model_labels() {
        let content = ResponseContent::new("**Important** & [source](https://example.com)")
            .with_model("a_[b]*");
        let html = content.html();
        let plain = content.plain_text(true);
        assert!(html.contains("<b>Important</b>"));
        assert!(html.contains("Model: a_[b]*"));
        assert!(plain.contains("source (https://example.com)"));
        assert!(!plain.contains("<b>"));
        assert!(content.markdown.starts_with("**Important**"));
    }
    #[test]
    fn fallback_obeys_tiny_and_unicode_limits() {
        for limit in [0, 1, 20, 100] {
            assert!(
                overflow_fallback_text(&"好".repeat(200), limit)
                    .chars()
                    .count()
                    <= limit
            );
        }
    }
}

#[cfg(test)]
mod delivery_tests {
    use super::*;
    #[tokio::test]
    async fn overflow_publishes_markdown_and_sends_a_link() {
        let content = ResponseContent::new("**bold** [source](https://example.com)\n".repeat(23));
        let mut sent = Vec::new();
        deliver_response(
            &content,
            4000,
            |markdown, _| async move {
                assert!(markdown.contains("**bold**"));
                assert!(!markdown.contains("<b>"));
                Some("https://telegra.ph/test".to_string())
            },
            |text, mode| {
                sent.push((text, mode));
                async { Ok(()) }
            },
        )
        .await
        .unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].0.contains("href=\"https://telegra.ph/test\""));
        assert_eq!(sent[0].1, Some(ParseMode::Html));
    }
    #[tokio::test]
    async fn formatting_rejection_uses_readable_plain_text() {
        let mut sent = Vec::new();
        deliver_response(
            &ResponseContent::new("**bold** [source](https://example.com)"),
            4000,
            |_, _| async { panic!("short answer should not publish") },
            |text, mode| {
                sent.push((text, mode));
                async move {
                    if mode.is_some() {
                        Err(anyhow::anyhow!("bad markup"))
                    } else {
                        Ok(())
                    }
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(sent.len(), 2);
        assert!(sent[0].0.contains("<b>bold</b>"));
        assert_eq!(sent[1].0, "bold source (https://example.com)");
        assert_eq!(sent[1].1, None);
    }
    #[tokio::test]
    async fn failed_publishing_truncates_plain_text_including_notice() {
        let mut sent = Vec::new();
        deliver_response(
            &ResponseContent::new(format!("**{}**", "好".repeat(400))),
            100,
            |_, _| async { None },
            |text, mode| {
                sent.push((text, mode));
                async { Ok(()) }
            },
        )
        .await
        .unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0.chars().count(), 100);
        assert!(!sent[0].0.contains("<b>"));
        assert_eq!(sent[0].1, None);
    }
}

#[cfg(test)]
mod footer_tests {
    use super::*;
    #[test]
    fn model_footer_is_not_consumed_by_an_unfinished_code_fence() {
        let content = ResponseContent::new("```rust\nlet x = 1;").with_model("model_[test]*");
        assert!(content.html().contains("</pre>\n\nModel: model_[test]*"));
        assert!(content.plain_text(true).ends_with("Model: model_[test]*"));
    }
}

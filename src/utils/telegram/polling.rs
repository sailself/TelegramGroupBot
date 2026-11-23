//! Long polling that preserves rich messages until teloxide supports them.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use futures_util::stream::{self, BoxStream};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use teloxide::backoff::exponential_backoff_strategy;
use teloxide::errors::AsResponseParameters;
use teloxide::payloads::GetUpdates;
use teloxide::prelude::*;
use teloxide::requests::{JsonRequest, Payload};
use teloxide::stop::{mk_stop_token, StopFlag, StopToken};
use teloxide::types::{AllowedUpdate, UpdateId, UpdateKind};
use teloxide::update_listeners::{AsUpdateStream, UpdateListener};
use teloxide::RequestError;
use tracing::warn;

use super::rich_message::RichUpdate;

/// Rich file references travel only as far as the dispatcher worker for this
/// update. Telegram includes the direct reply again on each new command.
#[derive(Clone, Default)]
pub(crate) struct RichUpdates(Arc<Mutex<HashMap<UpdateId, RichUpdate>>>);

impl RichUpdates {
    pub(crate) fn take(&self, id: UpdateId) -> RichUpdate {
        self.0.lock().remove(&id).unwrap_or_default()
    }
}

#[derive(Serialize)]
#[serde(transparent)]
struct RawGetUpdates(GetUpdates);

impl Payload for RawGetUpdates {
    type Output = Vec<Value>;
    const NAME: &'static str = GetUpdates::NAME;

    fn timeout_hint(&self) -> Option<Duration> {
        self.0.timeout_hint()
    }
}

fn update_kind_name(raw: &Value) -> &str {
    // Only log a single Bot API identifier. Never include the update payload.
    let Some(fields) = raw.as_object() else {
        return "unknown";
    };
    let mut kinds = fields.keys().filter(|key| key.as_str() != "update_id");
    kinds
        .next()
        .filter(|kind| {
            kinds.next().is_none()
                && kind.len() <= 64
                && kind.starts_with(|c: char| c.is_ascii_lowercase())
                && kind
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
        })
        .map_or("unknown", String::as_str)
}

fn decode_update(mut raw: Value) -> Option<(Update, RichUpdate)> {
    let rich = RichUpdate::from_update(&mut raw);
    // Teloxide's UpdateKind visitor reads borrowed keys.
    match Update::deserialize(&raw) {
        Ok(update) if matches!(update.kind, UpdateKind::Error(_)) => {
            warn!(
                update_id = update.id.0,
                kind = update_kind_name(&raw),
                "Skipping unsupported Telegram update"
            );
            None
        }
        Ok(update) => Some((update, rich)),
        Err(_) => {
            let id = raw["update_id"]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok());
            warn!(
                update_id = ?id,
                kind = update_kind_name(&raw),
                "Skipping malformed Telegram update"
            );
            None
        }
    }
}

pub(crate) struct RichPolling {
    bot: Bot,
    rich: RichUpdates,
    stop_token: StopToken,
    stop_flag: StopFlag,
    allowed_updates: Option<Vec<AllowedUpdate>>,
    offset: i32,
    pending: VecDeque<Value>,
    error_count: u32,
    retry_delay: Duration,
    acknowledged: bool,
}

impl RichPolling {
    pub(crate) async fn new(bot: Bot, rich: RichUpdates) -> Self {
        // Match the default teloxide polling listener's webhook behavior.
        match bot.get_webhook_info().await {
            Ok(info) if info.url.is_some() => {
                if let Err(err) = bot.delete_webhook().await {
                    warn!("Failed to delete Telegram webhook: {err}");
                }
            }
            Ok(_) => {}
            Err(err) => warn!("Failed to get Telegram webhook info: {err}"),
        }
        let (stop_token, stop_flag) = mk_stop_token();
        Self {
            bot,
            rich,
            stop_token,
            stop_flag,
            allowed_updates: None,
            offset: 0,
            pending: VecDeque::new(),
            error_count: 0,
            retry_delay: Duration::ZERO,
            acknowledged: false,
        }
    }

    async fn updates(&self, timeout: u32, limit: Option<u8>) -> Result<Vec<Value>, RequestError> {
        JsonRequest::new(
            self.bot.clone(),
            RawGetUpdates(GetUpdates {
                offset: Some(self.offset),
                limit,
                timeout: Some(timeout),
                allowed_updates: self.allowed_updates.clone(),
            }),
        )
        .await
    }

    async fn next_update(&mut self) -> Option<Result<Update, RequestError>> {
        loop {
            if let Some(raw) = self.pending.pop_front() {
                let Some((update, rich)) = decode_update(raw) else {
                    continue;
                };
                if matches!(update.kind, UpdateKind::Message(_))
                    && (rich.message.is_some() || rich.reply.is_some())
                {
                    self.rich.0.lock().insert(update.id, rich);
                }
                return Some(Ok(update));
            }

            if self.stop_flag.is_stopped() {
                // Confirm buffered updates delivered to the dispatcher, leaving
                // any newly received updates for the next process.
                if self.offset != 0 && !self.acknowledged {
                    self.acknowledged = true;
                    if let Err(err) = self.updates(0, Some(1)).await {
                        return Some(Err(err));
                    }
                }
                return None;
            }

            if !self.retry_delay.is_zero() {
                tokio::select! {
                    biased;
                    _ = self.stop_flag.clone() => continue,
                    _ = tokio::time::sleep(self.retry_delay) => {}
                }
            }
            let result = tokio::select! {
                biased;
                _ = self.stop_flag.clone() => continue,
                result = self.updates(10, None) => result,
            };
            match result {
                Ok(updates) => {
                    self.retry_delay = Duration::ZERO;
                    self.error_count = 0;
                    self.allowed_updates = None;
                    if let Some(last_id) = updates
                        .iter()
                        .rev()
                        .find_map(|update| update["update_id"].as_u64())
                        .and_then(|id| i32::try_from(id).ok())
                    {
                        self.offset = last_id.saturating_add(1);
                    }
                    self.pending = updates.into();
                }
                Err(err) => {
                    self.retry_delay = if let Some(retry_after) = err.retry_after() {
                        self.error_count = 0;
                        retry_after.duration()
                    } else {
                        let delay = exponential_backoff_strategy(self.error_count);
                        self.error_count = self.error_count.saturating_add(1);
                        delay
                    };
                    return Some(Err(err));
                }
            }
        }
    }
}

impl<'a> AsUpdateStream<'a> for RichPolling {
    type StreamErr = RequestError;
    type Stream = BoxStream<'a, Result<Update, RequestError>>;

    fn as_stream(&'a mut self) -> Self::Stream {
        Box::pin(stream::unfold(self, |listener| async {
            listener
                .next_update()
                .await
                .map(|update| (update, listener))
        }))
    }
}

impl UpdateListener for RichPolling {
    type Err = RequestError;

    fn stop_token(&mut self) -> StopToken {
        self.stop_token.clone()
    }

    fn hint_allowed_updates(&mut self, hint: &mut dyn Iterator<Item = AllowedUpdate>) {
        self.allowed_updates = Some(hint.collect());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn borrowed_updates_decode_regular_and_projected_messages() {
        let (regular, rich) = decode_update(json!({
            "update_id": 1,
            "message": {"message_id": 1, "date": 1, "chat": {"id": 5, "type": "private"}, "text": "ordinary"}
        }))
        .unwrap();
        let UpdateKind::Message(message) = regular.kind else {
            panic!("expected a message");
        };
        assert_eq!(message.text(), Some("ordinary"));
        assert!(rich.message.is_none());

        let (projected, rich) = decode_update(json!({
            "update_id": 2,
            "message": {
                "message_id": 2, "date": 1, "chat": {"id": 5, "type": "private"}, "text": "/factcheck",
                "reply_to_message": {
                    "message_id": 1, "date": 1, "chat": {"id": 5, "type": "private"},
                    "rich_message": {"blocks": [
                        {"type": "paragraph", "text": "projected text"},
                        {"type": "document", "document": {"file_id": "reference"}}
                    ]}
                }
            }
        }))
        .unwrap();
        let UpdateKind::Message(message) = projected.kind else {
            panic!("expected a message");
        };
        assert_eq!(message.text(), Some("/factcheck"));
        assert_eq!(
            message.reply_to_message().and_then(Message::text),
            Some("projected text")
        );
        assert_eq!(rich.reply.unwrap()[0].file_id.0, "reference");
    }

    #[test]
    fn unsupported_and_malformed_updates_only_expose_a_safe_kind() {
        let unsupported = json!({"update_id": 3, "future_update": {"text": "private text"}});
        assert_eq!(update_kind_name(&unsupported), "future_update");
        assert!(decode_update(unsupported).is_none());
        let malformed = json!({"update_id": "private value", "message": {"text": "private text"}});
        assert_eq!(update_kind_name(&malformed), "message");
        assert!(decode_update(malformed).is_none());

        for unsafe_kind in ["untrusted\nsecond line", "token:secret", &"a".repeat(65)] {
            let raw = json!({"update_id": 4, (unsafe_kind): {"text": "private text"}});
            assert_eq!(update_kind_name(&raw), "unknown");
            assert!(decode_update(raw).is_none());
        }
    }
}

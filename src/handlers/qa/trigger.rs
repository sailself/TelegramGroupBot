//! Auto-`/q` trigger detection: bot mentions, replies to this bot, and query extraction.

use std::ops::Range;

use teloxide::types::{Message, MessageEntityKind, MessageEntityRef};

use crate::config::CONFIG;
use crate::handlers::media::message_has_image;
use crate::utils::telegram::rich_message::RichMention;
use crate::utils::telegram::{message_entities_for_text, message_text_or_caption};

fn is_bot_mention_entity(
    entity: &MessageEntityRef<'_>,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> bool {
    match entity.kind() {
        MessageEntityKind::Mention => {
            if bot_username_lower.is_empty() {
                return false;
            }
            entity
                .text()
                .trim_start_matches('@')
                .eq_ignore_ascii_case(bot_username_lower)
        }
        MessageEntityKind::TextMention { user } => {
            i64::try_from(user.id.0).ok() == Some(bot_user_id)
        }
        _ => false,
    }
}

fn rich_bot_mention_ranges<'a>(
    mentions: &'a [RichMention],
    bot_username_lower: &'a str,
) -> impl Iterator<Item = Range<usize>> + 'a {
    mentions
        .iter()
        .filter(move |mention| mention.username.eq_ignore_ascii_case(bot_username_lower))
        .map(|mention| mention.range.clone())
}

fn strip_bot_mentions_from_query(
    text: &str,
    entities: Option<&[MessageEntityRef<'_>]>,
    bot_user_id: i64,
    bot_username_lower: &str,
    rich_mentions: &[RichMention],
) -> String {
    let mut ranges = entities
        .into_iter()
        .flatten()
        .filter(|entity| is_bot_mention_entity(entity, bot_user_id, bot_username_lower))
        .map(MessageEntityRef::range)
        .chain(rich_bot_mention_ranges(rich_mentions, bot_username_lower))
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return text.trim().to_string();
    }

    ranges.sort_unstable_by_key(|range| range.start);
    let mut stripped = String::with_capacity(text.len());
    let mut end = 0;
    for range in ranges {
        if range.end > end {
            stripped.push_str(&text[end..range.start.max(end)]);
            stripped.push(' ');
            end = range.end;
        }
    }
    stripped.push_str(&text[end..]);

    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_reply_to_this_bot(message: &Message, bot_user_id: i64) -> bool {
    let Some(reply) = message.reply_to_message() else {
        return false;
    };
    let Some(reply_from) = reply.from.as_ref() else {
        return false;
    };
    if !reply_from.is_bot {
        return false;
    }
    i64::try_from(reply_from.id.0).ok() == Some(bot_user_id)
}

/// Whether the message this update replies to contains an image or a video.
///
/// A reply to one of the bot's own image or video responses (`/img`, `/image`,
/// `/img2`, `/vid`, `/vid2`) is almost always a comment on the media rather
/// than a follow-up question, so the auto-`/q` reply trigger skips those unless
/// the user also mentions the bot explicitly.
fn reply_target_has_image_or_video(message: &Message) -> bool {
    message.reply_to_message().is_some_and(|reply| {
        message_has_image(reply) || reply.video().is_some() || reply.animation().is_some()
    })
}

fn is_mentioning_this_bot(message: &Message, bot_user_id: i64, bot_username_lower: &str) -> bool {
    let Some(entities) = message_entities_for_text(message) else {
        return false;
    };

    entities
        .iter()
        .any(|entity| is_bot_mention_entity(entity, bot_user_id, bot_username_lower))
}

pub fn should_auto_q_trigger(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
    rich_mentions: &[RichMention],
) -> bool {
    should_auto_q_trigger_with_config(
        message,
        bot_user_id,
        bot_username_lower,
        CONFIG.telegram.enable_bot_to_bot_auto_q,
        rich_mentions,
    )
}

pub(super) fn should_auto_q_trigger_with_config(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
    enable_bot_to_bot_auto_q: bool,
    rich_mentions: &[RichMention],
) -> bool {
    if message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        == Some(bot_user_id)
    {
        return false;
    }

    let Some(text) = message_text_or_caption(message) else {
        return false;
    };
    if text.trim_start().starts_with('/') {
        return false;
    }

    if !enable_bot_to_bot_auto_q
        && message
            .from
            .as_ref()
            .map(|user| user.is_bot)
            .unwrap_or(false)
    {
        return false;
    }

    rich_bot_mention_ranges(rich_mentions, bot_username_lower)
        .next()
        .is_some()
        || is_mentioning_this_bot(message, bot_user_id, bot_username_lower)
        || (is_reply_to_this_bot(message, bot_user_id) && !reply_target_has_image_or_video(message))
}

pub fn build_auto_q_query(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
    rich_mentions: &[RichMention],
) -> Option<String> {
    let text = message_text_or_caption(message)?;
    if text.trim().is_empty() {
        return None;
    }

    let entities = message_entities_for_text(message);
    let stripped = strip_bot_mentions_from_query(
        text,
        entities.as_deref(),
        bot_user_id,
        bot_username_lower,
        rich_mentions,
    );
    if stripped.is_empty() {
        None
    } else {
        Some(stripped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::telegram::rich_message::RichUpdate;
    use serde_json::{json, Value};

    fn project_rich_message(text: Value) -> (Message, RichUpdate) {
        let mut update = json!({"message": {
            "message_id": 1,
            "date": 1,
            "from": {"id": 10, "is_bot": false, "first_name": "Hum"},
            "chat": {"id": 10, "type": "private"},
            "rich_message": {"blocks": [{"type": "paragraph", "text": text}]}
        }});
        let rich = RichUpdate::from_update(&mut update);
        let message = serde_json::from_value(update["message"].take()).unwrap();
        (message, rich)
    }

    #[test]
    fn rich_mention_matches_username_and_strips_nested_display_spans() {
        let (message, rich) = project_rich_message(json!([
            "🧭 ",
            {"type": "mention", "username": "HelperBot", "text": [
                "the ",
                {"type": "text_mention", "text": "🛠 assistant", "user": {
                    "id": 42, "is_bot": true, "first_name": "Helper"
                }},
                " here"
            ]},
            " please review this"
        ]));

        assert_eq!(
            message.text(),
            Some("🧭 the 🛠 assistant here please review this")
        );
        assert!(should_auto_q_trigger_with_config(
            &message,
            42,
            "helperbot",
            false,
            &rich.message_mentions,
        ));
        assert_eq!(
            build_auto_q_query(&message, 42, "helperbot", &rich.message_mentions).as_deref(),
            Some("🧭 please review this")
        );
    }

    #[test]
    fn rich_mention_uses_target_even_when_display_names_this_bot() {
        let (message, rich) = project_rich_message(json!([
            {"type": "mention", "username": "OtherBot", "text": "@HelperBot"},
            " please review this"
        ]));

        assert!(!should_auto_q_trigger_with_config(
            &message,
            42,
            "helperbot",
            false,
            &rich.message_mentions,
        ));
        assert_eq!(
            build_auto_q_query(&message, 42, "helperbot", &rich.message_mentions).as_deref(),
            Some("@HelperBot please review this")
        );
    }
}

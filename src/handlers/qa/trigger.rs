//! Auto-`/q` trigger detection: bot mentions, replies to this bot, and query extraction.

use teloxide::types::{Message, MessageEntityKind, MessageEntityRef};

use crate::config::CONFIG;
use crate::handlers::media::message_has_image;
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

fn strip_bot_mentions_from_query(
    text: &str,
    entities: Option<&[MessageEntityRef<'_>]>,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> String {
    let Some(entities) = entities else {
        return text.trim().to_string();
    };

    let mut ranges = entities
        .iter()
        .filter(|entity| is_bot_mention_entity(entity, bot_user_id, bot_username_lower))
        .map(|entity| entity.start()..entity.end())
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return text.trim().to_string();
    }

    ranges.sort_by_key(|range| range.start);
    let mut stripped = text.to_string();
    for range in ranges.into_iter().rev() {
        stripped.replace_range(range, " ");
    }

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

/// Whether the message this update replies to contains an image.
///
/// A reply to one of the bot's own image responses (`/img`, `/image`, `/img2`)
/// is almost always a comment on the picture rather than a follow-up question,
/// so the auto-`/q` reply trigger skips those unless the user also mentions the
/// bot explicitly.
fn reply_target_has_image(message: &Message) -> bool {
    message
        .reply_to_message()
        .map(message_has_image)
        .unwrap_or(false)
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
) -> bool {
    should_auto_q_trigger_with_config(
        message,
        bot_user_id,
        bot_username_lower,
        CONFIG.enable_bot_to_bot_auto_q,
    )
}

pub(super) fn should_auto_q_trigger_with_config(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
    enable_bot_to_bot_auto_q: bool,
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

    is_mentioning_this_bot(message, bot_user_id, bot_username_lower)
        || (is_reply_to_this_bot(message, bot_user_id) && !reply_target_has_image(message))
}

pub fn build_auto_q_query(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> Option<String> {
    let text = message_text_or_caption(message)?;
    if text.trim().is_empty() {
        return None;
    }

    let entities = message_entities_for_text(message);
    let stripped =
        strip_bot_mentions_from_query(text, entities.as_deref(), bot_user_id, bot_username_lower);
    if stripped.is_empty() {
        None
    } else {
        Some(stripped)
    }
}

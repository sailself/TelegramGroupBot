use std::collections::HashMap;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use teloxide::prelude::*;
use teloxide::types::ReplyParameters;
use tracing::{info, warn};

use crate::acl::acl_manager;
use crate::config::CONFIG;

static RATE_LIMITS: Lazy<Mutex<HashMap<i64, Instant>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn resolve_actor_ids(message: &Message) -> (i64, i64) {
    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let chat_id = message.chat.id.0;
    (user_id, chat_id)
}

pub fn is_rate_limited(user_id: i64) -> bool {
    let mut limits = RATE_LIMITS.lock();
    let now = Instant::now();

    if let Some(last) = limits.get(&user_id) {
        if now.duration_since(*last) < Duration::from_secs(CONFIG.rate_limit_seconds) {
            return true;
        }
    }

    limits.insert(user_id, now);
    false
}

pub fn initialize_access_control() {
    acl_manager().initialize();
    let meta = acl_manager().snapshot_meta();
    info!(
        "ACL initialized: file='{}' loaded={} version={} chats={} owners={}",
        meta.path, meta.loaded, meta.version, meta.chat_rule_count, meta.owner_user_count
    );
    if let Some(err) = meta.last_error.as_deref() {
        warn!("ACL last_error: {}", err);
    }
}

pub fn is_owner(user_id: i64) -> bool {
    acl_manager().is_owner(user_id)
}

pub async fn check_access_control(bot: &Bot, message: &Message, command: &str) -> bool {
    let (user_id, chat_id) = resolve_actor_ids(message);
    let decision = acl_manager().authorize_command(chat_id, user_id, command);
    if decision.allowed {
        return true;
    }

    warn!(
        "ACL denied command '{}' for chat={} user={} reason={}",
        command, chat_id, user_id, decision.reason
    );
    let _ = bot
        .send_message(
            message.chat.id,
            "You are not authorized to use this command in this chat.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await;
    false
}

pub async fn check_admin_access(bot: &Bot, message: &Message, command: &str) -> bool {
    check_access_control(bot, message, command).await
}

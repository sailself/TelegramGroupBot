use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use teloxide::prelude::*;
use tracing::{info, warn};

use crate::config::CONFIG;

static RATE_LIMITS: Lazy<Mutex<HashMap<i64, Instant>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static WHITELIST_CACHE: Lazy<Mutex<Option<Vec<String>>>> = Lazy::new(|| Mutex::new(None));
static WHITELIST_LOADED: AtomicBool = AtomicBool::new(false);

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

pub fn load_whitelist() {
    if WHITELIST_LOADED.swap(true, Ordering::SeqCst) {
        return;
    }

    let path = &CONFIG.whitelist_file_path;
    let file = std::fs::read_to_string(path);
    let mut cache = WHITELIST_CACHE.lock();

    match file {
        Ok(content) => {
            let ids = content
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(|line| line.to_string())
                .collect::<Vec<_>>();
            *cache = Some(ids);
            info!("Loaded whitelist file {}", path);
        }
        Err(err) => {
            warn!("Whitelist file {} not found or failed to read: {}", path, err);
            *cache = None;
        }
    }
}

pub fn is_user_whitelisted(user_id: i64) -> bool {
    if !WHITELIST_LOADED.load(Ordering::SeqCst) {
        load_whitelist();
    }
    let cache = WHITELIST_CACHE.lock();
    match &*cache {
        None => true,
        Some(list) => list.contains(&user_id.to_string()),
    }
}

pub fn is_chat_whitelisted(chat_id: i64) -> bool {
    if !WHITELIST_LOADED.load(Ordering::SeqCst) {
        load_whitelist();
    }
    let cache = WHITELIST_CACHE.lock();
    match &*cache {
        None => true,
        Some(list) => list.contains(&chat_id.to_string()),
    }
}

pub fn is_access_allowed(user_id: i64, chat_id: i64) -> bool {
    is_user_whitelisted(user_id) || is_chat_whitelisted(chat_id)
}

pub fn requires_access_control(command: &str) -> bool {
    if CONFIG.access_controlled_commands.is_empty() {
        return false;
    }
    CONFIG
        .access_controlled_commands
        .iter()
        .any(|entry| entry == command)
}

pub async fn check_access_control(bot: &Bot, message: &Message, command: &str) -> bool {
    if !requires_access_control(command) {
        return true;
    }

    let user_id = message
        .from()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let chat_id = message.chat.id.0;

    if !is_access_allowed(user_id, chat_id) {
        let _ = bot
            .send_message(message.chat.id, "You are not authorized to use this command. Please contact the administrator.")
            .await;
        return false;
    }

    true
}

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use teloxide::types::{FileId, MediaGroupId};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::llm::media::MediaFile;
use crate::llm::openai_codex::{CodexReasoningEffortOption, CodexRemoteModel};
use crate::utils::timing::CommandTimer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QaCommandMode {
    Standard,
    ChatContext,
}

impl QaCommandMode {
    pub fn requires_custom_tools(self) -> bool {
        matches!(self, Self::ChatContext)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PendingQRequest {
    pub user_id: i64,
    pub username: String,
    pub query: String,
    pub original_query: String,
    pub db_query_text: String,
    pub telegram_language_code: Option<String>,
    pub media_files: Vec<MediaFile>,
    pub youtube_urls: Vec<String>,
    pub telegraph_contents: Vec<String>,
    pub twitter_contents: Vec<String>,
    pub chat_id: i64,
    pub message_id: i64,
    pub selection_message_id: i64,
    pub original_user_id: i64,
    pub reply_to_message_id: Option<i64>,
    pub llm_invocation_id: Option<i64>,
    pub timestamp: i64,
    pub command_timer: Option<CommandTimer>,
    pub mode: QaCommandMode,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PendingImageRequest {
    pub user_id: i64,
    pub chat_id: i64,
    pub message_id: i64,
    pub prompt: String,
    pub image_urls: Vec<String>,
    pub telegraph_contents: Vec<String>,
    pub original_message_text: String,
    pub selection_message_id: i64,
    pub llm_invocation_id: Option<i64>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PendingCodexModelRequest {
    pub admin_user_id: i64,
    pub chat_id: i64,
    pub selection_message_id: i64,
    pub timestamp: i64,
    pub page: usize,
    pub etag: Option<String>,
    pub models: Vec<CodexRemoteModel>,
}

#[derive(Debug, Clone)]
pub struct PendingCodexReasoningRequest {
    pub admin_user_id: i64,
    pub chat_id: i64,
    pub selection_message_id: i64,
    pub timestamp: i64,
    pub default_level: Option<String>,
    pub supported_levels: Vec<CodexReasoningEffortOption>,
}

#[derive(Debug, Clone)]
pub struct ActiveCodexLogin {
    pub admin_user_id: i64,
    pub chat_id: i64,
    pub status_message_id: i64,
    pub verification_url: String,
    pub user_code: String,
    pub started_at: i64,
    pub cancel_flag: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct MediaGroupItem {
    pub file_id: FileId,
}

#[derive(Debug, Clone)]
pub struct MediaGroupState {
    pub items: Vec<MediaGroupItem>,
    pub last_updated: Instant,
}

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub bot_user_id: i64,
    pub bot_username_lower: String,
    pub pending_q_requests: Arc<Mutex<HashMap<String, PendingQRequest>>>,
    pub pending_image_requests: Arc<Mutex<HashMap<String, PendingImageRequest>>>,
    pub pending_codex_model_requests: Arc<Mutex<HashMap<String, PendingCodexModelRequest>>>,
    pub pending_codex_reasoning_requests: Arc<Mutex<HashMap<String, PendingCodexReasoningRequest>>>,
    pub active_codex_login: Arc<Mutex<Option<ActiveCodexLogin>>>,
    pub media_groups: Arc<Mutex<HashMap<MediaGroupId, MediaGroupState>>>,
    pub heavy_command_semaphore: Arc<Semaphore>,
    pub heavy_command_waiters: Arc<AtomicUsize>,
}

impl AppState {
    pub fn new(db: Database, bot_user_id: i64, bot_username_lower: String) -> Self {
        AppState {
            db,
            bot_user_id,
            bot_username_lower,
            pending_q_requests: Arc::new(Mutex::new(HashMap::new())),
            pending_image_requests: Arc::new(Mutex::new(HashMap::new())),
            pending_codex_model_requests: Arc::new(Mutex::new(HashMap::new())),
            pending_codex_reasoning_requests: Arc::new(Mutex::new(HashMap::new())),
            active_codex_login: Arc::new(Mutex::new(None)),
            media_groups: Arc::new(Mutex::new(HashMap::new())),
            heavy_command_semaphore: Arc::new(Semaphore::new(CONFIG.heavy_command_max_concurrency)),
            heavy_command_waiters: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub async fn acquire_heavy_command_permit(&self) -> OwnedSemaphorePermit {
        self.heavy_command_waiters.fetch_add(1, Ordering::Relaxed);
        let permit = self
            .heavy_command_semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("heavy command semaphore should remain open");
        self.heavy_command_waiters.fetch_sub(1, Ordering::Relaxed);
        permit
    }

    pub fn heavy_command_active(&self) -> usize {
        CONFIG
            .heavy_command_max_concurrency
            .saturating_sub(self.heavy_command_semaphore.available_permits())
    }

    pub fn heavy_command_waiting(&self) -> usize {
        self.heavy_command_waiters.load(Ordering::Relaxed)
    }

    pub fn store_media_group_item(&self, media_group_id: &MediaGroupId, item: MediaGroupItem) {
        let mut groups = self.media_groups.lock();
        prune_media_groups(&mut groups);
        let entry = groups
            .entry(media_group_id.clone())
            .or_insert_with(|| MediaGroupState {
                items: Vec::new(),
                last_updated: Instant::now(),
            });
        entry.last_updated = Instant::now();
        entry.items.push(item);
    }

    pub fn media_group_items(&self, media_group_id: &MediaGroupId) -> Vec<MediaGroupItem> {
        let mut groups = self.media_groups.lock();
        prune_media_groups(&mut groups);
        groups
            .get_mut(media_group_id)
            .map(|group| {
                group.last_updated = Instant::now();
                group.items.clone()
            })
            .unwrap_or_default()
    }

    pub fn media_group_count(&self) -> usize {
        let mut groups = self.media_groups.lock();
        prune_media_groups(&mut groups);
        groups.len()
    }
}

fn prune_media_groups(groups: &mut HashMap<MediaGroupId, MediaGroupState>) {
    let max_items = CONFIG.media_group_max_items;
    if groups.len() <= max_items {
        return;
    }

    let mut ordered = groups
        .iter()
        .map(|(group_id, group)| (group_id.clone(), group.last_updated))
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(_, last_updated)| *last_updated);

    let remove_count = groups.len().saturating_sub(max_items);
    for (group_id, _) in ordered.into_iter().take(remove_count) {
        groups.remove(&group_id);
    }
}

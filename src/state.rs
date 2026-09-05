use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, MutexGuard};
use teloxide::types::{FileId, MediaGroupId};
use tokio::sync::{Mutex as AsyncMutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::AbortHandle;

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::llm::media::MediaFile;
use crate::llm::openai_codex::{CodexReasoningEffortOption, CodexRemoteModel};
use crate::utils::timing::CommandTimer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QaCommandMode {
    Standard,
    Quick,
    ChatContext,
    ChatSearch,
}

impl QaCommandMode {
    pub fn requires_custom_tools(self) -> bool {
        matches!(self, Self::ChatContext | Self::ChatSearch)
    }

    pub fn requires_chat_search_index(self) -> bool {
        matches!(self, Self::ChatContext | Self::ChatSearch)
    }
}

#[derive(Debug)]
pub struct PendingQRequest {
    pub user_id: i64,
    pub query: String,
    pub telegram_language_code: Option<String>,
    pub media_files: Vec<MediaFile>,
    pub youtube_urls: Vec<String>,
    pub telegraph_contents: Vec<String>,
    pub twitter_contents: Vec<String>,
    pub chat_id: i64,
    pub message_id: i64,
    pub selection_message_id: i64,
    pub original_user_id: i64,
    pub llm_invocation_id: Option<i64>,
    pub timestamp: i64,
    pub command_timer: Option<CommandTimer>,
    pub mode: QaCommandMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGenerationModel {
    Gemini,
    CodexGptImage2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingImageCommand {
    Img,
    Image,
}

#[derive(Debug, Clone)]
pub struct PendingImageRequest {
    pub user_id: i64,
    pub chat_id: i64,
    pub message_id: i64,
    pub command: PendingImageCommand,
    pub prompt: String,
    pub image_urls: Vec<String>,
    pub telegraph_contents: Vec<String>,
    pub selection_message_id: i64,
    pub llm_invocation_id: Option<i64>,
    pub model: Option<ImageGenerationModel>,
    pub codex_size: Option<String>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PendingCodexModelRequest {
    pub admin_user_id: i64,
    pub account_id: String,
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
    pub account_id: String,
    pub model_slug: String,
    pub chat_id: i64,
    pub selection_message_id: i64,
    pub timestamp: i64,
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

struct PendingEntry<T> {
    request: T,
    /// Handle of the timeout task armed by [`PendingRequests::insert_with_timeout`].
    timeout: Option<AbortHandle>,
}

impl<T> PendingEntry<T> {
    fn cancel_timeout(self) -> T {
        if let Some(timeout) = self.timeout {
            timeout.abort();
        }
        self.request
    }
}

/// Interactive requests waiting on a user's inline-keyboard choice, keyed by
/// the selection message. Each entry may carry a timeout task that removes
/// the request and hands it to a fallback once the user stops responding;
/// resolving the request through [`PendingEntryGuard::take`] cancels that
/// task so a late timeout can never double-process a request.
pub struct PendingRequests<T> {
    entries: Arc<Mutex<HashMap<String, PendingEntry<T>>>>,
}

impl<T> Clone for PendingRequests<T> {
    fn clone(&self) -> Self {
        Self {
            entries: Arc::clone(&self.entries),
        }
    }
}

impl<T> Default for PendingRequests<T> {
    fn default() -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<T> PendingRequests<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `request` under `key` with no deadline. Replaces (and cancels the
    /// timeout of) any request already stored under the same key.
    pub fn insert(&self, key: String, request: T) {
        let previous = self.entries.lock().insert(
            key,
            PendingEntry {
                request,
                timeout: None,
            },
        );
        if let Some(previous) = previous {
            previous.cancel_timeout();
        }
    }

    /// Store `request` under `key` and, unless it is taken first, remove it
    /// after `timeout` and pass it to `on_timeout`.
    pub fn insert_with_timeout<F, Fut>(
        &self,
        key: String,
        request: T,
        timeout: Duration,
        on_timeout: F,
    ) where
        T: Send + 'static,
        F: FnOnce(T) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        // Insert and arm under one lock so the timeout task can never observe
        // the map before its own handle is recorded.
        let mut entries = self.entries.lock();
        let previous = entries.insert(
            key.clone(),
            PendingEntry {
                request,
                timeout: None,
            },
        );
        let timeout_entries = Arc::clone(&self.entries);
        let timeout_key = key.clone();
        let task = tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            // This *is* the timeout task, so remove without cancelling.
            let expired = timeout_entries
                .lock()
                .remove(&timeout_key)
                .map(|entry| entry.request);
            if let Some(request) = expired {
                on_timeout(request).await;
            }
        });
        if let Some(entry) = entries.get_mut(&key) {
            entry.timeout = Some(task.abort_handle());
        }
        drop(entries);
        if let Some(previous) = previous {
            previous.cancel_timeout();
        }
    }

    /// Lock the entry under `key` for inspection, mutation, or removal.
    pub fn entry<'a>(&'a self, key: &'a str) -> PendingEntryGuard<'a, T> {
        PendingEntryGuard {
            key,
            entries: self.entries.lock(),
        }
    }

    /// Number of requests currently waiting on a selection.
    pub fn count(&self) -> usize {
        self.entries.lock().len()
    }
}

/// Locked view of one pending request; holds the map lock until dropped, so
/// a read-decide-take sequence is atomic with respect to other callbacks.
pub struct PendingEntryGuard<'a, T> {
    key: &'a str,
    entries: MutexGuard<'a, HashMap<String, PendingEntry<T>>>,
}

impl<T> PendingEntryGuard<'_, T> {
    pub fn get(&self) -> Option<&T> {
        self.entries.get(self.key).map(|entry| &entry.request)
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.entries
            .get_mut(self.key)
            .map(|entry| &mut entry.request)
    }

    /// Remove the request, cancelling its timeout task.
    pub fn take(&mut self) -> Option<T> {
        self.entries
            .remove(self.key)
            .map(PendingEntry::cancel_timeout)
    }
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
    pub pending_q_requests: PendingRequests<PendingQRequest>,
    pub pending_image_requests: PendingRequests<PendingImageRequest>,
    pub pending_codex_model_requests: PendingRequests<PendingCodexModelRequest>,
    pub pending_codex_reasoning_requests: PendingRequests<PendingCodexReasoningRequest>,
    pub active_codex_login: Arc<Mutex<Option<ActiveCodexLogin>>>,
    pub codex_auth_flow_lock: Arc<AsyncMutex<()>>,
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
            pending_q_requests: PendingRequests::new(),
            pending_image_requests: PendingRequests::new(),
            pending_codex_model_requests: PendingRequests::new(),
            pending_codex_reasoning_requests: PendingRequests::new(),
            active_codex_login: Arc::new(Mutex::new(None)),
            codex_auth_flow_lock: Arc::new(AsyncMutex::new(())),
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

    /// Reuse a permit the caller already holds, or acquire a fresh one. Lets a
    /// handler that throttled its own preparation hand the same permit to the
    /// request processor instead of taking a second slot (which could exhaust
    /// the semaphore and deadlock the heavy-command lane).
    pub async fn reuse_or_acquire_heavy_permit(
        &self,
        existing: Option<OwnedSemaphorePermit>,
    ) -> OwnedSemaphorePermit {
        match existing {
            Some(permit) => permit,
            None => self.acquire_heavy_command_permit().await,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_state(name: &str) -> AppState {
        let mut path = std::path::PathBuf::from("target");
        path.push("test-dbs");
        std::fs::create_dir_all(&path).expect("test db directory should exist");
        path.push(format!(
            "telegram-chat-bot-state-{}-{}-{}.db",
            name,
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::File::create(&path).expect("test db file should be creatable");
        let url = format!("sqlite://{}", path.to_string_lossy().replace('\\', "/"));
        let db = Database::init(&url)
            .await
            .expect("test database should initialize");
        AppState::new(db, 1, "test_bot".to_string())
    }

    #[tokio::test]
    async fn reusing_a_held_permit_does_not_take_a_second_slot() {
        let state = test_state("reuse-permit").await;
        let held = state.acquire_heavy_command_permit().await;
        assert_eq!(state.heavy_command_active(), 1);

        let reused = state.reuse_or_acquire_heavy_permit(Some(held)).await;
        assert_eq!(state.heavy_command_active(), 1, "no second permit taken");

        let fresh = state.reuse_or_acquire_heavy_permit(None).await;
        assert_eq!(state.heavy_command_active(), 2);

        drop(reused);
        drop(fresh);
        assert_eq!(state.heavy_command_active(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn pending_request_times_out_with_its_request_when_nobody_resolves_it() {
        let pending: PendingRequests<u32> = PendingRequests::new();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let sink = fired.clone();
        pending.insert_with_timeout(
            "k".to_string(),
            7,
            Duration::from_secs(30),
            move |request| async move {
                sink.lock().push(request);
            },
        );
        assert_eq!(pending.count(), 1);

        tokio::time::sleep(Duration::from_secs(31)).await;
        tokio::task::yield_now().await;

        assert_eq!(*fired.lock(), vec![7]);
        assert_eq!(pending.count(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn taking_a_pending_request_cancels_its_timeout() {
        let pending: PendingRequests<u32> = PendingRequests::new();
        let fired = Arc::new(AtomicBool::new(false));
        let flag = fired.clone();
        pending.insert_with_timeout(
            "k".to_string(),
            7,
            Duration::from_secs(30),
            move |_| async move {
                flag.store(true, Ordering::SeqCst);
            },
        );

        assert_eq!(pending.entry("k").take(), Some(7));
        assert_eq!(pending.entry("k").take(), None);

        tokio::time::sleep(Duration::from_secs(31)).await;
        tokio::task::yield_now().await;
        assert!(
            !fired.load(Ordering::SeqCst),
            "timeout must not fire after the request was resolved"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn re_inserting_a_key_cancels_the_previous_timeout() {
        let pending: PendingRequests<u32> = PendingRequests::new();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let first = fired.clone();
        pending.insert_with_timeout(
            "k".to_string(),
            1,
            Duration::from_secs(10),
            move |request| async move {
                first.lock().push(request);
            },
        );
        let second = fired.clone();
        pending.insert_with_timeout(
            "k".to_string(),
            2,
            Duration::from_secs(20),
            move |request| async move {
                second.lock().push(request);
            },
        );

        tokio::time::sleep(Duration::from_secs(21)).await;
        tokio::task::yield_now().await;
        assert_eq!(*fired.lock(), vec![2]);
    }

    #[tokio::test]
    async fn entry_guard_reads_mutates_and_takes_in_place() {
        let pending: PendingRequests<String> = PendingRequests::new();
        pending.insert("k".to_string(), "a".to_string());
        {
            let mut entry = pending.entry("k");
            assert_eq!(entry.get().map(String::as_str), Some("a"));
            entry.get_mut().unwrap().push('b');
        }
        assert_eq!(pending.entry("k").take(), Some("ab".to_string()));
        assert!(pending.entry("k").get().is_none());
        assert_eq!(pending.count(), 0);
    }
}

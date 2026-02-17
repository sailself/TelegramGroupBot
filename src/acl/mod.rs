use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::config::CONFIG;

#[derive(Debug, Clone)]
pub struct AclDecision {
    pub allowed: bool,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Default)]
pub struct AclMeta {
    pub path: String,
    pub loaded: bool,
    pub version: u32,
    pub source_exists: bool,
    pub owner_user_count: usize,
    pub full_access_chat_count: usize,
    pub chat_rule_count: usize,
    pub global_allow_command_count: usize,
    pub global_allow_tool_count: usize,
    pub last_loaded_unix_ms: Option<i64>,
    pub file_mtime_unix_ms: Option<i64>,
    pub last_error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct AclFileRaw {
    version: u32,
    owner_user_ids: Vec<i64>,
    full_access_chat_ids: Vec<i64>,
    global: GlobalAclRaw,
    chats: HashMap<String, ChatAclRaw>,
}

impl Default for AclFileRaw {
    fn default() -> Self {
        Self {
            version: 1,
            owner_user_ids: Vec::new(),
            full_access_chat_ids: Vec::new(),
            global: GlobalAclRaw::default(),
            chats: HashMap::new(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct GlobalAclRaw {
    allow_commands: Vec<String>,
    allow_tools: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct ChatAclRaw {
    full_access: bool,
    allow_commands: Vec<String>,
    deny_commands: Vec<String>,
    allow_tools: Vec<String>,
    deny_tools: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct GlobalAclCompiled {
    allow_commands: HashSet<String>,
    allow_tools: HashSet<String>,
}

#[derive(Debug, Clone, Default)]
struct ChatAclCompiled {
    full_access: bool,
    allow_commands: HashSet<String>,
    deny_commands: HashSet<String>,
    allow_tools: HashSet<String>,
    deny_tools: HashSet<String>,
}

#[derive(Debug, Clone, Default)]
struct AclCompiled {
    version: u32,
    owner_user_ids: HashSet<i64>,
    full_access_chat_ids: HashSet<i64>,
    global: GlobalAclCompiled,
    chats: HashMap<i64, ChatAclCompiled>,
}

#[derive(Debug)]
struct AclState {
    snapshot: Arc<AclCompiled>,
    has_attempted_load: bool,
    last_checked: Option<Instant>,
    file_mtime: Option<SystemTime>,
    meta: AclMeta,
}

pub struct AclManager {
    path: PathBuf,
    ttl: Duration,
    state: Mutex<AclState>,
}

static ACL_MANAGER: Lazy<AclManager> = Lazy::new(|| {
    AclManager::new(
        PathBuf::from(&CONFIG.acl_file_path),
        Duration::from_secs(CONFIG.acl_reload_ttl_seconds),
    )
});

pub fn acl_manager() -> &'static AclManager {
    &ACL_MANAGER
}

fn normalize_permission_name(raw: &str) -> String {
    raw.trim().trim_start_matches('/').to_ascii_lowercase()
}

fn normalize_permission_set(raw: Vec<String>) -> HashSet<String> {
    raw.into_iter()
        .map(|value| normalize_permission_name(&value))
        .filter(|value| !value.is_empty())
        .collect()
}

fn system_time_to_unix_ms(value: SystemTime) -> Option<i64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
}

fn now_unix_ms() -> i64 {
    system_time_to_unix_ms(SystemTime::now()).unwrap_or(0)
}

fn compile_acl(raw: AclFileRaw) -> Result<AclCompiled> {
    let mut chats = HashMap::new();
    for (chat_id_raw, chat_acl_raw) in raw.chats {
        let chat_id = chat_id_raw
            .trim()
            .parse::<i64>()
            .map_err(|err| anyhow!("Invalid chat id key '{}': {}", chat_id_raw, err))?;
        let chat_acl = ChatAclCompiled {
            full_access: chat_acl_raw.full_access,
            allow_commands: normalize_permission_set(chat_acl_raw.allow_commands),
            deny_commands: normalize_permission_set(chat_acl_raw.deny_commands),
            allow_tools: normalize_permission_set(chat_acl_raw.allow_tools),
            deny_tools: normalize_permission_set(chat_acl_raw.deny_tools),
        };
        chats.insert(chat_id, chat_acl);
    }

    Ok(AclCompiled {
        version: raw.version,
        owner_user_ids: raw.owner_user_ids.into_iter().collect(),
        full_access_chat_ids: raw.full_access_chat_ids.into_iter().collect(),
        global: GlobalAclCompiled {
            allow_commands: normalize_permission_set(raw.global.allow_commands),
            allow_tools: normalize_permission_set(raw.global.allow_tools),
        },
        chats,
    })
}

fn load_acl_snapshot(path: &Path) -> Result<AclCompiled> {
    let body = fs::read_to_string(path)
        .map_err(|err| anyhow!("Failed to read ACL file '{}': {}", path.display(), err))?;
    let raw: AclFileRaw = serde_json::from_str(&body)
        .map_err(|err| anyhow!("Failed to parse ACL JSON '{}': {}", path.display(), err))?;
    compile_acl(raw)
}

impl AclManager {
    pub fn new(path: PathBuf, ttl: Duration) -> Self {
        let meta = AclMeta {
            path: path.display().to_string(),
            ..AclMeta::default()
        };
        Self {
            path,
            ttl,
            state: Mutex::new(AclState {
                snapshot: Arc::new(AclCompiled::default()),
                has_attempted_load: false,
                last_checked: None,
                file_mtime: None,
                meta,
            }),
        }
    }

    fn current_mtime(&self) -> Option<SystemTime> {
        fs::metadata(&self.path)
            .ok()
            .and_then(|meta| meta.modified().ok())
    }

    fn build_meta(
        &self,
        snapshot: &AclCompiled,
        source_exists: bool,
        file_mtime: Option<SystemTime>,
        last_error: Option<String>,
    ) -> AclMeta {
        AclMeta {
            path: self.path.display().to_string(),
            loaded: true,
            version: snapshot.version,
            source_exists,
            owner_user_count: snapshot.owner_user_ids.len(),
            full_access_chat_count: snapshot.full_access_chat_ids.len(),
            chat_rule_count: snapshot.chats.len(),
            global_allow_command_count: snapshot.global.allow_commands.len(),
            global_allow_tool_count: snapshot.global.allow_tools.len(),
            last_loaded_unix_ms: Some(now_unix_ms()),
            file_mtime_unix_ms: file_mtime.and_then(system_time_to_unix_ms),
            last_error,
        }
    }

    fn apply_reload_error(&self, mtime: Option<SystemTime>, error_text: String) {
        let mut state = self.state.lock();
        state.has_attempted_load = true;
        state.file_mtime = mtime;
        state.meta.source_exists = mtime.is_some();
        state.meta.file_mtime_unix_ms = mtime.and_then(system_time_to_unix_ms);
        state.meta.last_error = Some(error_text.clone());
        if state.meta.last_loaded_unix_ms.is_none() {
            state.meta.last_loaded_unix_ms = Some(now_unix_ms());
        }
        warn!("{}", error_text);
    }

    fn maybe_reload_with_ttl(&self) {
        let now = Instant::now();
        let mtime = self.current_mtime();
        let should_attempt = {
            let mut state = self.state.lock();
            if let Some(last_checked) = state.last_checked {
                if now.duration_since(last_checked) < self.ttl {
                    return;
                }
            }
            state.last_checked = Some(now);
            !state.has_attempted_load || state.file_mtime != mtime
        };
        if !should_attempt {
            return;
        }

        match load_acl_snapshot(&self.path) {
            Ok(snapshot) => {
                let meta = self.build_meta(&snapshot, mtime.is_some(), mtime, None);
                let mut state = self.state.lock();
                state.snapshot = Arc::new(snapshot);
                state.file_mtime = mtime;
                state.has_attempted_load = true;
                state.meta = meta;
                info!(
                    "ACL reloaded from {} (version={}, chats={})",
                    state.meta.path, state.meta.version, state.meta.chat_rule_count
                );
            }
            Err(err) => {
                self.apply_reload_error(mtime, err.to_string());
            }
        }
    }

    fn snapshot(&self) -> Arc<AclCompiled> {
        self.maybe_reload_with_ttl();
        self.state.lock().snapshot.clone()
    }

    fn is_command_allowed_in_snapshot(
        snapshot: &AclCompiled,
        chat_id: i64,
        user_id: i64,
        command: &str,
    ) -> AclDecision {
        if snapshot.owner_user_ids.contains(&user_id) {
            return AclDecision {
                allowed: true,
                reason: "owner_bypass",
            };
        }
        if snapshot.full_access_chat_ids.contains(&chat_id) {
            return AclDecision {
                allowed: true,
                reason: "full_access_chat",
            };
        }
        if let Some(chat_acl) = snapshot.chats.get(&chat_id) {
            if chat_acl.full_access {
                return AclDecision {
                    allowed: true,
                    reason: "chat_full_access",
                };
            }
            if chat_acl.deny_commands.contains(command) {
                return AclDecision {
                    allowed: false,
                    reason: "chat_deny",
                };
            }
            if chat_acl.allow_commands.contains(command) {
                return AclDecision {
                    allowed: true,
                    reason: "chat_allow",
                };
            }
        }
        if snapshot.global.allow_commands.contains(command) {
            return AclDecision {
                allowed: true,
                reason: "global_allow",
            };
        }
        AclDecision {
            allowed: false,
            reason: "not_allowed",
        }
    }

    fn is_tool_allowed_in_snapshot(
        snapshot: &AclCompiled,
        chat_id: i64,
        user_id: i64,
        tool_name: &str,
    ) -> AclDecision {
        if snapshot.owner_user_ids.contains(&user_id) {
            return AclDecision {
                allowed: true,
                reason: "owner_bypass",
            };
        }
        if snapshot.full_access_chat_ids.contains(&chat_id) {
            return AclDecision {
                allowed: true,
                reason: "full_access_chat",
            };
        }
        if let Some(chat_acl) = snapshot.chats.get(&chat_id) {
            if chat_acl.full_access {
                return AclDecision {
                    allowed: true,
                    reason: "chat_full_access",
                };
            }
            if chat_acl.deny_tools.contains(tool_name) {
                return AclDecision {
                    allowed: false,
                    reason: "chat_deny",
                };
            }
            if chat_acl.allow_tools.contains(tool_name) {
                return AclDecision {
                    allowed: true,
                    reason: "chat_allow",
                };
            }
        }
        if snapshot.global.allow_tools.contains(tool_name) {
            return AclDecision {
                allowed: true,
                reason: "global_allow",
            };
        }
        AclDecision {
            allowed: false,
            reason: "not_allowed",
        }
    }

    pub fn initialize(&self) {
        if let Err(err) = self.reload_now() {
            warn!("ACL initial load failed: {}", err);
        }
    }

    pub fn reload_now(&self) -> Result<AclMeta> {
        let mtime = self.current_mtime();
        let snapshot = match load_acl_snapshot(&self.path) {
            Ok(snapshot) => snapshot,
            Err(err) => {
                self.apply_reload_error(mtime, err.to_string());
                return Err(err);
            }
        };
        let meta = self.build_meta(&snapshot, mtime.is_some(), mtime, None);
        let mut state = self.state.lock();
        state.snapshot = Arc::new(snapshot);
        state.file_mtime = mtime;
        state.has_attempted_load = true;
        state.last_checked = Some(Instant::now());
        state.meta = meta.clone();
        info!(
            "ACL force reloaded from {} (version={}, chats={})",
            meta.path, meta.version, meta.chat_rule_count
        );
        Ok(meta)
    }

    pub fn snapshot_meta(&self) -> AclMeta {
        self.maybe_reload_with_ttl();
        self.state.lock().meta.clone()
    }

    pub fn is_owner(&self, user_id: i64) -> bool {
        self.snapshot().owner_user_ids.contains(&user_id)
    }

    pub fn authorize_command(&self, chat_id: i64, user_id: i64, command: &str) -> AclDecision {
        if !CONFIG.acl_enforced {
            return AclDecision {
                allowed: true,
                reason: "acl_disabled",
            };
        }
        let normalized = normalize_permission_name(command);
        if normalized.is_empty() {
            return AclDecision {
                allowed: false,
                reason: "empty_command",
            };
        }
        let snapshot = self.snapshot();
        Self::is_command_allowed_in_snapshot(&snapshot, chat_id, user_id, &normalized)
    }

    pub fn authorize_tool(&self, chat_id: i64, user_id: i64, tool_name: &str) -> AclDecision {
        if !CONFIG.acl_enforced {
            return AclDecision {
                allowed: true,
                reason: "acl_disabled",
            };
        }
        let normalized = normalize_permission_name(tool_name);
        if normalized.is_empty() {
            return AclDecision {
                allowed: false,
                reason: "empty_tool",
            };
        }
        let snapshot = self.snapshot();
        Self::is_tool_allowed_in_snapshot(&snapshot, chat_id, user_id, &normalized)
    }

    pub fn filter_allowed_tools(
        &self,
        chat_id: i64,
        user_id: i64,
        candidate_tool_names: &[String],
    ) -> Vec<String> {
        if candidate_tool_names.is_empty() {
            return Vec::new();
        }
        if !CONFIG.acl_enforced {
            let mut all = candidate_tool_names
                .iter()
                .map(|tool| normalize_permission_name(tool))
                .filter(|tool| !tool.is_empty())
                .collect::<Vec<_>>();
            all.sort();
            all.dedup();
            return all;
        }

        let snapshot = self.snapshot();
        let mut seen = HashSet::new();
        let mut allowed = Vec::new();
        for tool in candidate_tool_names {
            let normalized = normalize_permission_name(tool);
            if normalized.is_empty() || !seen.insert(normalized.clone()) {
                continue;
            }
            let decision =
                Self::is_tool_allowed_in_snapshot(&snapshot, chat_id, user_id, &normalized);
            if decision.allowed {
                allowed.push(normalized);
            } else {
                debug!(
                    "Tool '{}' excluded by ACL in chat {} for user {} ({})",
                    normalized, chat_id, user_id, decision.reason
                );
            }
        }
        allowed.sort();
        allowed
    }
}

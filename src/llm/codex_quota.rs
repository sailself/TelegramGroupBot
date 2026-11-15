//! The Codex quota hold. When the ChatGPT account's usage window is spent the
//! backend answers 429 with the time it resets; until then Codex is left out
//! of model selection instead of failing every request that reaches it.
//!
//! The hold is one timestamp bound to the account that hit the limit. It is
//! compared lazily (nothing has to run when it expires) and kept in a small
//! file beside the selected-model record so a restart does not forget it.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::llm::transport::UsageLimitNotice;
use crate::utils::timing::now_unix_seconds;

/// Slack added to the reported reset, so the first request after it does not
/// arrive a moment early.
const RESET_MARGIN_SECS: i64 = 60;
/// Hold used when the backend named no reset time. Short on purpose: the next
/// usage-limit response replaces it with a real one.
const UNKNOWN_RESET_HOLD_SECS: i64 = 15 * 60;
/// Longest hold accepted from the backend; the weekly window is the longest
/// real one. Guards against a nonsense reset hiding Codex indefinitely.
const MAX_HOLD_SECS: i64 = 8 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodexQuotaHold {
    /// Unix seconds until which Codex stays out of model selection.
    pub until: i64,
    /// ChatGPT account whose window is spent. Another account is not held.
    #[serde(default)]
    pub account_id: Option<String>,
}

impl CodexQuotaHold {
    /// Whether the hold still applies at `now` for the signed-in account. An
    /// unknown account on either side cannot rule the hold out.
    fn covers(&self, now: i64, current_account: Option<&str>) -> bool {
        if now >= self.until {
            return false;
        }
        match (self.account_id.as_deref(), current_account) {
            (Some(held), Some(current)) => held.trim() == current.trim(),
            _ => true,
        }
    }
}

/// When a hold set at `now` for `notice` ends. The relative reset is preferred
/// over the absolute one because it does not depend on this host's clock
/// agreeing with the backend's.
pub(crate) fn hold_until(notice: UsageLimitNotice, now: i64) -> i64 {
    let wait = notice
        .resets_in_seconds
        .filter(|seconds| *seconds > 0)
        .or_else(|| {
            notice
                .resets_at
                .map(|at| at.saturating_sub(now))
                .filter(|seconds| *seconds > 0)
        });
    match wait {
        Some(seconds) => {
            now.saturating_add(seconds.saturating_add(RESET_MARGIN_SECS).min(MAX_HOLD_SECS))
        }
        None => now.saturating_add(UNKNOWN_RESET_HOLD_SECS),
    }
}

/// A Codex request refused because the account's usage window is spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("OpenAI Codex usage limit reached; on hold until unix time {until}")]
pub struct CodexUsageLimitError {
    pub until: i64,
}

impl CodexUsageLimitError {
    /// How long until Codex is back, e.g. `3h 25m`.
    pub(crate) fn wait_label(&self, now: i64) -> String {
        let minutes = self.until.saturating_sub(now) / 60;
        let (days, hours, minutes) = (minutes / 1_440, minutes % 1_440 / 60, minutes % 60);
        if days > 0 {
            format!("{days}d {hours}h")
        } else if hours > 0 {
            format!("{hours}h {minutes}m")
        } else if minutes > 0 {
            format!("{minutes}m")
        } else {
            "under a minute".to_string()
        }
    }

    /// What happened and when it ends, without advice on what to do next.
    pub fn availability_note(&self) -> String {
        format!(
            "Codex has reached its usage limit and should be back in about {}.",
            self.wait_label(now_unix_seconds())
        )
    }

    pub fn user_message(&self) -> String {
        format!(
            "{} Please try again then or pick another model.",
            self.availability_note()
        )
    }
}

/// The current hold and, outside tests, the file that keeps it across restarts.
#[derive(Debug, Default)]
struct HoldStore {
    hold: Option<CodexQuotaHold>,
    path: Option<PathBuf>,
}

impl HoldStore {
    fn open(path: PathBuf) -> Self {
        Self {
            hold: read_hold_file(&path),
            path: Some(path),
        }
    }

    fn set(&mut self, hold: CodexQuotaHold) {
        if let Some(path) = self.path.as_deref() {
            if let Err(err) = write_hold_file(path, &hold) {
                warn!(
                    "Failed to save the Codex quota hold to {}: {err}",
                    path.display()
                );
            }
        }
        self.hold = Some(hold);
    }

    fn clear(&mut self) {
        self.hold = None;
        if let Some(path) = self.path.as_deref() {
            if let Err(err) = fs::remove_file(path) {
                if err.kind() != std::io::ErrorKind::NotFound {
                    warn!(
                        "Failed to remove the Codex quota hold file {}: {err}",
                        path.display()
                    );
                }
            }
        }
    }

    /// The hold if it has not expired. An expired one is dropped along with
    /// its file, so expiry needs no timer.
    fn unexpired(&mut self, now: i64) -> Option<CodexQuotaHold> {
        let hold = self.hold.as_ref()?;
        if now >= hold.until {
            info!("The Codex quota hold expired; Codex is selectable again");
            self.clear();
            return None;
        }
        self.hold.clone()
    }
}

fn read_hold_file(path: &Path) -> Option<CodexQuotaHold> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                warn!(
                    "Failed to read the Codex quota hold file {}: {err}",
                    path.display()
                );
            }
            return None;
        }
    };
    match serde_json::from_str(&raw) {
        Ok(hold) => Some(hold),
        Err(err) => {
            warn!(
                "Ignoring the unreadable Codex quota hold file {}: {err}",
                path.display()
            );
            None
        }
    }
}

fn write_hold_file(path: &Path, hold: &CodexQuotaHold) -> std::io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec(hold)?)
}

#[cfg(not(test))]
fn with_store<R>(f: impl FnOnce(&mut HoldStore) -> R) -> R {
    use std::sync::LazyLock;

    static STORE: LazyLock<parking_lot::Mutex<HoldStore>> = LazyLock::new(|| {
        let model_path = Path::new(&crate::config::CONFIG.codex.model_path);
        parking_lot::Mutex::new(HoldStore::open(
            model_path.with_file_name("openai_codex_quota.json"),
        ))
    });
    f(&mut STORE.lock())
}

/// Tests get one in-memory store per thread, so a test that sets a hold
/// cannot hide Codex from tests running beside it or touch the real file.
#[cfg(test)]
fn with_store<R>(f: impl FnOnce(&mut HoldStore) -> R) -> R {
    thread_local! {
        static STORE: std::cell::RefCell<HoldStore> = std::cell::RefCell::new(HoldStore::default());
    }
    STORE.with(|store| f(&mut store.borrow_mut()))
}

/// Record that `account_id`'s usage window is spent and return the error the
/// failed request reports.
pub fn note_usage_limit(
    notice: UsageLimitNotice,
    account_id: Option<&str>,
) -> CodexUsageLimitError {
    let until = hold_until(notice, now_unix_seconds());
    let account_id = account_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    with_store(|store| store.set(CodexQuotaHold { until, account_id }));
    info!("Codex usage limit reached; Codex is on hold until unix time {until}");
    CodexUsageLimitError { until }
}

/// The hold in force for the signed-in ChatGPT account, if any.
pub fn active_hold() -> Option<CodexQuotaHold> {
    // The account id costs an auth-file read, so it is only looked up while a
    // hold exists.
    let hold = with_store(|store| store.unexpired(now_unix_seconds()))?;
    let current_account = crate::llm::runtime_models::current_codex_account_id();
    hold.covers(now_unix_seconds(), current_account.as_deref())
        .then_some(hold)
}

/// The hold in force, as the error a Codex request would fail with.
pub fn active_limit() -> Option<CodexUsageLimitError> {
    active_hold().map(|hold| CodexUsageLimitError { until: hold.until })
}

/// The hold this test thread recorded, expired or not.
#[cfg(test)]
pub(crate) fn stored_hold_for_test() -> Option<CodexQuotaHold> {
    with_store(|store| store.hold.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_794_712_835;

    fn notice(resets_at: Option<i64>, resets_in_seconds: Option<i64>) -> UsageLimitNotice {
        UsageLimitNotice {
            resets_at,
            resets_in_seconds,
        }
    }

    fn hold(until: i64, account_id: Option<&str>) -> CodexQuotaHold {
        CodexQuotaHold {
            until,
            account_id: account_id.map(str::to_string),
        }
    }

    fn temp_hold_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "telegram_bot_codex_quota_{label}_{}_{}.json",
            std::process::id(),
            NOW
        ))
    }

    #[test]
    fn hold_ends_a_margin_after_the_reported_reset() {
        // The logged response: 13974s until the reset.
        assert_eq!(
            hold_until(notice(Some(1_794_726_809), Some(13_974)), NOW),
            NOW + 13_974 + RESET_MARGIN_SECS
        );
        // Only an absolute reset.
        assert_eq!(
            hold_until(notice(Some(NOW + 600), None), NOW),
            NOW + 600 + RESET_MARGIN_SECS
        );
    }

    #[test]
    fn relative_reset_wins_when_the_local_clock_disagrees_with_the_backend() {
        // This host's clock runs an hour ahead of the backend's.
        let skewed_now = 1_794_726_809 + 3_600;
        assert_eq!(
            hold_until(notice(Some(1_794_726_809), Some(13_974)), skewed_now),
            skewed_now + 13_974 + RESET_MARGIN_SECS
        );
    }

    #[test]
    fn missing_or_unusable_resets_get_a_short_hold() {
        for unusable in [
            notice(None, None),
            notice(Some(NOW - 10), None),
            notice(None, Some(0)),
            notice(None, Some(-5)),
        ] {
            assert_eq!(
                hold_until(unusable, NOW),
                NOW + UNKNOWN_RESET_HOLD_SECS,
                "{unusable:?}"
            );
        }
    }

    #[test]
    fn absurd_resets_are_capped() {
        assert_eq!(
            hold_until(notice(None, Some(400 * 24 * 60 * 60)), NOW),
            NOW + MAX_HOLD_SECS
        );
    }

    #[test]
    fn hold_covers_its_account_until_it_expires() {
        let held = hold(NOW + 100, Some("account-a"));
        assert!(held.covers(NOW, Some("account-a")));
        assert!(held.covers(NOW + 99, Some("account-a")));
        assert!(!held.covers(NOW + 100, Some("account-a")), "expired");
        assert!(!held.covers(NOW, Some("account-b")), "another account");
    }

    #[test]
    fn hold_covers_when_either_account_is_unknown() {
        assert!(hold(NOW + 100, None).covers(NOW, Some("account-a")));
        assert!(hold(NOW + 100, Some("account-a")).covers(NOW, None));
    }

    #[test]
    fn wait_label_reads_as_a_rough_duration() {
        let limit = CodexUsageLimitError {
            until: NOW + 12_313,
        };
        assert_eq!(limit.wait_label(NOW), "3h 25m");
        assert_eq!(limit.wait_label(NOW + 12_313 - 1_500), "25m");
        assert_eq!(limit.wait_label(NOW + 12_313 - 20), "under a minute");
        assert_eq!(limit.wait_label(NOW + 12_313 + 50), "under a minute");
        let weekly = CodexUsageLimitError {
            until: NOW + 5 * 86_400 + 3 * 3_600 + 59,
        };
        assert_eq!(weekly.wait_label(NOW), "5d 3h");
    }

    #[test]
    fn store_keeps_the_hold_across_a_restart() {
        let path = temp_hold_path("restart");
        let _ = fs::remove_file(&path);

        let mut store = HoldStore::open(path.clone());
        assert_eq!(store.unexpired(NOW), None);
        store.set(hold(NOW + 100, Some("account-a")));

        let mut restarted = HoldStore::open(path.clone());
        assert_eq!(
            restarted.unexpired(NOW),
            Some(hold(NOW + 100, Some("account-a")))
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn store_drops_an_expired_hold_and_its_file() {
        let path = temp_hold_path("expired");
        let _ = fs::remove_file(&path);

        let mut store = HoldStore::open(path.clone());
        store.set(hold(NOW + 100, None));
        assert!(path.exists());

        assert_eq!(store.unexpired(NOW + 100), None);
        assert!(!path.exists(), "the expired hold's file is removed");
        assert_eq!(HoldStore::open(path.clone()).unexpired(NOW), None);
    }

    #[test]
    fn store_ignores_an_unreadable_file() {
        let path = temp_hold_path("garbage");
        fs::write(&path, "not json").unwrap();
        assert_eq!(HoldStore::open(path.clone()).unexpired(NOW), None);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn noting_a_usage_limit_sets_the_hold_and_reports_it() {
        let limit = note_usage_limit(notice(None, Some(600)), Some(" account-a "));
        let stored = with_store(|store| store.unexpired(now_unix_seconds())).expect("hold is set");
        assert_eq!(stored.until, limit.until);
        assert_eq!(stored.account_id.as_deref(), Some("account-a"));
        assert!(limit.until >= now_unix_seconds() + 600);
    }
}

use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use crate::config::CONFIG;
use crate::db::schema::{
    count_messages, current_search_schema_version, ensure_llm_audit_schema, ensure_messages_schema,
    ensure_search_fts_exists, ensure_search_support_schema, recreate_search_fts,
    reset_search_versions, set_search_schema_version,
};
use crate::db::search::CURRENT_SEARCH_SCHEMA_VERSION;
use crate::db::search_index::{count_pending_search_rows, spawn_search_rebuild};
use crate::db::writer::{db_writer, WriterCommand};
use anyhow::Result;
use parking_lot::Mutex;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::info;

#[derive(Clone)]
pub struct Database {
    pub(super) pool: SqlitePool,
    pub(super) sender: mpsc::Sender<WriterCommand>,
    pub(super) search_ready: Arc<AtomicBool>,
    pub(super) writer_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl Database {
    pub async fn init(database_url: &str) -> Result<Self> {
        // Connection-level settings go on the connect options so every pooled
        // connection gets them, not just whichever one ran a one-off PRAGMA.
        let connect_options = SqliteConnectOptions::from_str(database_url)?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(5))
            .foreign_keys(true)
            .pragma("cache_size", "-65536")
            .pragma("mmap_size", "134217728");
        let pool = SqlitePoolOptions::new()
            .max_connections(CONFIG.db_max_connections)
            .connect_with(connect_options)
            .await?;
        let search_ready = Arc::new(AtomicBool::new(false));

        ensure_messages_schema(&pool).await?;
        ensure_search_support_schema(&pool).await?;
        ensure_llm_audit_schema(&pool).await?;
        sqlx::query("PRAGMA optimize").execute(&pool).await?;

        let schema_version = current_search_schema_version(&pool).await?;
        if schema_version != CURRENT_SEARCH_SCHEMA_VERSION {
            recreate_search_fts(&pool).await?;
            reset_search_versions(&pool).await?;
        } else {
            ensure_search_fts_exists(&pool).await?;
        }

        info!("Database tables created successfully");

        let (sender, receiver) = mpsc::channel(CONFIG.db_queue_capacity);
        let writer_task = tokio::spawn(db_writer(pool.clone(), receiver));

        info!("Database writer task started");

        let total_rows = count_messages(&pool).await?;
        let pending_rows = count_pending_search_rows(&pool).await?;
        if total_rows == 0 || pending_rows == 0 {
            set_search_schema_version(&pool, CURRENT_SEARCH_SCHEMA_VERSION).await?;
            search_ready.store(true, Ordering::Relaxed);
        } else {
            search_ready.store(false, Ordering::Relaxed);
            spawn_search_rebuild(pool.clone(), search_ready.clone());
        }

        Ok(Database {
            pool,
            sender,
            search_ready,
            writer_task: Arc::new(Mutex::new(Some(writer_task))),
        })
    }

    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    #[cfg(test)]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

pub use crate::db::messages::build_message_insert;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::db::models::{LlmInvocationInsert, LlmRequestInsert};
    use chrono::Utc;

    use std::path::PathBuf;
    use tokio::time::{sleep, Duration};

    pub(crate) fn test_db_path(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("target");
        path.push("test-dbs");
        std::fs::create_dir_all(&path).expect("test db directory should exist");
        path.push(format!(
            "telegram-chat-bot-{}-{}-{}.db",
            test_name,
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let _ = std::fs::File::create(&path).expect("test db file should be creatable");
        path
    }

    pub(crate) fn sqlite_url_for_path(path: &std::path::Path) -> String {
        format!("sqlite://{}", path.to_string_lossy().replace('\\', "/"))
    }

    pub(crate) async fn init_test_db(test_name: &str) -> Database {
        let path = test_db_path(test_name);
        let db = Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("test database should initialize");
        wait_for_search_ready(&db).await;
        db
    }

    pub(crate) async fn wait_for_search_ready(db: &Database) {
        for _ in 0..200 {
            if db.is_search_ready() {
                return;
            }
            sleep(Duration::from_millis(25)).await;
        }
        panic!("search index did not become ready in time");
    }

    async fn wait_for_message_row(db: &Database, chat_id: i64, message_id: i64) {
        for _ in 0..100 {
            let count = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND message_id = ?",
            )
            .bind(chat_id)
            .bind(message_id)
            .fetch_one(db.pool())
            .await
            .expect("message row lookup should succeed");
            if count > 0 {
                return;
            }
            sleep(Duration::from_millis(20)).await;
        }
        panic!("message row did not become visible in time");
    }

    #[tokio::test]
    async fn init_creates_a_missing_database_file() {
        let path = test_db_path("create-if-missing");
        std::fs::remove_file(&path).expect("test db placeholder should be removable");
        assert!(!path.exists());

        let db = Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("init should create the database file when it is missing");

        assert!(path.exists());
        db.health_check()
            .await
            .expect("fresh database should answer queries");
    }

    #[tokio::test]
    async fn connection_pragmas_apply_to_every_pooled_connection() {
        let db = init_test_db("pragmas-per-connection").await;
        let mut first = db.pool().acquire().await.expect("first connection");
        let mut second = db.pool().acquire().await.expect("second connection");

        // sqlx already applies foreign_keys/busy_timeout per connection; these
        // three are the ones a one-off `PRAGMA` via the pool leaves unset on
        // every connection but the first.
        for conn in [&mut first, &mut second] {
            let synchronous: i64 = sqlx::query_scalar("PRAGMA synchronous")
                .fetch_one(&mut **conn)
                .await
                .expect("synchronous pragma should be readable");
            assert_eq!(
                synchronous, 1,
                "synchronous must be NORMAL on every connection"
            );

            let cache_size: i64 = sqlx::query_scalar("PRAGMA cache_size")
                .fetch_one(&mut **conn)
                .await
                .expect("cache_size pragma should be readable");
            assert_eq!(
                cache_size, -65_536,
                "cache_size must be set on every connection"
            );

            let mmap_size: i64 = sqlx::query_scalar("PRAGMA mmap_size")
                .fetch_one(&mut **conn)
                .await
                .expect("mmap_size pragma should be readable");
            assert_eq!(
                mmap_size, 134_217_728,
                "mmap_size must be set on every connection"
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn insert_invocation_with_usage(
        db: &Database,
        chat_id: i64,
        user_id: Option<i64>,
        username: Option<&str>,
        message_id: i64,
        provider: &str,
        model: &str,
        input_tokens: Option<i64>,
        output_tokens: Option<i64>,
        total_tokens: Option<i64>,
    ) {
        let invocation_id = db
            .insert_llm_invocation(LlmInvocationInsert {
                trigger_kind: "command".to_string(),
                trigger_name: "q".to_string(),
                chat_id,
                user_id,
                username: username.map(str::to_string),
                message_id,
                reply_to_message_id: None,
                message_text: Some("/q tokens".to_string()),
                created_at: Utc::now(),
            })
            .await
            .expect("invocation insert should succeed");

        db.insert_llm_request(LlmRequestInsert {
            invocation_id,
            provider: provider.to_string(),
            model: model.to_string(),
            operation: "call_model".to_string(),
            response_id: None,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            duration_ms: 50,
            input_tokens,
            output_tokens,
            total_tokens,
            reasoning_tokens: None,
            cached_input_tokens: None,
            cache_write_tokens: None,
            raw_usage_json: None,
            status: "success".to_string(),
            error_summary: None,
        })
        .await
        .expect("request insert should succeed");
    }

    pub(crate) async fn queue_message(
        db: &Database,
        message_id: i64,
        chat_id: i64,
        username: &str,
        text: &str,
    ) {
        let insert = build_message_insert(
            Some(123_i64),
            Some(username.to_string()),
            Some(text.to_string()),
            Some("en".to_string()),
            Utc::now(),
            None,
            Some(chat_id),
            Some(message_id),
            None,
            false,
            None,
            text.trim_start().starts_with('/'),
            false,
        );
        db.queue_message_insert(insert)
            .await
            .expect("message queue should succeed");
        wait_for_message_row(db, chat_id, message_id).await;
    }

    pub(crate) async fn queue_ai_request(
        db: &Database,
        message_id: i64,
        chat_id: i64,
        username: &str,
        wrapper_text: &str,
        search_source_text: &str,
        ai_command: &str,
    ) {
        let insert = build_message_insert(
            Some(123_i64),
            Some(username.to_string()),
            Some(wrapper_text.to_string()),
            Some("en".to_string()),
            Utc::now(),
            None,
            Some(chat_id),
            Some(message_id),
            Some(search_source_text.to_string()),
            true,
            Some(ai_command.to_string()),
            true,
            true,
        );
        db.queue_message_insert(insert)
            .await
            .expect("ai request queue should succeed");
        wait_for_message_row(db, chat_id, message_id).await;
    }

    pub(crate) async fn insert_legacy_message(
        pool: &SqlitePool,
        message_id: i64,
        chat_id: i64,
        username: &str,
        text: &str,
    ) {
        sqlx::query(
            "INSERT INTO messages (message_id, chat_id, user_id, username, text, language, date, reply_to_message_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(message_id)
        .bind(chat_id)
        .bind(123_i64)
        .bind(username)
        .bind(text)
        .bind("en")
        .bind(Utc::now())
        .bind(None::<i64>)
        .execute(pool)
        .await
        .expect("message insert should succeed");
    }

    /// Helper that lets the caller specify a custom `user_id`.
    pub(crate) async fn queue_message_with_user(
        db: &Database,
        message_id: i64,
        chat_id: i64,
        user_id: i64,
        username: &str,
        text: &str,
    ) {
        let insert = build_message_insert(
            Some(user_id),
            Some(username.to_string()),
            Some(text.to_string()),
            Some("en".to_string()),
            Utc::now(),
            None,
            Some(chat_id),
            Some(message_id),
            None,
            false,
            None,
            text.trim_start().starts_with('/'),
            false,
        );
        db.queue_message_insert(insert)
            .await
            .expect("message queue should succeed");
        wait_for_message_row(db, chat_id, message_id).await;
    }

    // ─── analytics helpers ────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn insert_count_message(
        db: &Database,
        message_id: i64,
        chat_id: i64,
        user_id: Option<i64>,
        username: Option<&str>,
        text: &str,
        date: chrono::DateTime<chrono::Utc>,
        is_command: bool,
        is_synthetic: bool,
    ) {
        let insert = build_message_insert(
            user_id,
            username.map(|s| s.to_string()),
            Some(text.to_string()),
            Some("en".to_string()),
            date,
            None,
            Some(chat_id),
            Some(message_id),
            None,
            false,
            None,
            is_command,
            is_synthetic,
        );
        db.queue_message_insert(insert).await.expect("queue");
        wait_for_message_row(db, chat_id, message_id).await;
    }

    pub(crate) fn at(s: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(s)
            .expect("rfc3339")
            .with_timezone(&chrono::Utc)
    }
}

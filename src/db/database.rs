use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use crate::config::CONFIG;
use crate::db::schema::{
    self, count_messages, current_search_schema_version, ensure_search_fts_exists,
    recreate_search_fts, reset_search_versions, set_search_schema_version,
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

        schema::migrate(&pool).await?;
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
mod tests {
    use super::*;
    use crate::db::test_support::{init_test_db, sqlite_url_for_path, test_db_path};

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
}

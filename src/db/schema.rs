//! Schema creation, column backfills, FTS bootstrapping, and versioned
//! migrations for the SQLite database.

use anyhow::{Context, Result};
use sqlx::{FromRow, SqliteConnection, SqlitePool};

#[derive(Debug, Clone, FromRow)]
struct TableInfoRow {
    name: String,
}

pub(super) const LATEST_SCHEMA_VERSION: i32 = 2;

/// Bring `pool`'s schema up to [`LATEST_SCHEMA_VERSION`], applying any
/// migrations in order. Refuses to run against a database stamped with a
/// version newer than this binary understands.
pub(super) async fn migrate(pool: &SqlitePool) -> Result<()> {
    let mut version: i32 = sqlx::query_scalar("PRAGMA user_version")
        .fetch_one(pool)
        .await?;
    if version > LATEST_SCHEMA_VERSION {
        anyhow::bail!(
            "database schema version {version} is newer than this binary supports ({LATEST_SCHEMA_VERSION})"
        );
    }
    while version < LATEST_SCHEMA_VERSION {
        let next = version + 1;
        let mut tx = pool.begin().await?;
        apply_migration(&mut tx, next)
            .await
            .with_context(|| format!("applying schema migration {next}"))?;
        sqlx::query(&format!("PRAGMA user_version = {next}"))
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        version = next;
    }
    Ok(())
}

async fn apply_migration(conn: &mut SqliteConnection, version: i32) -> Result<()> {
    match version {
        // v1 = today's idempotent schema (CREATE IF NOT EXISTS + column backfills),
        // so a legacy database at user_version 0 is brought to v1 without data loss.
        1 => {
            ensure_messages_schema(&mut *conn).await?;
            ensure_search_support_schema(&mut *conn).await?;
            ensure_llm_audit_schema(&mut *conn).await?;
            Ok(())
        }
        2 => {
            for statement in [
                "DROP TRIGGER IF EXISTS messages_ai",
                "DROP TRIGGER IF EXISTS messages_ad",
                "DROP TRIGGER IF EXISTS messages_au",
                "DROP TABLE IF EXISTS messages_fts",
                "DROP INDEX IF EXISTS idx_messages_chat_id",
                "DROP INDEX IF EXISTS idx_messages_message_id",
                "DROP INDEX IF EXISTS idx_messages_date",
            ] {
                sqlx::query(statement)
                    .persistent(false)
                    .execute(&mut *conn)
                    .await?;
            }
            prepare_search_fts_on(conn).await
        }
        other => anyhow::bail!("unknown schema migration {other}"),
    }
}

async fn ensure_messages_schema(conn: &mut SqliteConnection) -> Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS messages (\
            id INTEGER PRIMARY KEY AUTOINCREMENT,\
            message_id INTEGER NOT NULL,\
            chat_id INTEGER NOT NULL,\
            user_id INTEGER,\
            username TEXT,\
            text TEXT,\
            search_text TEXT,\
            search_tags TEXT,\
            search_version INTEGER NOT NULL DEFAULT 0,\
            language TEXT,\
            date TEXT NOT NULL,\
            reply_to_message_id INTEGER,\
            is_command INTEGER NOT NULL DEFAULT 0,\
            asks_ai INTEGER NOT NULL DEFAULT 0,\
            ai_command TEXT,\
            is_synthetic_record INTEGER NOT NULL DEFAULT 0,\
            UNIQUE(chat_id, message_id)\
        );",
    )
    .execute(&mut *conn)
    .await?;

    ensure_messages_column(&mut *conn, "search_text", "TEXT").await?;
    ensure_messages_column(&mut *conn, "search_tags", "TEXT").await?;
    ensure_messages_column(&mut *conn, "search_version", "INTEGER NOT NULL DEFAULT 0").await?;
    ensure_messages_column(&mut *conn, "is_command", "INTEGER NOT NULL DEFAULT 0").await?;
    ensure_messages_column(&mut *conn, "asks_ai", "INTEGER NOT NULL DEFAULT 0").await?;
    ensure_messages_column(&mut *conn, "ai_command", "TEXT").await?;
    ensure_messages_column(
        &mut *conn,
        "is_synthetic_record",
        "INTEGER NOT NULL DEFAULT 0",
    )
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
        .execute(&mut *conn)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);")
        .execute(&mut *conn)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);")
        .execute(&mut *conn)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_date ON messages(chat_id, date);")
        .execute(&mut *conn)
        .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_messages_chat_user_date \
         ON messages(chat_id, user_id, date DESC, message_id DESC);",
    )
    .execute(&mut *conn)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_messages_user_date \
         ON messages(user_id, date DESC, message_id DESC);",
    )
    .execute(&mut *conn)
    .await?;

    Ok(())
}

async fn ensure_search_support_schema(conn: &mut SqliteConnection) -> Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS app_meta (\
            key TEXT PRIMARY KEY,\
            value TEXT NOT NULL\
        );",
    )
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn ensure_llm_audit_schema(conn: &mut SqliteConnection) -> Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS llm_invocations (\
            id INTEGER PRIMARY KEY AUTOINCREMENT,\
            trigger_kind TEXT NOT NULL,\
            trigger_name TEXT NOT NULL,\
            chat_id INTEGER NOT NULL,\
            user_id INTEGER,\
            username TEXT,\
            message_id INTEGER NOT NULL,\
            reply_to_message_id INTEGER,\
            message_text TEXT,\
            created_at TEXT NOT NULL\
        );",
    )
    .execute(&mut *conn)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS llm_requests (\
            id INTEGER PRIMARY KEY AUTOINCREMENT,\
            invocation_id INTEGER NOT NULL,\
            provider TEXT NOT NULL,\
            model TEXT NOT NULL,\
            operation TEXT NOT NULL,\
            response_id TEXT,\
            started_at TEXT NOT NULL,\
            completed_at TEXT NOT NULL,\
            duration_ms INTEGER NOT NULL,\
            input_tokens INTEGER,\
            output_tokens INTEGER,\
            total_tokens INTEGER,\
            reasoning_tokens INTEGER,\
            cached_input_tokens INTEGER,\
            cache_write_tokens INTEGER,\
            raw_usage_json TEXT,\
            FOREIGN KEY(invocation_id) REFERENCES llm_invocations(id) ON DELETE CASCADE\
        );",
    )
    .execute(&mut *conn)
    .await?;
    ensure_llm_requests_column(&mut *conn, "cache_write_tokens", "INTEGER").await?;
    ensure_llm_requests_column(&mut *conn, "status", "TEXT NOT NULL DEFAULT 'success'").await?;
    ensure_llm_requests_column(&mut *conn, "error_summary", "TEXT").await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_invocations_chat_message \
         ON llm_invocations(chat_id, message_id);",
    )
    .execute(&mut *conn)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_requests_invocation_completed \
         ON llm_requests(invocation_id, completed_at);",
    )
    .execute(&mut *conn)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_invocations_chat_user \
         ON llm_invocations(chat_id, user_id);",
    )
    .execute(&mut *conn)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_invocations_user \
         ON llm_invocations(user_id);",
    )
    .execute(&mut *conn)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_requests_provider_model \
         ON llm_requests(provider, model);",
    )
    .execute(&mut *conn)
    .await?;

    Ok(())
}

async fn ensure_llm_requests_column(
    conn: &mut SqliteConnection,
    column_name: &str,
    column_sql: &str,
) -> Result<()> {
    let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(llm_requests)")
        .fetch_all(&mut *conn)
        .await?;
    if columns.iter().any(|column| column.name == column_name) {
        return Ok(());
    }

    sqlx::query(&format!(
        "ALTER TABLE llm_requests ADD COLUMN {column_name} {column_sql}"
    ))
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn ensure_messages_column(
    conn: &mut SqliteConnection,
    column_name: &str,
    column_sql: &str,
) -> Result<()> {
    let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(messages)")
        .fetch_all(&mut *conn)
        .await?;
    if columns.iter().any(|column| column.name == column_name) {
        return Ok(());
    }

    sqlx::query(&format!(
        "ALTER TABLE messages ADD COLUMN {column_name} {column_sql}"
    ))
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// Repair/bootstrap derived FTS state on one connection and transaction.
/// Normalization upgrades retain current rows; only a missing table resets them.
pub(super) async fn prepare_search_fts(pool: &SqlitePool) -> Result<()> {
    let mut tx = pool.begin().await?;
    prepare_search_fts_on(&mut tx).await?;
    tx.commit().await?;
    Ok(())
}

async fn prepare_search_fts_on(conn: &mut SqliteConnection) -> Result<()> {
    let exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'",
    )
    .fetch_one(&mut *conn)
    .await?;
    if exists == 0 {
        // Old triggers must be gone before resetting versions: their delete operations
        // refer to a missing or discarded index, not the new external-content table.
        for statement in [
            "DROP TRIGGER IF EXISTS messages_ai",
            "DROP TRIGGER IF EXISTS messages_ad",
            "DROP TRIGGER IF EXISTS messages_au",
            "UPDATE messages SET search_version = 0 WHERE search_version != 0",
            "DELETE FROM app_meta WHERE key = 'search_index_schema_version'",
        ] {
            sqlx::query(statement)
                .persistent(false)
                .execute(&mut *conn)
                .await?;
        }
    }
    sqlx::query("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(search_text, search_tags, content='messages', content_rowid='id')").persistent(false).execute(&mut *conn).await?;
    let version = crate::db::search::CURRENT_SEARCH_SCHEMA_VERSION;
    // Explicit 'delete' supplies OLD values even though the AFTER trigger runs after
    // the content row changes. Never delete a row the rebuild has not indexed yet.
    for statement in [
        format!("CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages WHEN NEW.search_version = {version} BEGIN INSERT INTO messages_fts(rowid, search_text, search_tags) VALUES(NEW.id, NEW.search_text, NEW.search_tags); END"),
        format!("CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages WHEN OLD.search_version = {version} BEGIN INSERT INTO messages_fts(messages_fts, rowid, search_text, search_tags) VALUES('delete', OLD.id, OLD.search_text, OLD.search_tags); END"),
        format!("CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN INSERT INTO messages_fts(messages_fts, rowid, search_text, search_tags) SELECT 'delete', OLD.id, OLD.search_text, OLD.search_tags WHERE OLD.search_version = {version}; INSERT INTO messages_fts(rowid, search_text, search_tags) SELECT NEW.id, NEW.search_text, NEW.search_tags WHERE NEW.search_version = {version}; END"),
    ] { sqlx::query(&statement).persistent(false).execute(&mut *conn).await?; }
    Ok(())
}

pub(super) async fn current_search_schema_version(pool: &SqlitePool) -> Result<i64> {
    let value = sqlx::query_scalar::<_, Option<String>>("SELECT value FROM app_meta WHERE key = ?")
        .bind(SEARCH_INDEX_META_KEY)
        .fetch_optional(pool)
        .await?
        .flatten();
    Ok(value
        .and_then(|raw| raw.parse::<i64>().ok())
        .unwrap_or_default())
}

pub(super) async fn set_search_schema_version(pool: &SqlitePool, version: i64) -> Result<()> {
    sqlx::query(
        "INSERT INTO app_meta(key, value) VALUES(?, ?) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    )
    .bind(SEARCH_INDEX_META_KEY)
    .bind(version.to_string())
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
pub(super) async fn reset_search_versions(pool: &SqlitePool) -> Result<()> {
    sqlx::query("UPDATE messages SET search_version = 0 WHERE search_version != 0")
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM app_meta WHERE key = ?")
        .bind(SEARCH_INDEX_META_KEY)
        .execute(pool)
        .await?;
    Ok(())
}

pub(super) const SEARCH_INDEX_META_KEY: &str = "search_index_schema_version";

pub(super) async fn count_messages(pool: &SqlitePool) -> Result<i64> {
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM messages")
        .fetch_one(pool)
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_support::{init_test_db, sqlite_url_for_path, test_db_path};
    use sqlx::sqlite::SqlitePoolOptions;

    async fn v1_fixture(name: &str) -> (String, SqlitePool) {
        let url = sqlite_url_for_path(&test_db_path(name));
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .unwrap();
        apply_migration(&mut pool.acquire().await.unwrap(), 1)
            .await
            .unwrap();
        for statement in [
            "PRAGMA user_version = 1",
            "CREATE VIRTUAL TABLE messages_fts USING fts5(search_text, search_tags)",
            "CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN INSERT INTO messages_fts(rowid, search_text, search_tags) VALUES(NEW.id, NEW.search_text, NEW.search_tags); END",
            "INSERT INTO messages(message_id, chat_id, text, search_text, search_tags, search_version, date, asks_ai, ai_command) VALUES(1, -123, '原始 alpha', 'alpha obsolete', 'askai', 2, '2026-01-01T00:00:00Z', 1, 'q')",
        ] { sqlx::query(statement).execute(&pool).await.unwrap(); }
        (url, pool)
    }

    #[tokio::test]
    async fn v1_upgrade_reopens_and_checks_external_content_integrity() {
        let (url, pool) = v1_fixture("external-upgrade").await;
        migrate(&pool).await.unwrap();
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT search_version FROM messages")
                .fetch_one(&pool)
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'obsolete'"
            )
            .fetch_one(&pool)
            .await
            .unwrap(),
            0
        );
        pool.close().await;
        let db = crate::db::database::Database::init(&url).await.unwrap();
        crate::db::test_support::wait_for_search_ready(&db).await;
        let hits = db.search_chat_messages(-123, "alpha", 10, 0).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "原始 alpha");
        assert!(hits[0].asks_ai);
        assert_eq!(hits[0].ai_command.as_deref(), Some("q"));
        sqlx::query("INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)")
            .execute(db.pool())
            .await
            .unwrap();
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM sqlite_master WHERE name = 'messages_fts_content'"
            )
            .fetch_one(db.pool())
            .await
            .unwrap(),
            0
        );
        db.shutdown().await;
        let reopened = crate::db::database::Database::init(&url).await.unwrap();
        assert!(reopened.is_search_ready());
        assert_eq!(
            reopened
                .search_chat_messages(-123, "alpha", 10, 0)
                .await
                .unwrap()
                .len(),
            1
        );
        reopened.shutdown().await;
    }

    #[tokio::test]
    async fn failed_migration_rolls_back_ddl_data_and_version_together() {
        let (_, pool) = v1_fixture("external-rollback").await;
        sqlx::query("CREATE TRIGGER block_upgrade BEFORE UPDATE OF search_version ON messages BEGIN SELECT RAISE(ABORT, 'synthetic migration failure'); END").execute(&pool).await.unwrap();
        assert!(migrate(&pool).await.is_err());
        assert_eq!(
            sqlx::query_scalar::<_, i64>("PRAGMA user_version")
                .fetch_one(&pool)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT search_version FROM messages")
                .fetch_one(&pool)
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'obsolete'"
            )
            .fetch_one(&pool)
            .await
            .unwrap(),
            1
        );
        assert_eq!(sqlx::query_scalar::<_, i64>("SELECT count(*) FROM sqlite_master WHERE name IN ('messages_ai','idx_messages_chat_id','idx_messages_message_id','idx_messages_date')").fetch_one(&pool).await.unwrap(), 4);
        sqlx::query("DROP TRIGGER block_upgrade")
            .execute(&pool)
            .await
            .unwrap();
        migrate(&pool).await.unwrap();
        pool.close().await;
    }

    #[tokio::test]
    async fn remaining_indexes_cover_current_message_queries() {
        use sqlx::Row;
        let db = init_test_db("replacement-indexes").await;
        for query in [
            "SELECT * FROM messages WHERE chat_id = -1 ORDER BY date DESC LIMIT 20",
            "SELECT * FROM messages WHERE chat_id = -1 AND message_id < 100 ORDER BY message_id DESC LIMIT 5",
            "SELECT * FROM messages WHERE user_id = 1 ORDER BY date DESC, message_id DESC LIMIT 1",
            "SELECT * FROM messages WHERE chat_id = -1 AND user_id = 1 ORDER BY date DESC LIMIT 20",
        ] {
            let plan = sqlx::query(&format!("EXPLAIN QUERY PLAN {query}")).fetch_all(db.pool()).await.unwrap();
            let detail = plan.iter().map(|row| row.get::<String,_>("detail")).collect::<Vec<_>>().join("; ");
            assert!(detail.contains("USING INDEX"), "{detail}");
            assert!(!detail.contains("TEMP B-TREE"), "{detail}");
        }
        db.shutdown().await;
    }

    #[tokio::test]
    async fn llm_audit_schema_restores_cache_write_tokens_on_existing_table() {
        let db = init_test_db("llm-audit-cache-write-migration").await;
        let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(llm_requests)")
            .fetch_all(db.pool())
            .await
            .expect("llm request columns should load");
        if columns
            .iter()
            .any(|column| column.name == "cache_write_tokens")
        {
            sqlx::query("ALTER TABLE llm_requests DROP COLUMN cache_write_tokens")
                .execute(db.pool())
                .await
                .expect("cache write column should be removable for the migration fixture");
        }

        ensure_llm_audit_schema(&mut db.pool().acquire().await.unwrap())
            .await
            .expect("audit schema should migrate");
        let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(llm_requests)")
            .fetch_all(db.pool())
            .await
            .expect("migrated llm request columns should load");

        assert!(columns
            .iter()
            .any(|column| column.name == "cache_write_tokens"));
    }

    #[tokio::test]
    async fn fresh_database_is_stamped_with_the_latest_schema_version() {
        let db = init_test_db("fresh-schema-version").await;

        let version: i32 = sqlx::query_scalar("PRAGMA user_version")
            .fetch_one(db.pool())
            .await
            .expect("user_version should be readable");

        assert_eq!(version, LATEST_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn legacy_database_without_user_version_is_migrated_in_place() {
        let path = test_db_path("legacy-schema-migration");
        let url = sqlite_url_for_path(&path);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("raw pool should initialize");

        // Old column set: no search_text/search_tags/search_version/is_command/
        // asks_ai/ai_command/is_synthetic_record, and no PRAGMA user_version stamp.
        sqlx::query(
            "CREATE TABLE messages (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                message_id INTEGER NOT NULL,\
                chat_id INTEGER NOT NULL,\
                user_id INTEGER,\
                username TEXT,\
                text TEXT,\
                language TEXT,\
                date TEXT NOT NULL,\
                reply_to_message_id INTEGER,\
                UNIQUE(chat_id, message_id)\
            );",
        )
        .execute(&pool)
        .await
        .expect("legacy messages table should be creatable");

        sqlx::query(
            "INSERT INTO messages (message_id, chat_id, user_id, username, text, language, date, reply_to_message_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(1_i64)
        .bind(-1001374348669_i64)
        .bind(123_i64)
        .bind("alice")
        .bind("legacy row survives migration")
        .bind("en")
        .bind(chrono::Utc::now())
        .bind(None::<i64>)
        .execute(&pool)
        .await
        .expect("legacy row should insert");

        migrate(&pool).await.expect("migration should succeed");

        let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(messages)")
            .fetch_all(&pool)
            .await
            .expect("migrated messages columns should load");
        assert!(columns.iter().any(|column| column.name == "search_version"));

        let text: String = sqlx::query_scalar("SELECT text FROM messages WHERE message_id = 1")
            .fetch_one(&pool)
            .await
            .expect("legacy row should still exist after migration");
        assert_eq!(text, "legacy row survives migration");

        let version: i32 = sqlx::query_scalar("PRAGMA user_version")
            .fetch_one(&pool)
            .await
            .expect("user_version should be readable");
        assert_eq!(version, LATEST_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn migrate_is_idempotent() {
        let path = test_db_path("migrate-idempotent");
        let url = sqlite_url_for_path(&path);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("raw pool should initialize");

        migrate(&pool)
            .await
            .expect("first migration should succeed");
        migrate(&pool)
            .await
            .expect("second migration should be a no-op");

        let version: i32 = sqlx::query_scalar("PRAGMA user_version")
            .fetch_one(&pool)
            .await
            .expect("user_version should be readable");
        assert_eq!(version, LATEST_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn newer_database_is_refused() {
        let path = test_db_path("migrate-newer-refused");
        let url = sqlite_url_for_path(&path);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("raw pool should initialize");

        sqlx::query("PRAGMA user_version = 99")
            .execute(&pool)
            .await
            .expect("setting user_version should succeed");

        let err = migrate(&pool)
            .await
            .expect_err("a newer schema version must be refused");

        assert!(err.to_string().contains("newer than this binary supports"));
    }
}

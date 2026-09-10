//! Schema creation, column backfills, FTS bootstrapping, and versioned
//! migrations for the SQLite database.

use anyhow::{Context, Result};
use sqlx::{FromRow, SqlitePool};

#[derive(Debug, Clone, FromRow)]
struct TableInfoRow {
    name: String,
}

pub(super) const LATEST_SCHEMA_VERSION: i32 = 1;

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
        apply_migration(pool, next)
            .await
            .with_context(|| format!("applying schema migration {next}"))?;
        sqlx::query(&format!("PRAGMA user_version = {next}"))
            .execute(pool)
            .await?;
        version = next;
    }
    Ok(())
}

async fn apply_migration(pool: &SqlitePool, version: i32) -> Result<()> {
    match version {
        // v1 = today's idempotent schema (CREATE IF NOT EXISTS + column backfills),
        // so a legacy database at user_version 0 is brought to v1 without data loss.
        1 => {
            ensure_messages_schema(pool).await?;
            ensure_search_support_schema(pool).await?;
            ensure_llm_audit_schema(pool).await?;
            Ok(())
        }
        other => anyhow::bail!("unknown schema migration {other}"),
    }
}

async fn ensure_messages_schema(pool: &SqlitePool) -> Result<()> {
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
    .execute(pool)
    .await?;

    ensure_messages_column(pool, "search_text", "TEXT").await?;
    ensure_messages_column(pool, "search_tags", "TEXT").await?;
    ensure_messages_column(pool, "search_version", "INTEGER NOT NULL DEFAULT 0").await?;
    ensure_messages_column(pool, "is_command", "INTEGER NOT NULL DEFAULT 0").await?;
    ensure_messages_column(pool, "asks_ai", "INTEGER NOT NULL DEFAULT 0").await?;
    ensure_messages_column(pool, "ai_command", "TEXT").await?;
    ensure_messages_column(pool, "is_synthetic_record", "INTEGER NOT NULL DEFAULT 0").await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_date ON messages(chat_id, date);")
        .execute(pool)
        .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_messages_chat_user_date \
         ON messages(chat_id, user_id, date DESC, message_id DESC);",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_messages_user_date \
         ON messages(user_id, date DESC, message_id DESC);",
    )
    .execute(pool)
    .await?;

    Ok(())
}

async fn ensure_search_support_schema(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS app_meta (\
            key TEXT PRIMARY KEY,\
            value TEXT NOT NULL\
        );",
    )
    .execute(pool)
    .await?;
    Ok(())
}

async fn ensure_llm_audit_schema(pool: &SqlitePool) -> Result<()> {
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
    .execute(pool)
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
    .execute(pool)
    .await?;
    ensure_llm_requests_column(pool, "cache_write_tokens", "INTEGER").await?;
    ensure_llm_requests_column(pool, "status", "TEXT NOT NULL DEFAULT 'success'").await?;
    ensure_llm_requests_column(pool, "error_summary", "TEXT").await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_invocations_chat_message \
         ON llm_invocations(chat_id, message_id);",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_requests_invocation_completed \
         ON llm_requests(invocation_id, completed_at);",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_invocations_chat_user \
         ON llm_invocations(chat_id, user_id);",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_invocations_user \
         ON llm_invocations(user_id);",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_llm_requests_provider_model \
         ON llm_requests(provider, model);",
    )
    .execute(pool)
    .await?;

    Ok(())
}

async fn ensure_llm_requests_column(
    pool: &SqlitePool,
    column_name: &str,
    column_sql: &str,
) -> Result<()> {
    let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(llm_requests)")
        .fetch_all(pool)
        .await?;
    if columns.iter().any(|column| column.name == column_name) {
        return Ok(());
    }

    sqlx::query(&format!(
        "ALTER TABLE llm_requests ADD COLUMN {column_name} {column_sql}"
    ))
    .execute(pool)
    .await?;
    Ok(())
}

async fn ensure_messages_column(
    pool: &SqlitePool,
    column_name: &str,
    column_sql: &str,
) -> Result<()> {
    let columns = sqlx::query_as::<_, TableInfoRow>("PRAGMA table_info(messages)")
        .fetch_all(pool)
        .await?;
    if columns.iter().any(|column| column.name == column_name) {
        return Ok(());
    }

    sqlx::query(&format!(
        "ALTER TABLE messages ADD COLUMN {column_name} {column_sql}"
    ))
    .execute(pool)
    .await?;
    Ok(())
}

pub(super) async fn ensure_search_fts_exists(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(search_text, search_tags);",
    )
    .execute(pool)
    .await?;
    create_search_fts_triggers(pool).await?;
    Ok(())
}

pub(super) async fn recreate_search_fts(pool: &SqlitePool) -> Result<()> {
    drop_search_fts(pool).await?;
    ensure_search_fts_exists(pool).await?;
    Ok(())
}

async fn drop_search_fts(pool: &SqlitePool) -> Result<()> {
    sqlx::query("DROP TRIGGER IF EXISTS messages_ai;")
        .execute(pool)
        .await?;
    sqlx::query("DROP TRIGGER IF EXISTS messages_ad;")
        .execute(pool)
        .await?;
    sqlx::query("DROP TRIGGER IF EXISTS messages_au;")
        .execute(pool)
        .await?;
    sqlx::query("DROP TABLE IF EXISTS messages_fts;")
        .execute(pool)
        .await?;
    Ok(())
}

async fn create_search_fts_triggers(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages \
         WHEN NEW.search_text IS NOT NULL OR NEW.search_tags IS NOT NULL BEGIN \
         INSERT INTO messages_fts(rowid, search_text, search_tags) VALUES (NEW.id, NEW.search_text, NEW.search_tags); \
         END;",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN \
         DELETE FROM messages_fts WHERE rowid = OLD.id; \
         END;",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN \
         DELETE FROM messages_fts WHERE rowid = OLD.id; \
         INSERT INTO messages_fts(rowid, search_text, search_tags) \
         SELECT NEW.id, NEW.search_text, NEW.search_tags \
         WHERE NEW.search_text IS NOT NULL OR NEW.search_tags IS NOT NULL; \
         END;",
    )
    .execute(pool)
    .await?;
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

pub(super) async fn reset_search_versions(pool: &SqlitePool) -> Result<()> {
    sqlx::query("UPDATE messages SET search_version = 0")
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

        ensure_llm_audit_schema(db.pool())
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

//! Schema creation, column backfills, and FTS bootstrapping for the SQLite database.

use anyhow::Result;
use sqlx::{FromRow, SqlitePool};

#[derive(Debug, Clone, FromRow)]
struct TableInfoRow {
    name: String,
}

pub(super) async fn ensure_messages_schema(pool: &SqlitePool) -> Result<()> {
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

pub(super) async fn ensure_search_support_schema(pool: &SqlitePool) -> Result<()> {
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

pub(super) async fn ensure_llm_audit_schema(pool: &SqlitePool) -> Result<()> {
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
    use crate::db::database::tests::init_test_db;

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
}

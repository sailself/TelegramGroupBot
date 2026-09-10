//! Shared test-only database fixtures used across `db` submodule tests.

use std::path::PathBuf;
use tokio::time::{sleep, Duration};

use crate::db::database::Database;
use crate::db::messages::build_message_insert;
use crate::db::models::{LlmInvocationInsert, LlmRequestInsert};
use chrono::Utc;
use sqlx::SqlitePool;

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

pub(crate) async fn wait_for_message_row(db: &Database, chat_id: i64, message_id: i64) {
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

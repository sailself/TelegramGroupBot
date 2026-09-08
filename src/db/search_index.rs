//! Staged full-text search retrieval and the background FTS rebuild.

use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use crate::db::database::Database;
use crate::db::models::ChatSearchHit;
use crate::db::schema::set_search_schema_version;
use crate::db::search::{
    clean_text_for_display, normalize_message_document, normalize_search_query, SearchMatchStage,
    SearchProvenance, CURRENT_SEARCH_SCHEMA_VERSION, SEARCH_INDEX_REBUILDING_ERROR,
};
use crate::utils::telegram::build_message_link;
use anyhow::{anyhow, Result};
use sqlx::{FromRow, SqlitePool};
use tracing::{info, warn};

const SEARCH_LIMIT_MAX: i64 = 20;
const SEARCH_OFFSET_MAX: i64 = 250;
pub(super) const WINDOW_LIMIT_MAX: i64 = 5;
const SNIPPET_LIMIT: usize = 140;
const SEARCH_REBUILD_BATCH_SIZE: i64 = 5_000;

#[derive(Debug, Clone, FromRow)]
struct SearchRow {
    id: i64,
    message_id: i64,
    chat_id: i64,
    user_id: Option<i64>,
    username: Option<String>,
    text: Option<String>,
    language: Option<String>,
    date: chrono::DateTime<chrono::Utc>,
    reply_to_message_id: Option<i64>,
    asks_ai: bool,
    ai_command: Option<String>,
    is_synthetic_record: bool,
    score: f64,
}

#[derive(Debug, Clone, FromRow)]
struct RebuildRow {
    id: i64,
    text: Option<String>,
    asks_ai: bool,
    ai_command: Option<String>,
    is_command: bool,
    is_synthetic_record: bool,
}

#[derive(Debug, Clone)]
struct StageHit {
    hit: ChatSearchHit,
}

fn find_snippet_offset(text: &str, terms: &[String]) -> usize {
    let lower = text.to_lowercase();
    terms
        .iter()
        .filter_map(|term| lower.find(term))
        .min()
        .unwrap_or(0)
}

fn build_snippet(text: &str, terms: &[String]) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return String::new();
    }

    if normalized.chars().count() <= SNIPPET_LIMIT {
        return normalized;
    }

    let start = find_snippet_offset(&normalized, terms);
    let prefix_char_count = normalized[..start.min(normalized.len())].chars().count();
    let snippet_start = prefix_char_count.saturating_sub(SNIPPET_LIMIT / 3);
    let snippet_body: String = normalized
        .chars()
        .skip(snippet_start)
        .take(SNIPPET_LIMIT)
        .collect();
    let mut snippet = snippet_body.trim().to_string();

    if snippet_start > 0 {
        snippet.insert_str(0, "...");
    }
    if snippet_start + snippet_body.chars().count() < normalized.chars().count() {
        snippet.push_str("...");
    }

    snippet
}

impl Database {
    pub fn is_search_ready(&self) -> bool {
        self.search_ready.load(Ordering::Relaxed)
    }

    pub async fn search_chat_messages(
        &self,
        chat_id: i64,
        query: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ChatSearchHit>> {
        if !self.is_search_ready() {
            return Err(anyhow!(SEARCH_INDEX_REBUILDING_ERROR));
        }

        let query_spec = normalize_search_query(query);
        if query_spec.semantic_tokens.is_empty() && query_spec.tag_tokens.is_empty() {
            return Err(anyhow!("query must contain searchable text"));
        }

        let limit = limit.clamp(1, SEARCH_LIMIT_MAX);
        let offset = offset.clamp(0, SEARCH_OFFSET_MAX) as usize;
        let stage_limit = (limit * 2).min(40);
        let mut merged = BTreeMap::new();

        if let Some(stage_query) = build_phrase_stage_query(&query_spec) {
            for hit in self
                .fetch_stage_hits(
                    chat_id,
                    stage_limit,
                    &stage_query,
                    SearchMatchStage::Phrase,
                    &query_spec.snippet_terms,
                )
                .await?
            {
                insert_stage_hit(&mut merged, hit);
            }
        }

        if let Some(stage_query) = build_and_stage_query(&query_spec) {
            for hit in self
                .fetch_stage_hits(
                    chat_id,
                    stage_limit,
                    &stage_query,
                    SearchMatchStage::And,
                    &query_spec.snippet_terms,
                )
                .await?
            {
                insert_stage_hit(&mut merged, hit);
            }
        }

        if let Some(stage_query) = build_or_prefix_stage_query(&query_spec) {
            for hit in self
                .fetch_stage_hits(
                    chat_id,
                    stage_limit,
                    &stage_query,
                    SearchMatchStage::OrPrefix,
                    &query_spec.snippet_terms,
                )
                .await?
            {
                insert_stage_hit(&mut merged, hit);
            }
        }

        let mut hits = merged
            .into_values()
            .map(|stage_hit| stage_hit.hit)
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            left.match_stage
                .cmp(&right.match_stage)
                .then_with(|| {
                    left.score
                        .partial_cmp(&right.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| right.date.cmp(&left.date))
                .then_with(|| right.message_id.cmp(&left.message_id))
        });

        Ok(hits.into_iter().skip(offset).take(limit as usize).collect())
    }

    async fn fetch_stage_hits(
        &self,
        chat_id: i64,
        limit: i64,
        stage_query: &str,
        match_stage: SearchMatchStage,
        snippet_terms: &[String],
    ) -> Result<Vec<StageHit>> {
        let rows = sqlx::query_as::<_, SearchRow>(
            "SELECT \
                 m.id, \
                 m.message_id, \
                 m.chat_id, \
                 m.user_id, \
                 m.username, \
                 m.text, \
                 m.language, \
                 m.date, \
                 m.reply_to_message_id, \
                 m.asks_ai, \
                 m.ai_command, \
                 m.is_synthetic_record, \
                 bm25(messages_fts, 1.0, 0.2) AS score \
             FROM messages_fts \
             JOIN messages m ON m.id = messages_fts.rowid \
             WHERE m.chat_id = ? AND messages_fts MATCH ? \
             ORDER BY score ASC, m.date DESC, m.message_id DESC \
             LIMIT ?",
        )
        .bind(chat_id)
        .bind(stage_query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let _ = row.id;
                let text = row.text.unwrap_or_default();
                if text.trim().is_empty() {
                    return None;
                }
                let snippet_source = clean_text_for_display(&text, row.is_synthetic_record);
                Some(StageHit {
                    hit: ChatSearchHit {
                        message_id: row.message_id,
                        chat_id: row.chat_id,
                        user_id: row.user_id,
                        username: row.username,
                        text,
                        language: row.language,
                        date: row.date,
                        reply_to_message_id: row.reply_to_message_id,
                        snippet: build_snippet(&snippet_source, snippet_terms),
                        link: build_message_link(row.chat_id, row.message_id),
                        score: row.score,
                        asks_ai: row.asks_ai,
                        ai_command: row.ai_command,
                        is_synthetic_record: row.is_synthetic_record,
                        match_stage,
                    },
                })
            })
            .collect())
    }
}

fn insert_stage_hit(merged: &mut BTreeMap<i64, StageHit>, stage_hit: StageHit) {
    let message_id = stage_hit.hit.message_id;
    match merged.get(&message_id) {
        None => {
            merged.insert(message_id, stage_hit);
        }
        Some(existing) => {
            let should_replace = stage_hit.hit.match_stage < existing.hit.match_stage
                || (stage_hit.hit.match_stage == existing.hit.match_stage
                    && stage_hit.hit.score < existing.hit.score)
                || (stage_hit.hit.match_stage == existing.hit.match_stage
                    && (stage_hit.hit.score - existing.hit.score).abs() < f64::EPSILON
                    && stage_hit.hit.date > existing.hit.date);
            if should_replace {
                merged.insert(message_id, stage_hit);
            }
        }
    }
}

fn build_phrase_stage_query(query_spec: &crate::db::search::SearchQuery) -> Option<String> {
    if !query_spec.phrase_eligible {
        return None;
    }
    query_spec
        .phrase_text
        .as_ref()
        .map(|phrase| phrase.trim())
        .filter(|phrase| !phrase.is_empty())
        .map(|phrase| format!("search_text : \"{}\"", phrase.replace('"', "\"\"")))
}

fn build_and_stage_query(query_spec: &crate::db::search::SearchQuery) -> Option<String> {
    crate::db::search::build_and_match_expression(query_spec)
}

fn build_or_prefix_stage_query(query_spec: &crate::db::search::SearchQuery) -> Option<String> {
    let mut terms = query_spec
        .semantic_tokens
        .iter()
        .map(|token| {
            if token.chars().count() >= 2 {
                format!("search_text : {token}*")
            } else {
                format!("search_text : {token}")
            }
        })
        .collect::<Vec<_>>();
    terms.extend(
        query_spec
            .tag_tokens
            .iter()
            .map(|token| format!("search_tags : {token}")),
    );
    if terms.is_empty() {
        return None;
    }
    Some(terms.join(" OR "))
}

#[cfg(test)]
fn sanitize_chat_search_query(query: &str) -> Option<String> {
    let query_spec = normalize_search_query(query);
    build_or_prefix_stage_query(&query_spec)
}

pub(super) fn spawn_search_rebuild(pool: SqlitePool, search_ready: Arc<AtomicBool>) {
    tokio::spawn(async move {
        if let Err(err) = rebuild_search_index(pool.clone(), search_ready.clone()).await {
            warn!("Search index rebuild failed: {err}");
            search_ready.store(false, Ordering::Relaxed);
        }
    });
}

async fn rebuild_search_index(pool: SqlitePool, search_ready: Arc<AtomicBool>) -> Result<()> {
    search_ready.store(false, Ordering::Relaxed);

    loop {
        let rows = sqlx::query_as::<_, RebuildRow>(
            "SELECT id, text, asks_ai, ai_command, is_command, is_synthetic_record \
             FROM messages \
             WHERE search_version != ? \
             ORDER BY id ASC \
             LIMIT ?",
        )
        .bind(CURRENT_SEARCH_SCHEMA_VERSION)
        .bind(SEARCH_REBUILD_BATCH_SIZE)
        .fetch_all(&pool)
        .await?;

        if rows.is_empty() {
            set_search_schema_version(&pool, CURRENT_SEARCH_SCHEMA_VERSION).await?;
            search_ready.store(true, Ordering::Relaxed);
            info!("Search index rebuild completed");
            break;
        }

        let mut tx = pool.begin().await?;
        for row in rows {
            let explicit = SearchProvenance {
                asks_ai: row.asks_ai,
                ai_command: row.ai_command,
                is_command: row.is_command,
                is_synthetic_record: row.is_synthetic_record,
            };
            let document = normalize_message_document(row.text.as_deref(), None, &explicit);
            sqlx::query(
                "UPDATE messages SET \
                     search_text = ?, \
                     search_tags = ?, \
                     search_version = ?, \
                     asks_ai = ?, \
                     ai_command = ?, \
                     is_command = ?, \
                     is_synthetic_record = ? \
                 WHERE id = ?",
            )
            .bind(document.search_text)
            .bind(document.search_tags)
            .bind(CURRENT_SEARCH_SCHEMA_VERSION)
            .bind(document.provenance.asks_ai)
            .bind(document.provenance.ai_command)
            .bind(document.provenance.is_command)
            .bind(document.provenance.is_synthetic_record)
            .bind(row.id)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    Ok(())
}

pub(super) async fn count_pending_search_rows(pool: &SqlitePool) -> Result<i64> {
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM messages WHERE search_version != ?")
        .bind(CURRENT_SEARCH_SCHEMA_VERSION)
        .fetch_one(pool)
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::database::tests::{
        init_test_db, insert_legacy_message, queue_ai_request, queue_message, sqlite_url_for_path,
        test_db_path, wait_for_search_ready,
    };
    use crate::db::schema::reset_search_versions;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::sync::atomic::Ordering;

    #[test]
    fn sanitize_chat_search_query_removes_unsafe_operators() {
        assert_eq!(
            sanitize_chat_search_query("hello; DROP TABLE messages --"),
            Some(
                "search_text : hello* OR search_text : drop* OR search_text : table* OR search_text : messages*"
                    .to_string()
            )
        );
    }

    #[tokio::test]
    async fn search_chat_messages_stays_within_the_requested_chat() {
        let db = init_test_db("chat-scope").await;
        queue_message(&db, 1, -1001374348669, "alice", "Bitcoin treasury update").await;
        queue_message(&db, 2, -1001374348669, "bob", "Bitcoin treasury memo").await;
        queue_message(&db, 3, -1002631835259, "mallory", "Bitcoin treasury leak").await;

        let hits = db
            .search_chat_messages(-1001374348669, "bitcoin treasury", 10, 0)
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.chat_id == -1001374348669));
        assert!(hits.iter().all(|hit| hit
            .link
            .as_deref()
            .unwrap_or_default()
            .starts_with("https://t.me/c/1374348669/")));
    }

    #[tokio::test]
    async fn staged_retrieval_orders_phrase_then_and_then_or_prefix() {
        let db = init_test_db("stage-order").await;
        queue_message(&db, 1, -1001374348669, "alice", "alpha beta exact phrase").await;
        queue_message(&db, 2, -1001374348669, "bob", "alpha noise beta context").await;
        queue_message(&db, 3, -1001374348669, "carol", "alpha only fallback").await;

        let hits = db
            .search_chat_messages(-1001374348669, "alpha beta", 10, 0)
            .await
            .expect("search should succeed");

        assert!(hits.len() >= 3);
        assert_eq!(hits[0].message_id, 1);
        assert_eq!(hits[0].match_stage, SearchMatchStage::Phrase);
        assert_eq!(hits[1].message_id, 2);
        assert_eq!(hits[1].match_stage, SearchMatchStage::And);
        assert_eq!(hits[2].message_id, 3);
        assert_eq!(hits[2].match_stage, SearchMatchStage::OrPrefix);
    }

    #[tokio::test]
    async fn staged_retrieval_applies_offset_after_merge() {
        let db = init_test_db("stage-offset").await;
        queue_message(&db, 1, -1001374348669, "alice", "alpha beta exact phrase").await;
        queue_message(&db, 2, -1001374348669, "bob", "alpha noise beta context").await;
        queue_message(&db, 3, -1001374348669, "carol", "alpha only fallback").await;

        let hits = db
            .search_chat_messages(-1001374348669, "alpha beta", 2, 1)
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].message_id, 2);
        assert_eq!(hits[0].match_stage, SearchMatchStage::And);
        assert_eq!(hits[1].message_id, 3);
        assert_eq!(hits[1].match_stage, SearchMatchStage::OrPrefix);
    }

    #[tokio::test]
    async fn link_tag_queries_and_ai_provenance_are_searchable() {
        let db = init_test_db("tag-search").await;
        queue_message(
            &db,
            1,
            -1001374348669,
            "alice",
            "Shared this link https://x.com/example/status/123",
        )
        .await;
        queue_ai_request(
            &db,
            2,
            -1001374348669,
            "alice",
            "Ask about chat AI bot: Context from replied message: \"old\"\n\nQuestion: stocks",
            "Context from replied message: \"old\"\n\nQuestion: stocks",
            "qc",
        )
        .await;

        let twitter_hits = db
            .search_chat_messages(-1001374348669, "twitter links", 10, 0)
            .await
            .expect("tag search should succeed");
        assert!(twitter_hits.iter().any(|hit| hit.message_id == 1));

        let qc_hits = db
            .search_chat_messages(-1001374348669, "/qc", 10, 0)
            .await
            .expect("ai tag search should succeed");
        let hit = qc_hits
            .iter()
            .find(|hit| hit.message_id == 2)
            .expect("qc-tagged hit should exist");
        assert!(hit.asks_ai);
        assert_eq!(hit.ai_command.as_deref(), Some("qc"));
        assert!(!hit.snippet.to_lowercase().contains("ask about chat"));
    }

    #[tokio::test]
    async fn search_returns_rebuilding_error_when_index_is_not_ready() {
        let db = init_test_db("rebuilding-error").await;
        reset_search_versions(db.pool())
            .await
            .expect("search versions should reset");
        db.search_ready.store(false, Ordering::Relaxed);

        let err = db
            .search_chat_messages(-1001374348669, "alpha", 10, 0)
            .await
            .expect_err("search should fail while rebuilding");

        assert!(err.to_string().contains(SEARCH_INDEX_REBUILDING_ERROR));
    }

    #[tokio::test]
    async fn init_backfills_fts_for_existing_rows() {
        let path = test_db_path("fts-backfill");
        let url = sqlite_url_for_path(&path);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("raw pool should initialize");

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
        .expect("messages table should exist");
        insert_legacy_message(
            &pool,
            77,
            -1001374348669,
            "alice",
            "Retroactive FTS backfill works",
        )
        .await;
        pool.close().await;

        let db = Database::init(&url)
            .await
            .expect("database should initialize");
        wait_for_search_ready(&db).await;
        let hits = db
            .search_chat_messages(-1001374348669, "retroactive backfill", 10, 0)
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].message_id, 77);
    }
}

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use crate::config::CONFIG;
use crate::db::models::{
    AnalyticsRow, ChatSearchHit, LlmInvocationInsert, LlmRequestInsert, MessageInsert, MessageRow,
    ModelTokenStat, TokenUserStat,
};
use crate::db::search::{
    clean_text_for_display, normalize_message_document, normalize_search_query, SearchMatchStage,
    SearchProvenance, CURRENT_SEARCH_SCHEMA_VERSION, SEARCH_INDEX_REBUILDING_ERROR,
};
use crate::utils::telegram::build_message_link;
use anyhow::{anyhow, Result};
use serde::Serialize;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{FromRow, SqlitePool};
use teloxide::types::Message;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

const SEARCH_LIMIT_MAX: i64 = 20;
const SEARCH_OFFSET_MAX: i64 = 250;
const WINDOW_LIMIT_MAX: i64 = 5;
const SNIPPET_LIMIT: usize = 140;
const SEARCH_REBUILD_BATCH_SIZE: i64 = 5_000;
const SEARCH_INDEX_META_KEY: &str = "search_index_schema_version";
const TOKEN_TOTAL_EXPR: &str = "COALESCE(r.total_tokens, r.input_tokens + r.output_tokens, 0)";
const DB_WRITE_RETRY_DELAY_MS: u64 = 100;
const DB_WRITE_DEAD_LETTER_PATH: &str = "data/db_writer_dead_letters.jsonl";

#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
    sender: mpsc::Sender<MessageInsert>,
    search_ready: Arc<AtomicBool>,
}

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

#[derive(Debug, Clone, FromRow)]
struct TableInfoRow {
    name: String,
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
    pub async fn init(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(CONFIG.db_max_connections)
            .connect(database_url)
            .await?;
        let search_ready = Arc::new(AtomicBool::new(false));

        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA busy_timeout = 5000")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA cache_size = -65536")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA mmap_size = 134217728")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await?;
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
        let writer_pool = pool.clone();
        tokio::spawn(async move {
            db_writer(writer_pool, receiver).await;
        });

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
        })
    }

    pub async fn queue_message_insert(&self, insert: MessageInsert) -> Result<()> {
        self.sender
            .send(insert)
            .await
            .map_err(|err| anyhow!("Failed to queue message insert: {err}"))
    }

    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    pub async fn create_llm_invocation_from_message(
        &self,
        trigger_kind: &str,
        trigger_name: &str,
        message: &Message,
    ) -> Result<i64> {
        let insert = LlmInvocationInsert {
            trigger_kind: trigger_kind.to_string(),
            trigger_name: trigger_name.to_string(),
            chat_id: message.chat.id.0,
            user_id: message
                .from
                .as_ref()
                .and_then(|user| i64::try_from(user.id.0).ok()),
            username: message.from.as_ref().map(|user| {
                if !user.full_name().is_empty() {
                    user.full_name()
                } else {
                    user.username
                        .clone()
                        .unwrap_or_else(|| "Anonymous".to_string())
                }
            }),
            message_id: message.id.0 as i64,
            reply_to_message_id: message.reply_to_message().map(|reply| reply.id.0 as i64),
            message_text: message
                .text()
                .map(|value| value.to_string())
                .or_else(|| message.caption().map(|value| value.to_string())),
            created_at: message.date,
        };

        self.insert_llm_invocation(insert).await
    }

    pub async fn insert_llm_invocation(&self, insert: LlmInvocationInsert) -> Result<i64> {
        let result = sqlx::query(
            "INSERT INTO llm_invocations (\
                 trigger_kind, \
                 trigger_name, \
                 chat_id, \
                 user_id, \
                 username, \
                 message_id, \
                 reply_to_message_id, \
                 message_text, \
                 created_at\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(insert.trigger_kind)
        .bind(insert.trigger_name)
        .bind(insert.chat_id)
        .bind(insert.user_id)
        .bind(insert.username)
        .bind(insert.message_id)
        .bind(insert.reply_to_message_id)
        .bind(insert.message_text)
        .bind(insert.created_at)
        .execute(&self.pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    pub async fn insert_llm_request(&self, insert: LlmRequestInsert) -> Result<()> {
        sqlx::query(
            "INSERT INTO llm_requests (\
                 invocation_id, \
                 provider, \
                 model, \
                 operation, \
                 response_id, \
                 started_at, \
                 completed_at, \
                 duration_ms, \
                 input_tokens, \
                 output_tokens, \
                 total_tokens, \
                 reasoning_tokens, \
                 cached_input_tokens, \
                 raw_usage_json\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(insert.invocation_id)
        .bind(insert.provider)
        .bind(insert.model)
        .bind(insert.operation)
        .bind(insert.response_id)
        .bind(insert.started_at)
        .bind(insert.completed_at)
        .bind(insert.duration_ms)
        .bind(insert.input_tokens)
        .bind(insert.output_tokens)
        .bind(insert.total_tokens)
        .bind(insert.reasoning_tokens)
        .bind(insert.cached_input_tokens)
        .bind(insert.raw_usage_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn select_chat_token_total_for_user(
        &self,
        chat_id: i64,
        user_id: i64,
    ) -> Result<i64> {
        let query = format!(
            "SELECT COALESCE(SUM({TOKEN_TOTAL_EXPR}), 0) AS total_tokens \
             FROM llm_requests r \
             JOIN llm_invocations i ON i.id = r.invocation_id \
             WHERE i.chat_id = ? AND i.user_id = ?"
        );

        sqlx::query_scalar::<_, i64>(&query)
            .bind(chat_id)
            .bind(user_id)
            .fetch_one(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn select_top_chat_token_users(
        &self,
        chat_id: i64,
        limit: i64,
    ) -> Result<Vec<TokenUserStat>> {
        let limit = limit.clamp(1, 20);
        let query = format!(
            "SELECT \
                 i.user_id AS user_id, \
                 COALESCE( \
                     ( \
                         SELECT m.username \
                         FROM messages m \
                         WHERE m.chat_id = i.chat_id \
                           AND m.user_id = i.user_id \
                           AND m.username IS NOT NULL \
                         ORDER BY m.date DESC, m.message_id DESC \
                         LIMIT 1 \
                     ), \
                     MAX(i.username) \
                 ) AS username, \
                 SUM({TOKEN_TOTAL_EXPR}) AS total_tokens \
             FROM llm_requests r \
             JOIN llm_invocations i ON i.id = r.invocation_id \
             WHERE i.chat_id = ? AND i.user_id IS NOT NULL \
             GROUP BY i.chat_id, i.user_id \
             ORDER BY total_tokens DESC, user_id ASC \
             LIMIT ?"
        );

        sqlx::query_as::<_, TokenUserStat>(&query)
            .bind(chat_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    #[allow(dead_code)]
    pub async fn run_chat_analytics(
        &self,
        chat_id: i64,
        spec: &crate::llm::analytics::QuerySpec,
    ) -> Result<Vec<AnalyticsRow>> {
        use crate::llm::analytics::Bind;
        let (sql, binds) = crate::llm::analytics::compile(spec, chat_id);
        let mut q = sqlx::query_as::<_, AnalyticsRow>(&sql);
        for b in binds {
            q = match b {
                Bind::Int(i) => q.bind(i),
                Bind::Text(s) => q.bind(s),
            };
        }
        // Per-query timeout so a leading-wildcard scan or a pathological grouping
        // can't pin a connection.
        let dur = std::time::Duration::from_secs(CONFIG.qc_analytics_query_timeout_secs);
        match tokio::time::timeout(dur, q.fetch_all(&self.pool)).await {
            Ok(res) => res.map_err(Into::into),
            Err(_) => Err(anyhow::anyhow!("analytics query exceeded the time budget")),
        }
    }

    pub async fn select_global_token_total(&self) -> Result<i64> {
        let query = format!(
            "SELECT COALESCE(SUM({TOKEN_TOTAL_EXPR}), 0) AS total_tokens \
             FROM llm_requests r"
        );

        sqlx::query_scalar::<_, i64>(&query)
            .fetch_one(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn select_global_token_totals_by_model(&self) -> Result<Vec<ModelTokenStat>> {
        let query = format!(
            "SELECT \
                 r.provider AS provider, \
                 r.model AS model, \
                 SUM({TOKEN_TOTAL_EXPR}) AS total_tokens \
             FROM llm_requests r \
             GROUP BY r.provider, r.model \
             ORDER BY total_tokens DESC, provider ASC, model ASC"
        );

        sqlx::query_as::<_, ModelTokenStat>(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn select_global_token_totals_by_user(&self) -> Result<Vec<TokenUserStat>> {
        let query = format!(
            "SELECT \
                 i.user_id AS user_id, \
                 COALESCE( \
                     ( \
                         SELECT m.username \
                         FROM messages m \
                         WHERE m.user_id = i.user_id \
                           AND m.username IS NOT NULL \
                         ORDER BY m.date DESC, m.message_id DESC \
                         LIMIT 1 \
                     ), \
                     MAX(i.username) \
                 ) AS username, \
                 SUM({TOKEN_TOTAL_EXPR}) AS total_tokens \
             FROM llm_requests r \
             JOIN llm_invocations i ON i.id = r.invocation_id \
             WHERE i.user_id IS NOT NULL \
             GROUP BY i.user_id \
             ORDER BY total_tokens DESC, user_id ASC"
        );

        sqlx::query_as::<_, TokenUserStat>(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub fn queue_max_capacity(&self) -> usize {
        self.sender.max_capacity()
    }

    pub fn queue_available_capacity(&self) -> usize {
        self.sender.capacity()
    }

    pub fn queue_len(&self) -> usize {
        self.queue_max_capacity()
            .saturating_sub(self.queue_available_capacity())
    }

    pub fn is_search_ready(&self) -> bool {
        self.search_ready.load(Ordering::Relaxed)
    }

    pub async fn select_messages(&self, chat_id: i64, limit: i64) -> Result<Vec<MessageRow>> {
        self.get_last_n_text_messages(chat_id, limit, true).await
    }

    pub async fn select_messages_by_user(
        &self,
        chat_id: i64,
        user_id: i64,
        limit: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages WHERE chat_id = ? AND user_id = ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC LIMIT ?");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(user_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
    }

    pub async fn select_messages_from_id(
        &self,
        chat_id: i64,
        message_id: i64,
    ) -> Result<Vec<MessageRow>> {
        self.get_messages_from_id(chat_id, message_id, true).await
    }

    pub async fn get_last_n_text_messages(
        &self,
        chat_id: i64,
        limit: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages WHERE chat_id = ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC LIMIT ?");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
    }

    pub async fn get_messages_from_id(
        &self,
        chat_id: i64,
        from_message_id: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages WHERE chat_id = ? AND message_id >= ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(from_message_id)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
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

    pub async fn get_message_window(
        &self,
        chat_id: i64,
        message_id: i64,
        context_before: i64,
        context_after: i64,
    ) -> Result<Option<Vec<MessageRow>>> {
        let context_before = context_before.clamp(0, WINDOW_LIMIT_MAX);
        let context_after = context_after.clamp(0, WINDOW_LIMIT_MAX);

        let center = sqlx::query_as::<_, MessageRow>(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages \
             WHERE chat_id = ? AND message_id = ? AND text IS NOT NULL",
        )
        .bind(chat_id)
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(center) = center else {
            return Ok(None);
        };

        let mut before = sqlx::query_as::<_, MessageRow>(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages \
             WHERE chat_id = ? AND message_id < ? AND text IS NOT NULL \
             ORDER BY message_id DESC LIMIT ?",
        )
        .bind(chat_id)
        .bind(message_id)
        .bind(context_before)
        .fetch_all(&self.pool)
        .await?;
        before.reverse();

        let after = sqlx::query_as::<_, MessageRow>(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages \
             WHERE chat_id = ? AND message_id > ? AND text IS NOT NULL \
             ORDER BY message_id ASC LIMIT ?",
        )
        .bind(chat_id)
        .bind(message_id)
        .bind(context_after)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = before;
        messages.push(center);
        messages.extend(after);
        Ok(Some(messages))
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
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
    let mut terms = query_spec
        .semantic_tokens
        .iter()
        .map(|token| token.trim())
        .filter(|token| !token.is_empty())
        .map(|token| format!("search_text : {token}"))
        .collect::<Vec<_>>();
    terms.extend(
        query_spec
            .tag_tokens
            .iter()
            .map(|token| token.trim())
            .filter(|token| !token.is_empty())
            .map(|token| format!("search_tags : {token}")),
    );
    if terms.is_empty() {
        return None;
    }
    Some(terms.join(" AND "))
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

fn spawn_search_rebuild(pool: SqlitePool, search_ready: Arc<AtomicBool>) {
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
            raw_usage_json TEXT,\
            FOREIGN KEY(invocation_id) REFERENCES llm_invocations(id) ON DELETE CASCADE\
        );",
    )
    .execute(pool)
    .await?;
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

async fn ensure_search_fts_exists(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(search_text, search_tags);",
    )
    .execute(pool)
    .await?;
    create_search_fts_triggers(pool).await?;
    Ok(())
}

async fn recreate_search_fts(pool: &SqlitePool) -> Result<()> {
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

async fn current_search_schema_version(pool: &SqlitePool) -> Result<i64> {
    let value = sqlx::query_scalar::<_, Option<String>>("SELECT value FROM app_meta WHERE key = ?")
        .bind(SEARCH_INDEX_META_KEY)
        .fetch_optional(pool)
        .await?
        .flatten();
    Ok(value
        .and_then(|raw| raw.parse::<i64>().ok())
        .unwrap_or_default())
}

async fn set_search_schema_version(pool: &SqlitePool, version: i64) -> Result<()> {
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

async fn reset_search_versions(pool: &SqlitePool) -> Result<()> {
    sqlx::query("UPDATE messages SET search_version = 0")
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM app_meta WHERE key = ?")
        .bind(SEARCH_INDEX_META_KEY)
        .execute(pool)
        .await?;
    Ok(())
}

async fn count_messages(pool: &SqlitePool) -> Result<i64> {
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM messages")
        .fetch_one(pool)
        .await
        .map_err(Into::into)
}

async fn count_pending_search_rows(pool: &SqlitePool) -> Result<i64> {
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM messages WHERE search_version != ?")
        .bind(CURRENT_SEARCH_SCHEMA_VERSION)
        .fetch_one(pool)
        .await
        .map_err(Into::into)
}

async fn db_writer(pool: SqlitePool, mut receiver: mpsc::Receiver<MessageInsert>) {
    let flush_deadline = Duration::from_millis(CONFIG.db_write_flush_ms);
    let batch_size = CONFIG.db_write_batch_size.max(1);
    let mut buffer = Vec::with_capacity(batch_size);

    loop {
        let Some(message) = receiver.recv().await else {
            break;
        };
        buffer.push(message);
        let flush_at = tokio::time::Instant::now() + flush_deadline;

        while buffer.len() < batch_size {
            match tokio::time::timeout_at(flush_at, receiver.recv()).await {
                Ok(Some(message)) => buffer.push(message),
                Ok(None) | Err(_) => break,
            }
        }

        if let Err(err) =
            write_message_batch_with_recovery(&pool, &buffer, Path::new(DB_WRITE_DEAD_LETTER_PATH))
                .await
        {
            error!("Error in db_writer batch after recovery attempts: {err}");
        }
        buffer.clear();
    }

    let _ = pool.close().await;
    info!("Database writer task stopped");
}

async fn write_message_batch_with_recovery(
    pool: &SqlitePool,
    batch: &[MessageInsert],
    dead_letter_path: &Path,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }

    match write_message_batch(pool, batch).await {
        Ok(()) => Ok(()),
        Err(first_err) => {
            warn!(
                "db_writer batch failed for {} message(s), retrying once: {first_err}",
                batch.len()
            );
            tokio::time::sleep(Duration::from_millis(DB_WRITE_RETRY_DELAY_MS)).await;

            match write_message_batch(pool, batch).await {
                Ok(()) => {
                    warn!(
                        "db_writer batch recovered after retry for {} message(s)",
                        batch.len()
                    );
                    Ok(())
                }
                Err(retry_err) => {
                    error!(
                        "db_writer retry failed for {} message(s), attempting per-message salvage: {retry_err}",
                        batch.len()
                    );
                    let mut failed = Vec::new();
                    for message in batch {
                        if let Err(err) =
                            write_message_batch(pool, std::slice::from_ref(message)).await
                        {
                            failed.push((message.clone(), err.to_string()));
                        }
                    }

                    if failed.is_empty() {
                        warn!(
                            "db_writer salvaged {} message(s) after batch retry failure",
                            batch.len()
                        );
                        return Ok(());
                    }

                    write_dead_letter_messages(dead_letter_path, &failed)?;
                    Err(anyhow!(
                        "Failed to write {} of {} message(s); dead-lettered to {}",
                        failed.len(),
                        batch.len(),
                        dead_letter_path.display()
                    ))
                }
            }
        }
    }
}

#[derive(Serialize)]
struct DeadLetterMessage<'a> {
    failed_at: chrono::DateTime<chrono::Utc>,
    error: &'a str,
    message: &'a MessageInsert,
}

fn write_dead_letter_messages(
    dead_letter_path: &Path,
    failed: &[(MessageInsert, String)],
) -> Result<()> {
    if let Some(parent) = dead_letter_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dead_letter_path)?;
    let failed_at = chrono::Utc::now();
    for (message, error) in failed {
        let entry = DeadLetterMessage {
            failed_at,
            error,
            message,
        };
        serde_json::to_writer(&mut file, &entry)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

async fn write_message_batch(pool: &SqlitePool, batch: &[MessageInsert]) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }

    let mut tx = pool.begin().await?;
    for message in batch {
        let explicit = SearchProvenance {
            asks_ai: message.asks_ai,
            ai_command: message.ai_command.clone(),
            is_command: message.is_command,
            is_synthetic_record: message.is_synthetic_record,
        };
        let document = normalize_message_document(
            message.text.as_deref(),
            message.search_source_text.as_deref(),
            &explicit,
        );
        sqlx::query(
            "INSERT INTO messages (\
                 message_id, \
                 chat_id, \
                 user_id, \
                 username, \
                 text, \
                 search_text, \
                 search_tags, \
                 search_version, \
                 language, \
                 date, \
                 reply_to_message_id, \
                 is_command, \
                 asks_ai, \
                 ai_command, \
                 is_synthetic_record\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(chat_id, message_id) DO UPDATE SET \
             user_id = excluded.user_id, \
             username = excluded.username, \
             text = excluded.text, \
             search_text = excluded.search_text, \
             search_tags = excluded.search_tags, \
             search_version = excluded.search_version, \
             language = excluded.language, \
             date = excluded.date, \
             reply_to_message_id = excluded.reply_to_message_id, \
             is_command = excluded.is_command, \
             asks_ai = excluded.asks_ai, \
             ai_command = excluded.ai_command, \
             is_synthetic_record = excluded.is_synthetic_record",
        )
        .bind(message.message_id)
        .bind(message.chat_id)
        .bind(message.user_id)
        .bind(message.username.clone())
        .bind(message.text.clone())
        .bind(document.search_text)
        .bind(document.search_tags)
        .bind(CURRENT_SEARCH_SCHEMA_VERSION)
        .bind(message.language.clone())
        .bind(message.date)
        .bind(message.reply_to_message_id)
        .bind(document.provenance.is_command)
        .bind(document.provenance.asks_ai)
        .bind(document.provenance.ai_command)
        .bind(document.provenance.is_synthetic_record)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn build_message_insert(
    user_id: Option<i64>,
    username: Option<String>,
    text: Option<String>,
    language: Option<String>,
    date: chrono::DateTime<chrono::Utc>,
    reply_to_message_id: Option<i64>,
    chat_id: Option<i64>,
    message_id: Option<i64>,
    search_source_text: Option<String>,
    asks_ai: bool,
    ai_command: Option<String>,
    is_command: bool,
    is_synthetic_record: bool,
) -> MessageInsert {
    let resolved_user_id = user_id.unwrap_or_default();
    let resolved_chat_id = chat_id.unwrap_or(resolved_user_id);
    MessageInsert {
        message_id: message_id.unwrap_or_default(),
        chat_id: resolved_chat_id,
        user_id,
        username,
        text,
        search_source_text,
        language,
        date,
        reply_to_message_id,
        asks_ai,
        ai_command,
        is_command,
        is_synthetic_record,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::models::{
        LlmInvocationInsert, LlmInvocationRow, LlmRequestInsert, LlmRequestRow,
    };
    use chrono::Utc;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use tokio::time::{sleep, Duration};

    fn test_db_path(test_name: &str) -> PathBuf {
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

    fn sqlite_url_for_path(path: &std::path::Path) -> String {
        format!("sqlite://{}", path.to_string_lossy().replace('\\', "/"))
    }

    async fn init_test_db(test_name: &str) -> Database {
        let path = test_db_path(test_name);
        let db = Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("test database should initialize");
        wait_for_search_ready(&db).await;
        db
    }

    async fn wait_for_search_ready(db: &Database) {
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
    async fn failed_message_batches_are_dead_lettered() {
        let path = test_db_path("db-writer-dead-letter");
        let db = Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("test database should initialize");
        wait_for_search_ready(&db).await;
        sqlx::query("DROP TABLE messages")
            .execute(db.pool())
            .await
            .expect("messages table should be dropped for failure test");

        let dead_letter_path = path.with_extension("dead-letter.jsonl");
        let insert = build_message_insert(
            Some(123_i64),
            Some("alice".to_string()),
            Some("message that cannot be inserted".to_string()),
            Some("en".to_string()),
            Utc::now(),
            None,
            Some(-1001374348669),
            Some(777),
            None,
            false,
            None,
            false,
            false,
        );

        let err = write_message_batch_with_recovery(db.pool(), &[insert], &dead_letter_path)
            .await
            .expect_err("unrecoverable batch should return an error");

        assert!(err.to_string().contains("dead-lettered"));
        let dead_letter =
            std::fs::read_to_string(&dead_letter_path).expect("dead-letter file should be written");
        assert!(dead_letter.contains("\"message_id\":777"));
        assert!(dead_letter.contains("\"chat_id\":-1001374348669"));
    }

    #[tokio::test]
    async fn llm_audit_rows_persist_and_link() {
        let db = init_test_db("llm-audit").await;
        let invocation_id = db
            .insert_llm_invocation(LlmInvocationInsert {
                trigger_kind: "command".to_string(),
                trigger_name: "q".to_string(),
                chat_id: -1001234567890,
                user_id: Some(42),
                username: Some("alice".to_string()),
                message_id: 321,
                reply_to_message_id: Some(320),
                message_text: Some("/q what happened?".to_string()),
                created_at: Utc::now(),
            })
            .await
            .expect("invocation insert should succeed");

        db.insert_llm_request(LlmRequestInsert {
            invocation_id,
            provider: "gemini".to_string(),
            model: "gemini-2.5-pro".to_string(),
            operation: "call_gemini".to_string(),
            response_id: Some("resp_123".to_string()),
            started_at: Utc::now(),
            completed_at: Utc::now(),
            duration_ms: 456,
            input_tokens: Some(12),
            output_tokens: Some(34),
            total_tokens: Some(46),
            reasoning_tokens: Some(5),
            cached_input_tokens: Some(3),
            raw_usage_json: Some("{\"totalTokenCount\":46}".to_string()),
        })
        .await
        .expect("request insert should succeed");

        let invocation =
            sqlx::query_as::<_, LlmInvocationRow>("SELECT * FROM llm_invocations WHERE id = ?")
                .bind(invocation_id)
                .fetch_one(db.pool())
                .await
                .expect("invocation row should exist");
        let request = sqlx::query_as::<_, LlmRequestRow>(
            "SELECT * FROM llm_requests WHERE invocation_id = ?",
        )
        .bind(invocation_id)
        .fetch_one(db.pool())
        .await
        .expect("request row should exist");

        assert_eq!(invocation.trigger_name, "q");
        assert_eq!(request.invocation_id, invocation_id);
        assert_eq!(request.provider, "gemini");
        assert_eq!(request.total_tokens, Some(46));
        assert_eq!(request.reasoning_tokens, Some(5));
        assert_eq!(request.cached_input_tokens, Some(3));
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_invocation_with_usage(
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
            raw_usage_json: None,
        })
        .await
        .expect("request insert should succeed");
    }

    async fn queue_message(
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

    async fn queue_ai_request(
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

    async fn insert_legacy_message(
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
    async fn get_message_window_rejects_cross_chat_requests() {
        let db = init_test_db("window-scope").await;
        queue_message(&db, 1, -1001374348669, "alice", "Alpha keyword").await;
        queue_message(&db, 2, -1002631835259, "mallory", "Alpha keyword").await;

        let window = db
            .get_message_window(-1001374348669, 2, 1, 1)
            .await
            .expect("window lookup should succeed");

        assert!(window.is_none());
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
    async fn token_usage_queries_aggregate_chat_model_and_user_totals() {
        let db = init_test_db("token-usage-queries").await;
        let chat_a = -1001374348669_i64;
        let chat_b = -1002631835259_i64;

        insert_invocation_with_usage(
            &db,
            chat_a,
            Some(101),
            Some("Alice"),
            10,
            "gemini",
            "gemini-2.5-pro",
            Some(45),
            Some(55),
            Some(100),
        )
        .await;
        insert_invocation_with_usage(
            &db,
            chat_a,
            Some(101),
            Some("Alice"),
            11,
            "openai",
            "gpt-4.1",
            Some(40),
            Some(2),
            None,
        )
        .await;
        insert_invocation_with_usage(
            &db,
            chat_a,
            Some(202),
            Some("Bob"),
            12,
            "gemini",
            "gemini-2.5-pro",
            Some(30),
            Some(50),
            Some(80),
        )
        .await;
        insert_invocation_with_usage(
            &db,
            chat_b,
            Some(101),
            Some("Alice"),
            13,
            "openrouter",
            "gpt-4.1",
            Some(150),
            Some(150),
            Some(300),
        )
        .await;

        assert_eq!(
            db.select_chat_token_total_for_user(chat_a, 101)
                .await
                .expect("chat token total should succeed"),
            142
        );
        assert_eq!(
            db.select_global_token_total()
                .await
                .expect("global token total should succeed"),
            522
        );

        let top_chat_users = db
            .select_top_chat_token_users(chat_a, 5)
            .await
            .expect("chat ranking should succeed");
        assert_eq!(top_chat_users.len(), 2);
        assert_eq!(top_chat_users[0].user_id, 101);
        assert_eq!(top_chat_users[0].total_tokens, 142);
        assert_eq!(top_chat_users[1].user_id, 202);
        assert_eq!(top_chat_users[1].total_tokens, 80);

        let model_totals = db
            .select_global_token_totals_by_model()
            .await
            .expect("model totals should succeed");
        assert_eq!(model_totals.len(), 3);
        assert_eq!(model_totals[0].provider, "openrouter");
        assert_eq!(model_totals[0].model, "gpt-4.1");
        assert_eq!(model_totals[0].total_tokens, 300);
        assert_eq!(model_totals[1].provider, "gemini");
        assert_eq!(model_totals[1].model, "gemini-2.5-pro");
        assert_eq!(model_totals[1].total_tokens, 180);
        assert_eq!(model_totals[2].provider, "openai");
        assert_eq!(model_totals[2].model, "gpt-4.1");
        assert_eq!(model_totals[2].total_tokens, 42);

        let user_totals = db
            .select_global_token_totals_by_user()
            .await
            .expect("user totals should succeed");
        assert_eq!(user_totals.len(), 2);
        assert_eq!(user_totals[0].user_id, 101);
        assert_eq!(user_totals[0].total_tokens, 442);
        assert_eq!(user_totals[1].user_id, 202);
        assert_eq!(user_totals[1].total_tokens, 80);
    }

    #[tokio::test]
    async fn token_usage_queries_prefer_latest_username_from_messages() {
        let db = init_test_db("token-usage-usernames").await;
        let chat = -1001374348669_i64;

        queue_message_with_user(&db, 1, chat, 7001, "Old Name", "first").await;
        queue_message_with_user(&db, 2, chat, 7001, "New Name", "second").await;

        insert_invocation_with_usage(
            &db,
            chat,
            Some(7001),
            Some("Stale Name"),
            10,
            "gemini",
            "gemini-2.5-pro",
            Some(10),
            Some(20),
            Some(30),
        )
        .await;

        let chat_totals = db
            .select_top_chat_token_users(chat, 5)
            .await
            .expect("chat totals should succeed");
        assert_eq!(chat_totals[0].username.as_deref(), Some("New Name"));

        let global_totals = db
            .select_global_token_totals_by_user()
            .await
            .expect("global totals should succeed");
        assert_eq!(global_totals[0].username.as_deref(), Some("New Name"));
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

    /// Helper that lets the caller specify a custom `user_id`.
    async fn queue_message_with_user(
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
    async fn insert_count_message(
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

    fn at(s: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(s)
            .expect("rfc3339")
            .with_timezone(&chrono::Utc)
    }

    // ─── invariant property test (security gate) ─────────────────────────────

    #[tokio::test]
    async fn analytics_never_leaks_other_chats_tables_or_writes() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-invariant").await;
        let a = -1001374348669_i64;
        let b = -1002631835259_i64;
        insert_count_message(
            &db,
            1,
            a,
            Some(11),
            Some("alice"),
            "hello",
            at("2026-03-01T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            2,
            a,
            Some(11),
            Some("alice"),
            "world",
            at("2026-03-02T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            3,
            a,
            Some(12),
            Some("bob"),
            "hi",
            at("2026-03-03T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        // Sentinel in chat B — must never influence chat A results.
        insert_count_message(
            &db,
            4,
            b,
            Some(11),
            Some("alice"),
            "SENTINEL_CHATB",
            at("2026-03-04T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        // Seed an llm_invocations row with text "SENTINEL_AUDIT" (another table).
        db.insert_llm_invocation(crate::db::models::LlmInvocationInsert {
            trigger_kind: "command".to_string(),
            trigger_name: "q".to_string(),
            chat_id: a,
            user_id: Some(11),
            username: Some("alice".to_string()),
            message_id: 999,
            reply_to_message_id: None,
            message_text: Some("SENTINEL_AUDIT".to_string()),
            created_at: at("2026-03-01T00:00:00+00:00"),
        })
        .await
        .expect("invocation insert");

        // Adversarial / fuzzed specs the model might emit.
        let specs = [
            r#"{"metric":"count","group_by":"user"}"#,
            r#"{"metric":"count","filters":{"text_contains":"SENTINEL"}}"#,
            r#"{"metric":"count","filters":{"term":"SENTINEL_AUDIT"}}"#,
            r#"{"metric":"max_date","filters":{"text_contains":"SENTINEL_CHATB"}}"#,
            r#"{"metric":"count","chat_id":-1002631835259}"#, // unknown field must be ignored
            r#"{"metric":"count","filters":{"text_contains":"x'; DROP TABLE messages;--"}}"#,
            r#"{"metric":"count","filters":{"username":"a' UNION SELECT value FROM app_meta--"}}"#,
            r#"{"metric":"count","filters":{"term":"search_tags:* OR 1=1"}}"#,
        ];
        for raw in specs {
            let spec: QuerySpec = parse_lenient_json(raw).expect("spec parses");
            let rows = db
                .run_chat_analytics(a, &spec)
                .await
                .expect("query ok (inert, not executed SQL)");
            for r in &rows {
                assert_ne!(r.group_key.as_deref(), Some("SENTINEL_CHATB"));
                assert!(!r.value_text.as_deref().unwrap_or("").contains("SENTINEL"));
            }
        }
        // chat_id-in-spec is ignored: total count == chat A's 3 messages, never includes B.
        let total: QuerySpec =
            parse_lenient_json(r#"{"metric":"count","chat_id":-1002631835259}"#).unwrap();
        assert_eq!(
            db.run_chat_analytics(a, &total).await.unwrap()[0].value_num,
            Some(3.0)
        );
        // Write-impossibility: messages table unchanged after injection attempts.
        let after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages")
            .fetch_one(db.pool())
            .await
            .unwrap();
        assert_eq!(after, 4);

        // Cross-table isolation: the llm_invocations SENTINEL_AUDIT text must NOT be counted.
        let audit_probe: QuerySpec =
            parse_lenient_json(r#"{"metric":"count","filters":{"term":"SENTINEL_AUDIT"}}"#)
                .unwrap();
        let r = db.run_chat_analytics(a, &audit_probe).await.unwrap();
        assert_eq!(
            r.first().and_then(|row| row.value_num),
            Some(0.0),
            "llm_invocations text leaked into analytics"
        );

        // Cross-chat isolation: chat B's message text must NOT be reachable from chat A.
        let chatb_probe: QuerySpec = parse_lenient_json(
            r#"{"metric":"count","filters":{"text_contains":"SENTINEL_CHATB"}}"#,
        )
        .unwrap();
        let r2 = db.run_chat_analytics(a, &chatb_probe).await.unwrap();
        assert_eq!(
            r2.first().and_then(|row| row.value_num),
            Some(0.0),
            "chat B text leaked into chat A analytics"
        );
    }

    // ─── focused per-metric / scope / date-range tests ───────────────────────

    #[tokio::test]
    async fn run_chat_analytics_count_by_user_ranks_and_is_chat_scoped() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-count-user").await;
        let chat_a = -1001374348669_i64;
        let chat_b = -1002631835259_i64;
        // alice: 3 messages in chat A; bob: 1 message in chat A; eve: 1 message in chat B (sentinel).
        for mid in 1i64..=3 {
            insert_count_message(
                &db,
                mid,
                chat_a,
                Some(11),
                Some("alice"),
                &format!("msg {mid}"),
                at("2026-04-01T10:00:00+00:00"),
                false,
                false,
            )
            .await;
        }
        insert_count_message(
            &db,
            4,
            chat_a,
            Some(12),
            Some("bob"),
            "bob msg",
            at("2026-04-01T11:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            5,
            chat_b,
            Some(99),
            Some("eve"),
            "sentinel",
            at("2026-04-01T12:00:00+00:00"),
            false,
            false,
        )
        .await;

        let spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"count","group_by":"user","order":"value_desc"}"#)
                .unwrap();
        let rows = db
            .run_chat_analytics(chat_a, &spec)
            .await
            .expect("query ok");

        // Only chat A rows; alice first (3), bob second (1).
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].group_user_id, Some(11));
        assert_eq!(rows[0].value_num, Some(3.0));
        assert_eq!(rows[1].group_user_id, Some(12));
        assert_eq!(rows[1].value_num, Some(1.0));
        // eve (chat B) must not appear.
        assert!(rows.iter().all(|r| r.group_user_id != Some(99)));
    }

    #[tokio::test]
    async fn run_chat_analytics_term_filter_counts_matches_only() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-term").await;
        let chat = -1001374348669_i64;
        insert_count_message(
            &db,
            1,
            chat,
            Some(10),
            Some("alice"),
            "bitcoin rally today",
            at("2026-04-01T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            2,
            chat,
            Some(10),
            Some("alice"),
            "ethereum news",
            at("2026-04-02T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            3,
            chat,
            Some(11),
            Some("bob"),
            "bitcoin dip",
            at("2026-04-03T00:00:00+00:00"),
            false,
            false,
        )
        .await;

        let spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"count","filters":{"term":"bitcoin"}}"#).unwrap();
        let rows = db.run_chat_analytics(chat, &spec).await.expect("query ok");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value_num, Some(2.0));
    }

    #[tokio::test]
    async fn run_chat_analytics_group_by_day_buckets_correctly() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-day").await;
        let chat = -1001374348669_i64;
        // 2 messages on day 1, 1 on day 2.
        for mid in 1i64..=2 {
            insert_count_message(
                &db,
                mid,
                chat,
                Some(10),
                Some("alice"),
                "day1 msg",
                at("2026-05-01T09:00:00+00:00"),
                false,
                false,
            )
            .await;
        }
        insert_count_message(
            &db,
            3,
            chat,
            Some(10),
            Some("alice"),
            "day2 msg",
            at("2026-05-02T09:00:00+00:00"),
            false,
            false,
        )
        .await;

        let spec: QuerySpec = parse_lenient_json(
            r#"{"metric":"count","group_by":"day","order":"group_asc","limit":10}"#,
        )
        .unwrap();
        let rows = db.run_chat_analytics(chat, &spec).await.expect("query ok");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].group_key.as_deref(), Some("2026-05-01"));
        assert_eq!(rows[0].value_num, Some(2.0));
        assert_eq!(rows[1].group_key.as_deref(), Some("2026-05-02"));
        assert_eq!(rows[1].value_num, Some(1.0));
    }

    #[tokio::test]
    async fn run_chat_analytics_distinct_count_group_by_none() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-distinct").await;
        let chat = -1001374348669_i64;
        // 3 messages from 2 distinct users.
        insert_count_message(
            &db,
            1,
            chat,
            Some(10),
            Some("alice"),
            "hello",
            at("2026-06-01T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            2,
            chat,
            Some(10),
            Some("alice"),
            "world",
            at("2026-06-02T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            3,
            chat,
            Some(11),
            Some("bob"),
            "hi",
            at("2026-06-03T00:00:00+00:00"),
            false,
            false,
        )
        .await;

        let spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"distinct_count","group_by":"none"}"#).unwrap();
        let rows = db.run_chat_analytics(chat, &spec).await.expect("query ok");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value_num, Some(2.0));
        assert!(rows[0].group_key.is_none());
    }

    #[tokio::test]
    async fn run_chat_analytics_min_max_date_returns_value_text() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-minmax").await;
        let chat = -1001374348669_i64;
        insert_count_message(
            &db,
            1,
            chat,
            Some(10),
            Some("alice"),
            "early",
            at("2026-01-15T08:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            2,
            chat,
            Some(10),
            Some("alice"),
            "late",
            at("2026-06-20T22:00:00+00:00"),
            false,
            false,
        )
        .await;

        let min_spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"min_date","group_by":"none"}"#).unwrap();
        let min_rows = db
            .run_chat_analytics(chat, &min_spec)
            .await
            .expect("min_date ok");
        assert_eq!(min_rows.len(), 1);
        assert!(min_rows[0].value_text.is_some());
        assert!(min_rows[0]
            .value_text
            .as_deref()
            .unwrap()
            .starts_with("2026-01-15"));

        let max_spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"max_date","group_by":"none"}"#).unwrap();
        let max_rows = db
            .run_chat_analytics(chat, &max_spec)
            .await
            .expect("max_date ok");
        assert_eq!(max_rows.len(), 1);
        assert!(max_rows[0]
            .value_text
            .as_deref()
            .unwrap()
            .starts_with("2026-06-20"));
    }

    #[tokio::test]
    async fn run_chat_analytics_date_range_excludes_out_of_range() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-daterange").await;
        let chat = -1001374348669_i64;
        // message before range
        insert_count_message(
            &db,
            1,
            chat,
            Some(10),
            Some("alice"),
            "before",
            at("2026-02-28T00:00:00+00:00"),
            false,
            false,
        )
        .await;
        // 2 messages within range [2026-03-01, 2026-04-01)
        for mid in 2i64..=3 {
            insert_count_message(
                &db,
                mid,
                chat,
                Some(10),
                Some("alice"),
                "during",
                at("2026-03-15T00:00:00+00:00"),
                false,
                false,
            )
            .await;
        }
        // message after range
        insert_count_message(
            &db,
            4,
            chat,
            Some(10),
            Some("alice"),
            "after",
            at("2026-04-05T00:00:00+00:00"),
            false,
            false,
        )
        .await;

        let spec: QuerySpec = parse_lenient_json(
            r#"{"metric":"count","filters":{"date_from":"2026-03-01T00:00:00+00:00","date_to":"2026-04-01T00:00:00+00:00"}}"#,
        )
        .unwrap();
        let rows = db.run_chat_analytics(chat, &spec).await.expect("query ok");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value_num, Some(2.0));
    }

    #[tokio::test]
    async fn run_chat_analytics_avg_len_computes_mean_char_length() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-avglen").await;
        let chat = -1001374348669_i64;
        insert_count_message(
            &db,
            1,
            chat,
            Some(11),
            Some("a"),
            "abc",
            at("2026-03-01T00:00:00+00:00"),
            false,
            false,
        )
        .await; // len 3
        insert_count_message(
            &db,
            2,
            chat,
            Some(11),
            Some("a"),
            "abcdefg",
            at("2026-03-02T00:00:00+00:00"),
            false,
            false,
        )
        .await; // len 7
        let spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"avg_len","group_by":"none"}"#).unwrap();
        let rows = db.run_chat_analytics(chat, &spec).await.unwrap();
        assert_eq!(rows.first().and_then(|r| r.value_num), Some(5.0)); // (3+7)/2
    }

    #[tokio::test]
    async fn display_labels_disambiguate_same_name_users_in_chat() {
        use crate::handlers::{build_display_label_map, format_tldr_chat_content};

        let db = init_test_db("disambiguate-names").await;
        let chat = -1001374348669_i64;

        // Two different users with the same display name "John".
        queue_message_with_user(&db, 1, chat, 1001, "John", "Hello from first John").await;
        queue_message_with_user(&db, 2, chat, 1002, "John", "Hello from second John").await;
        // A third user with a unique name.
        queue_message_with_user(&db, 3, chat, 1003, "Alice", "Hello from Alice").await;

        let messages = db
            .select_messages(chat, 10)
            .await
            .expect("select should work");
        assert_eq!(messages.len(), 3);

        let label_map = build_display_label_map(messages.iter().filter_map(|m| {
            m.user_id
                .map(|uid| (uid, m.username.as_deref().unwrap_or("Anonymous")))
        }));

        // The two Johns should be disambiguated with ordinal suffixes.
        assert_eq!(label_map[&1001], "John (1)");
        assert_eq!(label_map[&1002], "John (2)");
        // Alice is unique — no suffix.
        assert_eq!(label_map[&1003], "Alice");

        let chat_content = format_tldr_chat_content(&messages);

        assert!(chat_content.contains("[message_id=1] John (1): Hello from first John"));
        assert!(chat_content.contains("[message_id=2] John (2): Hello from second John"));
        assert!(chat_content.contains("[message_id=3] Alice: Hello from Alice"));
    }
}

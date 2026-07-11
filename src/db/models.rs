use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

use crate::db::search::SearchMatchStage;

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct MessageRow {
    pub id: i64,
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: Option<String>,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
    pub asks_ai: bool,
    pub ai_command: Option<String>,
    pub is_synthetic_record: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicWindowSpec {
    pub date_from: DateTime<Utc>,
    pub date_to: DateTime<Utc>,
    pub user_id: Option<i64>,
    pub exclude_commands: bool,
    pub exclude_synthetic: bool,
    pub limit: i64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TopicWindow {
    pub messages: Vec<MessageRow>,
    pub total_eligible: i64,
    pub capped: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageInsert {
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: Option<String>,
    pub search_source_text: Option<String>,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
    pub asks_ai: bool,
    pub ai_command: Option<String>,
    pub is_command: bool,
    pub is_synthetic_record: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSearchHit {
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: String,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
    pub snippet: String,
    pub link: Option<String>,
    pub score: f64,
    pub asks_ai: bool,
    pub ai_command: Option<String>,
    pub is_synthetic_record: bool,
    pub match_stage: SearchMatchStage,
}

#[allow(dead_code)]
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct LlmInvocationRow {
    pub id: i64,
    pub trigger_kind: String,
    pub trigger_name: String,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub message_id: i64,
    pub reply_to_message_id: Option<i64>,
    pub message_text: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct LlmInvocationInsert {
    pub trigger_kind: String,
    pub trigger_name: String,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub message_id: i64,
    pub reply_to_message_id: Option<i64>,
    pub message_text: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct LlmRequestRow {
    pub id: i64,
    pub invocation_id: i64,
    pub provider: String,
    pub model: String,
    pub operation: String,
    pub response_id: Option<String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub cached_input_tokens: Option<i64>,
    pub raw_usage_json: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmRequestInsert {
    pub invocation_id: i64,
    pub provider: String,
    pub model: String,
    pub operation: String,
    pub response_id: Option<String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub cached_input_tokens: Option<i64>,
    pub raw_usage_json: Option<String>,
}

#[derive(Debug, Clone, FromRow, PartialEq, Eq)]
pub struct TokenUserStat {
    pub user_id: i64,
    pub username: Option<String>,
    pub total_tokens: i64,
}

#[derive(Debug, Clone, FromRow)]
pub struct AnalyticsRow {
    pub group_user_id: Option<i64>,
    pub group_key: Option<String>,
    pub value_num: Option<f64>,
    pub value_text: Option<String>,
}

#[derive(Debug, Clone, FromRow, PartialEq, Eq)]
pub struct ModelTokenStat {
    pub provider: String,
    pub model: String,
    pub total_tokens: i64,
}

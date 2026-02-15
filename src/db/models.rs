use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

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
}

#[derive(Debug, Clone)]
pub struct MessageInsert {
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: Option<String>,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct AgentMemoryRow {
    pub id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub session_id: Option<i64>,
    pub source_role: String,
    pub category: String,
    pub content: String,
    pub summary: Option<String>,
    pub importance: f64,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct AgentMemoryInsert<'a> {
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub session_id: Option<i64>,
    pub source_role: &'a str,
    pub category: &'a str,
    pub content: &'a str,
    pub summary: Option<&'a str>,
    pub importance: f64,
}

#[derive(Debug, Clone)]
pub struct AgentMemorySearchRow {
    pub memory: AgentMemoryRow,
    pub lexical_score: f64,
    pub recency_days: f64,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct AgentSessionRow {
    pub id: i64,
    pub chat_id: i64,
    pub user_id: i64,
    pub model_name: String,
    pub prompt: String,
    pub selected_skills_json: Option<String>,
    pub status: String,
    pub final_response: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct AgentStepRow {
    pub id: i64,
    pub session_id: i64,
    pub role: String,
    pub content: Option<String>,
    pub raw_json: String,
    pub created_at: String,
}

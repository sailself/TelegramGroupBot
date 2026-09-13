use chrono::{DateTime, Utc};
use serde_json::Value;
use teloxide::types::Message;
use tracing::{info, warn};

use crate::db::database::Database;
use crate::db::models::LlmRequestInsert;
use crate::state::AppState;
use crate::utils::text::truncate_for_log;

pub const LLM_TRIGGER_KIND_AUTO_Q: &str = "auto_q";
pub const LLM_TRIGGER_KIND_COMMAND: &str = "command";
pub const LLM_REQUEST_STATUS_SUCCESS: &str = "success";
pub const LLM_REQUEST_STATUS_ERROR: &str = "error";

#[derive(Clone)]
pub struct LlmAuditContext {
    pub db: Database,
    pub invocation_id: i64,
}

impl LlmAuditContext {
    pub fn new(db: Database, invocation_id: i64) -> Self {
        Self { db, invocation_id }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlmUsageRecord {
    pub response_id: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub cached_input_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
    pub raw_usage_json: Option<String>,
}

pub fn audit_context_from_id(db: &Database, invocation_id: Option<i64>) -> Option<LlmAuditContext> {
    invocation_id.map(|invocation_id| LlmAuditContext::new(db.clone(), invocation_id))
}

pub async fn create_audit_context_from_message(
    db: &Database,
    trigger_kind: &str,
    trigger_name: &str,
    message: &Message,
) -> Option<LlmAuditContext> {
    match db
        .create_llm_invocation_from_message(trigger_kind, trigger_name, message)
        .await
    {
        Ok(invocation_id) => Some(LlmAuditContext::new(db.clone(), invocation_id)),
        Err(err) => {
            warn!(
                "Failed to create llm invocation record: trigger_kind={}, trigger_name={}, chat_id={}, message_id={}, error={err}",
                trigger_kind,
                trigger_name,
                message.chat.id.0,
                message.id.0
            );
            None
        }
    }
}

pub(crate) async fn create_command_audit_context(
    state: &AppState,
    message: &Message,
    trigger_name: &str,
) -> Option<LlmAuditContext> {
    create_audit_context_from_message(&state.db, LLM_TRIGGER_KIND_COMMAND, trigger_name, message)
        .await
}

fn json_text(value: Option<&Value>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "{}".to_string())
}

pub fn log_llm_request_started(
    provider: &str,
    model: &str,
    operation: &str,
    started_at: DateTime<Utc>,
    metadata: Option<&Value>,
) {
    info!(
        target: "bot.timing",
        event = "llm_request",
        provider,
        model,
        operation,
        started_at = %started_at.to_rfc3339(),
        metadata = %json_text(metadata),
    );
}

pub async fn record_llm_request_success(
    audit_context: Option<&LlmAuditContext>,
    provider: &str,
    model: &str,
    operation: &str,
    started_at: DateTime<Utc>,
    completed_at: DateTime<Utc>,
    usage: LlmUsageRecord,
) {
    let duration_ms = (completed_at - started_at).num_milliseconds().max(0);
    info!(
        target: "bot.timing",
        event = "llm_response",
        provider,
        model,
        operation,
        completed_at = %completed_at.to_rfc3339(),
        duration_ms,
        status = "success",
        response_id = usage.response_id.as_deref(),
        usage = usage.raw_usage_json.as_deref().unwrap_or("{}"),
    );

    let Some(audit_context) = audit_context else {
        return;
    };

    let insert = LlmRequestInsert {
        invocation_id: audit_context.invocation_id,
        provider: provider.to_string(),
        model: model.to_string(),
        operation: operation.to_string(),
        response_id: usage.response_id,
        started_at,
        completed_at,
        duration_ms,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_write_tokens: usage.cache_write_tokens,
        raw_usage_json: usage.raw_usage_json,
        status: LLM_REQUEST_STATUS_SUCCESS.to_string(),
        error_summary: None,
    };

    persist_llm_request(audit_context, insert).await;
}

/// Record a provider call that failed after its retries were exhausted, so
/// the audit trail shows attempts and not only successes. `error_summary`
/// must already be safe to store (no credentials, bounded length).
pub async fn record_llm_request_failure(
    audit_context: Option<&LlmAuditContext>,
    provider: &str,
    model: &str,
    operation: &str,
    started_at: DateTime<Utc>,
    completed_at: DateTime<Utc>,
    error_summary: &str,
) {
    let duration_ms = (completed_at - started_at).num_milliseconds().max(0);
    info!(
        target: "bot.timing",
        event = "llm_response",
        provider,
        model,
        operation,
        completed_at = %completed_at.to_rfc3339(),
        duration_ms,
        status = "error",
        error = error_summary,
    );

    let Some(audit_context) = audit_context else {
        return;
    };

    let insert = LlmRequestInsert {
        invocation_id: audit_context.invocation_id,
        provider: provider.to_string(),
        model: model.to_string(),
        operation: operation.to_string(),
        response_id: None,
        started_at,
        completed_at,
        duration_ms,
        input_tokens: None,
        output_tokens: None,
        total_tokens: None,
        reasoning_tokens: None,
        cached_input_tokens: None,
        cache_write_tokens: None,
        raw_usage_json: None,
        status: LLM_REQUEST_STATUS_ERROR.to_string(),
        error_summary: Some(truncate_for_log(error_summary, 1_000)),
    };

    persist_llm_request(audit_context, insert).await;
}

async fn persist_llm_request(audit_context: &LlmAuditContext, insert: LlmRequestInsert) {
    let (provider, model, operation) = (
        insert.provider.clone(),
        insert.model.clone(),
        insert.operation.clone(),
    );
    if let Err(err) = audit_context.db.insert_llm_request(insert).await {
        warn!(
            "Failed to persist llm request audit row: invocation_id={}, provider={}, model={}, operation={}, error={err}",
            audit_context.invocation_id,
            provider,
            model,
            operation
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::models::LlmInvocationInsert;
    use crate::utils::log_capture::capture_json_events;

    async fn test_database(name: &str) -> Database {
        let mut path = std::path::PathBuf::from("target");
        path.push("test-dbs");
        std::fs::create_dir_all(&path).expect("test db directory should exist");
        path.push(format!(
            "telegram-chat-bot-audit-{}-{}-{}.db",
            name,
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let url = format!(
            "sqlite://{}",
            path.to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/")
        );
        Database::init(&url)
            .await
            .expect("test database should initialize")
    }

    async fn test_invocation(db: &Database) -> i64 {
        db.insert_llm_invocation(LlmInvocationInsert {
            trigger_kind: "command".to_string(),
            trigger_name: "q".to_string(),
            chat_id: -100,
            user_id: Some(1),
            username: Some("alice".to_string()),
            message_id: 5,
            reply_to_message_id: None,
            message_text: Some("/q hi".to_string()),
            created_at: Utc::now(),
        })
        .await
        .expect("invocation should insert")
    }

    async fn request_row(
        db: &Database,
        invocation_id: i64,
    ) -> (String, Option<String>, Option<i64>) {
        sqlx::query_as(
            "SELECT status, error_summary, total_tokens FROM llm_requests WHERE invocation_id = ?",
        )
        .bind(invocation_id)
        .fetch_one(db.pool())
        .await
        .expect("exactly one request row")
    }

    #[tokio::test]
    async fn record_llm_request_failure_persists_an_error_row() {
        let db = test_database("failure-row").await;
        let invocation_id = test_invocation(&db).await;
        let context = LlmAuditContext::new(db.clone(), invocation_id);

        record_llm_request_failure(
            Some(&context),
            "gemini",
            "gemini-2.5-flash",
            "generate_content",
            Utc::now(),
            Utc::now(),
            "status 503: overloaded",
        )
        .await;

        let (status, error_summary, total_tokens) = request_row(&db, invocation_id).await;
        assert_eq!(status, "error");
        assert_eq!(error_summary.as_deref(), Some("status 503: overloaded"));
        assert_eq!(total_tokens, None);
        db.shutdown().await;
    }

    #[tokio::test]
    async fn record_llm_request_success_marks_the_row_successful() {
        let db = test_database("success-row").await;
        let invocation_id = test_invocation(&db).await;
        let context = LlmAuditContext::new(db.clone(), invocation_id);

        record_llm_request_success(
            Some(&context),
            "gemini",
            "gemini-2.5-flash",
            "generate_content",
            Utc::now(),
            Utc::now(),
            LlmUsageRecord {
                total_tokens: Some(46),
                ..LlmUsageRecord::default()
            },
        )
        .await;

        let (status, error_summary, total_tokens) = request_row(&db, invocation_id).await;
        assert_eq!(status, "success");
        assert_eq!(error_summary, None);
        assert_eq!(total_tokens, Some(46));
        db.shutdown().await;
    }

    #[test]
    fn llm_request_started_emits_structured_fields_for_the_json_layer() {
        let events = capture_json_events(|| {
            log_llm_request_started(
                "gemini",
                "gemini-2.5-flash",
                "generate_content",
                Utc::now(),
                Some(&serde_json::json!({"thinking": "low"})),
            );
        });

        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0]["target"], "bot.timing");
        let fields = &events[0]["fields"];
        assert_eq!(fields["event"], "llm_request");
        assert_eq!(fields["provider"], "gemini");
        assert_eq!(fields["model"], "gemini-2.5-flash");
        assert_eq!(fields["operation"], "generate_content");
        assert_eq!(fields["metadata"], r#"{"thinking":"low"}"#);
        assert!(fields["started_at"].is_string(), "{fields:?}");
    }
}

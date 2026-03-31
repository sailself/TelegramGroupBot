use chrono::{DateTime, Utc};
use serde_json::Value;
use teloxide::types::Message;
use tracing::{info, warn};

use crate::db::database::Database;
use crate::db::models::LlmRequestInsert;

pub const LLM_TRIGGER_KIND_AUTO_Q: &str = "auto_q";
pub const LLM_TRIGGER_KIND_COMMAND: &str = "command";

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
        "event=llm_request provider={} model={} operation={} started_at={} metadata={}",
        provider,
        model,
        operation,
        started_at.to_rfc3339(),
        json_text(metadata)
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
        "event=llm_response provider={} model={} operation={} completed_at={} duration_ms={} status=success response_id={:?} usage={}",
        provider,
        model,
        operation,
        completed_at.to_rfc3339(),
        duration_ms,
        usage.response_id,
        usage.raw_usage_json.as_deref().unwrap_or("{}")
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
        raw_usage_json: usage.raw_usage_json,
    };

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

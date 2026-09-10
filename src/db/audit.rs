//! LLM invocation/request audit logging and token-usage queries.

use crate::db::database::Database;
use crate::db::models::{LlmInvocationInsert, LlmRequestInsert, ModelTokenStat, TokenUserStat};
use anyhow::Result;
use teloxide::types::Message;

const TOKEN_TOTAL_EXPR: &str = "COALESCE(r.total_tokens, r.input_tokens + r.output_tokens, 0)";

impl Database {
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
                 cache_write_tokens, \
                 raw_usage_json, \
                 status, \
                 error_summary\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        .bind(insert.cache_write_tokens)
        .bind(insert.raw_usage_json)
        .bind(insert.status)
        .bind(insert.error_summary)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_support::{
        init_test_db, insert_invocation_with_usage, queue_message_with_user,
    };
    use chrono::Utc;
    use sqlx::FromRow;

    /// The audit columns the tests assert on; `SELECT *` maps by name, so
    /// unlisted columns are simply ignored.
    #[derive(Debug, FromRow)]
    struct LlmInvocationRow {
        trigger_name: String,
    }

    #[derive(Debug, FromRow)]
    struct LlmRequestRow {
        invocation_id: i64,
        provider: String,
        total_tokens: Option<i64>,
        reasoning_tokens: Option<i64>,
        cached_input_tokens: Option<i64>,
        cache_write_tokens: Option<i64>,
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
            cache_write_tokens: Some(4),
            raw_usage_json: Some("{\"totalTokenCount\":46}".to_string()),
            status: "success".to_string(),
            error_summary: None,
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
        assert_eq!(request.cache_write_tokens, Some(4));
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
}

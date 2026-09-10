//! Message row queries: recent history, id-anchored windows, topic windows,
//! and the free-form chat analytics used by `/qc`.

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::db::models::{AnalyticsRow, MessageInsert, MessageRow, TopicWindow, TopicWindowSpec};
use crate::db::search_index::WINDOW_LIMIT_MAX;
use anyhow::{anyhow, Result};
use std::time::Duration;

fn topic_window_is_capped(total_eligible: i64, selected_messages: usize) -> bool {
    total_eligible > selected_messages as i64
}

impl Database {
    pub async fn run_chat_analytics(
        &self,
        chat_id: i64,
        spec: &crate::llm::analytics::QuerySpec,
    ) -> Result<(crate::llm::analytics::QuerySpec, Vec<AnalyticsRow>)> {
        use crate::llm::analytics::Bind;
        let spec = crate::llm::analytics::normalize_and_validate(spec.clone())
            .map_err(anyhow::Error::msg)?;
        // Term filters hit the FTS index; refuse while it's rebuilding (mirror search_chat_messages).
        if spec
            .filters
            .term
            .as_deref()
            .is_some_and(|t| !t.trim().is_empty())
            && !self.is_search_ready()
        {
            return Err(anyhow::anyhow!(
                crate::db::search::SEARCH_INDEX_REBUILDING_ERROR
            ));
        }
        let (sql, binds) = crate::llm::analytics::compile(&spec, chat_id);
        let mut q = sqlx::query_as::<_, AnalyticsRow>(&sql);
        for b in binds {
            q = match b {
                Bind::Int(i) => q.bind(i),
                Bind::Text(s) => q.bind(s),
            };
        }
        // Per-query timeout so a leading-wildcard scan or a pathological grouping
        // can't pin a connection.
        let dur = std::time::Duration::from_secs(CONFIG.agents.qc_analytics_query_timeout_secs);
        match tokio::time::timeout(dur, q.fetch_all(&self.pool)).await {
            Ok(Ok(rows)) => Ok((spec, rows)),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err(anyhow::anyhow!("analytics query exceeded the time budget")),
        }
    }

    pub async fn select_topic_window(
        &self,
        chat_id: i64,
        spec: &TopicWindowSpec,
    ) -> Result<TopicWindow> {
        let mut where_sql = String::from(
            " FROM messages m WHERE m.chat_id = ? \
             AND m.date >= ? AND m.date < ? \
             AND m.text IS NOT NULL AND TRIM(m.text) <> '' \
             AND m.user_id IS NOT NULL AND m.user_id <> 1087968824",
        );
        if spec.exclude_commands {
            where_sql.push_str(" AND m.is_command = 0");
        }
        if spec.exclude_synthetic {
            where_sql.push_str(" AND m.is_synthetic_record = 0");
        }
        if spec.user_id.is_some() {
            where_sql.push_str(" AND m.user_id = ?");
        }

        let mut transaction = self.pool.begin().await?;
        let count_sql = format!("SELECT COUNT(*){where_sql}");
        let mut count_query = sqlx::query_scalar::<_, i64>(&count_sql)
            .bind(chat_id)
            .bind(spec.date_from.to_rfc3339())
            .bind(spec.date_to.to_rfc3339());
        if let Some(user_id) = spec.user_id {
            count_query = count_query.bind(user_id);
        }
        let timeout = Duration::from_secs(CONFIG.agents.qc_analytics_query_timeout_secs);
        let total_eligible =
            tokio::time::timeout(timeout, count_query.fetch_one(&mut *transaction))
                .await
                .map_err(|_| anyhow!("topic count query exceeded the time budget"))??;

        let select_sql = format!(
            "SELECT m.id, m.message_id, m.chat_id, m.user_id, m.username, m.text, \
             m.language, m.date, m.reply_to_message_id, m.asks_ai, m.ai_command, \
             m.is_synthetic_record{where_sql} ORDER BY m.date DESC, m.message_id DESC LIMIT ?"
        );
        let mut select_query = sqlx::query_as::<_, MessageRow>(&select_sql)
            .bind(chat_id)
            .bind(spec.date_from.to_rfc3339())
            .bind(spec.date_to.to_rfc3339());
        if let Some(user_id) = spec.user_id {
            select_query = select_query.bind(user_id);
        }
        let limit = spec.limit.max(1);
        let mut messages = tokio::time::timeout(
            timeout,
            select_query.bind(limit).fetch_all(&mut *transaction),
        )
        .await
        .map_err(|_| anyhow!("topic window query exceeded the time budget"))??;
        transaction.commit().await?;
        messages.reverse();

        Ok(TopicWindow {
            capped: topic_window_is_capped(total_eligible, messages.len()),
            total_eligible,
            messages,
        })
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

    /// Messages from `message_id` onward, newest `limit` only, in ascending order.
    pub async fn select_messages_from_id(
        &self,
        chat_id: i64,
        message_id: i64,
        limit: i64,
    ) -> Result<Vec<MessageRow>> {
        self.get_messages_from_id(chat_id, message_id, limit, true)
            .await
    }

    async fn get_last_n_text_messages(
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

    async fn get_messages_from_id(
        &self,
        chat_id: i64,
        from_message_id: i64,
        limit: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id, asks_ai, ai_command, is_synthetic_record \
             FROM messages WHERE chat_id = ? AND message_id >= ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC LIMIT ?");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(from_message_id)
            .bind(limit.max(1))
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
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
    use crate::db::models::TopicWindowSpec;
    use crate::db::test_support::{
        at, init_test_db, insert_count_message, queue_message, queue_message_with_user,
    };

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

    #[test]
    fn topic_window_coverage_invariants_match_selected_snapshot_rows() {
        assert!(!topic_window_is_capped(0, 0));
        assert!(!topic_window_is_capped(2, 2));
        assert!(topic_window_is_capped(3, 2));
    }

    #[tokio::test]
    async fn topic_window_is_chat_scoped_excludes_non_topics_and_reports_cap() {
        let db = init_test_db("topic-window").await;
        let chat_a = -1001374348669_i64;
        let chat_b = -1002631835259_i64;
        let alice_id = 11_i64;

        insert_count_message(
            &db,
            1,
            chat_a,
            Some(alice_id),
            Some("alice"),
            "eligible alice one",
            at("2026-07-01T01:00:00Z"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            2,
            chat_a,
            Some(12),
            Some("bob"),
            "eligible bob",
            at("2026-07-02T01:00:00Z"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            3,
            chat_a,
            Some(alice_id),
            Some("alice"),
            "eligible alice two",
            at("2026-07-03T01:00:00Z"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            4,
            chat_b,
            Some(99),
            Some("mallory"),
            "cross-chat sentinel",
            at("2026-07-04T01:00:00Z"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            5,
            chat_a,
            Some(alice_id),
            Some("alice"),
            "command sentinel",
            at("2026-07-05T01:00:00Z"),
            true,
            false,
        )
        .await;
        insert_count_message(
            &db,
            6,
            chat_a,
            Some(alice_id),
            Some("alice"),
            "synthetic sentinel",
            at("2026-07-06T01:00:00Z"),
            false,
            true,
        )
        .await;
        insert_count_message(
            &db,
            7,
            chat_a,
            None,
            Some("channel"),
            "null-user channel sentinel",
            at("2026-07-06T02:00:00Z"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            8,
            chat_a,
            Some(1_087_968_824),
            Some("anonymous-admin"),
            "anonymous-admin sentinel",
            at("2026-07-06T03:00:00Z"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            9,
            chat_a,
            Some(alice_id),
            Some("alice"),
            "   ",
            at("2026-07-06T04:00:00Z"),
            false,
            false,
        )
        .await;

        let spec = TopicWindowSpec {
            date_from: at("2026-07-01T00:00:00Z"),
            date_to: at("2026-07-08T00:00:00Z"),
            user_id: None,
            exclude_commands: true,
            exclude_synthetic: true,
            limit: 2,
        };
        let window = db.select_topic_window(chat_a, &spec).await.unwrap();

        assert_eq!(window.total_eligible, 3);
        assert_eq!(window.messages.len(), 2);
        assert!(window.capped);
        assert_eq!(
            window
                .messages
                .iter()
                .map(|row| row.message_id)
                .collect::<Vec<_>>(),
            vec![2, 3],
            "the cap must retain the newest eligible rows before restoring chronological order"
        );
        assert!(window.total_eligible >= window.messages.len() as i64);
        assert_eq!(
            window.capped,
            topic_window_is_capped(window.total_eligible, window.messages.len())
        );
        assert!(window.messages.iter().all(|row| row.chat_id == chat_a));
        assert!(window.messages[0].date <= window.messages[1].date);
        assert!(window.messages.iter().all(|row| {
            !matches!(
                row.text.as_deref(),
                Some(
                    "cross-chat sentinel"
                        | "command sentinel"
                        | "synthetic sentinel"
                        | "null-user channel sentinel"
                        | "anonymous-admin sentinel"
                        | "   "
                )
            )
        }));

        let alice_spec = TopicWindowSpec {
            user_id: Some(alice_id),
            limit: 10,
            ..spec
        };
        let alice_window = db.select_topic_window(chat_a, &alice_spec).await.unwrap();

        assert_eq!(alice_window.total_eligible, 2);
        assert_eq!(alice_window.messages.len(), 2);
        assert!(!alice_window.capped);
        assert!(alice_window
            .messages
            .iter()
            .all(|row| row.user_id == Some(alice_id)));
        assert!(alice_window.messages[0].date <= alice_window.messages[1].date);
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
            let (_normalized_spec, rows) = db
                .run_chat_analytics(a, &spec)
                .await
                .expect("query ok (inert, not executed SQL)");
            for r in &rows {
                assert_ne!(r.group_key.as_deref(), Some("SENTINEL_CHATB"));
                assert!(!r.value_text.as_deref().unwrap_or("").contains("SENTINEL"));
                // chat A has only 3 messages, so any count above that means a leak.
                assert!(
                    r.value_num.is_none_or(|v| v <= 3.0),
                    "adversarial spec returned an impossible count (possible leak)"
                );
            }
        }
        // chat_id-in-spec is ignored: total count == chat A's 3 messages, never includes B.
        let total: QuerySpec =
            parse_lenient_json(r#"{"metric":"count","chat_id":-1002631835259}"#).unwrap();
        let (_normalized_spec, rows) = db.run_chat_analytics(a, &total).await.unwrap();
        assert_eq!(rows[0].value_num, Some(3.0));
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
        let (_normalized_spec, r) = db.run_chat_analytics(a, &audit_probe).await.unwrap();
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
        let (_normalized_spec, r2) = db.run_chat_analytics(a, &chatb_probe).await.unwrap();
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
        let (_normalized_spec, rows) = db
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
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.expect("query ok");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value_num, Some(2.0));
    }

    #[tokio::test]
    async fn run_chat_analytics_text_contains_treats_wildcards_literally_and_counts_messages() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-literal-wildcards").await;
        let chat = -1001374348669_i64;
        for (message_id, text) in [
            (1, "sale 50%_off now"),
            (2, "50%_off repeated 50%_off"),
            (3, "50percentXoff wildcard lookalike"),
        ] {
            insert_count_message(
                &db,
                message_id,
                chat,
                Some(10),
                Some("alice"),
                text,
                at("2026-04-01T00:00:00+00:00"),
                false,
                false,
            )
            .await;
        }

        let spec: QuerySpec =
            parse_lenient_json(r#"{"metric":"count","filters":{"text_contains":"50%_off"}}"#)
                .unwrap();
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.expect("query ok");

        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].value_num,
            Some(2.0),
            "count matching messages once each; do not expand %/_ or count occurrences"
        );
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
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.expect("query ok");

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
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.expect("query ok");

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
        let (_normalized_spec, min_rows) = db
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
        let (_normalized_spec, max_rows) = db
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
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.expect("query ok");

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
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.unwrap();
        assert_eq!(rows.first().and_then(|r| r.value_num), Some(5.0)); // (3+7)/2
    }

    #[tokio::test]
    async fn run_chat_analytics_group_by_hour_of_day_buckets_in_utc() {
        use crate::agents::step::parse_lenient_json;
        use crate::llm::analytics::QuerySpec;
        let db = init_test_db("analytics-hour").await;
        let chat = -1001374348669_i64;
        insert_count_message(
            &db,
            1,
            chat,
            Some(11),
            Some("a"),
            "x",
            at("2026-03-01T09:00:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            2,
            chat,
            Some(11),
            Some("a"),
            "y",
            at("2026-03-02T09:30:00+00:00"),
            false,
            false,
        )
        .await;
        insert_count_message(
            &db,
            3,
            chat,
            Some(12),
            Some("b"),
            "z",
            at("2026-03-03T14:00:00+00:00"),
            false,
            false,
        )
        .await;
        let spec: QuerySpec = parse_lenient_json(
            r#"{"metric":"count","group_by":"hour_of_day","order":"value_desc"}"#,
        )
        .unwrap();
        let (_normalized_spec, rows) = db.run_chat_analytics(chat, &spec).await.unwrap();
        let top = rows.first().unwrap();
        assert_eq!(top.group_key.as_deref(), Some("09")); // two messages in the 09:00 UTC hour
        assert_eq!(top.value_num, Some(2.0));
    }

    #[tokio::test]
    async fn display_labels_disambiguate_same_name_users_in_chat() {
        use crate::llm::prompting::{build_display_label_map, format_tldr_chat_content};

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

    #[tokio::test]
    async fn select_messages_from_id_keeps_only_the_newest_limit_rows() {
        let db = init_test_db("select-from-id-limit").await;
        let chat = -1001374348670_i64;
        for message_id in 1..=5_i64 {
            insert_count_message(
                &db,
                message_id,
                chat,
                Some(1001),
                Some("Alice"),
                &format!("message {message_id}"),
                at(&format!("2026-09-01T10:0{message_id}:00Z")),
                false,
                false,
            )
            .await;
        }

        let rows = db
            .select_messages_from_id(chat, 1, 3)
            .await
            .expect("select should work");

        let ids: Vec<i64> = rows.iter().map(|row| row.message_id).collect();
        assert_eq!(ids, vec![3, 4, 5], "newest three, ascending");
    }
}

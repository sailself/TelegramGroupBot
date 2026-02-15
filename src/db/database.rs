use crate::db::models::{
    AgentMemoryInsert, AgentMemoryRow, AgentMemorySearchRow, AgentSessionRow, AgentStepRow,
    MessageInsert, MessageRow,
};
use anyhow::Result;
use serde_json::Value;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};
use tokio::sync::mpsc;
use tracing::{info, warn};

#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
    sender: mpsc::Sender<MessageInsert>,
}

impl Database {
    pub async fn init(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS messages (\
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
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
            .execute(&pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);")
            .execute(&pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);")
            .execute(&pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS agent_sessions (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                chat_id INTEGER NOT NULL,\
                user_id INTEGER NOT NULL,\
                model_name TEXT NOT NULL,\
                prompt TEXT NOT NULL,\
                selected_skills_json TEXT,\
                status TEXT NOT NULL DEFAULT 'running',\
                final_response TEXT,\
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\
            );",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS agent_steps (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                session_id INTEGER NOT NULL,\
                role TEXT NOT NULL,\
                content TEXT,\
                raw_json TEXT NOT NULL,\
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                FOREIGN KEY(session_id) REFERENCES agent_sessions(id)\
            );",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS agent_tool_calls (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                session_id INTEGER NOT NULL,\
                step_id INTEGER NOT NULL,\
                tool_call_id TEXT NOT NULL,\
                tool_name TEXT NOT NULL,\
                args_json TEXT NOT NULL,\
                status TEXT NOT NULL,\
                requires_confirmation INTEGER NOT NULL DEFAULT 0,\
                confirmed_by INTEGER,\
                result_json TEXT,\
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                completed_at TEXT,\
                FOREIGN KEY(session_id) REFERENCES agent_sessions(id),\
                FOREIGN KEY(step_id) REFERENCES agent_steps(id)\
            );",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS agent_session_skills (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                session_id INTEGER NOT NULL,\
                selected_skills_json TEXT NOT NULL,\
                selection_reason TEXT,\
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\
                FOREIGN KEY(session_id) REFERENCES agent_sessions(id)\
            );",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS agent_memories (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                chat_id INTEGER NOT NULL,\
                user_id INTEGER,\
                session_id INTEGER,\
                source_role TEXT NOT NULL,\
                category TEXT NOT NULL,\
                content TEXT NOT NULL,\
                summary TEXT,\
                importance REAL NOT NULL DEFAULT 0.5,\
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\
            );",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS agent_memories_fts USING fts5(content, summary);",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS agent_memories_ai AFTER INSERT ON agent_memories BEGIN \
                INSERT INTO agent_memories_fts(rowid, content, summary) \
                VALUES (new.id, new.content, COALESCE(new.summary, '')); \
            END;",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS agent_memories_au AFTER UPDATE ON agent_memories BEGIN \
                INSERT INTO agent_memories_fts(agent_memories_fts, rowid, content, summary) \
                VALUES('delete', old.id, old.content, COALESCE(old.summary, '')); \
                INSERT INTO agent_memories_fts(rowid, content, summary) \
                VALUES (new.id, new.content, COALESCE(new.summary, '')); \
            END;",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS agent_memories_ad AFTER DELETE ON agent_memories BEGIN \
                INSERT INTO agent_memories_fts(agent_memories_fts, rowid, content, summary) \
                VALUES('delete', old.id, old.content, COALESCE(old.summary, '')); \
            END;",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "INSERT INTO agent_memories_fts(rowid, content, summary) \
             SELECT id, content, COALESCE(summary, '') \
             FROM agent_memories \
             WHERE id NOT IN (SELECT rowid FROM agent_memories_fts);",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_chat_user ON agent_sessions(chat_id, user_id);",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_agent_steps_session_id ON agent_steps(session_id);",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_session_id ON agent_tool_calls(session_id);",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_agent_session_skills_session_id ON agent_session_skills(session_id);",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_chat_created ON agent_memories(chat_id, created_at);",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_chat_category ON agent_memories(chat_id, category);",
        )
        .execute(&pool)
        .await?;

        info!("Database tables created successfully");

        let (sender, receiver) = mpsc::channel(1000);
        let writer_pool = pool.clone();
        tokio::spawn(async move {
            db_writer(writer_pool, receiver).await;
        });

        info!("Database writer task started");

        Ok(Database { pool, sender })
    }

    pub async fn queue_message_insert(&self, insert: MessageInsert) -> Result<()> {
        self.sender
            .send(insert)
            .await
            .map_err(|err| anyhow::anyhow!("Failed to queue message insert: {err}"))
    }

    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
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
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
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
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
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
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
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

    pub async fn create_agent_session(
        &self,
        chat_id: i64,
        user_id: i64,
        model_name: &str,
        prompt: &str,
        selected_skills_json: &str,
    ) -> Result<i64> {
        let result = sqlx::query(
            "INSERT INTO agent_sessions (chat_id, user_id, model_name, prompt, selected_skills_json, status) \
             VALUES (?, ?, ?, ?, ?, 'running')",
        )
        .bind(chat_id)
        .bind(user_id)
        .bind(model_name)
        .bind(prompt)
        .bind(selected_skills_json)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    pub async fn complete_agent_session(
        &self,
        session_id: i64,
        status: &str,
        final_response: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE agent_sessions \
             SET status = ?, final_response = COALESCE(?, final_response), updated_at = CURRENT_TIMESTAMP \
             WHERE id = ?",
        )
        .bind(status)
        .bind(final_response)
        .bind(session_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_agent_step(
        &self,
        session_id: i64,
        role: &str,
        content: &str,
        raw_json: &Value,
    ) -> Result<i64> {
        let result = sqlx::query(
            "INSERT INTO agent_steps (session_id, role, content, raw_json) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(role)
        .bind(content)
        .bind(raw_json.to_string())
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    pub async fn insert_agent_tool_call(
        &self,
        session_id: i64,
        step_id: i64,
        tool_call_id: &str,
        tool_name: &str,
        args_json: &Value,
        status: &str,
        requires_confirmation: bool,
    ) -> Result<i64> {
        let result = sqlx::query(
            "INSERT INTO agent_tool_calls \
             (session_id, step_id, tool_call_id, tool_name, args_json, status, requires_confirmation) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(step_id)
        .bind(tool_call_id)
        .bind(tool_name)
        .bind(args_json.to_string())
        .bind(status)
        .bind(if requires_confirmation { 1 } else { 0 })
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    pub async fn update_agent_tool_call_status(
        &self,
        tool_call_row_id: i64,
        status: &str,
        result_json: Option<&Value>,
        confirmed_by: Option<i64>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE agent_tool_calls \
             SET status = ?, result_json = COALESCE(?, result_json), confirmed_by = COALESCE(?, confirmed_by), completed_at = CURRENT_TIMESTAMP \
             WHERE id = ?",
        )
        .bind(status)
        .bind(result_json.map(|value| value.to_string()))
        .bind(confirmed_by)
        .bind(tool_call_row_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn record_agent_session_skills(
        &self,
        session_id: i64,
        selected_skills: &[String],
        selection_reason: &str,
    ) -> Result<()> {
        let selected_skills_json = serde_json::to_string(selected_skills)?;
        sqlx::query(
            "INSERT INTO agent_session_skills (session_id, selected_skills_json, selection_reason) \
             VALUES (?, ?, ?)",
        )
        .bind(session_id)
        .bind(selected_skills_json)
        .bind(selection_reason)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_recent_agent_sessions_for_user(
        &self,
        chat_id: i64,
        user_id: i64,
        limit: usize,
    ) -> Result<Vec<AgentSessionRow>> {
        let rows = sqlx::query_as::<_, AgentSessionRow>(
            "SELECT \
                id, chat_id, user_id, model_name, prompt, selected_skills_json, \
                status, final_response, created_at, updated_at \
             FROM agent_sessions \
             WHERE chat_id = ? AND user_id = ? \
             ORDER BY datetime(updated_at) DESC \
             LIMIT ?",
        )
        .bind(chat_id)
        .bind(user_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn get_agent_session_for_user(
        &self,
        session_id: i64,
        chat_id: i64,
        user_id: i64,
    ) -> Result<Option<AgentSessionRow>> {
        let row = sqlx::query_as::<_, AgentSessionRow>(
            "SELECT \
                id, chat_id, user_id, model_name, prompt, selected_skills_json, \
                status, final_response, created_at, updated_at \
             FROM agent_sessions \
             WHERE id = ? AND chat_id = ? AND user_id = ? \
             LIMIT 1",
        )
        .bind(session_id)
        .bind(chat_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    pub async fn list_agent_steps(
        &self,
        session_id: i64,
        limit: usize,
    ) -> Result<Vec<AgentStepRow>> {
        let mut rows = sqlx::query_as::<_, AgentStepRow>(
            "SELECT \
                id, session_id, role, content, raw_json, created_at \
             FROM agent_steps \
             WHERE session_id = ? \
             ORDER BY id DESC \
             LIMIT ?",
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.reverse();
        Ok(rows)
    }

    pub async fn supersede_active_agent_sessions(
        &self,
        chat_id: i64,
        user_id: i64,
    ) -> Result<usize> {
        let result = sqlx::query(
            "UPDATE agent_sessions \
             SET status = 'superseded', updated_at = CURRENT_TIMESTAMP \
             WHERE chat_id = ? AND user_id = ? AND status IN ('running', 'awaiting_confirmation')",
        )
        .bind(chat_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() as usize)
    }

    pub async fn prune_agent_memories_older_than(&self, retention_days: u64) -> Result<usize> {
        if retention_days == 0 {
            return Ok(0);
        }

        let retention_expr = format!("-{} days", retention_days);
        let result = sqlx::query(
            "DELETE FROM agent_memories \
             WHERE datetime(created_at) < datetime('now', ?)",
        )
        .bind(retention_expr)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() as usize)
    }

    pub async fn prune_agent_sessions_older_than(
        &self,
        retention_days: u64,
    ) -> Result<(usize, usize, usize, usize)> {
        if retention_days == 0 {
            return Ok((0, 0, 0, 0));
        }

        let retention_expr = format!("-{} days", retention_days);
        let rows = sqlx::query(
            "SELECT id FROM agent_sessions \
             WHERE status NOT IN ('running', 'awaiting_confirmation') \
               AND datetime(updated_at) < datetime('now', ?)",
        )
        .bind(retention_expr)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok((0, 0, 0, 0));
        }

        let session_ids = rows
            .into_iter()
            .filter_map(|row| row.try_get::<i64, _>("id").ok())
            .collect::<Vec<_>>();
        if session_ids.is_empty() {
            return Ok((0, 0, 0, 0));
        }

        let placeholders = std::iter::repeat("?")
            .take(session_ids.len())
            .collect::<Vec<_>>()
            .join(", ");

        let delete_steps_sql = format!(
            "DELETE FROM agent_steps WHERE session_id IN ({})",
            placeholders
        );
        let delete_calls_sql = format!(
            "DELETE FROM agent_tool_calls WHERE session_id IN ({})",
            placeholders
        );
        let delete_skills_sql = format!(
            "DELETE FROM agent_session_skills WHERE session_id IN ({})",
            placeholders
        );
        let delete_sessions_sql =
            format!("DELETE FROM agent_sessions WHERE id IN ({})", placeholders);

        let mut delete_steps_query = sqlx::query(&delete_steps_sql);
        let mut delete_calls_query = sqlx::query(&delete_calls_sql);
        let mut delete_skills_query = sqlx::query(&delete_skills_sql);
        let mut delete_sessions_query = sqlx::query(&delete_sessions_sql);

        for session_id in &session_ids {
            delete_steps_query = delete_steps_query.bind(session_id);
            delete_calls_query = delete_calls_query.bind(session_id);
            delete_skills_query = delete_skills_query.bind(session_id);
            delete_sessions_query = delete_sessions_query.bind(session_id);
        }

        let steps_deleted = delete_steps_query
            .execute(&self.pool)
            .await?
            .rows_affected() as usize;
        let calls_deleted = delete_calls_query
            .execute(&self.pool)
            .await?
            .rows_affected() as usize;
        let skills_deleted = delete_skills_query
            .execute(&self.pool)
            .await?
            .rows_affected() as usize;
        let sessions_deleted = delete_sessions_query
            .execute(&self.pool)
            .await?
            .rows_affected() as usize;

        Ok((
            sessions_deleted,
            steps_deleted,
            calls_deleted,
            skills_deleted,
        ))
    }

    pub async fn insert_agent_memory(&self, insert: AgentMemoryInsert<'_>) -> Result<i64> {
        let result = sqlx::query(
            "INSERT INTO agent_memories \
             (chat_id, user_id, session_id, source_role, category, content, summary, importance) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(insert.chat_id)
        .bind(insert.user_id)
        .bind(insert.session_id)
        .bind(insert.source_role)
        .bind(insert.category)
        .bind(insert.content)
        .bind(insert.summary)
        .bind(insert.importance)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    pub async fn recent_agent_memories(
        &self,
        chat_id: i64,
        limit: usize,
    ) -> Result<Vec<AgentMemoryRow>> {
        let rows = sqlx::query_as::<_, AgentMemoryRow>(
            "SELECT \
                id, chat_id, user_id, session_id, source_role, category, content, summary, importance, created_at \
             FROM agent_memories \
             WHERE chat_id = ? \
             ORDER BY datetime(created_at) DESC \
             LIMIT ?",
        )
        .bind(chat_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn search_agent_memories(
        &self,
        chat_id: i64,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<AgentMemorySearchRow>> {
        let trimmed = query_text.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        let rows = sqlx::query(
            "SELECT \
                m.id, m.chat_id, m.user_id, m.session_id, m.source_role, m.category, \
                m.content, m.summary, m.importance, m.created_at, \
                bm25(agent_memories_fts) AS bm25_rank, \
                MAX(0.0, julianday('now') - julianday(m.created_at)) AS recency_days \
             FROM agent_memories_fts \
             JOIN agent_memories m ON m.id = agent_memories_fts.rowid \
             WHERE m.chat_id = ? AND agent_memories_fts MATCH ? \
             ORDER BY bm25_rank ASC \
             LIMIT ?",
        )
        .bind(chat_id)
        .bind(trimmed)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let memory = AgentMemoryRow {
                id: row.try_get("id")?,
                chat_id: row.try_get("chat_id")?,
                user_id: row.try_get("user_id")?,
                session_id: row.try_get("session_id")?,
                source_role: row.try_get("source_role")?,
                category: row.try_get("category")?,
                content: row.try_get("content")?,
                summary: row.try_get("summary")?,
                importance: row.try_get::<f64, _>("importance").unwrap_or(0.5),
                created_at: row.try_get("created_at")?,
            };
            let lexical_score = row.try_get::<f64, _>("bm25_rank").unwrap_or(1000.0);
            let recency_days = row.try_get::<f64, _>("recency_days").unwrap_or(365.0);
            out.push(AgentMemorySearchRow {
                memory,
                lexical_score,
                recency_days,
            });
        }

        Ok(out)
    }

    pub async fn delete_agent_memories(&self, chat_id: i64, ids: &[i64]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }

        let placeholders = std::iter::repeat("?")
            .take(ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "DELETE FROM agent_memories WHERE chat_id = ? AND id IN ({})",
            placeholders
        );

        let mut query = sqlx::query(&sql).bind(chat_id);
        for id in ids {
            query = query.bind(id);
        }
        let result = query.execute(&self.pool).await?;
        Ok(result.rows_affected() as usize)
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

async fn db_writer(pool: SqlitePool, mut receiver: mpsc::Receiver<MessageInsert>) {
    while let Some(message) = receiver.recv().await {
        let result = sqlx::query(
            "INSERT INTO messages (message_id, chat_id, user_id, username, text, language, date, reply_to_message_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(chat_id, message_id) DO UPDATE SET \
             user_id = excluded.user_id, \
             username = excluded.username, \
             text = excluded.text, \
             language = excluded.language, \
             date = excluded.date, \
             reply_to_message_id = excluded.reply_to_message_id",
        )
        .bind(message.message_id)
        .bind(message.chat_id)
        .bind(message.user_id)
        .bind(message.username)
        .bind(message.text)
        .bind(message.language)
        .bind(message.date)
        .bind(message.reply_to_message_id)
        .execute(&pool)
        .await;

        if let Err(err) = result {
            warn!("Error in db_writer: {err}");
        }
    }

    let _ = pool.close().await;
    info!("Database writer task stopped");
}

pub fn build_message_insert(
    user_id: Option<i64>,
    username: Option<String>,
    text: Option<String>,
    language: Option<String>,
    date: chrono::DateTime<chrono::Utc>,
    reply_to_message_id: Option<i64>,
    chat_id: Option<i64>,
    message_id: Option<i64>,
) -> MessageInsert {
    let resolved_user_id = user_id.unwrap_or_default();
    let resolved_chat_id = chat_id.unwrap_or(resolved_user_id);
    MessageInsert {
        message_id: message_id.unwrap_or_default(),
        chat_id: resolved_chat_id,
        user_id,
        username,
        text,
        language,
        date,
        reply_to_message_id,
    }
}

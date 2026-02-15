use crate::db::models::{MessageInsert, MessageRow};
use anyhow::Result;
use serde_json::Value;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
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

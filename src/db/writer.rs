//! The background write queue that batches message inserts onto the pool.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::db::models::MessageInsert;
use crate::db::search::{
    normalize_message_document, SearchProvenance, CURRENT_SEARCH_SCHEMA_VERSION,
};
use anyhow::{anyhow, Result};
use serde::Serialize;
use sqlx::SqlitePool;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

const DB_WRITE_RETRY_DELAY_MS: u64 = 100;
const DB_WRITE_DEAD_LETTER_PATH: &str = "data/db_writer_dead_letters.jsonl";

/// Work items for the background writer task.
pub(super) enum WriterCommand {
    Insert(MessageInsert),
    #[cfg(test)]
    Flush(tokio::sync::oneshot::Sender<std::result::Result<(), String>>),
    /// Flush everything queued so far (and anything that races in), close the
    /// pool, and exit.
    Shutdown,
}

impl Database {
    pub async fn queue_message_insert(&self, insert: MessageInsert) -> Result<()> {
        self.sender
            .send(WriterCommand::Insert(insert))
            .await
            .map_err(|_| anyhow!("Failed to queue message insert: writer is not accepting work"))
    }

    #[cfg(test)]
    pub(crate) async fn flush_for_test(&self) -> Result<()> {
        let (send, receive) = tokio::sync::oneshot::channel();
        self.sender
            .send(WriterCommand::Flush(send))
            .await
            .map_err(|_| anyhow!("writer closed"))?;
        receive
            .await
            .map_err(|_| anyhow!("writer dropped acknowledgement"))?
            .map_err(anyhow::Error::msg)
    }

    /// Flush everything queued for the background writer and close the pool.
    /// Safe to call while other `Database` clones are still alive; only the
    /// first caller waits for the writer, later calls return immediately.
    pub async fn shutdown(&self) {
        // A closed channel means the writer already finished a previous
        // shutdown; there is nothing left to flush.
        let _ = self.sender.send(WriterCommand::Shutdown).await;
        let task = self.writer_task.lock().take();
        if let Some(task) = task {
            if let Err(err) = task.await {
                error!("Database writer task ended abnormally during shutdown: {err}");
            }
        }
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
}

struct WriterSettings {
    flush_deadline: Duration,
    batch_size: usize,
    dead_letter_path: PathBuf,
    #[cfg(test)]
    batch_written: Option<mpsc::UnboundedSender<usize>>,
}

pub(super) async fn db_writer(pool: SqlitePool, receiver: mpsc::Receiver<WriterCommand>) {
    run_writer(
        pool,
        receiver,
        WriterSettings {
            flush_deadline: Duration::from_millis(CONFIG.db.write_flush_ms),
            batch_size: CONFIG.db.write_batch_size.max(1),
            dead_letter_path: PathBuf::from(DB_WRITE_DEAD_LETTER_PATH),
            #[cfg(test)]
            batch_written: None,
        },
    )
    .await;
}

async fn run_writer(
    pool: SqlitePool,
    mut receiver: mpsc::Receiver<WriterCommand>,
    settings: WriterSettings,
) {
    let batch_size = settings.batch_size;
    let flush_deadline = settings.flush_deadline;
    let mut buffer = Vec::with_capacity(batch_size);
    let mut shutting_down = false;

    while !shutting_down {
        let Some(command) = receiver.recv().await else {
            break;
        };
        #[cfg(test)]
        let mut acknowledgements = Vec::new();
        #[cfg(test)]
        let mut flush_requested = false;
        #[cfg(not(test))]
        let flush_requested = false;
        match command {
            #[cfg(test)]
            WriterCommand::Flush(sender) => {
                acknowledgements.push(sender);
                flush_requested = true;
            }
            WriterCommand::Insert(message) => buffer.push(message),
            WriterCommand::Shutdown => shutting_down = true,
        }

        if !shutting_down && !flush_requested {
            let flush_at = tokio::time::Instant::now() + flush_deadline;
            while buffer.len() < batch_size {
                match tokio::time::timeout_at(flush_at, receiver.recv()).await {
                    Ok(Some(WriterCommand::Insert(message))) => buffer.push(message),
                    #[cfg(test)]
                    Ok(Some(WriterCommand::Flush(sender))) => {
                        acknowledgements.push(sender);
                        break;
                    }
                    Ok(Some(WriterCommand::Shutdown)) => {
                        shutting_down = true;
                        break;
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        }

        if shutting_down {
            // Refuse new work, then drain inserts that raced the shutdown
            // request so nothing already accepted is lost.
            receiver.close();
            while let Some(command) = receiver.recv().await {
                match command {
                    WriterCommand::Insert(message) => buffer.push(message),
                    #[cfg(test)]
                    WriterCommand::Flush(sender) => acknowledgements.push(sender),
                    WriterCommand::Shutdown => {}
                }
            }
        }

        let result =
            write_message_batch_with_recovery(&pool, &buffer, &settings.dead_letter_path).await;
        if let Err(err) = &result {
            error!("Error in db_writer batch after recovery attempts: {err}");
        }
        #[cfg(test)]
        for acknowledgement in acknowledgements {
            let _ = acknowledgement.send(result.as_ref().map(|_| ()).map_err(ToString::to_string));
        }
        #[cfg(test)]
        if let Some(observer) = &settings.batch_written {
            let _ = observer.send(buffer.len());
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
            message.chat_id,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::database::build_message_insert;
    use crate::db::test_support::{sqlite_url_for_path, test_db_path, wait_for_search_ready};
    use chrono::Utc;
    use sqlx::sqlite::SqlitePoolOptions;

    fn fixture_message(id: i64, text: &str) -> MessageInsert {
        build_message_insert(
            Some(1),
            Some("alice".into()),
            Some(text.into()),
            Some("en".into()),
            Utc::now(),
            None,
            Some(-100),
            Some(id),
            None,
            false,
            None,
            false,
            false,
        )
    }

    #[tokio::test]
    async fn writer_batches_messages_and_flushes_on_deadline() {
        let db = crate::db::test_support::init_test_db("writer-deadline").await;
        let (send, recv) = mpsc::channel(10);
        let (written, mut observed) = mpsc::unbounded_channel();
        let settings = WriterSettings {
            flush_deadline: Duration::from_secs(10),
            batch_size: 3,
            dead_letter_path: test_db_path("writer-unused").with_extension("jsonl"),
            batch_written: Some(written),
        };
        let task = tokio::spawn(run_writer(db.pool().clone(), recv, settings));
        send.send(WriterCommand::Insert(fixture_message(1, "one")))
            .await
            .unwrap();
        send.send(WriterCommand::Insert(fixture_message(2, "two")))
            .await
            .unwrap();
        send.send(WriterCommand::Insert(fixture_message(3, "three")))
            .await
            .unwrap();
        assert_eq!(observed.recv().await, Some(3));
        // Once real SQLite I/O is idle, use paused time only for the batching timer.
        tokio::time::pause();
        send.send(WriterCommand::Insert(fixture_message(4, "four")))
            .await
            .unwrap();
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(9)).await;
        assert!(observed.try_recv().is_err());
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::time::resume();
        assert_eq!(observed.recv().await, Some(1));
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages")
            .fetch_one(db.pool())
            .await
            .unwrap();
        assert_eq!(count, 4);
        send.send(WriterCommand::Shutdown).await.unwrap();
        task.await.unwrap();
        db.shutdown().await;
    }

    #[tokio::test]
    async fn mixed_batch_salvages_valid_messages_and_edits_upsert() {
        let db = crate::db::test_support::init_test_db("writer-salvage").await;
        sqlx::query("CREATE TRIGGER reject_fixture BEFORE INSERT ON messages WHEN NEW.message_id=2 BEGIN SELECT RAISE(ABORT,'invalid fixture'); END").execute(db.pool()).await.unwrap();
        let path = test_db_path("writer-mixed").with_extension("jsonl");
        let batch = vec![
            fixture_message(1, "original"),
            fixture_message(2, "invalid"),
            fixture_message(3, "valid"),
        ];
        assert!(write_message_batch_with_recovery(db.pool(), &batch, &path)
            .await
            .is_err());
        let ids: Vec<i64> =
            sqlx::query_scalar("SELECT message_id FROM messages ORDER BY message_id")
                .fetch_all(db.pool())
                .await
                .unwrap();
        assert_eq!(ids, vec![1, 3]);
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 1);
        let row_id: i64 = sqlx::query_scalar("SELECT id FROM messages WHERE message_id=1")
            .fetch_one(db.pool())
            .await
            .unwrap();
        write_message_batch(db.pool(), &[fixture_message(1, "edited")])
            .await
            .unwrap();
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT id FROM messages WHERE message_id=1")
                .fetch_one(db.pool())
                .await
                .unwrap(),
            row_id
        );
        assert!(db
            .search_chat_messages(-100, "original", 10, 0)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            db.search_chat_messages(-100, "edited", 10, 0)
                .await
                .unwrap()
                .len(),
            1
        );
        db.shutdown().await;
    }

    #[tokio::test]
    async fn writer_retry_succeeds_after_observed_lock_failure() {
        use sqlx::sqlite::SqliteConnectOptions;
        use std::str::FromStr;
        let path = test_db_path("writer-retry");
        let url = sqlite_url_for_path(&path);
        let db = Database::init(&url).await.unwrap();
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::from_str(&url)
                    .unwrap()
                    .busy_timeout(Duration::ZERO),
            )
            .await
            .unwrap();
        let mut lock = db.pool().acquire().await.unwrap();
        sqlx::query("BEGIN IMMEDIATE")
            .execute(&mut *lock)
            .await
            .unwrap();
        let seen = std::sync::Arc::new(tokio::sync::Notify::new());
        let release = async {
            tokio::time::timeout(Duration::from_secs(10), seen.notified())
                .await
                .expect("writer must report the retry before releasing the lock");
            sqlx::query("ROLLBACK").execute(&mut *lock).await.unwrap();
        };
        let batch = [fixture_message(1, "retry recovered")];
        let dead_letter = path.with_extension("jsonl");
        let work = crate::utils::log_capture::capture_json_events_on(
            write_message_batch_with_recovery(&pool, &batch, &dead_letter),
            "retrying once",
            seen.clone(),
        );
        let ((result, _), ()) = tokio::join!(work, release);
        result.unwrap();
        assert!(!dead_letter.exists());
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM messages")
                .fetch_one(&pool)
                .await
                .unwrap(),
            1
        );
        drop(lock);
        pool.close().await;
        db.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_flushes_queued_inserts_before_returning() {
        let path = test_db_path("shutdown-flush");
        let url = sqlite_url_for_path(&path);
        let db = Database::init(&url)
            .await
            .expect("database should initialize");
        // Handlers still running at shutdown hold their own Database clones.
        let still_alive = db.clone();

        for message_id in 1..=3 {
            let insert = build_message_insert(
                Some(1),
                Some("alice".to_string()),
                Some(format!("message {message_id}")),
                Some("en".to_string()),
                Utc::now(),
                None,
                Some(-100),
                Some(message_id),
                None,
                false,
                None,
                false,
                false,
            );
            db.queue_message_insert(insert)
                .await
                .expect("queue should accept inserts before shutdown");
        }

        db.shutdown().await;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("raw pool should reopen the database");
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE chat_id = -100")
            .fetch_one(&pool)
            .await
            .expect("count query should succeed");
        assert_eq!(count, 3, "every queued insert must be flushed by shutdown");
        drop(still_alive);
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
}

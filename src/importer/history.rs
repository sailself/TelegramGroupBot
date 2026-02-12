use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use serde::de::{self, Deserializer as _, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::task;
use tracing::{info, warn};

use crate::config::CONFIG;
use crate::db::database::{build_message_insert, Database};
use crate::rag::client::{self, build_ingest_item, RagMessageItem};
use crate::utils::language::detect_language_or_fallback;

#[derive(Debug, Clone)]
pub struct ImportHistoryArgs {
    pub file_path: PathBuf,
    pub chat_id: i64,
    pub batch_size: usize,
    pub dry_run: bool,
    pub resume: bool,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Default)]
pub struct ImportSummary {
    pub total_records: usize,
    pub skipped_by_resume: usize,
    pub skipped_by_date_filter: usize,
    pub invalid_records: usize,
    pub db_upserts: usize,
    pub rag_candidates: usize,
    pub rag_accepted: usize,
    pub rag_skipped: usize,
    pub rag_failed: usize,
}

#[derive(Debug, Deserialize)]
struct HistoryRecord {
    id: i64,
    #[serde(default)]
    user_id: Option<HistoryUserId>,
    #[serde(default)]
    username: Option<String>,
    datetime: String,
    #[serde(default)]
    reply_to_message_id: Option<i64>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HistoryUserId {
    Number(i64),
    Text(String),
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct ImportCheckpoint {
    last_message_id: i64,
    updated_at: String,
}

struct HistoryRecordStreamVisitor {
    sender: mpsc::Sender<HistoryRecord>,
}

impl<'de> Visitor<'de> for HistoryRecordStreamVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON array of history records")
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while let Some(record) = seq.next_element::<HistoryRecord>()? {
            self.sender.blocking_send(record).map_err(|_| {
                de::Error::custom("import consumer stopped before parsing finished")
            })?;
        }
        Ok(())
    }
}

fn stream_history_records(path: &Path, sender: mpsc::Sender<HistoryRecord>) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open import file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    deserializer
        .deserialize_seq(HistoryRecordStreamVisitor { sender })
        .with_context(|| {
            format!(
                "Failed to parse import file {} as JSON array",
                path.display()
            )
        })?;
    Ok(())
}

fn parse_datetime_utc(value: &str) -> Result<DateTime<Utc>> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Ok(parsed.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return Ok(DateTime::from_naive_utc_and_offset(parsed, Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Ok(DateTime::from_naive_utc_and_offset(parsed, Utc));
    }

    Err(anyhow!("Unsupported datetime format: {value}"))
}

pub fn parse_filter_datetime(value: &str, end_of_day: bool) -> Result<DateTime<Utc>> {
    if let Ok(parsed) = parse_datetime_utc(value) {
        return Ok(parsed);
    }
    let day = NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .with_context(|| format!("Unsupported date format: {value}"))?;
    let time = if end_of_day {
        day.and_hms_opt(23, 59, 59)
            .ok_or_else(|| anyhow!("Invalid end-of-day timestamp for {value}"))?
    } else {
        day.and_hms_opt(0, 0, 0)
            .ok_or_else(|| anyhow!("Invalid start-of-day timestamp for {value}"))?
    };
    Ok(DateTime::from_naive_utc_and_offset(time, Utc))
}

fn parse_user_id(user_id: Option<HistoryUserId>) -> Option<i64> {
    match user_id {
        Some(HistoryUserId::Number(value)) => Some(value),
        Some(HistoryUserId::Text(value)) => value.trim().parse::<i64>().ok(),
        None => None,
    }
}

fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|text| {
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    })
}

fn normalize_optional_username(value: Option<String>) -> Option<String> {
    value.and_then(|username| {
        if username.trim().is_empty() {
            None
        } else {
            Some(username)
        }
    })
}

fn checkpoint_path(chat_id: i64) -> PathBuf {
    PathBuf::from(&CONFIG.rag_import_resume_dir).join(format!("import_state_{chat_id}.json"))
}

fn load_checkpoint(path: &Path) -> Result<Option<ImportCheckpoint>> {
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)?;
    let checkpoint = serde_json::from_str::<ImportCheckpoint>(&content)?;
    Ok(Some(checkpoint))
}

fn write_checkpoint(path: &Path, last_message_id: i64) -> Result<()> {
    let checkpoint = ImportCheckpoint {
        last_message_id,
        updated_at: Utc::now().to_rfc3339(),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let serialized = serde_json::to_string_pretty(&checkpoint)?;
    fs::write(path, serialized)?;
    Ok(())
}

async fn flush_rag_batch(
    chat_id: i64,
    rag_batch: &mut Vec<RagMessageItem>,
    summary: &mut ImportSummary,
) -> Result<()> {
    if rag_batch.is_empty() {
        return Ok(());
    }

    let payload = std::mem::take(rag_batch);
    let response = client::ingest_messages(chat_id, payload).await?;
    summary.rag_accepted += response.accepted;
    summary.rag_skipped += response.skipped;
    summary.rag_failed += response.failed;

    if !response.errors.is_empty() {
        warn!(
            "RAG ingest returned {} error(s): {}",
            response.errors.len(),
            response.errors.join(" | ")
        );
    }

    Ok(())
}

pub async fn run_import_history(args: ImportHistoryArgs) -> Result<ImportSummary> {
    let channel_capacity = args.batch_size.max(1).saturating_mul(4).max(64);
    let (record_sender, mut record_receiver) = mpsc::channel::<HistoryRecord>(channel_capacity);
    let stream_path = args.file_path.clone();
    let parser_handle =
        task::spawn_blocking(move || stream_history_records(&stream_path, record_sender));

    let db = Database::init(&CONFIG.database_url).await?;
    let checkpoint_file = checkpoint_path(args.chat_id);
    let checkpoint = if args.resume {
        load_checkpoint(&checkpoint_file)?
    } else {
        None
    };
    let mut last_processed_message_id = checkpoint
        .as_ref()
        .map(|cp| cp.last_message_id)
        .unwrap_or(0);

    info!(
        "Starting history import: file={}, chat_id={}, batch_size={}, dry_run={}, resume={}",
        args.file_path.display(),
        args.chat_id,
        args.batch_size,
        args.dry_run,
        args.resume
    );
    if let Some(checkpoint) = checkpoint.as_ref() {
        info!(
            "Resuming from checkpoint message_id={} ({})",
            checkpoint.last_message_id, checkpoint.updated_at
        );
    }

    let mut summary = ImportSummary::default();
    let mut rag_batch = Vec::with_capacity(args.batch_size.max(1));
    let rag_enabled = client::is_rag_enabled() && !args.dry_run;

    while let Some(record) = record_receiver.recv().await {
        summary.total_records += 1;

        if checkpoint
            .as_ref()
            .map(|cp| record.id <= cp.last_message_id)
            .unwrap_or(false)
        {
            summary.skipped_by_resume += 1;
            continue;
        }

        let date = match parse_datetime_utc(&record.datetime) {
            Ok(date) => date,
            Err(err) => {
                summary.invalid_records += 1;
                warn!(
                    "Skipping invalid record id={} datetime='{}': {}",
                    record.id, record.datetime, err
                );
                continue;
            }
        };

        if args.from_date.map(|from| date < from).unwrap_or(false)
            || args.to_date.map(|to| date > to).unwrap_or(false)
        {
            summary.skipped_by_date_filter += 1;
            continue;
        }

        let normalized_text = normalize_optional_text(record.text);
        let language = normalized_text
            .as_ref()
            .map(|text| detect_language_or_fallback(&[text.as_str()], None, "Chinese"));
        let user_id = parse_user_id(record.user_id);
        let username = normalize_optional_username(record.username);

        let insert = build_message_insert(
            user_id,
            username.clone(),
            normalized_text.clone(),
            language,
            date,
            record.reply_to_message_id,
            Some(args.chat_id),
            Some(record.id),
        );

        if !args.dry_run {
            db.upsert_message_direct(&insert).await?;
        }
        summary.db_upserts += 1;

        if rag_enabled {
            if let Some(text) = normalized_text.as_deref() {
                if let Some(item) = build_ingest_item(
                    record.id,
                    user_id,
                    username.clone(),
                    date,
                    record.reply_to_message_id,
                    text,
                ) {
                    summary.rag_candidates += 1;
                    rag_batch.push(item);
                    if rag_batch.len() >= args.batch_size.max(1) {
                        flush_rag_batch(args.chat_id, &mut rag_batch, &mut summary).await?;
                    }
                }
            }
        }

        last_processed_message_id = record.id;
        if args.resume && !args.dry_run && summary.db_upserts % 1000 == 0 {
            write_checkpoint(&checkpoint_file, last_processed_message_id)?;
        }
    }

    let parser_outcome = parser_handle
        .await
        .map_err(|err| anyhow!("History parser task failed: {err}"))?;
    parser_outcome?;

    if rag_enabled {
        flush_rag_batch(args.chat_id, &mut rag_batch, &mut summary).await?;
    }

    if args.resume && !args.dry_run && last_processed_message_id > 0 {
        write_checkpoint(&checkpoint_file, last_processed_message_id)?;
    }

    info!(
        "History import complete: total={} db_upserts={} invalid={} resume_skips={} date_skips={} rag_candidates={} rag_accepted={} rag_skipped={} rag_failed={}",
        summary.total_records,
        summary.db_upserts,
        summary.invalid_records,
        summary.skipped_by_resume,
        summary.skipped_by_date_filter,
        summary.rag_candidates,
        summary.rag_accepted,
        summary.rag_skipped,
        summary.rag_failed
    );

    Ok(summary)
}

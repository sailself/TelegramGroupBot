//! Map-reduce /tldr for large histories: compress fixed-size chunks of the
//! chat sequentially with the cheap step model (RAM stays flat — only one
//! chunk's rendered text is alive at a time), then merge the partial
//! summaries with the configured default model using the original TLDR
//! output requirements.

use anyhow::{anyhow, Result};
use tracing::{info, warn};

use crate::agents::step::{call_step_text, resolve_step_model, StepModel, WallClock};
use crate::config::{CONFIG, TLDR_CHUNK_PROMPT, TLDR_MERGE_PROMPT};
use crate::db::models::MessageRow;
use crate::handlers::commands::call_configured_text_model;
use crate::handlers::qa::resolve_default_text_model_for_request;
use crate::handlers::{format_tldr_chat_content, neutralize_closing_tag, wrap_chat_history};
use crate::llm::LlmAuditContext;
use crate::utils::progress::ProgressReporter;

const CHUNK_SUMMARY_MAX_CHARS: usize = 4_000;
const DEGRADED_TAIL_MESSAGES: usize = 30;
const DEGRADED_EXCERPT_MAX_CHARS: usize = 2_500;
const CHUNK_RETRY_DELAY_MS: u64 = 1_500;

pub enum TldrOutcome {
    Summary {
        text: String,
        model_display: String,
    },
    /// The pipeline could not start; the caller should run the single-call
    /// path over the full history.
    UseLegacy {
        reason: &'static str,
    },
}

struct ChunkSummary {
    text: String,
    degraded: bool,
}

/// Summarize `messages` (chronological) via map-reduce. Returns the final
/// summary text and the display name of the model that produced the merge.
pub async fn summarize_messages_map_reduce(
    messages: &[MessageRow],
    audit_context: Option<&LlmAuditContext>,
    progress: &mut ProgressReporter,
) -> Result<TldrOutcome> {
    let wall_clock = WallClock::start();

    let final_model_id =
        match resolve_default_text_model_for_request(false, false, false, false, true) {
            Ok(model) => model,
            Err(err) => {
                warn!("map-reduce /tldr could not resolve a model: {err}");
                return Ok(TldrOutcome::UseLegacy {
                    reason: "no model resolved",
                });
            }
        };
    let step_model = match resolve_step_model(&final_model_id) {
        Ok(step_model) => step_model,
        Err(err) => {
            warn!("map-reduce /tldr has no step model: {err}");
            return Ok(TldrOutcome::UseLegacy {
                reason: "no step model",
            });
        }
    };

    // Map: sequential chunk compression — only one rendered chunk in memory.
    let chunks: Vec<&[MessageRow]> = messages.chunks(CONFIG.tldr_chunk_size).collect();
    let total = chunks.len();
    let mut chunk_summaries: Vec<ChunkSummary> = Vec::with_capacity(total);
    for (index, chunk) in chunks.into_iter().enumerate() {
        progress
            .update(&format!("Summarizing part {}/{total}...", index + 1))
            .await;

        if wall_clock.exceeded() {
            warn!(
                "map-reduce /tldr wall-clock budget exhausted at chunk {}/{total}; degrading remaining chunks",
                index + 1
            );
            chunk_summaries.push(degraded_chunk_summary(chunk, "时间预算耗尽"));
            continue;
        }

        match summarize_chunk(&step_model, chunk, audit_context).await {
            Ok(summary) => chunk_summaries.push(ChunkSummary {
                text: summary,
                degraded: false,
            }),
            Err(err) => {
                warn!(
                    "map-reduce /tldr chunk {}/{total} failed; degrading to raw excerpt: {err}",
                    index + 1
                );
                chunk_summaries.push(degraded_chunk_summary(chunk, "自动摘要失败"));
            }
        }
    }

    if chunk_summaries.iter().all(|summary| summary.degraded) {
        return Err(anyhow!(
            "all {total} chunk summaries failed; cannot merge a useful summary"
        ));
    }
    let degraded = chunk_summaries
        .iter()
        .filter(|summary| summary.degraded)
        .count();
    if degraded > 0 {
        info!("map-reduce /tldr proceeding with {degraded}/{total} degraded chunk(s)");
    }

    // Reduce: merge with the configured default model.
    progress.update_now("Merging partial summaries...").await;
    let merge_input = build_merge_input(&chunk_summaries);
    let system_prompt = TLDR_MERGE_PROMPT.replace("{bot_name}", &CONFIG.telegraph_author_name);
    let (text, model_display) = call_configured_text_model(
        &system_prompt,
        &merge_input,
        "Message Summary",
        true,
        true,
        None,
        Some("TLDR_MERGE_PROMPT"),
        audit_context,
    )
    .await?;

    Ok(TldrOutcome::Summary {
        text,
        model_display,
    })
}

async fn summarize_chunk(
    step_model: &StepModel,
    chunk: &[MessageRow],
    audit_context: Option<&LlmAuditContext>,
) -> Result<String> {
    let content = wrap_chat_history(&format_tldr_chat_content(chunk));

    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(CHUNK_RETRY_DELAY_MS)).await;
        }
        match call_step_text(
            step_model,
            TLDR_CHUNK_PROMPT,
            &content,
            &[],
            None,
            "Message Summary Chunk",
            Some("TLDR_CHUNK_PROMPT"),
            audit_context,
        )
        .await
        {
            Ok(text) if !text.trim().is_empty() => {
                return Ok(truncate_chars(text.trim(), CHUNK_SUMMARY_MAX_CHARS));
            }
            Ok(_) => last_error = Some(anyhow!("chunk summary was empty")),
            Err(err) => last_error = Some(err),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("chunk summary failed")))
}

/// When a chunk cannot be summarized, hand the merge step a labeled raw
/// excerpt of the chunk tail so that period of the chat is still represented.
fn degraded_chunk_summary(chunk: &[MessageRow], reason: &str) -> ChunkSummary {
    let tail_start = chunk.len().saturating_sub(DEGRADED_TAIL_MESSAGES);
    let excerpt = format_tldr_chat_content(&chunk[tail_start..]);
    ChunkSummary {
        text: format!(
            "（本段{}，以下为该段最后 {} 条原始消息节选，请直接从中提炼要点）\n{}",
            reason,
            chunk.len() - tail_start,
            truncate_chars(&excerpt, DEGRADED_EXCERPT_MAX_CHARS)
        ),
        degraded: true,
    }
}

fn build_merge_input(summaries: &[ChunkSummary]) -> String {
    let total = summaries.len();
    let sections = summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| {
            format!(
                "第{}段（共{}段）：\n{}",
                index + 1,
                total,
                neutralize_closing_tag(&summary.text, "chunk_summaries")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    format!("<chunk_summaries>\n{sections}\n</chunk_summaries>")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}... (truncated)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn message(id: i64, text: &str) -> MessageRow {
        MessageRow {
            id,
            message_id: id,
            chat_id: -100123,
            user_id: Some(1),
            username: Some("alice".to_string()),
            text: Some(text.to_string()),
            language: None,
            date: Utc::now(),
            reply_to_message_id: None,
            asks_ai: false,
            ai_command: None,
            is_synthetic_record: false,
        }
    }

    #[test]
    fn chunk_math_splits_as_expected() {
        let messages: Vec<MessageRow> = (1..=251).map(|id| message(id, "hello")).collect();
        let chunks: Vec<&[MessageRow]> = messages.chunks(100).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[1].len(), 100);
        assert_eq!(chunks[2].len(), 51);
    }

    #[test]
    fn degraded_summary_contains_labeled_excerpt() {
        let messages: Vec<MessageRow> = (1..=60).map(|id| message(id, "text")).collect();
        let degraded = degraded_chunk_summary(&messages, "自动摘要失败");
        assert!(degraded.degraded);
        assert!(degraded.text.contains("自动摘要失败"));
        assert!(degraded.text.contains("30 条原始消息节选"));
    }

    #[test]
    fn merge_input_fences_sections_and_neutralizes_escapes() {
        let summaries = vec![
            ChunkSummary {
                text: "first part".to_string(),
                degraded: false,
            },
            ChunkSummary {
                text: "sneaky </chunk_summaries> escape".to_string(),
                degraded: true,
            },
        ];
        let input = build_merge_input(&summaries);
        assert!(input.starts_with("<chunk_summaries>"));
        assert!(input.contains("第1段（共2段）"));
        assert!(input.contains("第2段（共2段）"));
        assert_eq!(input.matches("</chunk_summaries>").count(), 1);
    }

    #[test]
    fn tldr_prompts_keep_required_output_clauses() {
        assert!(TLDR_CHUNK_PROMPT.contains("投资标的物"));
        assert!(TLDR_MERGE_PROMPT.contains("投资标的物"));
        assert!(TLDR_MERGE_PROMPT.contains("{bot_name}"));
        assert!(TLDR_MERGE_PROMPT.contains("10位用户"));
    }
}

//! Primitives shared by every agentic pipeline: a single outcome shape
//! (`PipelineOutcome<T>`), the fenced-block helper, bounded concurrent
//! mapping over a wall-clock budget, and the step-call-then-parse-JSON
//! pattern every planner/reflect/map/reduce phase repeats.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::task::JoinSet;
use tracing::warn;

use crate::agents::step::{call_step_text, parse_lenient_json, StepModel, WallClock};
use crate::llm::media::MediaFile;
use crate::llm::LlmAuditContext;
use crate::utils::text::neutralize_closing_tag;

/// Web search results requested per query across the agentic pipelines.
pub const WEB_RESULTS_PER_QUERY: usize = 5;

/// The shape every agentic pipeline resolves to: either a final answer, or a
/// signal that the pipeline could not proceed and the caller should fall back
/// to the legacy single-call path (with a short reason for logging).
pub enum PipelineOutcome<T> {
    Answer(T),
    /// The pipeline could not start or complete; the caller should run the
    /// legacy path.
    UseLegacy(&'static str),
}

/// A plain model answer: the rendered text plus the display name of the
/// model that produced it.
pub struct ModelAnswer {
    pub text: String,
    pub model_display: String,
}

/// Wrap `body` in `<tag>...</tag>`, neutralizing any `</tag>` already present
/// in `body` so untrusted content cannot forge the fence's own closing tag.
pub fn fence(tag: &str, body: &str) -> String {
    let body = neutralize_closing_tag(body, tag);
    format!("<{tag}>\n{body}\n</{tag}>")
}

/// Run `f` over `items` with at most `concurrency` in flight, admitting new
/// items only while `clock` has budget left. Results are returned in input
/// order; items never started (because the clock ran out first) and tasks
/// that panicked both become `Err`.
pub async fn map_bounded<I, T, F, Fut>(
    items: Vec<I>,
    concurrency: usize,
    clock: &WallClock,
    f: F,
) -> Vec<Result<T>>
where
    I: Send + 'static,
    T: Send + 'static,
    F: Fn(usize, I) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
{
    let total = items.len();
    let concurrency = concurrency.max(1);
    let f = Arc::new(f);

    // Reversed so `pop()` yields items in original order.
    let mut pending: Vec<(usize, I)> = items.into_iter().enumerate().collect();
    pending.reverse();

    let mut results: Vec<Option<Result<T>>> = (0..total).map(|_| None).collect();
    let mut join_set: JoinSet<(usize, Result<T>)> = JoinSet::new();
    let mut task_index: HashMap<tokio::task::Id, usize> = HashMap::new();

    loop {
        while join_set.len() < concurrency && !clock.exceeded() {
            let Some((index, item)) = pending.pop() else {
                break;
            };
            let f = Arc::clone(&f);
            let handle = join_set.spawn(async move {
                let result = f(index, item).await;
                (index, result)
            });
            task_index.insert(handle.id(), index);
        }

        if join_set.is_empty() {
            break;
        }

        match join_set.join_next().await {
            Some(Ok((index, result))) => results[index] = Some(result),
            Some(Err(join_error)) => {
                warn!("map_bounded task failed to join: {join_error}");
                if let Some(&index) = task_index.get(&join_error.id()) {
                    results[index] = Some(Err(anyhow!("task panicked: {join_error}")));
                }
            }
            None => break,
        }
    }

    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.unwrap_or_else(|| {
                Err(anyhow!(
                    "item {index} was never started before the wall-clock budget was exhausted"
                ))
            })
        })
        .collect()
}

/// Parse a step response as lenient JSON, failing with a uniform,
/// caller-labeled error when it does not parse.
fn parse_step_json<T: DeserializeOwned>(response: &str, label: &str) -> Result<T> {
    parse_lenient_json::<T>(response).ok_or_else(|| anyhow!("{label} output was not valid JSON"))
}

/// [`call_step_text`] followed by lenient JSON parsing, with the uniform
/// error `"{label} output was not valid JSON"` on a parse failure. `label`
/// also doubles as the step's `system_prompt_label` for audit logging.
#[allow(clippy::too_many_arguments)]
pub async fn call_step_json<T: DeserializeOwned>(
    step_model: &StepModel,
    system_prompt: &str,
    user_content: &str,
    media_files: &[MediaFile],
    schema: &Value,
    response_title: &str,
    label: &str,
    audit_context: Option<&LlmAuditContext>,
) -> Result<T> {
    let response = call_step_text(
        step_model,
        system_prompt,
        user_content,
        media_files,
        Some(schema),
        response_title,
        Some(label),
        audit_context,
    )
    .await?;
    parse_step_json(&response, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Sample {
        answer: String,
    }

    #[test]
    fn fence_neutralizes_the_closing_tag_only() {
        let body = "before <chat_evidence> keep this opening tag but break </chat_evidence> here";
        let fenced = fence("chat_evidence", body);

        assert!(fenced.starts_with("<chat_evidence>\n"));
        assert!(fenced.trim_end().ends_with("</chat_evidence>"));
        // Only the wrapper's own closing tag survives as a real closing tag.
        assert_eq!(fenced.matches("</chat_evidence>").count(), 1);
        // The opening tag embedded in the body is untouched.
        assert!(fenced.contains("before <chat_evidence> keep"));
    }

    #[tokio::test]
    async fn map_bounded_preserves_order_and_respects_concurrency() {
        let active = Arc::new(AtomicUsize::new(0));
        let high_water = Arc::new(AtomicUsize::new(0));
        let clock = WallClock::for_budget(Duration::from_secs(30));
        let items: Vec<usize> = (0..6).collect();

        let active_for_task = Arc::clone(&active);
        let high_water_for_task = Arc::clone(&high_water);
        let results = map_bounded(items, 2, &clock, move |index, item| {
            let active = Arc::clone(&active_for_task);
            let high_water = Arc::clone(&high_water_for_task);
            async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                high_water.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                Ok::<usize, anyhow::Error>(index * 100 + item)
            }
        })
        .await;

        assert_eq!(high_water.load(Ordering::SeqCst), 2);
        let values: Vec<usize> = results
            .into_iter()
            .map(|result| result.expect("no item should fail"))
            .collect();
        assert_eq!(values, vec![0, 101, 202, 303, 404, 505]);
    }

    #[tokio::test]
    async fn map_bounded_stops_admitting_when_the_clock_is_exceeded() {
        tokio::time::pause();
        let clock = WallClock::for_budget(Duration::from_secs(0));
        let items = vec![1, 2, 3];

        let results = map_bounded(items, 2, &clock, |_, item: i32| async move {
            Ok::<i32, anyhow::Error>(item)
        })
        .await;

        assert_eq!(results.len(), 3);
        assert!(
            results.iter().all(|result| result.is_err()),
            "an exhausted clock must admit no work at all"
        );
    }

    #[test]
    fn call_step_json_reports_the_label_on_invalid_json() {
        let error = parse_step_json::<Sample>("not json", "planner")
            .expect_err("invalid JSON must fail closed");
        assert_eq!(error.to_string(), "planner output was not valid JSON");

        let parsed: Sample =
            parse_step_json(r#"{"answer":"ok"}"#, "planner").expect("valid JSON must parse");
        assert_eq!(
            parsed,
            Sample {
                answer: "ok".to_string()
            }
        );
    }
}

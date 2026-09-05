//! One tool-calling loop shared by every provider adapter.
//!
//! The loop owns the policy: which tools are offered, how tool calls are
//! charged against the [`ToolRuntime`] budget, when tools are withdrawn, how
//! many model turns may run, the final tool-less pass, and the wall-clock
//! deadline for the whole turn. A [`ToolProtocol`] supplies only the wire
//! format: how to make one model request and how tool results are written
//! back into the transcript.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::Value;
use tokio::time::Instant;
use tracing::{debug, warn};

use crate::llm::tool_runtime::ToolRuntime;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Model turns allowed beyond the successful-call budget: one for the call
/// that gets refused once the budget is spent, one for the answer.
const EXTRA_MODEL_TURNS: usize = 2;

/// A tool call requested by the model in one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Provider-assigned id echoed back with the result (may be empty).
    pub id: String,
    pub name: String,
    /// Parsed arguments; malformed argument text becomes an empty object.
    pub arguments: Value,
}

impl ToolCall {
    /// Build a call from the argument text providers send, tolerating
    /// malformed JSON the way every adapter used to.
    pub fn from_argument_text(id: impl Into<String>, name: impl Into<String>, text: &str) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::from_str(text)
                .unwrap_or_else(|_| Value::Object(Default::default())),
        }
    }
}

/// What one model request produced.
pub struct ModelTurn<Item> {
    /// Assistant text, already post-processed by the protocol.
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    /// Transcript items for this assistant turn, appended before the tool
    /// results when the loop continues.
    pub transcript: Vec<Item>,
}

/// Wall-clock budget for a whole tool-calling turn, plus the cap for any
/// single request. Later requests are clamped to whatever time remains.
#[derive(Debug, Clone, Copy)]
pub struct TurnDeadline {
    started: Instant,
    total: Duration,
    per_request: Duration,
}

impl TurnDeadline {
    pub fn new(total: Duration, per_request: Duration) -> Self {
        Self {
            started: Instant::now(),
            total,
            per_request,
        }
    }

    /// A deadline spanning `requests` full request timeouts.
    pub fn for_requests(per_request: Duration, requests: u32) -> Self {
        Self::new(per_request.saturating_mul(requests), per_request)
    }

    /// The deadline for one tool-calling turn of `runtime`: every model turn
    /// the loop may run, each at the provider's full request timeout. Tool
    /// execution time eats into the same budget.
    pub fn for_runtime(per_request: Duration, runtime: &ToolRuntime) -> Self {
        let turns = runtime
            .max_total_successful_calls()
            .saturating_add(EXTRA_MODEL_TURNS);
        Self::for_requests(per_request, u32::try_from(turns).unwrap_or(u32::MAX))
    }

    pub fn total(&self) -> Duration {
        self.total
    }

    pub fn remaining(&self) -> Duration {
        self.total.saturating_sub(self.started.elapsed())
    }

    pub fn is_expired(&self) -> bool {
        self.remaining().is_zero()
    }

    /// Timeout for the next request: the per-request cap, or less when the
    /// turn is running out of time.
    pub fn request_timeout(&self) -> Duration {
        self.per_request.min(self.remaining())
    }
}

/// Provider-specific half of the loop.
pub trait ToolProtocol {
    /// Transcript item type: a chat message, a Responses input item, a
    /// Gemini content.
    type Item: Clone;

    /// Tool declarations in this provider's wire format for the tools the
    /// runtime offers. Empty means the turn runs without tools.
    fn tool_declarations(&self, runtime: &ToolRuntime) -> Vec<Value>;

    /// One model request over `transcript`. `tools` is `None` when tools are
    /// withdrawn for this request.
    fn complete<'a>(
        &'a mut self,
        transcript: &'a [Self::Item],
        tools: Option<&'a [Value]>,
        request_timeout: Duration,
    ) -> BoxFuture<'a, Result<ModelTurn<Self::Item>>>;

    /// Transcript items carrying the (fenced) tool results back to the model.
    fn tool_results(&self, results: Vec<(ToolCall, String)>) -> Vec<Self::Item>;

    /// Item appended once when the budget is spent and tools are withdrawn,
    /// for protocols that tell the model in-band. `None` relies on the
    /// system prompt alone.
    fn budget_exhausted_notice(&self) -> Option<Self::Item> {
        None
    }

    /// Whether a text-only reply must still be followed by the final pass
    /// (for example to apply a JSON response schema).
    fn requires_final_pass(&self) -> bool {
        false
    }

    /// Called once before the final tool-less pass so the protocol can adjust
    /// its instructions or generation config.
    fn begin_final_pass(&mut self) {}
}

/// Drive `protocol` until the model answers without tool calls, the model
/// turn budget is spent, or the deadline passes.
pub async fn run_tool_loop<P: ToolProtocol>(
    protocol: &mut P,
    runtime: &mut ToolRuntime,
    mut transcript: Vec<P::Item>,
    deadline: &TurnDeadline,
) -> Result<String> {
    let declarations = protocol.tool_declarations(runtime);
    let mut tools_enabled = !declarations.is_empty();
    let max_turns = runtime
        .max_total_successful_calls()
        .saturating_add(EXTRA_MODEL_TURNS);
    let mut final_pass_reason = "model turn budget exhausted";

    for turn in 0..max_turns {
        ensure_time_remains(deadline)?;
        debug!(
            "tool loop turn {}/{} (tools_enabled={}, remaining={:?})",
            turn + 1,
            max_turns,
            tools_enabled,
            deadline.remaining()
        );
        let model_turn = protocol
            .complete(
                &transcript,
                tools_enabled.then_some(declarations.as_slice()),
                deadline.request_timeout(),
            )
            .await?;
        let tool_calls = if tools_enabled {
            model_turn.tool_calls
        } else {
            Vec::new()
        };

        if tool_calls.is_empty() {
            if !protocol.requires_final_pass() {
                return Ok(model_turn.text);
            }
            final_pass_reason = "the protocol requires a final pass";
            break;
        }

        transcript.extend(model_turn.transcript);
        let mut results = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            let output = runtime.execute_tool(&call.name, &call.arguments).await;
            results.push((call, output));
        }
        transcript.extend(protocol.tool_results(results));

        // Any refusal forces the final answer: withdraw the tools so the next
        // reply is the answer, telling the model in-band where the protocol
        // supports it.
        if runtime.force_final_answer() && tools_enabled {
            tools_enabled = false;
            if let Some(notice) = protocol.budget_exhausted_notice() {
                transcript.push(notice);
            }
        }
    }

    ensure_time_remains(deadline)?;
    protocol.begin_final_pass();
    debug!(
        "tool loop final pass without tools: {} (remaining={:?})",
        final_pass_reason,
        deadline.remaining()
    );
    let final_turn = protocol
        .complete(&transcript, None, deadline.request_timeout())
        .await?;
    if !final_turn.tool_calls.is_empty() {
        warn!(
            "model returned {} tool call(s) in the final tool-less pass; using its text",
            final_turn.tool_calls.len()
        );
    }
    Ok(final_turn.text)
}

/// Clamp a provider's configured request timeout (seconds) to the time the
/// loop allows for this request, never going below one second.
pub fn clamp_request_timeout_secs(configured_secs: u64, request_timeout: Duration) -> u64 {
    configured_secs.min(request_timeout.as_secs()).max(1)
}

fn ensure_time_remains(deadline: &TurnDeadline) -> Result<()> {
    if deadline.is_expired() {
        return Err(anyhow!(
            "tool loop exceeded its {:?} deadline",
            deadline.total()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::tool_runtime::test_support::init_test_db;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// Scripted protocol: returns the next queued turn on every `complete`
    /// and records what it was asked so tests can inspect the loop.
    struct FakeProtocol {
        turns: VecDeque<ModelTurn<Value>>,
        requests: Arc<Mutex<Vec<Request>>>,
        needs_final_pass: bool,
        final_pass_started: bool,
    }

    #[derive(Debug, Clone)]
    struct Request {
        transcript: Vec<Value>,
        tool_names: Option<Vec<String>>,
        timeout: Duration,
    }

    impl FakeProtocol {
        fn scripted(turns: Vec<ModelTurn<Value>>) -> Self {
            Self {
                turns: turns.into(),
                requests: Arc::new(Mutex::new(Vec::new())),
                needs_final_pass: false,
                final_pass_started: false,
            }
        }

        fn requests(&self) -> Vec<Request> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    fn text_turn(text: &str) -> ModelTurn<Value> {
        ModelTurn {
            text: text.to_string(),
            tool_calls: Vec::new(),
            transcript: vec![json!({"role": "assistant", "content": text})],
        }
    }

    fn tool_turn(calls: &[(&str, &str, Value)]) -> ModelTurn<Value> {
        ModelTurn {
            text: String::new(),
            tool_calls: calls
                .iter()
                .map(|(id, name, arguments)| ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: arguments.clone(),
                })
                .collect(),
            transcript: vec![json!({"role": "assistant", "tool_calls": calls.len()})],
        }
    }

    impl ToolProtocol for FakeProtocol {
        type Item = Value;

        fn tool_declarations(&self, runtime: &ToolRuntime) -> Vec<Value> {
            runtime.build_openai_function_tools()
        }

        fn complete<'a>(
            &'a mut self,
            transcript: &'a [Value],
            tools: Option<&'a [Value]>,
            request_timeout: Duration,
        ) -> BoxFuture<'a, Result<ModelTurn<Value>>> {
            Box::pin(async move {
                self.requests.lock().expect("requests lock").push(Request {
                    transcript: transcript.to_vec(),
                    tool_names: tools.map(|tools| {
                        tools
                            .iter()
                            .filter_map(|tool| tool.pointer("/function/name"))
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    }),
                    timeout: request_timeout,
                });
                self.turns
                    .pop_front()
                    .ok_or_else(|| anyhow!("the script ran out of turns"))
            })
        }

        fn tool_results(&self, results: Vec<(ToolCall, String)>) -> Vec<Value> {
            results
                .into_iter()
                .map(|(call, output)| json!({"role": "tool", "tool_call_id": call.id, "content": output}))
                .collect()
        }

        fn budget_exhausted_notice(&self) -> Option<Value> {
            Some(json!({"role": "system", "content": "budget exhausted"}))
        }

        fn requires_final_pass(&self) -> bool {
            self.needs_final_pass
        }

        fn begin_final_pass(&mut self) {
            self.final_pass_started = true;
        }
    }

    fn generous_deadline() -> TurnDeadline {
        TurnDeadline::new(Duration::from_secs(600), Duration::from_secs(60))
    }

    fn user_turn() -> Vec<Value> {
        vec![json!({"role": "user", "content": "question"})]
    }

    #[tokio::test]
    async fn returns_the_text_when_the_model_makes_no_tool_calls() {
        let db = init_test_db("loop-plain-text").await;
        let mut runtime = ToolRuntime::for_quick(db, -100).with_web_search_available(true);
        let mut protocol = FakeProtocol::scripted(vec![text_turn("done")]);

        let answer = run_tool_loop(
            &mut protocol,
            &mut runtime,
            user_turn(),
            &generous_deadline(),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(answer, "done");
        let requests = protocol.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].tool_names.as_deref(),
            Some(&["web_search".to_string()][..]),
            "the first turn offers the runtime's tools"
        );
        assert_eq!(requests[0].timeout, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn executes_requested_tools_and_feeds_fenced_results_back() {
        let db = init_test_db("loop-tool-results").await;
        let mut runtime = ToolRuntime::for_quick(db, -100).with_web_search_available(true);
        // An empty query fails validation inside the runtime, so no network is touched.
        let mut protocol = FakeProtocol::scripted(vec![
            tool_turn(&[("call_1", "web_search", json!({"query": ""}))]),
            text_turn("answer"),
        ]);

        let answer = run_tool_loop(
            &mut protocol,
            &mut runtime,
            user_turn(),
            &generous_deadline(),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(answer, "answer");
        let requests = protocol.requests();
        assert_eq!(requests.len(), 2);
        let second = &requests[1].transcript;
        assert_eq!(
            second.len(),
            3,
            "user turn, assistant turn, tool result: {second:?}"
        );
        assert_eq!(second[2]["tool_call_id"], "call_1");
        let content = second[2]["content"].as_str().expect("tool content is text");
        assert!(
            content.starts_with(r#"<tool_result tool="web_search">"#),
            "{content}"
        );
        assert!(content.contains("invalid_arguments"), "{content}");
        assert!(runtime.web_search_attempted());
    }

    #[tokio::test]
    async fn stops_offering_tools_once_the_budget_is_exhausted() {
        let db = init_test_db("loop-budget-exhausted").await;
        let mut runtime = ToolRuntime::for_quick(db, -100).with_web_search_available(true);
        let mut protocol = FakeProtocol::scripted(vec![
            tool_turn(&[
                ("call_1", "web_search", json!({"query": ""})),
                ("call_2", "web_search", json!({"query": ""})),
            ]),
            text_turn("answer"),
        ]);

        let answer = run_tool_loop(
            &mut protocol,
            &mut runtime,
            user_turn(),
            &generous_deadline(),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(answer, "answer");
        let requests = protocol.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1].tool_names, None,
            "tools are withdrawn after the budget is spent"
        );
        let transcript = &requests[1].transcript;
        assert_eq!(
            transcript.last().map(|item| item["content"].as_str()),
            Some(Some("budget exhausted")),
            "the protocol's notice is appended once: {transcript:?}"
        );
        assert!(runtime.force_final_answer());
    }

    #[tokio::test]
    async fn withdraws_tools_after_a_refusal_and_takes_the_next_reply_as_the_answer() {
        let db = init_test_db("loop-withdraw").await;
        let mut runtime = ToolRuntime::for_quick(db, -100).with_web_search_available(true);
        // Budget 1: the second call is refused, so the third turn runs
        // without tools and its reply is the answer, whatever it contains.
        let mut protocol = FakeProtocol::scripted(vec![
            tool_turn(&[("call_1", "web_search", json!({"query": ""}))]),
            tool_turn(&[("call_2", "web_search", json!({"query": ""}))]),
            ModelTurn {
                text: "answer".to_string(),
                tool_calls: vec![ToolCall::from_argument_text("call_3", "web_search", "{}")],
                transcript: Vec::new(),
            },
            text_turn("never reached"),
        ]);

        let answer = run_tool_loop(
            &mut protocol,
            &mut runtime,
            user_turn(),
            &generous_deadline(),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(answer, "answer");
        assert!(
            !protocol.final_pass_started,
            "no separate final pass is needed"
        );
        let requests = protocol.requests();
        assert_eq!(requests.len(), 3, "budget + 2 model turns at most");
        assert!(requests[0].tool_names.is_some());
        assert!(requests[1].tool_names.is_some());
        assert_eq!(requests[2].tool_names, None);
        assert_eq!(
            requests[2]
                .transcript
                .last()
                .map(|item| item["content"].as_str()),
            Some(Some("budget exhausted"))
        );
        assert_eq!(protocol.turns.len(), 1);
    }

    #[tokio::test]
    async fn a_protocol_that_requires_a_final_pass_gets_one_even_for_a_text_reply() {
        let db = init_test_db("loop-schema-final-pass").await;
        let mut runtime = ToolRuntime::for_search(db, -100);
        let mut protocol =
            FakeProtocol::scripted(vec![text_turn("prose"), text_turn("{\"json\":true}")]);
        protocol.needs_final_pass = true;

        let answer = run_tool_loop(
            &mut protocol,
            &mut runtime,
            user_turn(),
            &generous_deadline(),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(answer, "{\"json\":true}");
        assert!(protocol.final_pass_started);
        let requests = protocol.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1].transcript,
            user_turn(),
            "the prose turn is not appended before the final pass"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_turn_deadline_clamps_request_timeouts_to_the_remaining_time() {
        let deadline = TurnDeadline::new(Duration::from_secs(100), Duration::from_secs(60));
        assert_eq!(deadline.request_timeout(), Duration::from_secs(60));

        tokio::time::advance(Duration::from_secs(70)).await;
        assert_eq!(deadline.request_timeout(), Duration::from_secs(30));
        assert!(!deadline.is_expired());

        tokio::time::advance(Duration::from_secs(40)).await;
        assert!(deadline.is_expired());
        assert_eq!(deadline.request_timeout(), Duration::ZERO);
    }

    #[tokio::test]
    async fn an_expired_deadline_fails_the_loop_before_any_request() {
        let db = init_test_db("loop-deadline").await;
        let mut runtime = ToolRuntime::for_quick(db, -100);
        let mut protocol = FakeProtocol::scripted(vec![text_turn("too late")]);
        let deadline = TurnDeadline::new(Duration::ZERO, Duration::from_secs(60));

        let err = run_tool_loop(&mut protocol, &mut runtime, user_turn(), &deadline)
            .await
            .expect_err("an expired deadline fails before any request");

        assert!(err.to_string().contains("deadline"), "{err}");
        assert!(protocol.requests().is_empty());
    }

    #[tokio::test]
    async fn a_turn_deadline_spans_every_model_turn_the_loop_may_run() {
        let deadline = TurnDeadline::for_requests(Duration::from_secs(30), 4);
        assert_eq!(deadline.total(), Duration::from_secs(120));
        assert_eq!(deadline.request_timeout(), Duration::from_secs(30));

        let db = init_test_db("loop-deadline-runtime").await;
        let quick = ToolRuntime::for_quick(db, -100);
        let deadline = TurnDeadline::for_runtime(Duration::from_secs(30), &quick);
        assert_eq!(
            deadline.total(),
            Duration::from_secs(90),
            "budget 1 allows 3 model turns"
        );
    }

    #[test]
    fn request_timeouts_are_clamped_to_the_remaining_time_but_never_zero() {
        assert_eq!(clamp_request_timeout_secs(60, Duration::from_secs(30)), 30);
        assert_eq!(clamp_request_timeout_secs(20, Duration::from_secs(30)), 20);
        assert_eq!(clamp_request_timeout_secs(20, Duration::ZERO), 1);
    }
}

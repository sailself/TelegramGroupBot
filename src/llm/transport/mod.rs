//! Shared HTTP transport for every LLM provider: one retry loop with
//! `Retry-After`/rate-limit awareness and jitter, one error type, bounded body
//! reading, SSE decoding, usage extraction, and audit/timing bookkeeping for
//! successes and failures alike.

pub mod body;
pub mod error;
pub mod retry;
pub mod sse;
pub mod usage;

use std::future::Future;

use chrono::{DateTime, Utc};
use reqwest::StatusCode;
use serde_json::Value;
use tracing::{debug, error, warn};

pub use body::{read_body_limited, read_json};
pub use error::ProviderError;
pub use retry::RetryPolicy;

use self::body::{summarize_error_body, ERROR_BODY_LIMIT};
use self::retry::retry_after_from_headers;
use crate::llm::audit::{
    log_llm_request_started, record_llm_request_failure, record_llm_request_success,
    LlmAuditContext, LlmUsageRecord,
};

/// Per-attempt context handed to the request builder.
#[derive(Debug, Clone, Copy)]
pub struct Attempt {
    /// 1-based attempt number.
    pub number: usize,
    pub max_attempts: usize,
    /// The previous attempt was rejected with 401 and the policy allows a
    /// credential refresh; the builder should refresh before building.
    pub previous_unauthorized: bool,
}

fn no_redaction(text: &str) -> String {
    text.to_string()
}

/// One logical provider call: emits the `llm_request` timing event when it
/// begins and records the audit row (success with usage, or failure with an
/// error summary) when it ends.
pub struct LlmCall<'a> {
    /// Audit/stats key (`gemini`, `openai-codex`, ...).
    provider: &'a str,
    /// Human-readable name for log lines and error text; defaults to `provider`.
    label: &'a str,
    model: &'a str,
    operation: &'a str,
    audit: Option<&'a LlmAuditContext>,
    started_at: DateTime<Utc>,
    redact: fn(&str) -> String,
    /// `false` for plain fetches (media downloads) that share the retry loop
    /// but are neither LLM requests nor audited.
    tracked: bool,
}

impl<'a> LlmCall<'a> {
    pub fn begin(
        provider: &'a str,
        model: &'a str,
        operation: &'a str,
        audit: Option<&'a LlmAuditContext>,
        metadata: Option<&Value>,
    ) -> Self {
        let started_at = Utc::now();
        log_llm_request_started(provider, model, operation, started_at, metadata);
        Self {
            provider,
            label: provider,
            model,
            operation,
            audit,
            started_at,
            redact: no_redaction,
            tracked: true,
        }
    }

    /// A fetch that uses the shared retry loop and logging but emits no
    /// timing events and writes no audit rows. `label` stands in for the model
    /// name in log lines (a redacted URL, for instance).
    pub fn untracked(provider: &'a str, label: &'a str) -> Self {
        Self {
            provider,
            label: provider,
            model: label,
            operation: "",
            audit: None,
            started_at: Utc::now(),
            redact: no_redaction,
            tracked: false,
        }
    }

    /// Use a different name than the audit key in logs and error messages
    /// (e.g. `OpenAI Codex` while audit rows stay keyed `openai-codex`).
    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = label;
        self
    }

    /// Scrub secrets (API keys or tokens embedded in URLs) from error text
    /// before it is logged or stored.
    pub fn with_redaction(mut self, redact: fn(&str) -> String) -> Self {
        self.redact = redact;
        self
    }

    pub fn label(&self) -> &'a str {
        self.label
    }

    pub fn model(&self) -> &'a str {
        self.model
    }

    pub fn redact(&self, text: &str) -> String {
        (self.redact)(text)
    }

    pub async fn succeed(&self, usage: LlmUsageRecord) {
        if !self.tracked {
            return;
        }
        record_llm_request_success(
            self.audit,
            self.provider,
            self.model,
            self.operation,
            self.started_at,
            Utc::now(),
            usage,
        )
        .await;
    }

    pub async fn fail(&self, error: &ProviderError) {
        if !self.tracked {
            return;
        }
        record_llm_request_failure(
            self.audit,
            self.provider,
            self.model,
            self.operation,
            self.started_at,
            Utc::now(),
            &error.to_string(),
        )
        .await;
    }
}

/// Run one provider call with retries.
///
/// * `build_request` produces the request for each attempt (so per-attempt
///   credentials and multipart bodies can be rebuilt); its errors are permanent.
/// * `observe_response` sees every response's headers before the body is
///   touched (turn-state and rate-limit header capture).
/// * `read_response` consumes a 2xx response; a retryable [`ProviderError`]
///   from it (stream interruption, incomplete SSE) triggers another attempt.
/// * `usage_of` extracts token usage from the successful result for the audit row.
///
/// Non-2xx statuses are classified by [`RetryPolicy`]: transient ones wait
/// for `Retry-After`/rate-limit hints (or the backoff) and retry, permanent
/// ones return at once. A 401 retries once through the builder when the
/// policy has `refresh_auth_on_unauthorized`.
pub async fn call_with_retry<T, B, BFut, R, RFut, U>(
    call: &LlmCall<'_>,
    policy: &RetryPolicy,
    mut build_request: B,
    mut observe_response: impl FnMut(&reqwest::Response),
    mut read_response: R,
    usage_of: U,
) -> Result<T, ProviderError>
where
    B: FnMut(Attempt) -> BFut,
    BFut: Future<Output = Result<reqwest::RequestBuilder, ProviderError>>,
    R: FnMut(reqwest::Response) -> RFut,
    RFut: Future<Output = Result<T, ProviderError>>,
    U: Fn(&T) -> LlmUsageRecord,
{
    let provider = call.label();
    let model = call.model();
    let mut previous_unauthorized = false;
    let mut attempt = 0usize;

    loop {
        attempt += 1;
        let context = Attempt {
            number: attempt,
            max_attempts: policy.max_attempts,
            previous_unauthorized,
        };
        previous_unauthorized = false;
        if context.previous_unauthorized {
            debug!(
                "{provider} rebuilding request after a credential refresh: model={model}, attempt={attempt}/{}",
                policy.max_attempts
            );
        }

        let request = match build_request(context).await {
            Ok(request) => request,
            Err(err) => {
                call.fail(&err).await;
                return Err(err);
            }
        };

        let response = match request.send().await {
            Ok(response) => response,
            Err(err) => {
                let error = ProviderError::transport(provider, &err, call.redact(&err.to_string()));
                let retrying = error.is_retryable() && policy.allows_retry(attempt);
                let message = format!(
                    "{provider} request failed to send: model={model}, attempt={attempt}/{}, timeout={}, connect={}, retrying={retrying}, error={}",
                    policy.max_attempts,
                    err.is_timeout(),
                    err.is_connect(),
                    call.redact(&err.to_string())
                );
                if retrying {
                    warn!("{message}");
                    tokio::time::sleep(policy.delay(attempt, None)).await;
                    continue;
                }
                error!("{message}");
                call.fail(&error).await;
                return Err(error);
            }
        };

        observe_response(&response);
        let status = response.status();

        if !status.is_success() {
            if status == StatusCode::UNAUTHORIZED
                && policy.refresh_auth_on_unauthorized
                && policy.allows_retry(attempt)
            {
                warn!(
                    "{provider} request unauthorized: model={model}, attempt={attempt}/{}; refreshing credentials and retrying",
                    policy.max_attempts
                );
                previous_unauthorized = true;
                continue;
            }

            let retry_after = retry_after_from_headers(response.headers());
            let body = match read_body_limited(response, provider, ERROR_BODY_LIMIT).await {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(_) => String::new(),
            };
            let summary = summarize_error_body(&body);
            let error = ProviderError::http(
                provider,
                status,
                summary.message.map(|message| call.redact(&message)),
                call.redact(&summary.snippet),
                retry_after,
            );
            let retrying = error.is_retryable() && policy.allows_retry(attempt);
            let message = format!(
                "{provider} API error: model={model}, status={status}, attempt={attempt}/{}, retry_after={retry_after:?}, retrying={retrying}, body={}",
                policy.max_attempts,
                call.redact(&summary.snippet)
            );
            if retrying {
                warn!("{message}");
                tokio::time::sleep(policy.delay(attempt, retry_after)).await;
                continue;
            }
            error!("{message}");
            call.fail(&error).await;
            return Err(error);
        }

        match read_response(response).await {
            Ok(value) => {
                call.succeed(usage_of(&value)).await;
                return Ok(value);
            }
            Err(error) => {
                let retrying = error.is_retryable() && policy.allows_retry(attempt);
                let message = format!(
                    "{provider} response rejected: model={model}, attempt={attempt}/{}, retrying={retrying}, error={error}",
                    policy.max_attempts
                );
                if retrying {
                    warn!("{message}");
                    tokio::time::sleep(policy.delay(attempt, None)).await;
                    continue;
                }
                error!("{message}");
                call.fail(&error).await;
                return Err(error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};

    use super::*;
    use crate::tools::twitter_extractor::test_support::{
        response_with_headers, ExpectedRequest, TestServer,
    };
    use crate::utils::http::get_http_client;

    fn raw_response(status: u16, headers: &[(&str, &str)], body: &str) -> Vec<u8> {
        response_with_headers(status, headers, body.as_bytes().to_vec())
    }

    fn fast_policy(max_attempts: usize) -> RetryPolicy {
        RetryPolicy {
            jitter: false,
            ..RetryPolicy::linear(max_attempts, Duration::from_millis(1))
        }
    }

    fn no_usage(_: &Value) -> LlmUsageRecord {
        LlmUsageRecord::default()
    }

    #[test]
    fn tracked_calls_emit_the_request_event_and_untracked_ones_stay_silent() {
        let events = crate::utils::log_capture::capture_json_events(|| {
            let _tracked = LlmCall::begin("gemini", "m", "op", None, None);
            let _untracked = LlmCall::untracked("media", "telegram-file");
        });
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0]["fields"]["event"], "llm_request");
        assert_eq!(events[0]["fields"]["provider"], "gemini");
    }

    async fn read_json_value(response: reqwest::Response) -> Result<Value, ProviderError> {
        read_json::<Value>(response, "test-provider").await
    }

    #[tokio::test]
    async fn read_json_reports_decode_failures_with_a_snippet() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/j",
            response_with_headers(
                200,
                &[("content-type", "text/html")],
                b"<html>not json</html>".to_vec(),
            ),
        )]);
        let response = get_http_client()
            .get(server.url("/j"))
            .send()
            .await
            .expect("request");

        let err = read_json::<Value>(response, "test-provider")
            .await
            .expect_err("html is not json");
        match err {
            ProviderError::Decode {
                detail, retryable, ..
            } => {
                assert!(!retryable);
                assert!(detail.contains("text/html"), "{detail}");
                assert!(detail.contains("not json"), "{detail}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn read_json_rejects_an_empty_body_without_retrying() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/j",
            response_with_headers(200, &[], Vec::new()),
        )]);
        let response = get_http_client()
            .get(server.url("/j"))
            .send()
            .await
            .expect("request");

        let err = read_json::<Value>(response, "test-provider")
            .await
            .expect_err("empty body is not json");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("empty response body"), "{err}");
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn retries_a_retryable_status_and_honours_retry_after() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/v1",
                raw_response(429, &[("Retry-After", "0")], "{\"error\":\"slow down\"}"),
            ),
            ExpectedRequest::new("POST", "/v1", raw_response(200, &[], "{\"ok\":true}")),
        ]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        // A 5s backoff that is *not* replaced by Retry-After: 0 would make
        // this test take seconds.
        let policy = RetryPolicy {
            jitter: false,
            ..RetryPolicy::linear(3, Duration::from_secs(5))
        };
        let call = LlmCall::begin("test-provider", "model", "op", None, None);
        let started = Instant::now();

        let value: Value = call_with_retry(
            &call,
            &policy,
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            |_| {},
            read_json_value,
            no_usage,
        )
        .await
        .expect("second attempt succeeds");

        assert_eq!(value, json!({"ok": true}));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        server.join().expect("both requests were served");
    }

    #[tokio::test]
    async fn error_text_uses_the_display_label_not_the_audit_key() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1",
            raw_response(400, &[], "nope"),
        )]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let call =
            LlmCall::begin("openai-codex", "model", "op", None, None).with_label("OpenAI Codex");

        let err = call_with_retry(
            &call,
            &fast_policy(1),
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            |_| {},
            read_json_value,
            no_usage,
        )
        .await
        .expect_err("400 is permanent");

        assert!(
            err.to_string()
                .starts_with("OpenAI Codex request failed with status 400"),
            "{err}"
        );
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn permanent_http_errors_are_returned_without_retrying() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1",
            raw_response(400, &[], "{\"error\":{\"message\":\"bad model\"}}"),
        )]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let call = LlmCall::begin("test-provider", "model", "op", None, None);

        let err = call_with_retry(
            &call,
            &fast_policy(3),
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            |_| {},
            read_json_value,
            no_usage,
        )
        .await
        .expect_err("400 is permanent");

        match &err {
            ProviderError::Http {
                status, message, ..
            } => {
                assert_eq!(*status, StatusCode::BAD_REQUEST);
                assert_eq!(message.as_deref(), Some("bad model"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        server.join().expect("exactly one request was served");
    }

    #[tokio::test]
    async fn gives_up_after_the_attempt_budget_on_server_errors() {
        let server = TestServer::new(vec![
            ExpectedRequest::new("POST", "/v1", raw_response(503, &[], "busy")),
            ExpectedRequest::new("POST", "/v1", raw_response(503, &[], "busy")),
            ExpectedRequest::new("POST", "/v1", raw_response(503, &[], "busy")),
        ]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let call = LlmCall::begin("test-provider", "model", "op", None, None);

        let err = call_with_retry(
            &call,
            &fast_policy(3),
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            |_| {},
            read_json_value,
            no_usage,
        )
        .await
        .expect_err("all attempts fail");

        assert_eq!(err.status(), Some(StatusCode::SERVICE_UNAVAILABLE));
        assert!(err.is_retryable());
        server.join().expect("three requests were served");
    }

    #[tokio::test]
    async fn retryable_reader_failures_trigger_another_attempt() {
        let server = TestServer::new(vec![
            ExpectedRequest::new("POST", "/v1", raw_response(200, &[], "{\"n\":1}")),
            ExpectedRequest::new("POST", "/v1", raw_response(200, &[], "{\"n\":2}")),
        ]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let call = LlmCall::begin("test-provider", "model", "op", None, None);
        let reads = Arc::new(AtomicUsize::new(0));
        let reads_in_reader = reads.clone();

        let value: Value = call_with_retry(
            &call,
            &fast_policy(3),
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            |_| {},
            move |response| {
                let reads = reads_in_reader.clone();
                async move {
                    let value = read_json_value(response).await?;
                    if reads.fetch_add(1, Ordering::SeqCst) == 0 {
                        return Err(ProviderError::decode(
                            "test-provider",
                            "stream ended early",
                            true,
                        ));
                    }
                    Ok(value)
                }
            },
            no_usage,
        )
        .await
        .expect("second read succeeds");

        assert_eq!(value, json!({"n": 2}));
        assert_eq!(reads.load(Ordering::SeqCst), 2);
        server.join().expect("two requests were served");
    }

    #[tokio::test]
    async fn unauthorized_triggers_a_refresh_attempt_when_the_policy_allows() {
        let server = TestServer::new(vec![
            ExpectedRequest::new("POST", "/v1", raw_response(401, &[], "expired")),
            ExpectedRequest::new("POST", "/v1", raw_response(200, &[], "{\"ok\":true}")),
        ]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let policy = RetryPolicy {
            refresh_auth_on_unauthorized: true,
            ..fast_policy(3)
        };
        let call = LlmCall::begin("test-provider", "model", "op", None, None);
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let seen_in_builder = seen.clone();

        call_with_retry(
            &call,
            &policy,
            move |attempt: Attempt| {
                seen_in_builder.lock().push(attempt.previous_unauthorized);
                async move { Ok(get_http_client().post(url).json(&json!({}))) }
            },
            |_| {},
            read_json_value,
            no_usage,
        )
        .await
        .expect("refresh then success");

        assert_eq!(*seen.lock(), vec![false, true]);
        server.join().expect("two requests were served");
    }

    #[tokio::test]
    async fn unauthorized_is_permanent_by_default() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1",
            raw_response(401, &[], "expired"),
        )]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let call = LlmCall::begin("test-provider", "model", "op", None, None);

        let err = call_with_retry(
            &call,
            &fast_policy(3),
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            |_| {},
            read_json_value,
            no_usage,
        )
        .await
        .expect_err("401 without refresh is permanent");

        assert_eq!(err.status(), Some(StatusCode::UNAUTHORIZED));
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn response_observer_sees_headers_before_the_body_is_read() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/v1",
            raw_response(200, &[("x-request-id", "req_42")], "{\"ok\":true}"),
        )]);
        let url = server.url("/v1").to_string();
        let url = url.as_str();
        let call = LlmCall::begin("test-provider", "model", "op", None, None);
        let observed = Arc::new(parking_lot::Mutex::new(None));
        let observed_in_hook = observed.clone();

        call_with_retry(
            &call,
            &fast_policy(1),
            |_| async move { Ok(get_http_client().post(url).json(&json!({}))) },
            move |response| {
                *observed_in_hook.lock() = response
                    .headers()
                    .get("x-request-id")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string);
            },
            read_json_value,
            no_usage,
        )
        .await
        .expect("success");

        assert_eq!(observed.lock().as_deref(), Some("req_42"));
        server.join().expect("one request was served");
    }

    #[tokio::test]
    async fn read_body_limited_rejects_oversized_bodies() {
        let big = "x".repeat(100);
        let server = TestServer::new(vec![
            ExpectedRequest::new("GET", "/b", raw_response(200, &[], &big)),
            ExpectedRequest::new("GET", "/b", raw_response(200, &[], &big)),
        ]);

        let response = get_http_client()
            .get(server.url("/b"))
            .send()
            .await
            .unwrap();
        let err = read_body_limited(response, "test-provider", 10)
            .await
            .expect_err("100 bytes exceed a 10-byte limit");
        assert!(matches!(err, ProviderError::BodyTooLarge { limit: 10, .. }));

        let response = get_http_client()
            .get(server.url("/b"))
            .send()
            .await
            .unwrap();
        let bytes = read_body_limited(response, "test-provider", 100)
            .await
            .expect("100 bytes fit a 100-byte limit");
        assert_eq!(bytes.len(), 100);
        server.join().expect("two requests were served");
    }
}

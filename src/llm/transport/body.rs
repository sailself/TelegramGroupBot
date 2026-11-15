//! Response-body helpers: bounded reads, JSON decoding with diagnostics, and
//! error-body summaries.

use reqwest::header::CONTENT_TYPE;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::error::ProviderError;
use crate::utils::text::truncate_for_log;

/// Generous ceiling for JSON API responses (image payloads are base64).
pub const DEFAULT_JSON_BODY_LIMIT: usize = 16 * 1024 * 1024;
/// Error bodies are only summarized for logs; anything larger is noise.
pub const ERROR_BODY_LIMIT: usize = 64 * 1024;
const ERROR_SNIPPET_CHARS: usize = 2_000;
const DECODE_SNIPPET_CHARS: usize = 4_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorBodySummary {
    /// `error.message` or top-level `message` when the body is JSON.
    pub message: Option<String>,
    /// Bounded excerpt of the raw body for logs.
    pub snippet: String,
}

/// A provider's statement, in a 429 body, that the account's usage window is
/// spent. Waiting out the retry budget cannot help; only the reset can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageLimitNotice {
    /// Unix seconds at which the window resets.
    pub resets_at: Option<i64>,
    /// Seconds until the reset, counted from when the response was produced.
    pub resets_in_seconds: Option<i64>,
}

/// `error.type` Codex sends when a usage window (5-hour or weekly) is spent.
const USAGE_LIMIT_ERROR_TYPE: &str = "usage_limit_reached";

/// The usage-limit notice in an error body, when the body is one.
pub fn usage_limit_notice(body: &str) -> Option<UsageLimitNotice> {
    let value = serde_json::from_str::<Value>(body.trim()).ok()?;
    let error = value.get("error")?;
    if error.get("type").and_then(Value::as_str) != Some(USAGE_LIMIT_ERROR_TYPE) {
        return None;
    }
    Some(UsageLimitNotice {
        resets_at: error.get("resets_at").and_then(Value::as_i64),
        resets_in_seconds: error.get("resets_in_seconds").and_then(Value::as_i64),
    })
}

pub fn summarize_error_body(body: &str) -> ErrorBodySummary {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return ErrorBodySummary {
            message: None,
            snippet: "empty response body".to_string(),
        };
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let message = value
            .pointer("/error/message")
            .and_then(|v| v.as_str())
            .or_else(|| value.get("message").and_then(|v| v.as_str()))
            .map(str::to_string);
        return ErrorBodySummary {
            message,
            snippet: truncate_for_log(&value.to_string(), ERROR_SNIPPET_CHARS),
        };
    }

    ErrorBodySummary {
        message: None,
        snippet: truncate_for_log(trimmed, ERROR_SNIPPET_CHARS),
    }
}

/// Outcome of reading a body whose stream may be cut before the end.
#[derive(Debug)]
pub enum BodyRead {
    Complete(Vec<u8>),
    /// The connection dropped mid-body. `partial` holds what arrived; `error`
    /// is the retryable transport error to surface if the partial body is not
    /// usable on its own.
    Interrupted {
        partial: Vec<u8>,
        error: ProviderError,
    },
}

/// Read the whole body, refusing anything above `limit` bytes (a declared
/// `Content-Length` over the limit is rejected before any byte is read) and
/// reporting a stream interruption together with the bytes read so far.
pub async fn read_body_limited_or_partial(
    mut response: reqwest::Response,
    provider: &str,
    limit: usize,
) -> Result<BodyRead, ProviderError> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(ProviderError::BodyTooLarge {
            provider: provider.to_string(),
            limit,
        });
    }

    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if body.len().saturating_add(chunk.len()) > limit {
                    return Err(ProviderError::BodyTooLarge {
                        provider: provider.to_string(),
                        limit,
                    });
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => return Ok(BodyRead::Complete(body)),
            Err(err) => {
                return Ok(BodyRead::Interrupted {
                    partial: body,
                    error: ProviderError::Transport {
                        provider: provider.to_string(),
                        message: format!("response body read failed: {err}"),
                        retryable: true,
                    },
                })
            }
        }
    }
}

/// Read the whole body, refusing anything above `limit` bytes. Stream
/// interruptions surface as retryable transport errors.
pub async fn read_body_limited(
    response: reqwest::Response,
    provider: &str,
    limit: usize,
) -> Result<Vec<u8>, ProviderError> {
    match read_body_limited_or_partial(response, provider, limit).await? {
        BodyRead::Complete(body) => Ok(body),
        BodyRead::Interrupted { error, .. } => Err(error),
    }
}

/// Decode a 2xx JSON response, reporting status, content type and a body
/// excerpt when the payload is empty or malformed.
pub async fn read_json<T: DeserializeOwned>(
    response: reqwest::Response,
    provider: &str,
) -> Result<T, ProviderError> {
    let status = response.status();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let bytes = read_body_limited(response, provider, DEFAULT_JSON_BODY_LIMIT).await?;
    if bytes.is_empty() {
        return Err(ProviderError::decode(
            provider,
            format!("empty response body (status {status}, content-type {content_type})"),
            false,
        ));
    }

    serde_json::from_slice::<T>(&bytes).map_err(|err| {
        let body = String::from_utf8_lossy(&bytes);
        ProviderError::decode(
            provider,
            format!(
                "{err} (status {status}, content-type {content_type}) | body={}",
                truncate_for_log(&body, DECODE_SNIPPET_CHARS)
            ),
            false,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_error_bodies_are_labelled() {
        let summary = summarize_error_body("   ");
        assert_eq!(summary.message, None);
        assert_eq!(summary.snippet, "empty response body");
    }

    #[test]
    fn json_error_bodies_expose_the_nested_or_top_level_message() {
        let nested = summarize_error_body(r#"{"error":{"message":"quota exceeded","code":429}}"#);
        assert_eq!(nested.message.as_deref(), Some("quota exceeded"));
        assert!(nested.snippet.contains("quota exceeded"));

        let flat = summarize_error_body(r#"{"message":"bad key"}"#);
        assert_eq!(flat.message.as_deref(), Some("bad key"));
    }

    #[test]
    fn usage_limit_notice_reads_the_reset_from_a_usage_limit_body() {
        let notice = usage_limit_notice(
            r#"{"error":{"eligible_promo":null,"limit_window_minutes":300,"message":"The usage limit has been reached","plan_type":"plus","resets_at":1794726809,"resets_in_seconds":13974,"type":"usage_limit_reached"}}"#,
        );
        assert_eq!(
            notice,
            Some(UsageLimitNotice {
                resets_at: Some(1794726809),
                resets_in_seconds: Some(13974),
            })
        );

        let bare = usage_limit_notice(r#"{"error":{"type":"usage_limit_reached"}}"#);
        assert_eq!(
            bare,
            Some(UsageLimitNotice {
                resets_at: None,
                resets_in_seconds: None,
            })
        );
    }

    #[test]
    fn other_error_bodies_carry_no_usage_limit_notice() {
        for body in [
            r#"{"error":{"message":"quota exceeded","code":429}}"#,
            r#"{"error":{"type":"rate_limit_exceeded","message":"slow down"}}"#,
            r#"{"error":"slow down"}"#,
            "usage_limit_reached",
            "",
        ] {
            assert_eq!(usage_limit_notice(body), None, "{body}");
        }
    }

    #[test]
    fn non_json_error_bodies_are_truncated_snippets() {
        let summary = summarize_error_body(&"x".repeat(3000));
        assert_eq!(summary.message, None);
        assert!(summary.snippet.ends_with("... (truncated)"));
        assert!(summary.snippet.chars().count() < 2100);
    }
}

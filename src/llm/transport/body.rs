//! Response-body helpers: bounded reads, JSON decoding with diagnostics, and
//! error-body summaries.

use serde_json::Value;

use super::error::ProviderError;
use crate::utils::text::truncate_for_log;

/// Error bodies are only summarized for logs; anything larger is noise.
pub const ERROR_BODY_LIMIT: usize = 64 * 1024;
const ERROR_SNIPPET_CHARS: usize = 2_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorBodySummary {
    /// `error.message` or top-level `message` when the body is JSON.
    pub message: Option<String>,
    /// Bounded excerpt of the raw body for logs.
    pub snippet: String,
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

/// Read the whole body, refusing anything above `limit` bytes. A declared
/// `Content-Length` over the limit is rejected before any byte is read.
/// Stream interruptions surface as retryable transport errors.
pub async fn read_body_limited(
    mut response: reqwest::Response,
    provider: &str,
    limit: usize,
) -> Result<Vec<u8>, ProviderError> {
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
            Ok(None) => return Ok(body),
            Err(err) => {
                return Err(ProviderError::Transport {
                    provider: provider.to_string(),
                    message: format!("response body read failed: {err}"),
                    retryable: true,
                })
            }
        }
    }
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
    fn non_json_error_bodies_are_truncated_snippets() {
        let summary = summarize_error_body(&"x".repeat(3000));
        assert_eq!(summary.message, None);
        assert!(summary.snippet.ends_with("... (truncated)"));
        assert!(summary.snippet.chars().count() < 2100);
    }
}

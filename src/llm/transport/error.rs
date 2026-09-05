//! One error type for every provider transport failure.

use std::time::Duration;

use reqwest::StatusCode;

use super::retry::{is_retryable_status, is_retryable_transport_error};

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// The request never produced a response (connect/timeout/body upload).
    #[error("{provider} request failed: {message}")]
    Transport {
        provider: String,
        message: String,
        retryable: bool,
    },
    /// The server answered with a non-2xx status.
    #[error("{provider} request failed with status {status}: {}", message.as_deref().unwrap_or(body_snippet))]
    Http {
        provider: String,
        status: StatusCode,
        /// The provider's own error message when the body carried one.
        message: Option<String>,
        /// Bounded, log-safe excerpt of the error body.
        body_snippet: String,
        retry_after: Option<Duration>,
    },
    /// A 2xx response whose body could not be read or parsed. `retryable`
    /// marks stream interruptions and incomplete SSE streams, which a fresh
    /// request can fix; malformed JSON is permanent.
    #[error("{provider} response could not be decoded: {detail}")]
    Decode {
        provider: String,
        detail: String,
        retryable: bool,
    },
    #[error("{provider} response body exceeded the {limit}-byte limit")]
    BodyTooLarge { provider: String, limit: usize },
    /// A consumer-defined permanent failure (disabled provider, expired
    /// credentials that could not be refreshed, changed account, ...).
    #[error("{0}")]
    Rejected(String),
}

impl ProviderError {
    pub fn transport(provider: &str, err: &reqwest::Error, message: String) -> Self {
        Self::Transport {
            provider: provider.to_string(),
            message,
            retryable: is_retryable_transport_error(err),
        }
    }

    pub fn http(
        provider: &str,
        status: StatusCode,
        message: Option<String>,
        body_snippet: String,
        retry_after: Option<Duration>,
    ) -> Self {
        Self::Http {
            provider: provider.to_string(),
            status,
            message,
            body_snippet,
            retry_after,
        }
    }

    pub fn decode(provider: &str, detail: impl Into<String>, retryable: bool) -> Self {
        Self::Decode {
            provider: provider.to_string(),
            detail: detail.into(),
            retryable,
        }
    }

    pub fn rejected(message: impl Into<String>) -> Self {
        Self::Rejected(message.into())
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { retryable, .. } | Self::Decode { retryable, .. } => *retryable,
            Self::Http { status, .. } => is_retryable_status(*status),
            Self::BodyTooLarge { .. } | Self::Rejected(_) => false,
        }
    }

    pub fn status(&self) -> Option<StatusCode> {
        match self {
            Self::Http { status, .. } => Some(*status),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_errors_follow_their_retryable_flag() {
        let transient = ProviderError::Transport {
            provider: "gemini".to_string(),
            message: "connection reset".to_string(),
            retryable: true,
        };
        let permanent = ProviderError::Transport {
            provider: "gemini".to_string(),
            message: "invalid url".to_string(),
            retryable: false,
        };
        assert!(transient.is_retryable());
        assert!(!permanent.is_retryable());
        assert_eq!(
            transient.to_string(),
            "gemini request failed: connection reset"
        );
    }

    #[test]
    fn http_errors_are_retryable_only_for_transient_statuses() {
        let overloaded = ProviderError::http(
            "OpenRouter",
            StatusCode::SERVICE_UNAVAILABLE,
            None,
            "busy".to_string(),
            None,
        );
        let bad_request = ProviderError::http(
            "OpenRouter",
            StatusCode::BAD_REQUEST,
            Some("model not found".to_string()),
            "{...}".to_string(),
            None,
        );
        assert!(overloaded.is_retryable());
        assert!(!bad_request.is_retryable());
        assert_eq!(bad_request.status(), Some(StatusCode::BAD_REQUEST));
        assert_eq!(
            bad_request.to_string(),
            "OpenRouter request failed with status 400 Bad Request: model not found"
        );
        assert_eq!(
            overloaded.to_string(),
            "OpenRouter request failed with status 503 Service Unavailable: busy"
        );
    }

    #[test]
    fn decode_body_and_rejected_errors_classify_as_expected() {
        let decode = ProviderError::decode("gemini", "expected value", false);
        let stream = ProviderError::decode("OpenAI Codex", "missing response.completed", true);
        let too_large = ProviderError::BodyTooLarge {
            provider: "OpenAI Codex".to_string(),
            limit: 10,
        };
        let rejected = ProviderError::rejected("Gemini is disabled");

        assert!(!decode.is_retryable());
        assert!(stream.is_retryable());
        assert!(!too_large.is_retryable());
        assert!(!rejected.is_retryable());
        assert_eq!(
            too_large.to_string(),
            "OpenAI Codex response body exceeded the 10-byte limit"
        );
        assert_eq!(rejected.to_string(), "Gemini is disabled");
        assert_eq!(decode.status(), None);
    }
}

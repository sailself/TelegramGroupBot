//! SSE parsing for the Responses API's streaming transport.

use std::fmt;

use serde_json::{json, Value};

use crate::llm::transport::sse::parse_sse_data_events;

#[derive(Debug)]
pub(super) enum SseParseError {
    InvalidPayload {
        bytes: usize,
        source: serde_json::Error,
    },
    MissingCompletion,
    Incomplete(String),
    Failed(String),
}

impl SseParseError {
    pub(super) fn is_retryable(&self) -> bool {
        matches!(self, Self::InvalidPayload { .. } | Self::MissingCompletion)
    }
}

impl fmt::Display for SseParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayload { bytes, source } => {
                write!(
                    formatter,
                    "invalid {bytes}-byte SSE event payload: {source}"
                )
            }
            Self::MissingCompletion => {
                formatter.write_str("SSE stream ended before response.completed")
            }
            Self::Incomplete(detail) => write!(formatter, "SSE response incomplete: {detail}"),
            Self::Failed(detail) => write!(formatter, "SSE response failed: {detail}"),
        }
    }
}

impl std::error::Error for SseParseError {}

fn classify_incomplete_reason(value: Option<&str>) -> &'static str {
    match value {
        Some("max_output_tokens") => "max_output_tokens",
        Some("content_filter") => "content_filter",
        _ => "other",
    }
}

/// A chunked SSE stream whose terminator was cut after a full
/// `response.completed` event is semantically complete; retrying it would
/// only repeat the turn.
pub(super) fn interrupted_body_is_complete_sse(partial: &[u8]) -> bool {
    std::str::from_utf8(partial)
        .ok()
        .is_some_and(|body| parse_sse_responses_body(body).is_ok())
}

pub(super) fn parse_sse_responses_body(body: &str) -> std::result::Result<Value, SseParseError> {
    let events = parse_sse_data_events(body).map_err(|err| SseParseError::InvalidPayload {
        bytes: err.bytes,
        source: err.source,
    })?;

    let mut output_items: Vec<Value> = Vec::new();
    let mut response_id: Option<String> = None;
    let mut usage: Option<Value> = None;
    let mut completed = false;

    for value in events {
        if response_id.is_none() {
            response_id = value
                .pointer("/response/id")
                .and_then(|value| value.as_str())
                .map(str::to_string);
        }
        match value.get("type").and_then(|value| value.as_str()) {
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item").cloned() {
                    output_items.push(item);
                }
            }
            Some("response.completed") => {
                completed = true;
                if usage.is_none() {
                    usage = value.pointer("/response/usage").cloned();
                }
                if output_items.is_empty() {
                    if let Some(items) = value
                        .get("response")
                        .and_then(|response| response.get("output"))
                        .and_then(|items| items.as_array())
                    {
                        output_items.extend(items.iter().cloned());
                    }
                }
            }
            Some("response.incomplete") => {
                let reason = value
                    .pointer("/response/incomplete_details/reason")
                    .and_then(|value| value.as_str());
                return Err(SseParseError::Incomplete(
                    classify_incomplete_reason(reason).to_string(),
                ));
            }
            Some("response.failed") => {
                let message_present = value
                    .pointer("/response/error/message")
                    .and_then(|value| value.as_str())
                    .or_else(|| {
                        value
                            .pointer("/error/message")
                            .and_then(|value| value.as_str())
                    })
                    .is_some();
                return Err(SseParseError::Failed(format!(
                    "response_failed(message_present={message_present})"
                )));
            }
            Some("error") => {
                let message_present = value
                    .pointer("/error/message")
                    .and_then(|value| value.as_str())
                    .or_else(|| value.get("message").and_then(|value| value.as_str()))
                    .is_some();
                return Err(SseParseError::Failed(format!(
                    "error_event(message_present={message_present})"
                )));
            }
            _ => {}
        }
    }

    if !completed {
        return Err(SseParseError::MissingCompletion);
    }

    Ok(json!({
        "id": response_id,
        "output": output_items,
        "usage": usage,
    }))
}

pub(super) fn extract_response_output_items(response: &Value) -> Vec<Value> {
    response
        .get("output")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default()
}

pub(super) fn extract_response_text(output_items: &[Value]) -> String {
    let mut text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    for item in output_items {
        match item.get("type").and_then(|value| value.as_str()) {
            Some("message") => {
                if let Some(content_items) = item.get("content").and_then(|value| value.as_array())
                {
                    for content_item in content_items {
                        let item_type = content_item.get("type").and_then(|value| value.as_str());
                        if matches!(item_type, Some("output_text") | Some("text")) {
                            if let Some(text) =
                                content_item.get("text").and_then(|value| value.as_str())
                            {
                                let trimmed = text.trim();
                                if !trimmed.is_empty() {
                                    text_parts.push(trimmed.to_string());
                                }
                            }
                        }
                    }
                }
            }
            Some("reasoning") => {
                if let Some(summary_items) = item.get("summary").and_then(|value| value.as_array())
                {
                    for summary_item in summary_items {
                        if let Some(text) =
                            summary_item.get("text").and_then(|value| value.as_str())
                        {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                reasoning_parts.push(trimmed.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        return text_parts.join("\n");
    }
    reasoning_parts.join("\n")
}

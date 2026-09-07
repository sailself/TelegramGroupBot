//! Server-sent-events decoding shared by the Responses and Codex image
//! transports: `data:` lines are joined per blank-line-separated event and
//! parsed as JSON.

use std::fmt;

use serde_json::Value;

#[derive(Debug)]
pub struct SseDecodeError {
    pub bytes: usize,
    pub source: serde_json::Error,
}

impl fmt::Display for SseDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid {}-byte SSE event payload: {}",
            self.bytes, self.source
        )
    }
}

impl std::error::Error for SseDecodeError {}

/// Decode every `data:` event in `body` as JSON, in order. `[DONE]` markers,
/// empty events, comments and non-`data` fields (`event:`, `id:`) are skipped.
pub fn parse_sse_data_events(body: &str) -> Result<Vec<Value>, SseDecodeError> {
    let mut events = Vec::new();
    let mut current_data_lines: Vec<String> = Vec::new();

    fn flush(lines: &mut Vec<String>, events: &mut Vec<Value>) -> Result<(), SseDecodeError> {
        if lines.is_empty() {
            return Ok(());
        }
        let payload = lines.join("\n");
        lines.clear();
        let trimmed = payload.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return Ok(());
        }
        let value = serde_json::from_str::<Value>(trimmed).map_err(|source| SseDecodeError {
            bytes: trimmed.len(),
            source,
        })?;
        events.push(value);
        Ok(())
    }

    for line in body.lines() {
        if line.trim().is_empty() {
            flush(&mut current_data_lines, &mut events)?;
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            current_data_lines.push(data.trim_start().to_string());
        }
    }
    flush(&mut current_data_lines, &mut events)?;

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn joins_multi_line_data_and_ignores_other_fields() {
        let body = "event: response.created\nid: 1\ndata: {\"type\":\n\
                    data:  \"response.created\"}\n\n\
                    : keepalive comment\n\n\
                    data: {\"type\":\"response.completed\"}\n";
        let events = parse_sse_data_events(body).expect("valid stream");
        assert_eq!(
            events,
            vec![
                json!({"type": "response.created"}),
                json!({"type": "response.completed"})
            ]
        );
    }

    #[test]
    fn skips_done_markers_and_empty_events() {
        let body = "data: [DONE]\n\ndata:\n\ndata: {\"a\":1}\n\n";
        let events = parse_sse_data_events(body).expect("valid stream");
        assert_eq!(events, vec![json!({"a": 1})]);
    }

    #[test]
    fn reports_invalid_json_payloads_with_their_size() {
        let err = parse_sse_data_events("data: {not json\n\n").expect_err("invalid payload");
        assert_eq!(err.bytes, "{not json".len());
        assert!(err
            .to_string()
            .starts_with("invalid 9-byte SSE event payload"));
    }
}

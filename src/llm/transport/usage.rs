//! Token-usage extraction for provider response shapes.

use serde_json::Value;

use crate::llm::audit::LlmUsageRecord;

fn i64_at(value: &Value, pointer: &str) -> Option<i64> {
    value.pointer(pointer).and_then(Value::as_i64)
}

fn response_id(response: &Value) -> Option<String> {
    response
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn usage_object(response: &Value) -> Option<&Value> {
    response.get("usage").filter(|usage| !usage.is_null())
}

/// OpenAI chat-completions shape (OpenRouter, NVIDIA, Ollama): `usage.prompt_tokens`,
/// `completion_tokens`, `total_tokens` plus the `*_tokens_details` breakdowns.
pub fn from_chat_completions(response: &Value) -> LlmUsageRecord {
    let usage = usage_object(response);
    LlmUsageRecord {
        response_id: response_id(response),
        input_tokens: usage.and_then(|u| i64_at(u, "/prompt_tokens")),
        output_tokens: usage.and_then(|u| i64_at(u, "/completion_tokens")),
        total_tokens: usage.and_then(|u| i64_at(u, "/total_tokens")),
        reasoning_tokens: usage
            .and_then(|u| i64_at(u, "/completion_tokens_details/reasoning_tokens")),
        cached_input_tokens: usage.and_then(|u| i64_at(u, "/prompt_tokens_details/cached_tokens")),
        cache_write_tokens: None,
        raw_usage_json: usage.map(Value::to_string),
    }
}

/// Gemini `usageMetadata`.
pub fn from_gemini(response: &Value) -> LlmUsageRecord {
    let Some(usage) = response
        .get("usageMetadata")
        .filter(|usage| !usage.is_null())
    else {
        return LlmUsageRecord::default();
    };
    LlmUsageRecord {
        response_id: None,
        input_tokens: i64_at(usage, "/promptTokenCount"),
        output_tokens: i64_at(usage, "/candidatesTokenCount"),
        total_tokens: i64_at(usage, "/totalTokenCount"),
        reasoning_tokens: i64_at(usage, "/thoughtsTokenCount"),
        cached_input_tokens: i64_at(usage, "/cachedContentTokenCount"),
        cache_write_tokens: None,
        raw_usage_json: Some(usage.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_completions_usage_reads_prompt_completion_and_total_tokens() {
        let response = json!({
            "id": "chatcmpl_123",
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 34,
                "total_tokens": 55,
                "prompt_tokens_details": { "cached_tokens": 8 },
                "completion_tokens_details": { "reasoning_tokens": 5 }
            }
        });

        let usage = from_chat_completions(&response);

        assert_eq!(usage.response_id.as_deref(), Some("chatcmpl_123"));
        assert_eq!(usage.input_tokens, Some(21));
        assert_eq!(usage.output_tokens, Some(34));
        assert_eq!(usage.total_tokens, Some(55));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.cached_input_tokens, Some(8));
        assert_eq!(usage.cache_write_tokens, None);
    }

    #[test]
    fn gemini_usage_reads_usage_metadata() {
        let response = json!({
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 34,
                "totalTokenCount": 46,
                "thoughtsTokenCount": 5,
                "cachedContentTokenCount": 3
            }
        });

        let usage = from_gemini(&response);

        assert_eq!(usage.response_id, None);
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(34));
        assert_eq!(usage.total_tokens, Some(46));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert!(usage
            .raw_usage_json
            .as_deref()
            .expect("usage json")
            .contains("\"promptTokenCount\":12"));
    }

    #[test]
    fn missing_usage_yields_an_empty_record() {
        assert_eq!(from_gemini(&json!({})), LlmUsageRecord::default());
        assert_eq!(
            from_chat_completions(&json!({"id": "x"})),
            LlmUsageRecord {
                response_id: Some("x".to_string()),
                ..LlmUsageRecord::default()
            }
        );
    }
}

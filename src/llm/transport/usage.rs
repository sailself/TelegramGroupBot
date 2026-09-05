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

/// OpenAI Responses shape: `usage.input_tokens`/`output_tokens` (total derived
/// when absent) with `input_tokens_details` and `output_tokens_details`.
pub fn from_responses(response: &Value) -> LlmUsageRecord {
    match usage_object(response) {
        Some(usage) => from_responses_usage_object(usage, response_id(response)),
        None => LlmUsageRecord {
            response_id: response_id(response),
            ..LlmUsageRecord::default()
        },
    }
}

/// A bare Responses `usage` object, as found on `response.completed` events and
/// on `image_generation_call` output items.
pub fn from_responses_usage_object(usage: &Value, response_id: Option<String>) -> LlmUsageRecord {
    let input_tokens = i64_at(usage, "/input_tokens");
    let output_tokens = i64_at(usage, "/output_tokens");
    let total_tokens =
        i64_at(usage, "/total_tokens").or_else(|| match (input_tokens, output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        });
    LlmUsageRecord {
        response_id,
        input_tokens,
        output_tokens,
        total_tokens,
        reasoning_tokens: i64_at(usage, "/output_tokens_details/reasoning_tokens"),
        cached_input_tokens: i64_at(usage, "/input_tokens_details/cached_tokens"),
        cache_write_tokens: i64_at(usage, "/input_tokens_details/cache_write_tokens"),
        raw_usage_json: Some(usage.to_string()),
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
    fn responses_usage_reads_token_counts_and_derives_the_total() {
        let response = json!({
            "id": "resp_123",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": { "cached_tokens": 3, "cache_write_tokens": 4 },
                "output_tokens_details": { "reasoning_tokens": 7 }
            }
        });

        let usage = from_responses(&response);

        assert_eq!(usage.response_id.as_deref(), Some("resp_123"));
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(30));
        assert_eq!(usage.reasoning_tokens, Some(7));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert_eq!(usage.cache_write_tokens, Some(4));
    }

    #[test]
    fn responses_usage_leaves_cache_write_tokens_none_when_absent() {
        let usage = from_responses(&json!({
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "input_tokens_details": { "cached_tokens": 0 }
            }
        }));
        assert_eq!(usage.cache_write_tokens, None);
        assert_eq!(usage.total_tokens, Some(3));
    }

    #[test]
    fn responses_usage_object_can_be_read_with_an_explicit_response_id() {
        let usage = from_responses_usage_object(
            &json!({ "input_tokens": 5, "output_tokens": 6 }),
            Some("resp_img".to_string()),
        );
        assert_eq!(usage.response_id.as_deref(), Some("resp_img"));
        assert_eq!(usage.total_tokens, Some(11));
        assert!(usage.raw_usage_json.is_some());
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

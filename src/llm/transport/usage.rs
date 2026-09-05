//! Token-usage extraction for provider response shapes.

use serde_json::Value;

use crate::llm::audit::LlmUsageRecord;

fn i64_at(value: &Value, pointer: &str) -> Option<i64> {
    value.pointer(pointer).and_then(Value::as_i64)
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
    }
}

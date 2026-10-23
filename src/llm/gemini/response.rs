//! Validate completed text/tool responses before exposing output or executing tools.
//! Image and music generation have their own response contracts.

use anyhow::Result;
use serde_json::Value;

use super::GeminiResponse;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum GeminiTextResponseError {
    #[error("Gemini returned an incomplete response ({reason}); single-hop generation stopped")]
    Incomplete { reason: &'static str },
    #[error("Gemini blocked the response")]
    Blocked,
    #[error("Gemini returned no visible answer or usable tool call")]
    Empty,
    #[error("Gemini returned an invalid text/tool response")]
    Invalid,
}

impl GeminiTextResponseError {
    pub(crate) fn user_message(&self) -> &'static str {
        match self {
            Self::Incomplete { .. } => {
                "Gemini stopped before completing its answer. Please try a smaller request."
            }
            Self::Blocked => {
                "Gemini could not provide an answer to this request. Please try rephrasing it."
            }
            Self::Empty => "Gemini returned no answer. Please try again.",
            Self::Invalid => "Gemini returned an invalid response. Please try again.",
        }
    }
}

/// Missing finishReason is tolerated for older responses only when usable
/// content exists and no continuation token indicates unfinished generation.
/// Never copy provider text, finish messages, or opaque tokens into errors.
pub(super) fn validate_text_response(response: &Value, allow_tools: bool) -> Result<()> {
    if let Some(reason) = response.pointer("/promptFeedback/blockReason") {
        match reason.as_str() {
            Some("" | "BLOCK_REASON_UNSPECIFIED") => {}
            Some(_) => return Err(GeminiTextResponseError::Blocked.into()),
            None => return Err(GeminiTextResponseError::Invalid.into()),
        }
    }
    let candidates = response
        .get("candidates")
        .and_then(Value::as_array)
        .filter(|candidates| !candidates.is_empty())
        .ok_or(GeminiTextResponseError::Empty)?;
    for candidate in candidates {
        match candidate.get("finishReason") {
            None => {
                if candidate.get("continuationToken").is_some() {
                    return Err(GeminiTextResponseError::Incomplete {
                        reason: "CONTINUATION",
                    }
                    .into());
                }
            }
            Some(reason) => match reason.as_str() {
                Some("STOP") => {}
                Some("MAX_TOKENS") => {
                    return Err(GeminiTextResponseError::Incomplete {
                        reason: "MAX_TOKENS",
                    }
                    .into());
                }
                Some("CONTINUATION") => {
                    return Err(GeminiTextResponseError::Incomplete {
                        reason: "CONTINUATION",
                    }
                    .into());
                }
                Some("SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII") => {
                    return Err(GeminiTextResponseError::Blocked.into());
                }
                _ => return Err(GeminiTextResponseError::Invalid.into()),
            },
        }
        let parts = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .ok_or(GeminiTextResponseError::Empty)?;
        let mut usable = false;
        for part in parts {
            let thought = part.get("thought").and_then(Value::as_bool) == Some(true);
            if let Some(call) = part.get("functionCall") {
                if !allow_tools
                    || thought
                    || call
                        .get("name")
                        .and_then(Value::as_str)
                        .is_none_or(|name| name.trim().is_empty())
                    || call.get("args").is_some_and(|args| !args.is_object())
                    || call.get("id").is_some_and(|id| !id.is_string())
                {
                    return Err(GeminiTextResponseError::Invalid.into());
                }
                usable = true;
            }
            if thought {
                continue;
            }
            usable |= [
                "/text",
                "/executableCode/code",
                "/codeExecutionResult/output",
            ]
            .iter()
            .any(|path| {
                part.pointer(path)
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty())
            });
        }
        if !usable {
            return Err(GeminiTextResponseError::Empty.into());
        }
    }
    Ok(())
}

pub(super) fn remove_thought_parts(response: &mut Value) {
    if let Some(candidates) = response.get_mut("candidates").and_then(Value::as_array_mut) {
        for candidate in candidates {
            if let Some(parts) = candidate
                .pointer_mut("/content/parts")
                .and_then(Value::as_array_mut)
            {
                parts.retain(|part| part.get("thought").and_then(Value::as_bool) != Some(true));
            }
        }
    }
}

pub(super) fn decode_text_response(mut response: Value) -> Result<GeminiResponse> {
    validate_text_response(&response, false)?;
    remove_thought_parts(&mut response);
    serde_json::from_value(response).map_err(|_| GeminiTextResponseError::Invalid.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::gemini::{
        decode_gemini_turn, extract_text_from_response, extract_text_from_response_value,
        gemini_error_allows_model_fallback,
    };
    use serde_json::json;

    #[test]
    fn incomplete_and_blocked_output_never_becomes_an_answer_or_tool_call() {
        for reason in [
            "MAX_TOKENS",
            "CONTINUATION",
            "SAFETY",
            "RECITATION",
            "MALFORMED_FUNCTION_CALL",
            "FINISH_REASON_UNSPECIFIED",
            "UNKNOWN_NEW_REASON",
        ] {
            let response = json!({"candidates": [{
                "finishReason": reason,
                "continuationToken": "private-checkpoint",
                "finishMessage": "private-provider-message",
                "content": {"role": "model", "parts": [{"text": "unfinished answer"}]}
            }]});
            let error = decode_text_response(response.clone()).unwrap_err();
            assert!(error.is::<GeminiTextResponseError>(), "{reason}");
            assert!(!error.to_string().contains("private"));
            assert!(!gemini_error_allows_model_fallback(&error));
            assert!(decode_gemini_turn(&response).is_err());
            assert!(validate_text_response(&response, false).is_err());
        }
        let blocked = json!({"promptFeedback": {"blockReason": "SAFETY"}});
        assert_eq!(
            validate_text_response(&blocked, false)
                .unwrap_err()
                .downcast_ref::<GeminiTextResponseError>(),
            Some(&GeminiTextResponseError::Blocked)
        );
    }

    #[test]
    fn empty_and_thought_only_results_are_rejected_on_all_text_paths() {
        for response in [
            json!({}),
            json!({"candidates": []}),
            json!({"candidates": [{"finishReason": "STOP"}]}),
            json!({"candidates": [{"content": {"parts": [{"text": "  "}]}}]}),
            json!({"candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"thought": true, "text": "private thought"},
                {"thought": true, "executableCode": {"code": "private code"}}
            ]}}]}),
        ] {
            assert!(decode_text_response(response.clone()).is_err());
            assert!(decode_gemini_turn(&response).is_err());
            assert!(validate_text_response(&response, false).is_err());
        }
        let checkpoint = json!({"candidates": [{"continuationToken": "private",
            "content": {"parts": [{"text": "partial"}]}}]});
        assert!(decode_text_response(checkpoint).is_err());
    }

    #[test]
    fn visible_output_excludes_thoughts_and_tool_history_retains_signatures() {
        let response = json!({"candidates": [{"finishReason": "STOP", "content": {
            "role": "model", "parts": [
                {"thought": true, "text": "private thought", "thoughtSignature": "signed-thought"},
                {"text": "answer", "thoughtSignature": "signed-answer"}
            ]
        }}]});
        assert_eq!(
            extract_text_from_response(decode_text_response(response.clone()).unwrap()),
            "answer"
        );
        assert_eq!(extract_text_from_response_value(&response), "answer");
        let turn = decode_gemini_turn(&response).unwrap();
        assert_eq!(turn.text, "answer");
        assert_eq!(turn.transcript[0], response["candidates"][0]["content"]);
        let legacy = json!({"candidates": [{"content": {"parts": [{"text": "legacy"}]}}]});
        assert_eq!(
            extract_text_from_response(decode_text_response(legacy).unwrap()),
            "legacy"
        );
    }

    #[test]
    fn only_complete_valid_function_calls_are_executable() {
        let mut response = json!({"candidates": [{"finishReason": "STOP", "content": {
            "role": "model", "parts": [{"thoughtSignature": "signed-call",
                "functionCall": {"id": "call1", "name": "web_search", "args": {"query": "example"}}}]
        }}]});
        let turn = decode_gemini_turn(&response).unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "call1");
        assert_eq!(turn.transcript[0], response["candidates"][0]["content"]);
        assert!(validate_text_response(&response, false).is_err());
        response["candidates"][0]["finishReason"] = json!("MAX_TOKENS");
        assert!(decode_gemini_turn(&response).is_err());
        response["candidates"][0]["finishReason"] = json!("STOP");
        for call in [
            json!({}),
            json!({"name":""}),
            json!({"name":"web_search","args":[]}),
            json!({"name":"web_search","id":4}),
        ] {
            response["candidates"][0]["content"]["parts"][0]["functionCall"] = call;
            assert!(decode_gemini_turn(&response).is_err());
        }
        response["candidates"][0]["content"]["parts"][0] = json!({
            "thought":true,"functionCall":{"name":"web_search","args":{}}});
        assert!(decode_gemini_turn(&response).is_err());
    }
}

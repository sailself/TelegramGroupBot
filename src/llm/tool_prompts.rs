//! Shared tool-runtime prompt fragments.
//!
//! Single source of truth for the tool-budget guidance that is appended to a
//! model's system prompt and the post-limit nudge sent once the budget is
//! exhausted. Both the OpenAI Responses provider and the third-party (chat
//! completions) provider compose these, so they live here to prevent drift.

/// System message pushed when the tool-call budget is exhausted, asking the
/// model to answer with what it already gathered instead of calling more tools.
pub const TOOL_LIMIT_SYSTEM_PROMPT: &str =
    "Tool call limit reached. Provide the best possible answer using the available information without requesting more tool calls.";

/// Advisory guidance appended to the system prompt describing the tool budget.
/// `{max_tool_calls}` is substituted by [`tool_limit_guidance`].
pub const TOOL_LIMIT_GUIDANCE: &str =
    "Tool usage limit: you may use tools for at most {max_tool_calls} rounds total in this conversation. Plan your searches efficiently, avoid redundant tool calls, and after the final allowed tool round you must answer using the information already gathered without requesting more tool calls.";

/// Render [`TOOL_LIMIT_GUIDANCE`] for a concrete maximum number of tool rounds.
/// Callers pass the same constant that gates their tool loop so the spoken
/// number can never drift from the enforced cap.
pub fn tool_limit_guidance(max_tool_calls: usize) -> String {
    TOOL_LIMIT_GUIDANCE.replace("{max_tool_calls}", &max_tool_calls.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_limit_guidance_substitutes_count() {
        let rendered = tool_limit_guidance(3);
        assert!(rendered.contains("at most 3 rounds total"));
        assert!(!rendered.contains("{max_tool_calls}"));
    }
}

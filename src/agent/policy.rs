use anyhow::Result;
use regex::Regex;
use serde_json::Value;

use crate::agent::tools::is_tool_allowed;
use crate::config::CONFIG;

fn contains_case_insensitive(values: &[String], item: &str) -> bool {
    values.iter().any(|value| value.eq_ignore_ascii_case(item))
}

fn command_matches_allowlist(command: &str, patterns: &[String]) -> Result<bool> {
    for pattern in patterns {
        let trimmed = pattern.trim();
        if trimmed.is_empty() {
            continue;
        }
        let regex = Regex::new(trimmed)?;
        if regex.is_match(command) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn evaluate_agent_tool_call(
    tool_name: &str,
    args: &Value,
    skill_allowed_tools: &[String],
) -> Result<(), String> {
    if !CONFIG.agent_tool_policy_enforced {
        return Ok(());
    }

    if !is_tool_allowed(tool_name, skill_allowed_tools) {
        return Err(format!(
            "Tool '{}' is not allowed by active skills.",
            tool_name
        ));
    }

    if contains_case_insensitive(&CONFIG.agent_tool_denylist, tool_name) {
        return Err(format!(
            "Tool '{}' is denied by AGENT_TOOL_DENYLIST.",
            tool_name
        ));
    }

    if !CONFIG.agent_tool_allowlist.is_empty()
        && !contains_case_insensitive(&CONFIG.agent_tool_allowlist, tool_name)
    {
        return Err(format!(
            "Tool '{}' is not in AGENT_TOOL_ALLOWLIST.",
            tool_name
        ));
    }

    if tool_name.eq_ignore_ascii_case("exec") && !CONFIG.agent_exec_allowlist_regex.is_empty() {
        let Some(command) = args.get("command").and_then(|value| value.as_str()) else {
            return Err("Tool 'exec' requires a string 'command' field.".to_string());
        };
        let trimmed = command.trim();
        if trimmed.is_empty() {
            return Err("Tool 'exec' received an empty command.".to_string());
        }
        match command_matches_allowlist(trimmed, &CONFIG.agent_exec_allowlist_regex) {
            Ok(true) => {}
            Ok(false) => {
                return Err(
                    "Tool 'exec' command does not match AGENT_EXEC_ALLOWLIST_REGEX.".to_string(),
                )
            }
            Err(err) => {
                return Err(format!(
                    "Tool policy misconfiguration in AGENT_EXEC_ALLOWLIST_REGEX: {}",
                    err
                ))
            }
        }
    }

    Ok(())
}

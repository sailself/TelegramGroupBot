use anyhow::Result;
use regex::Regex;
use serde_json::Value;

use crate::acl::acl_manager;
use crate::agent::tools::is_tool_allowed;
use crate::config::CONFIG;

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
    chat_id: i64,
    user_id: i64,
    tool_name: &str,
    args: &Value,
    declared_tools: &[String],
) -> Result<(), String> {
    let normalized_tool = tool_name.trim().to_ascii_lowercase();
    if normalized_tool.is_empty() {
        return Err("Tool name is required.".to_string());
    }

    if !is_tool_allowed(&normalized_tool, declared_tools) {
        return Err(format!(
            "Tool '{}' is not declared for this run.",
            normalized_tool
        ));
    }

    let decision = acl_manager().authorize_tool(chat_id, user_id, &normalized_tool);
    if !decision.allowed {
        return Err(format!(
            "Tool '{}' is denied by ACL ({}).",
            normalized_tool, decision.reason
        ));
    }

    if normalized_tool.eq_ignore_ascii_case("exec") && !CONFIG.agent_exec_allowlist_regex.is_empty()
    {
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

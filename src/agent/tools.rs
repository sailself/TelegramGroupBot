use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::config::CONFIG;
use crate::llm::web_search::web_search_tool;
use crate::tools::core_filesystem;
use crate::tools::core_shell;

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
    pub side_effect: bool,
}

pub const READ_FILE_TOOL: &str = "read_file";
pub const WRITE_FILE_TOOL: &str = "write_file";
pub const EDIT_FILE_TOOL: &str = "edit_file";
pub const EXEC_TOOL: &str = "exec";
pub const WEB_SEARCH_TOOL: &str = "web_search";

pub fn all_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: READ_FILE_TOOL,
            description: "Read file contents from a path.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read."
                    }
                },
                "required": ["path"]
            }),
            side_effect: false,
        },
        ToolSpec {
            name: WRITE_FILE_TOOL,
            description:
                "Write full content to a file path. Creates parent directories when needed.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content."
                    }
                },
                "required": ["path", "content"]
            }),
            side_effect: true,
        },
        ToolSpec {
            name: EDIT_FILE_TOOL,
            description:
                "Replace exact old_text with new_text in a file. old_text must appear exactly once.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text."
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }),
            side_effect: true,
        },
        ToolSpec {
            name: EXEC_TOOL,
            description:
                "Execute a shell command (PowerShell on Windows, bash on Unix) in workspace.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory."
                    }
                },
                "required": ["command"]
            }),
            side_effect: true,
        },
        ToolSpec {
            name: WEB_SEARCH_TOOL,
            description:
                "Search the web using configured providers and return a concise Markdown summary.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of results (default 5)."
                    }
                },
                "required": ["query"]
            }),
            side_effect: false,
        },
    ]
}

pub fn build_openrouter_tool_definitions(allowed_tools: &[String]) -> Vec<Value> {
    let allowed = allowed_tools
        .iter()
        .map(|tool| tool.trim().to_lowercase())
        .collect::<HashSet<_>>();

    all_tool_specs()
        .into_iter()
        .filter(|spec| allowed.contains(spec.name))
        .map(|spec| {
            json!({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                }
            })
        })
        .collect()
}

pub fn build_gemini_tool_definitions(allowed_tools: &[String]) -> Vec<Value> {
    let allowed = allowed_tools
        .iter()
        .map(|tool| tool.trim().to_lowercase())
        .collect::<HashSet<_>>();

    let declarations = all_tool_specs()
        .into_iter()
        .filter(|spec| allowed.contains(spec.name))
        .map(|spec| {
            json!({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        })
        .collect::<Vec<_>>();

    if declarations.is_empty() {
        Vec::new()
    } else {
        vec![json!({ "functionDeclarations": declarations })]
    }
}

pub fn is_tool_allowed(tool_name: &str, allowed_tools: &[String]) -> bool {
    allowed_tools
        .iter()
        .any(|tool| tool.eq_ignore_ascii_case(tool_name))
}

pub fn is_side_effect_tool(tool_name: &str) -> bool {
    all_tool_specs()
        .into_iter()
        .find(|spec| spec.name.eq_ignore_ascii_case(tool_name))
        .map(|spec| spec.side_effect)
        .unwrap_or(false)
}

pub fn requires_confirmation(tool_name: &str) -> bool {
    if !is_side_effect_tool(tool_name) {
        return false;
    }
    if tool_name.eq_ignore_ascii_case(WRITE_FILE_TOOL) {
        return CONFIG.agent_require_confirmation_for_write;
    }
    if tool_name.eq_ignore_ascii_case(EDIT_FILE_TOOL) {
        return CONFIG.agent_require_confirmation_for_edit;
    }
    if tool_name.eq_ignore_ascii_case(EXEC_TOOL) {
        return CONFIG.agent_require_confirmation_for_exec;
    }
    true
}

pub async fn execute_tool(tool_name: &str, args: &Value, workspace_root: &Path) -> Result<String> {
    match tool_name.to_lowercase().as_str() {
        READ_FILE_TOOL => {
            let path = args
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("read_file requires string field 'path'"))?;
            core_filesystem::read_file(workspace_root, path).await
        }
        WRITE_FILE_TOOL => {
            let path = args
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("write_file requires string field 'path'"))?;
            let content = args
                .get("content")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("write_file requires string field 'content'"))?;
            core_filesystem::write_file(workspace_root, path, content).await
        }
        EDIT_FILE_TOOL => {
            let path = args
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("edit_file requires string field 'path'"))?;
            let old_text = args
                .get("old_text")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("edit_file requires string field 'old_text'"))?;
            let new_text = args
                .get("new_text")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("edit_file requires string field 'new_text'"))?;
            core_filesystem::edit_file(workspace_root, path, old_text, new_text).await
        }
        EXEC_TOOL => {
            let command = args
                .get("command")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("exec requires string field 'command'"))?;
            let working_dir = args.get("working_dir").and_then(|value| value.as_str());
            core_shell::execute_command(
                workspace_root,
                command,
                working_dir,
                CONFIG.agent_exec_timeout_seconds,
                CONFIG.agent_exec_max_output_chars,
                CONFIG.agent_exec_restrict_to_workspace,
                &CONFIG.agent_exec_deny_patterns,
            )
            .await
        }
        WEB_SEARCH_TOOL => {
            let query = args
                .get("query")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("web_search requires string field 'query'"))?;
            let max_results = args
                .get("max_results")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize);
            web_search_tool(query, max_results).await
        }
        other => Err(anyhow!("Unknown tool '{}'", other)),
    }
}

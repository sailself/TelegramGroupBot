use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::db::models::AgentMemoryInsert;
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
pub const MEMORY_STORE_TOOL: &str = "memory_store";
pub const MEMORY_RECALL_TOOL: &str = "memory_recall";
pub const MEMORY_FORGET_TOOL: &str = "memory_forget";

fn normalize_memory_query(value: &str) -> String {
    value
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .take(24)
        .collect::<Vec<_>>()
        .join(" ")
}

fn summarize_for_memory(content: &str) -> String {
    let max = CONFIG.agent_memory_save_summary_chars.max(32);
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max {
        return normalized;
    }
    let mut out = String::new();
    for ch in normalized.chars().take(max) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn normalize_memory_category(raw: Option<&str>) -> String {
    let category = raw.unwrap_or("fact").trim().to_lowercase();
    match category.as_str() {
        "conversation" | "preference" | "task" | "fact" | "note" => category,
        _ => "note".to_string(),
    }
}

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
        ToolSpec {
            name: MEMORY_STORE_TOOL,
            description: "Store a durable memory note for this chat/session.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Memory content to store."
                    },
                    "category": {
                        "type": "string",
                        "description": "One of: conversation, preference, task, fact, note."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional short summary."
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Relative importance (0.0 to 1.0). Default 0.5."
                    }
                },
                "required": ["content"]
            }),
            side_effect: false,
        },
        ToolSpec {
            name: MEMORY_RECALL_TOOL,
            description:
                "Recall relevant memories from this chat/session using query or recency fallback.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional recall query; if omitted, returns recent memories."
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum memories to return (default 5)."
                    }
                }
            }),
            side_effect: false,
        },
        ToolSpec {
            name: MEMORY_FORGET_TOOL,
            description:
                "Delete memories by explicit ids or query match for this chat (destructive).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Optional explicit memory ids to delete."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query to select memories to delete."
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter when deleting by query."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Limit for query-based deletion (default 5)."
                    }
                }
            }),
            side_effect: true,
        },
    ]
}

pub fn all_tool_names() -> Vec<String> {
    let mut names = all_tool_specs()
        .into_iter()
        .map(|spec| spec.name.to_string())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
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

pub fn is_memory_tool(tool_name: &str) -> bool {
    tool_name.eq_ignore_ascii_case(MEMORY_STORE_TOOL)
        || tool_name.eq_ignore_ascii_case(MEMORY_RECALL_TOOL)
        || tool_name.eq_ignore_ascii_case(MEMORY_FORGET_TOOL)
}

pub async fn execute_memory_tool(
    db: &Database,
    session_id: i64,
    chat_id: i64,
    user_id: i64,
    tool_name: &str,
    args: &Value,
) -> Result<String> {
    match tool_name.to_lowercase().as_str() {
        MEMORY_STORE_TOOL => {
            let content = args
                .get("content")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("memory_store requires string field 'content'"))?
                .trim();
            if content.is_empty() {
                return Err(anyhow!("memory_store content cannot be empty"));
            }

            let category =
                normalize_memory_category(args.get("category").and_then(|value| value.as_str()));
            let importance = args
                .get("importance")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let summary_owned = args
                .get("summary")
                .and_then(|value| value.as_str())
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| summarize_for_memory(content));

            let memory_id = db
                .insert_agent_memory(AgentMemoryInsert {
                    chat_id,
                    user_id: Some(user_id),
                    session_id: Some(session_id),
                    source_role: "assistant",
                    category: &category,
                    content,
                    summary: Some(summary_owned.as_str()),
                    importance,
                })
                .await?;
            Ok(format!(
                "Stored memory id={} category={} importance={:.2}",
                memory_id, category, importance
            ))
        }
        MEMORY_RECALL_TOOL => {
            let limit = args
                .get("limit")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .unwrap_or(5)
                .clamp(1, 20);
            let category_filter =
                normalize_memory_category(args.get("category").and_then(|value| value.as_str()));
            let has_category = args
                .get("category")
                .and_then(|value| value.as_str())
                .is_some();

            let query = args
                .get("query")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let normalized_query = normalize_memory_query(query);
            let rows = if normalized_query.is_empty() {
                db.recent_agent_memories(chat_id, limit).await?
            } else {
                db.search_agent_memories(chat_id, &normalized_query, limit.saturating_mul(3))
                    .await?
                    .into_iter()
                    .map(|value| value.memory)
                    .collect::<Vec<_>>()
            };

            let mut filtered = rows
                .into_iter()
                .filter(|row| !has_category || row.category.eq_ignore_ascii_case(&category_filter))
                .take(limit)
                .collect::<Vec<_>>();

            if filtered.is_empty() {
                return Ok("No memories found.".to_string());
            }

            filtered.sort_by(|a, b| b.id.cmp(&a.id));
            let mut lines = vec![format!("Recalled {} memories:", filtered.len())];
            for memory in filtered {
                let summary = memory
                    .summary
                    .unwrap_or_else(|| summarize_for_memory(&memory.content));
                lines.push(format!(
                    "- id={} [{}|{}] {}",
                    memory.id, memory.category, memory.source_role, summary
                ));
            }
            Ok(lines.join("\n"))
        }
        MEMORY_FORGET_TOOL => {
            let mut ids = BTreeSet::new();
            if let Some(raw_ids) = args.get("ids").and_then(|value| value.as_array()) {
                for id in raw_ids {
                    if let Some(parsed) = id.as_i64() {
                        ids.insert(parsed);
                    }
                }
            }

            let limit = args
                .get("limit")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .unwrap_or(5)
                .clamp(1, 20);
            let category_filter =
                normalize_memory_category(args.get("category").and_then(|value| value.as_str()));
            let has_category = args
                .get("category")
                .and_then(|value| value.as_str())
                .is_some();

            let query = args
                .get("query")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let normalized_query = normalize_memory_query(query);
            if !normalized_query.is_empty() {
                let matches = db
                    .search_agent_memories(chat_id, &normalized_query, limit.saturating_mul(3))
                    .await?;
                for memory in matches.into_iter().map(|value| value.memory) {
                    if !has_category || memory.category.eq_ignore_ascii_case(&category_filter) {
                        ids.insert(memory.id);
                    }
                    if ids.len() >= limit {
                        break;
                    }
                }
            }

            if ids.is_empty() {
                return Err(anyhow!(
                    "memory_forget requires `ids` or a query that matches memories"
                ));
            }

            let ids_to_delete = ids.into_iter().collect::<Vec<_>>();
            let deleted = db.delete_agent_memories(chat_id, &ids_to_delete).await?;
            if deleted == 0 {
                return Ok("No matching memories were deleted.".to_string());
            }
            Ok(format!("Deleted {} memories.", deleted))
        }
        other => Err(anyhow!("Unknown memory tool '{}'", other)),
    }
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

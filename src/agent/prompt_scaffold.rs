use std::fs;
use std::path::Path;

use crate::config::CONFIG;

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut truncated = String::with_capacity(max_chars + 64);
    for ch in input.chars().take(max_chars) {
        truncated.push(ch);
    }
    truncated.push_str("\n\n[Truncated]");
    truncated
}

fn load_optional_file(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_chars(trimmed, CONFIG.agent_prompt_max_file_chars))
}

fn section(title: &str, body: &str) -> String {
    format!("{}\n{}\n", title, body)
}

pub fn build_agent_system_prompt(
    workspace_root: &Path,
    skill_index: &str,
    selected_skill_context: &str,
) -> String {
    let now = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    let mut sections = vec![format!(
        "You are an AI-native Telegram assistant that can reason in multiple steps and use tools.\n\
         Current time: {now}\n\
         Workspace root: {}\n\
         Work strictly inside the workspace and avoid unsafe side effects.\n\
         If you call side-effectful tools (write_file/edit_file/exec), execution may require confirmation.\n\
         Follow selected skills as operational procedures.\n",
        workspace_root.display()
    )];

    if CONFIG.agent_prompt_include_agents {
        let path = workspace_root.join("AGENTS.md");
        let body = load_optional_file(&path)
            .unwrap_or_else(|| "AGENTS.md not found in workspace root.".to_string());
        sections.push(section("Workspace agent guidelines (AGENTS.md):", &body));
    }

    if CONFIG.agent_prompt_include_memory_md {
        let path = workspace_root.join("MEMORY.md");
        let body = load_optional_file(&path)
            .unwrap_or_else(|| "MEMORY.md not found in workspace root.".to_string());
        sections.push(section("Persistent memory notes (MEMORY.md):", &body));
    }

    if CONFIG.agent_prompt_include_skills_index {
        sections.push(section("Skill catalog:", skill_index));
    }

    sections.push(section(
        "Active skill instructions:",
        selected_skill_context,
    ));

    sections.join("\n")
}

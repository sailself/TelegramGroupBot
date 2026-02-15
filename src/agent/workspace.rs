use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tracing::info;

use crate::config::CONFIG;

const DEFAULT_AGENTS_MD: &str = r#"# AGENTS

You are running inside a dedicated agent workspace.

Rules:
- Work only inside this workspace directory.
- Prefer small, reversible file edits.
- Explain what changed and why.
- Keep dangerous operations explicit and minimal.
"#;

const DEFAULT_MEMORY_MD: &str = r#"# MEMORY

Persistent notes for this workspace.

- Store important facts, preferences, and decisions.
- Remove stale or incorrect notes when needed.
"#;

fn absolute_base_workspace_root() -> Result<PathBuf> {
    let configured = &CONFIG.agent_workspace_root;
    if configured.as_os_str().is_empty() {
        return Err(anyhow!(
            "AGENT_WORKSPACE_ROOT cannot be empty. Configure a valid path."
        ));
    }

    if configured.is_absolute() {
        return Ok(configured.clone());
    }

    let cwd = std::env::current_dir().map_err(|err| anyhow!("Failed to read CWD: {}", err))?;
    Ok(cwd.join(configured))
}

fn ensure_text_file(path: &Path, default_content: &str) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    fs::write(path, default_content)
        .map_err(|err| anyhow!("Failed to create '{}': {}", path.display(), err))
}

fn ensure_workspace_markdown_files(workspace_path: &Path) -> Result<()> {
    ensure_text_file(&workspace_path.join("AGENTS.md"), DEFAULT_AGENTS_MD)?;
    ensure_text_file(&workspace_path.join("MEMORY.md"), DEFAULT_MEMORY_MD)?;
    Ok(())
}

fn chat_workspace_folder_name(chat_id: i64) -> String {
    if chat_id < 0 {
        format!("chat_neg_{}", chat_id.saturating_abs())
    } else {
        format!("chat_{}", chat_id)
    }
}

pub fn ensure_base_workspace() -> Result<PathBuf> {
    let base = absolute_base_workspace_root()?;
    fs::create_dir_all(&base).map_err(|err| {
        anyhow!(
            "Failed to create workspace root '{}': {}",
            base.display(),
            err
        )
    })?;
    ensure_workspace_markdown_files(&base)?;
    Ok(base)
}

pub fn ensure_chat_workspace(chat_id: i64) -> Result<PathBuf> {
    let base = ensure_base_workspace()?;
    let workspace_path = if CONFIG.agent_workspace_separate_by_chat {
        let chat_dir = base.join(chat_workspace_folder_name(chat_id));
        fs::create_dir_all(&chat_dir).map_err(|err| {
            anyhow!(
                "Failed to create chat workspace '{}': {}",
                chat_dir.display(),
                err
            )
        })?;
        chat_dir
    } else {
        base
    };

    ensure_workspace_markdown_files(&workspace_path)?;
    Ok(workspace_path)
}

pub fn bootstrap_workspace_on_startup() -> Result<PathBuf> {
    let root = ensure_base_workspace()?;
    info!("Agent workspace root: {}", root.display());
    Ok(root)
}

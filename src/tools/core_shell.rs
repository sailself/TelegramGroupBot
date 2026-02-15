use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Result};
use regex::Regex;
use tokio::process::Command;
use tokio::time::timeout;

use crate::tools::core_filesystem::resolve_workspace_path;

const DEFAULT_DENY_PATTERNS: [&str; 8] = [
    r"\brm\s+-[rf]{1,2}\b",
    r"\bdel\s+/[fq]\b",
    r"\brmdir\s+/s\b",
    r"\b(format|mkfs|diskpart)\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r"\b(shutdown|reboot|poweroff)\b",
    r":\(\)\s*\{.*\};\s*:",
];

fn resolve_working_dir(
    workspace_root: &Path,
    working_dir: Option<&str>,
    restrict_to_workspace: bool,
) -> Result<PathBuf> {
    match working_dir {
        Some(path) => {
            if restrict_to_workspace {
                let resolved = resolve_workspace_path(workspace_root, path)?;
                if !resolved.is_dir() {
                    return Err(anyhow!("Working directory is not a directory: {}", path));
                }
                Ok(resolved)
            } else {
                let candidate = PathBuf::from(path);
                let resolved = if candidate.is_absolute() {
                    candidate
                } else {
                    workspace_root.join(candidate)
                };
                Ok(resolved)
            }
        }
        None => Ok(workspace_root.to_path_buf()),
    }
}

fn command_is_blocked(command: &str, custom_deny_patterns: &[String]) -> Result<bool> {
    let lowered = command.trim().to_lowercase();
    for pattern in DEFAULT_DENY_PATTERNS {
        let regex = Regex::new(pattern)?;
        if regex.is_match(&lowered) {
            return Ok(true);
        }
    }

    for pattern in custom_deny_patterns {
        let trimmed = pattern.trim();
        if trimmed.is_empty() {
            continue;
        }
        let regex = Regex::new(trimmed)
            .map_err(|err| anyhow!("Invalid deny pattern '{}': {}", trimmed, err))?;
        if regex.is_match(&lowered) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn command_references_path_outside_workspace(command: &str, workspace_root: &Path) -> Result<bool> {
    let workspace_root = workspace_root.canonicalize().map_err(|err| {
        anyhow!(
            "Failed to resolve workspace root '{}': {}",
            workspace_root.display(),
            err
        )
    })?;

    if command.contains("../") || command.contains("..\\") {
        return Ok(true);
    }

    let win_regex = Regex::new(r#"[A-Za-z]:\\[^\\\"'\s]+"#)?;
    let posix_regex = Regex::new(r#"(?:^|[\s|>])(/[^\s"'>]+)"#)?;

    for m in win_regex.find_iter(command) {
        let path = PathBuf::from(m.as_str());
        if path.is_absolute() && !path.starts_with(&workspace_root) {
            return Ok(true);
        }
    }

    for captures in posix_regex.captures_iter(command) {
        let Some(group) = captures.get(1) else {
            continue;
        };
        let path = PathBuf::from(group.as_str().trim());
        if path.is_absolute() && !path.starts_with(&workspace_root) {
            return Ok(true);
        }
    }

    Ok(false)
}

pub async fn execute_command(
    workspace_root: &Path,
    command: &str,
    working_dir: Option<&str>,
    timeout_seconds: u64,
    max_output_chars: usize,
    restrict_to_workspace: bool,
    custom_deny_patterns: &[String],
) -> Result<String> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("Command is required"));
    }

    if command_is_blocked(trimmed, custom_deny_patterns)? {
        return Err(anyhow!(
            "Command blocked by safety guard (dangerous pattern detected)"
        ));
    }

    if restrict_to_workspace && command_references_path_outside_workspace(trimmed, workspace_root)?
    {
        return Err(anyhow!(
            "Command blocked by safety guard (path outside workspace detected)"
        ));
    }

    let cwd = resolve_working_dir(workspace_root, working_dir, restrict_to_workspace)?;

    let mut process = if cfg!(target_os = "windows") {
        let mut cmd = Command::new("powershell");
        cmd.arg("-NoProfile").arg("-Command").arg(trimmed);
        cmd
    } else {
        let mut cmd = Command::new("bash");
        cmd.arg("-lc").arg(trimmed);
        cmd
    };

    process.kill_on_drop(true);
    process.current_dir(cwd);

    let output = timeout(Duration::from_secs(timeout_seconds), process.output())
        .await
        .map_err(|_| anyhow!("Command timed out after {} seconds", timeout_seconds))?
        .map_err(|err| anyhow!("Failed to execute command: {}", err))?;

    let mut result = String::new();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.trim().is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.trim().is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("STDERR:\n");
        result.push_str(&stderr);
    }
    if result.is_empty() {
        result.push_str("(no output)");
    }
    if !output.status.success() {
        result.push_str(&format!(
            "\nExit code: {}",
            output.status.code().unwrap_or(-1)
        ));
    }

    if result.chars().count() > max_output_chars {
        let truncated: String = result.chars().take(max_output_chars).collect();
        let remaining = result.chars().count().saturating_sub(max_output_chars);
        return Ok(format!(
            "{}\n... (truncated, {} more chars)",
            truncated, remaining
        ));
    }

    Ok(result)
}
